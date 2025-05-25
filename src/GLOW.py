import sys
sys.path.insert(0,'../')
from Utils.GLOW_Bench import generate_targets_and_RC, kg_metadata
from Utils.evaluate import eval_predictions_Exact, llm_as_judge
from Utils.GNN_KGMeta import calc_acc_gnn, calc_GNN_predictions_acc
from Utils.ollamaAPI import chat
from glow_parser import args
import time
import pandas as pd
import pickle
from tqdm import tqdm

# model_name="qwen3:8b"
model_name = args.llm_model


def Answer_LLM_WOC_QPerPrompt(ds_name, ground_truth_dict, class_dict):
    system_prompt = "You are an expert open world question answer (QA) system."
    predictd_WOC_dict = {}
    predictd_WOC_time_dict = {}
    kg = ds_name.split("-")[0]
    for idx, (k, v) in enumerate(ground_truth_dict.items()):
        target_type = k.split('-')[0]
        col_title = k.split('-')[1]
        possible_predictions_str = None
        if class_dict and class_dict[k]['classes']:
            possible_predictions_str = ",".join(list(class_dict[k]['classes'].values()))
        Answers_lst = []
        Answers_time_lst = []
        Answers_usage_list = []
        for idx, vt in enumerate(tqdm(v["target_txt"])):
            print(f"Q_idx:{idx}/{len(v['target_txt'])}")
            question_messsage = f"""predict the {kg_metadata[kg][0]} {col_title} for the following {target_type} from the {kg_metadata[kg][1]}.
                            {"" if possible_predictions_str is None else "Help: The possible list of " + col_title + "s are [" + possible_predictions_str + "]"}
                            {target_type}: {vt}
                            return the answer only without any context, explaination, thinking or analysis.
                            Answer:
                            """
            # print("question_messsage=",question_messsage)
            # response =chat_engine.chat(question_messsage)
            start_time = time.time()
            response, usage, full_response = chat(model=model_name, prompt_in=question_messsage,
                                                  system_prompt=system_prompt)
            print(f"Answer:{response}")
            Answers_lst.append([vt, response.split("Answer:")[-1].strip()])
            elapsed_time = time.time() - start_time
            Answers_time_lst.append(elapsed_time)
            Answers_usage_list.append(usage)
            # tqdm.update(1)

        predictd_WOC_time_dict[k] = sum(Answers_time_lst)
        print(f"response for {k}={response}")
        try:
            ans_df = pd.DataFrame(Answers_lst)
        except:
            ans_df = pd.DataFrame([['None', 'None']], columns=["target", col_title])  # remove URLs Patterns
        # ans_df['target']=ans_df['target'].apply(lambda x:x.split("/")[-1].replace("_"," "))
        predictd_WOC_dict[k] = (ans_df, response, Answers_usage_list, full_response)
        # print("\n\n")
    return predictd_WOC_dict, predictd_WOC_time_dict


def Answer_LLM_WOC(ds_name, ground_truth_dict, class_dict):
    ################## Questions ###########################
    predictd_WOC_dict = {}
    predictd_WOC_time_dict = {}
    kg = ds_name.split("-")[0]
    system_prompt = "You are an expert open world question answer (QA) system."
    for idx, (k, v) in enumerate(ground_truth_dict.items()):
        start_time = time.time()
        target_type = k.split('-')[0]
        col_title = k.split('-')[1]
        # listOfTargetsStr="\n".join(v["target"].unique().tolist())
        listOfTargetsStr = "\n".join(v["target_txt"].unique().tolist())
        possible_predictions_str = None
        if class_dict and class_dict[k]['classes']:
            possible_predictions_str = ",".join(list(class_dict[k]['classes'].values()))
        question_messsage = f"""predict the {kg_metadata[kg][0]} {col_title} for each {target_type} in the following list of {target_type}s from the {kg_metadata[kg][1]}.
                          return in format: {target_type}||the Prediction per line.
                          return the {target_type} name while replace underscore with space.
                          do not return any context or analysis.
                          {"" if possible_predictions_str is None else "Help: The possible list of " + col_title + "s are [" + possible_predictions_str + "]"}
                          ---------------- list of {target_type}s  ----------------------
                          {listOfTargetsStr}"""
        # print("question_messsage=",question_messsage)
        response, usage, full_response = chat(model=model_name, prompt_in=question_messsage,
                                              system_prompt=system_prompt)
        elapsed_time = time.time() - start_time
        predictd_WOC_time_dict[k] = elapsed_time
        print(f"response for {k}={response}")
        try:
            ans_df = pd.DataFrame([elem.split("||") for elem in response.split("\n")], columns=["target", col_title])
        except:
            ans_df = pd.DataFrame([['None', 'None']], columns=["target", col_title])  # remove URLs Patterns
        ans_df['target'] = ans_df['target'].apply(lambda x: x.split("/")[-1].replace("_", " "))
        predictd_WOC_dict[k] = (ans_df, response, usage, full_response)
        # print("\n\n")
    return predictd_WOC_dict, predictd_WOC_time_dict


def eval_LLM_WOC(ground_truth_dict, predictd_WOC_dict):
    WOC_acc_res = {}
    merged_df_res = {}
    for idx, (k, v) in enumerate(ground_truth_dict.items()):
        col_title = k.split('-')[1]
        predictd_WOC_dict[k][0]['target'] = predictd_WOC_dict[k][0]['target'].apply(lambda x: str(x).strip())
        merged_df = pd.merge(ground_truth_dict[k], predictd_WOC_dict[k][0], left_on='target_txt', right_on='target',
                             how='inner')
        # print(merged_df.columns)
        if len(merged_df) > 0:
            merged_df[col_title + "_txt"] = merged_df[col_title + "_txt"].apply(lambda x: str(x).replace("_", " "))
            merged_df[col_title + "_y"] = merged_df[col_title + "_y"].apply(lambda x: str(x).replace("_", " "))
            ####################### LLM Judge ##########
            pairs = list(zip(list(merged_df[col_title + "_txt"].values), list(merged_df[col_title + "_y"].values)))
            res, response, usage, full_response = llm_as_judge(pairs)
            l1, l2 = zip(*res)
            print(len(res))
            merged_df = merged_df.head(len(res))
            merged_df["is_true_pred"] = list(l1)
            merged_df["pred_similarity_score"] = list(l2)
            WOC_acc_res[k] = [sum(merged_df["is_true_pred"]) / len(merged_df),
                              sum(merged_df["pred_similarity_score"]) / len(merged_df), sum(merged_df["is_true_pred"])]
            merged_df_res[k] = merged_df
        else:
            WOC_acc_res[k] = [0, 0, 0]
            merged_df_res[k] = None
    return WOC_acc_res, merged_df_res


def Answer_LLM_WC_QPerPrompt(ds_name, ground_truth_dict, ground_truth_context_dict, class_dict, GNN_Answers_dict=None):
    predictd_WC_dict = {}
    predictd_WC_time_dict = {}
    kg = ds_name.split("-")[0]
    system_prompt = "You are an expert open world question answer (QA) system."
    for idx, (k, v) in enumerate(ground_truth_dict.items()):
        target_type = k.split('-')[0]
        col_title = k.split('-')[1]
        target_lst = v["target"].unique().tolist()
        possible_predictions_str = None
        if class_dict[k]['classes']:
            possible_predictions_str = ",".join(list(class_dict[k]['classes'].values()))

        GNN_Answers_str = None
        if GNN_Answers_dict and k in GNN_Answers_dict.keys():
            GNN_Answers_str = str(GNN_Answers_dict[k])

        answers_lst = []
        usage_lst = []
        times_lst = []
        for idx, vt in enumerate(tqdm(target_lst)):
            # print(f"Q_idx:{idx}//{len(target_lst)}")
            question_messsage = f"""predict the {kg_metadata[kg][0]} {col_title} for the following {target_type} from Linked the {kg_metadata[kg][1]}.
 use the given information context per {target_type} to refine your prediction.
 {"" if possible_predictions_str is None else "Help: The possible list of " + col_title + "s are [" + possible_predictions_str + "]"}
 {"" if GNN_Answers_str is None else f"Verfy the following answer generated using a Graph Neural Network Model: {GNN_Answers_dict[k][idx]} ."}
 The Question Main Entity: {target_type}-> {vt}
 {"" if ground_truth_context_dict is None else "The main entity Related list of Information in format of (relation,value):" + ground_truth_context_dict[k][vt]}
 do not return any context or analysis.
 Answer:"""
            # print("question_messsage=",question_messsage)
            start_time = time.time()
            response, usage, full_reponse = chat(model=model_name, prompt_in=question_messsage,
                                                 system_prompt=system_prompt)
            answers_lst.append([vt, response.split("Answer:")[-1].strip()])
            usage_lst.append(usage)
            print(f"{response}")
            elapsed_time = time.time() - start_time
            times_lst.append(elapsed_time)
            # tqdm.update(1)

        predictd_WC_time_dict[k] = sum(times_lst)
        try:
            ans_df = pd.DataFrame(answers_lst, columns=["target", col_title])  # remove URLs Patterns
        except:
            ans_df = pd.DataFrame([['None', 'None']], columns=["target", col_title])  # remove URLs Patterns
        predictd_WC_dict[k] = (ans_df, usage_lst)
    return predictd_WC_dict, predictd_WC_time_dict


def Answer_LLM_WC(ds_name, ground_truth_dict, ground_truth_context_dict, class_dict, GNN_Answers_dict=None):
    ################## Questions ###########################
    predictd_WC_dict = {}
    predictd_WC_time_dict = {}
    kg = ds_name.split("-")[0]
    system_prompt = "You are an expert open world question answer (QA) system."
    for idx, (k, v) in enumerate(ground_truth_dict.items()):
        start_time = time.time()
        target_type = k.split('-')[0]
        col_title = k.split('-')[1]

        target_lst = v["target"].unique().tolist()
        target_title_df = ground_truth_dict[k][['target', 'target_txt']].drop_duplicates()
        target_title_dict = dict(zip(target_title_df['target'], target_title_df['target_txt']))
        targets_context_str = ""
        print("target_lst=", target_lst)
        for p in target_lst:
            if p in ground_truth_context_dict[k].keys():
                targets_context_str += f"{target_type}:{target_title_dict[p]} <tab> {target_type} Information: {ground_truth_context_dict[k][p]}\n"
            else:
                targets_context_str += f"{target_type}:{target_title_dict[p]} <tab> {target_type}\n"
        possible_predictions_str = None
        if class_dict[k]['classes']:
            possible_predictions_str = ",".join(list(class_dict[k]['classes'].values()))
        GNN_Answers_str = None
        if GNN_Answers_dict and k in GNN_Answers_dict.keys():
            GNN_Answers_str = str(GNN_Answers_dict[k])
        question_messsage = f"""predict the {kg_metadata[kg][0]}  {col_title} for each {target_type} in the following list of {target_type}s from {kg_metadata[kg][1]}. use the given information context per {target_type} to refine your prediction.
                          {"" if possible_predictions_str is None else "Help: The possible list of " + col_title + "s are [" + possible_predictions_str + "]"}
                          {"" if GNN_Answers_str is None else f"Verfy the following list of answers generated using a Graph Neural Network Model for the given list of {target_type}s. the answers are mapped to questions one to one. GNN Answers={GNN_Answers_str} ."}
                          do not return any context or analysis.
                          ---------------- {target_type}s and Their Information ----------------------
                          {targets_context_str}.\n
                          ---return answer in format  {target_type} name||the Prediction per line.
                          ---Note: return the {target_type} Name and replace underscore with space.
                          Answer:"""
        # print("question_messsage=",question_messsage)
        response, usage, full_reponse = chat(model=model_name, prompt_in=question_messsage, system_prompt=system_prompt)
        elapsed_time = time.time() - start_time
        predictd_WC_time_dict[k] = elapsed_time
        response = response.split("Here is the output:\n")[-1].split("the requested format:\n")[-1].replace("\n\n",
                                                                                                            "\n").replace(
            "```", "").replace("\n\n", "\n")
        print(f"response for {k}={response}")
        try:
            ans_df = pd.DataFrame([elem.split("||") for elem in response.split("\n")],
                                  columns=["target", col_title])  # remove URLs Patterns
        except:
            ans_df = pd.DataFrame([['None', 'None']], columns=["target", col_title])  # remove URLs Patterns
        ans_df['target'] = ans_df['target'].apply(lambda x: x.split("/")[-1].replace("_", " "))
        predictd_WC_dict[k] = (ans_df, usage)
    return predictd_WC_dict, predictd_WC_time_dict


def get_runs_mean_and_std(runs_lst, Accuracy=True):
    join_by_k = {}
    runs_acc_res_mean = {}
    runs_acc_res_std = {}
    elem_id = 0 if Accuracy else 1
    for run in runs_lst:
        for k, v in run[elem_id].items():
            if k not in join_by_k.keys():
                join_by_k[k] = [v]
            else:
                join_by_k[k].append(v)
    for k in join_by_k.keys():
        join_by_k_df = pd.DataFrame(join_by_k[k], columns=['EM', 'HM', 'count'] if Accuracy else ['T'])
        runs_acc_res_mean[k] = join_by_k_df.mean().tolist()
        runs_acc_res_std[k] = join_by_k_df.std(ddof=1).tolist()
    return runs_acc_res_mean, runs_acc_res_std, join_by_k


def save_pipeline_results(predictd_results_dic, predictd_time_dict, merged_results_dic, run_lst_dic, model_name,
                          KG="BioKG"):
    with open(f'../GLOW-QA_dataset/{KG}_GNNOnly_Materlized_answers_dict.pickle', 'wb') as file:
        pickle.dump(GNNOnly_Materlized_answers_dict, file)
    with open(f'../GLOW-QA_dataset/{KG}_{model_name}_predictd_results_dic.pickle', 'wb') as file:
        pickle.dump(predictd_results_dic, file)

    with open(f'../GLOW-QA_dataset/{KG}_{model_name}_predictd_time_dict.pickle', 'wb') as file:
        pickle.dump(predictd_time_dict, file)

    with open(f'../GLOW-QA_dataset/{KG}_{model_name}_merged_results_dic.pickle', 'wb') as file:
        pickle.dump(merged_results_dic, file)

    with open(f'../GLOW-QA_dataset/{KG}_{model_name}_run_lst_dic.pickle', 'wb') as file:
        pickle.dump(run_lst_dic, file)


def calc_tokens(piplines):
    for pipline in piplines:
        total_tokens = []
        for k in pipline:
            print(f'k={k}')
            tokens_lst = [elem['eval_count'] for elem in pipline[k][2]]
            total_tokens.append(sum(tokens_lst) / len(tokens_lst))
        print(f'avg_tokens={sum(total_tokens) / len(total_tokens)}')


def calc_answer_time(pred_lst, time_lst):
    for idx, time_pipline in enumerate(time_lst):
        time_lst = []
        for k in time_pipline:
            print(k)
            time_lst.append(time_pipline[k] / len(pred_lst[idx][k][0]))
            print(time_pipline[k], len(pred_lst[idx][k][0]))
        print(sum(time_lst) / len(time_lst))
        print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>")


if __name__ == '__main__':
    predictd_LLMOnly_dict, predictd_WOC_dict, predictd_WC_dict, predictd_GNN_dict, predictd_LLMGNN_dict = {}, {}, {}, {}, {}
    predictd_LLMOnly_time_dict, predictd_WOC_time_dict, predictd_WC_time_dict, predictd_GNN_time_dict, predictd_LLMGNN_time_dict = {}, {}, {}, {}, {}
    merged_LLMOnly_df, merged_WOC_df, merged_WC_df, GNN_merged_df, LLMGNN_merged_df = {}, {}, {}, {}, {}
    LLMOnly_runs_lst, WOC_runs_lst, WC_runs_lst, GNNOnly_run_lst, GNN_run_lst, GNNGRAG_run_lst = {}, {}, {}, {}, {}, {}
    #######################
    print(f"Args={args}")
    Glow_datasets = ['biokg', 'linkedIMDB', 'yago4-person', 'yago4-creativwork', 'crunchbase', 'arxiv2023', 'ogbnArxiv',
                     'ogbnProduct']
    if args.dataset_name == 'All':
        datasets = Glow_datasets
    else:
        datasets = [args.dataset_name]
    for ds in datasets:
        print(f"############## Glow Dataset:{ds} ########################")
        ground_truth_dict, pred_class_dict, ground_truth_context_dict = generate_targets_and_RC(ds,
                                                                                                load_from_disk=args.generate_GlowBench == False)
        WOC_runs_lst, WC_runs_lst, LLMOnly_runs_lst = [], [], []
        GNNOnly_run_lst = []
        GNN_run_lst = []
        GNNGRAG_run_lst = []
        for i in range(args.runs):
            print(f"####RUN:{i}#######")
            if args.glow_v in ['LLMOnly', 'All']:
                # ['L', 'GN', 'G', 'N', 'LLM', 'All']
                print("###########Start LLM Only Prompts########")
                predictd_LLMOnly_dict, predictd_LLMOnly_time_dict = Answer_LLM_WOC_QPerPrompt(ds, ground_truth_dict,
                                                                                              None)
                LLMOnly_acc_res, merged_LLMOnly_df = eval_predictions_Exact(ground_truth_dict, predictd_LLMOnly_dict)
                LLMOnly_runs_lst.append([LLMOnly_acc_res, predictd_LLMOnly_time_dict])
            if args.glow_v in ['L', 'All']:
                print("###########Start GLOW-L Prompts########")
                predictd_WOC_dict, predictd_WOC_time_dict = Answer_LLM_WOC_QPerPrompt(ds, ground_truth_dict,
                                                                                      pred_class_dict)
                WOC_acc_res, merged_WOC_df = eval_predictions_Exact(ground_truth_dict, predictd_WOC_dict)
                WOC_runs_lst.append([WOC_acc_res, predictd_WOC_time_dict])
            if args.glow_v in ['G', 'All']:
                print("###########Start GLOW-G Prompts########")
                predictd_WC_dict, predictd_WC_time_dict = Answer_LLM_WC_QPerPrompt(ds, ground_truth_dict,
                                                                                   ground_truth_context_dict,
                                                                                   pred_class_dict)
                WC_acc_res, merged_WC_df = eval_predictions_Exact(ground_truth_dict, predictd_WC_dict)
                WC_runs_lst.append([WC_acc_res, predictd_WC_time_dict])
            if args.glow_v in ['N', 'GN', 'All']:
                print("###########Start GNN Only ########")
                GNNOnly_acc_res_dict, predictd_GNN_dict = {}, {}
                GNNOnly_answers_dict = {}
                GNNOnly_times_dict, predictd_GNN_time_dict = {}, {}
                for k, v in ground_truth_dict.items():
                    GNNOnly_acc_res_dict[k], GNNOnly_answers_dict[k], GNNOnly_times_dict[k] = calc_GNN_predictions_acc(
                        ground_truth_dict, pred_class_dict, k=k)
                GNNOnly_run_lst.append([GNNOnly_acc_res_dict, GNNOnly_times_dict, GNNOnly_answers_dict])
                GNNOnly_Materlized_answers_dict = {}
                for k in ground_truth_dict.keys():
                    ground_truth_dict[k]['GNN_pred'] = ground_truth_dict[k]['target'].apply(
                        lambda x: pred_class_dict[k]['classes'][GNNOnly_answers_dict[k][x]] if GNNOnly_answers_dict[k][
                                                                                                   x] in
                                                                                               pred_class_dict[k][
                                                                                                   'classes'].keys() else None)
                    GNNOnly_Materlized_answers_dict[k] = list(ground_truth_dict[k]['GNN_pred'].values)
            if args.glow_v in ['N', 'All']:
                print("###########Start GLOW-N Prompts########")
                predictd_GNN_dict, predictd_GNN_time_dict = Answer_LLM_WC_QPerPrompt(ds,
                                                                                     ground_truth_dict,
                                                                                     None,
                                                                                     pred_class_dict,
                                                                                     GNNOnly_Materlized_answers_dict)
                GNN_acc_res, GNN_merged_df = eval_predictions_Exact(ground_truth_dict, predictd_GNN_dict)
                GNNGRAG_run_lst.append([GNN_acc_res, predictd_GNN_time_dict])

            if args.glow_v in ['GN', 'All']:
                print("###########Start GLOW-GN Prompts########")
                predictd_LLMGNN_dict, predictd_LLMGNN_time_dict = Answer_LLM_WC_QPerPrompt(ds, ground_truth_dict,
                                                                                           ground_truth_context_dict,
                                                                                           pred_class_dict,
                                                                                           GNNOnly_Materlized_answers_dict)
                LLMGNN_acc_res, LLMGNN_merged_df = eval_predictions_Exact(ground_truth_dict, predictd_LLMGNN_dict)
                GNNGRAG_run_lst.append([LLMGNN_acc_res, predictd_LLMGNN_time_dict])

        ############### Save The Pipeline Answers ################
        predictd_results_dic = {"predictd_LLMOnly_dict": predictd_LLMOnly_dict,
                                "predictd_LLMOnly_dict": predictd_WOC_dict,
                                "predictd_WC_dict": predictd_WC_dict, "predictd_GNN_dict": predictd_GNN_dict,
                                "predictd_LLMGNN_dict": predictd_LLMGNN_dict}
        predictd_time_dict = {"predictd_LLMOnly_time_dict": predictd_LLMOnly_time_dict,
                              "predictd_WOC_time_dict": predictd_WOC_time_dict,
                              "predictd_WC_time_dict": predictd_WC_time_dict,
                              "predictd_GNN_time_dict": predictd_GNN_time_dict,
                              "predictd_LLMGNN_time_dict": predictd_LLMGNN_time_dict}
        merged_results_dic = {"LLMOnly_merged_df": merged_LLMOnly_df, "merged_WOC_df": merged_WOC_df,
                              "merged_WC_df": merged_WC_df,
                              "GNN_merged_df": GNN_merged_df, "LLMGNN_merged_df": LLMGNN_merged_df}
        run_lst_dic = {"LLMOnly_runs_lst": LLMOnly_runs_lst, "WOC_runs_lst": WOC_runs_lst, "WC_runs_lst": WC_runs_lst,
                       "GNNOnly_runs_lst": GNNOnly_run_lst, "GNN_run_lst": GNN_run_lst,
                       "GNNGRAG_run_lst": GNNGRAG_run_lst}
        save_pipeline_results(predictd_results_dic, predictd_time_dict, merged_results_dic, run_lst_dic,
                              args.llm_model.split("/")[-1], KG=ds)
        ################ Print Results ################
        final_results_dict = {}
        for k in LLMOnly_acc_res:
            final_results_dict[k] = []
            for res_dic in [LLMOnly_acc_res, WOC_acc_res, WC_acc_res, GNNOnly_acc_res_dict, GNN_acc_res,
                            LLMGNN_acc_res]:
                final_results_dict[k].append(res_dic[k][0])
        res_df = pd.DataFrame(final_results_dict).transpose()
        res_df.columns = ['LLMOnly', 'Glow-L', 'Glow-G', 'GNN', 'Glow-N', 'Glow-GN']
        res_df["dataset_name"] = res_df.index
        res_df = res_df[['dataset_name', 'LLMOnly', 'Glow-L', 'Glow-G', 'GNN', 'Glow-N', 'Glow-GN']]
        res_df.to_csv(f'../GLOW-QA_dataset/{ds}_{args.llm_model.split("/")[-1]}_result.csv', index=False)
        print("Accuracy:\n", res_df)
        try:
            pred_lst = [predictd_LLMOnly_dict, predictd_WOC_dict, predictd_WC_dict, predictd_GNN_dict,
                        predictd_LLMGNN_dict]
            time_lst = [predictd_LLMOnly_time_dict, predictd_WOC_time_dict, predictd_WC_time_dict,
                        predictd_GNN_time_dict, predictd_LLMGNN_time_dict]
            print("Time:\n", calc_answer_time(pred_lst, time_lst))
            print("Tokens Count:\n", calc_tokens(
                piplines=[predictd_LLMOnly_dict, predictd_WOC_dict, predictd_WC_dict, predictd_GNN_dict,
                          predictd_LLMGNN_dict]))
        except:
            continue

from Utils.evaluate import eval_LLM_WC
from Utils.GLOW_Bench import generate_biokg_targets, generate_biokg_target_context,drug_dict_pred_class,drug_dict_pred
from Utils.evaluate import eval_predictions_Exact
from Utils.GNN_KGMeta import calc_acc_gnn
from glow_parser import args
import pandas as pd
model_names=["gpt-4o-mini","deepseek-chat","deepseek-r1","granite3.3","gemini-1.5-flash","llama3.2:3b","qwen3:8b","phi4-mini"]
model_name="qwen3:8b"

def Answer_LLM_WOC_QPerPrompt(ground_truth_dict,class_dict):
  ################## Questions ###########################
  predictd_WOC_dict={}
  predictd_WOC_time_dict={}
  for idx, (k,v) in enumerate(ground_truth_dict.items()):
    target_type=k.split('-')[0]
    col_title=k.split('-')[1]
    # listOfTargetsStr="\n".join(v["target"].unique().tolist())
    listOfTargetsStr="\n".join(v["target_txt"].unique().tolist())
    possible_predictions_str=None
    if class_dict and class_dict[k]['classes']:
       possible_predictions_str=",".join(list(class_dict[k]['classes'].values()))
    Answers_lst=[]
    Answers_time_lst=[]
    Answers_usage_list=[]
    for idx , vt in enumerate(v["target_txt"]):
      print(f"Q_idx:{idx}/{len(v['target_txt'])}")
      question_messsage=f"""predict the chemical {col_title} for the following {target_type} from the biomedical knowledge graph (BioKG).
                            {"" if possible_predictions_str is None else "Help: The possible list of "+col_title+"s are ["+possible_predictions_str+"]"}
                            {target_type}: {vt}
                            return the answer only without any context, explaination, thinking or analysis.
                            Answer:
                            """
      print("question_messsage=",question_messsage)
      # response =chat_engine.chat(question_messsage)
      start_time = time.time()
      response,usage,full_response=chat(model=model_name,prompt_in=question_messsage)
      print(f"Answer:{response}")
      Answers_lst.append([vt,response.split("Answer:")[-1].strip()])
      elapsed_time =time.time()-start_time
      Answers_time_lst.append(elapsed_time)
      Answers_usage_list.append(usage)

    predictd_WOC_time_dict[k]=sum(Answers_time_lst)
    print(f"response for {k}={response}")
    try:
      ans_df=pd.DataFrame(Answers_lst)
    except:
      ans_df=pd.DataFrame([['None','None']],columns=["target",col_title]) # remove URLs Patterns
    # ans_df['target']=ans_df['target'].apply(lambda x:x.split("/")[-1].replace("_"," "))
    predictd_WOC_dict[k]=(ans_df,response,Answers_usage_list,full_response)
    # print("\n\n")
  return predictd_WOC_dict,predictd_WOC_time_dict

import time
def Answer_LLM_WOC(ground_truth_dict,class_dict):
  ################## Questions ###########################
  predictd_WOC_dict={}
  predictd_WOC_time_dict={}
  for idx, (k,v) in enumerate(ground_truth_dict.items()):
    start_time = time.time()
    target_type=k.split('-')[0]
    col_title=k.split('-')[1]
    # listOfTargetsStr="\n".join(v["target"].unique().tolist())
    listOfTargetsStr="\n".join(v["target_txt"].unique().tolist())
    possible_predictions_str=None
    if class_dict and class_dict[k]['classes']:
       possible_predictions_str=",".join(list(class_dict[k]['classes'].values()))
    question_messsage=f"""predict the chemical {col_title} for each {target_type} in the following list of {target_type}s from the biomedical knowledge graph (BioKG).
                          return in format: {target_type}||the Prediction per line.
                          return the {target_type} name while replace underscore with space.
                          do not return any context or analysis.
                          {"" if possible_predictions_str is None else "Help: The possible list of "+col_title+"s are ["+possible_predictions_str+"]"}
                          ---------------- list of {target_type}s  ----------------------
                          {listOfTargetsStr}"""
    print("question_messsage=",question_messsage)
    # response =chat_engine.chat(question_messsage)
    response,usage,full_response=chat(model=model_name,prompt_in=question_messsage)
    elapsed_time =time.time()-start_time
    predictd_WOC_time_dict[k]=elapsed_time
    print(f"response for {k}={response}")
    try:
      ans_df=pd.DataFrame([elem.split("||") for elem in response.split("\n")],columns=["target",col_title])
    except:
      ans_df=pd.DataFrame([['None','None']],columns=["target",col_title]) # remove URLs Patterns
    ans_df['target']=ans_df['target'].apply(lambda x:x.split("/")[-1].replace("_"," "))
    predictd_WOC_dict[k]=(ans_df,response,usage,full_response)
    # print("\n\n")
  return predictd_WOC_dict,predictd_WOC_time_dict

def eval_LLM_WOC(ground_truth_dict,predictd_WOC_dict):
  WOC_acc_res={}
  merged_df_res={}
  for idx, (k,v) in enumerate(ground_truth_dict.items()):
    col_title=k.split('-')[1]
    predictd_WOC_dict[k][0]['target']=predictd_WOC_dict[k][0]['target'].apply(lambda x:str(x).strip())
    merged_df=pd.merge(ground_truth_dict[k], predictd_WOC_dict[k][0], left_on='target_txt',right_on='target', how='inner')
    # print(merged_df.columns)
    if len(merged_df)>0:
      merged_df[col_title+"_txt"]=merged_df[col_title+"_txt"].apply(lambda x: str(x).replace("_"," "))
      merged_df[col_title+"_y"]=merged_df[col_title+"_y"].apply(lambda x: str(x).replace("_"," "))
      ####################### LLM Judge ##########
      pairs=list(zip(list(merged_df[col_title+"_txt"].values),list(merged_df[col_title+"_y"].values)))
      res,response,usage,full_response=llm_as_judge(pairs)
      l1,l2=zip(*res)
      print(len(res))
      merged_df=merged_df.head(len(res))
      merged_df["is_true_pred"]=list(l1)
      merged_df["pred_similarity_score"]=list(l2)
      # print("################merged_df###############",merged_df)
      # merged_df["is_true_pred"]= merged_df.apply(lambda row: compare_strings(row,col_title+"_txt",col_title+"_y"),axis=1)
      # merged_df["pred_similarity_score"]= merged_df.apply(lambda row: get_similarity_score(row,col_title+"_txt",col_title+"_y"),axis=1)
      WOC_acc_res[k]=[sum(merged_df["is_true_pred"])/len(merged_df),sum(merged_df["pred_similarity_score"])/len(merged_df), sum(merged_df["is_true_pred"])]
      merged_df_res[k]=merged_df
    else:
      WOC_acc_res[k]=[0,0, 0]
      merged_df_res[k]=None
    # print(f"""{k} task: String Matching Accuracy={sum(merged_df["is_true_pred"])/len(merged_df)},Pred Similarity Score ={sum(merged_df["pred_similarity_score"])/len(merged_df)}, True answers Count={sum(merged_df["is_true_pred"])}""")
    # print(merged_df)
  return WOC_acc_res,merged_df_res

def Answer_LLM_WC_QPerPrompt(ground_truth_dict,ground_truth_context_dict,class_dict,GNN_Answers_dict=None):
  ################## Questions ###########################
  predictd_WC_dict={}
  predictd_WC_time_dict={}
  for idx, (k,v) in enumerate(ground_truth_dict.items()):
    target_type=k.split('-')[0]
    col_title=k.split('-')[1]
    target_lst=v["target"].unique().tolist()
    target_title_df=ground_truth_dict[k][['target','target_txt']].drop_duplicates()
    target_title_dict=dict(zip(target_title_df['target'],target_title_df['target_txt']))
    targets_context_str=""
    print("target_lst=",target_lst)
    for p in target_lst:
      if p in ground_truth_context_dict[k].keys():
        targets_context_str+=f"{target_type}:{target_title_dict[p]} <tab> {target_type} Information: {ground_truth_context_dict[k][p]}\n"
      else:
          targets_context_str+=f"{target_type}:{target_title_dict[p]} <tab> {target_type}\n"
    possible_predictions_str=None
    if class_dict[k]['classes']:
       possible_predictions_str=",".join(list(class_dict[k]['classes'].values()))

    GNN_Answers_str=None
    if GNN_Answers_dict and k in GNN_Answers_dict.keys():
      GNN_Answers_str=str(GNN_Answers_dict[k])

    answers_lst=[]
    usage_lst=[]
    times_lst=[]
    for idx,vt in enumerate(target_lst):
      print(f"Q_idx:{idx}/{len(target_lst)}")
      question_messsage=f"""predict the chemical {col_title} for the following {target_type} from Linked the biomedical knowledge graph (BioKG).
                            use the given information context per {target_type} to refine your prediction.
                            {"" if possible_predictions_str is None else "Help: The possible list of "+col_title+"s are ["+possible_predictions_str+"]"}
                            {"" if GNN_Answers_str is None else f"Verfy the following answer generated using a Graph Neural Network Model: {GNN_Answers_dict[k][idx]} ."}
                            {target_type}: {vt} \n
                            Entiy Related list of Information in format of (relation,value) : 
                                {ground_truth_context_dict[k][vt]} \n
                            do not return any context or analysis.
                            Answer:"""
      print("question_messsage=",question_messsage)
      start_time = time.time()
      response,usage,full_reponse=chat(model=model_name,prompt_in=question_messsage)
      answers_lst.append([vt,response.split("Answer:")[-1].strip()])
      usage_lst.append(usage)
      print(f"{response}")
      elapsed_time = time.time()-start_time
      times_lst.append(elapsed_time)

    predictd_WC_time_dict[k]=sum(times_lst)
    try:
      ans_df=pd.DataFrame(answers_lst,columns=["target",col_title]) # remove URLs Patterns
    except:
      ans_df=pd.DataFrame([['None','None']],columns=["target",col_title]) # remove URLs Patterns

    # ans_df['target']=ans_df['target'].apply(lambda x:x.split("/")[-1].replace("_"," "))
    predictd_WC_dict[k]=(ans_df,response,usage_lst,full_response)
    # print("\n\n")
  return predictd_WC_dict,predictd_WC_time_dict

def Answer_LLM_WC(ground_truth_dict,ground_truth_context_dict,class_dict,GNN_Answers_dict=None):
  ################## Questions ###########################
  predictd_WC_dict={}
  predictd_WC_time_dict={}
  for idx, (k,v) in enumerate(ground_truth_dict.items()):
    start_time = time.time()
    target_type=k.split('-')[0]
    col_title=k.split('-')[1]

    target_lst=v["target"].unique().tolist()
    target_title_df=ground_truth_dict[k][['target','target_txt']].drop_duplicates()
    target_title_dict=dict(zip(target_title_df['target'],target_title_df['target_txt']))
    targets_context_str=""
    print("target_lst=",target_lst)
    for p in target_lst:
      if p in ground_truth_context_dict[k].keys():
        targets_context_str+=f"{target_type}:{target_title_dict[p]} <tab> {target_type} Information: {ground_truth_context_dict[k][p]}\n"
      else:
          targets_context_str+=f"{target_type}:{target_title_dict[p]} <tab> {target_type}\n"
    possible_predictions_str=None
    if class_dict[k]['classes']:
       possible_predictions_str=",".join(list(class_dict[k]['classes'].values()))
    GNN_Answers_str=None
    if GNN_Answers_dict and k in GNN_Answers_dict.keys():
      GNN_Answers_str=str(GNN_Answers_dict[k])
    question_messsage=f"""predict the chemical {col_title} for each {target_type} in the following list of {target_type}s from Linked the biomedical knowledge graph (BioKG). use the given information context per {target_type} to refine your prediction.
                          {"" if possible_predictions_str is None else "Help: The possible list of "+col_title+"s are ["+possible_predictions_str+"]"}
                          {"" if GNN_Answers_str is None else f"Verfy the following list of answers generated using a Graph Neural Network Model for the given list of {target_type}s. the answers are mapped to questions one to one. GNN Answers={GNN_Answers_str} ."}
                          do not return any context or analysis.
                          ---------------- {target_type}s and Their Information ----------------------
                          {targets_context_str}.\n
                          ---return answer in format  {target_type} name||the Prediction per line.
                          ---Note: return the {target_type} Name and replace underscore with space.
                          \\\Answer:"""
    print("question_messsage=",question_messsage)
    # response =chat_engine.chat(question_messsage)
    response,usage,full_reponse=chat(model=model_name,prompt_in=question_messsage)
    elapsed_time = time.time()-start_time
    predictd_WC_time_dict[k]=elapsed_time
    response=response.split("Here is the output:\n")[-1].split("the requested format:\n")[-1].replace("\n\n","\n").replace("```","").replace("\n\n","\n")
    print(f"response for {k}={response}")
    try:
      ans_df=pd.DataFrame([elem.split("||") for elem in response.split("\n")],columns=["target",col_title]) # remove URLs Patterns
    except:
      ans_df=pd.DataFrame([['None','None']],columns=["target",col_title]) # remove URLs Patterns
    ans_df['target']=ans_df['target'].apply(lambda x:x.split("/")[-1].replace("_"," "))
    predictd_WC_dict[k]=(ans_df,response,usage,full_response)
    # print("\n\n")
  return predictd_WC_dict,predictd_WC_time_dict


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


import pickle


def save_dataset(KG="BioKG"):
    with open(f'GLOW-QA_dataset/{KG}_ground_truth_dict.pickle', 'wb') as file:
        pickle.dump(ground_truth_dict, file)
    with open(f'GLOW-QA_dataset/{KG}_ground_truth_context_dict.pickle', 'wb') as file:
        pickle.dump(ground_truth_context_dict, file)
    with open(f'GLOW-QA_dataset/{KG}_ground_dict_pred_class.pickle', 'wb') as file:
        pickle.dump(drug_dict_pred_class, file)
    with open(f'GLOW-QA_dataset/{KG}_GNN_Materlized_answers_dict.pickle', 'wb') as file:
        pickle.dump(GNN_Materlized_answers_dict, file)

    predictd_results_dic = {"predictd_LLMOnly_dict": predictd_LLMOnly_dict, "predictd_LLMOnly_dict": predictd_WOC_dict,
                            "predictd_WC_dict": predictd_WC_dict, "predictd_LLMGNN_dict": predictd_LLMGNN_dict}
    with open(f'GLOW-QA_dataset/{KG}_{model_name}_predictd_results_dic.pickle', 'wb') as file:
        pickle.dump(predictd_results_dic, file)

    predictd_time_dict = {"predictd_LLMOnly_time_dict": predictd_LLMOnly_time_dict,
                          "predictd_WOC_time_dict": predictd_WOC_time_dict,
                          "predictd_WC_time_dict": predictd_WC_time_dict,
                          "predictd_LLMGNN_time_dict": predictd_LLMGNN_time_dict}
    with open(f'GLOW-QA_dataset/{KG}_{model_name}_predictd_time_dict.pickle', 'wb') as file:
        pickle.dump(predictd_time_dict, file)

    merged_results_dic = {"LLMOnly_merged_df": None, "merged_WOC_df": merged_WOC_df, "merged_WC_df": merged_WC_df,
                          "LLMGNN_merged_df": LLMGNN_merged_df}
    with open(f'GLOW-QA_dataset/{KG}_{model_name}_merged_results_dic.pickle', 'wb') as file:
        pickle.dump(merged_results_dic, file)

    run_lst_dic = {"LLMOnly_runs_lst": LLMOnly_runs_lst, "WOC_runs_lst": WOC_runs_lst, "WC_runs_lst": WC_runs_lst,
                   "GNN_runs_lst": GNN_run_lst, "GNNGRAG_run_lst": GNNGRAG_run_lst}
    with open(f'GLOW-QA_dataset/{KG}_{model_name}_run_lst_dic.pickle', 'wb') as file:
        pickle.dump(run_lst_dic, file)


save_dataset(KG="BioKG")


def calc_tokens():
    for pipline in [predictd_LLMOnly_dict, predictd_WOC_dict, predictd_WC_dict, predictd_LLMGNN_dict]:
        total_tokens = []
        for k in pipline:
            print(f'k={k}')
            tokens_lst = [elem['eval_count'] for elem in pipline[k][2]]
            total_tokens.append(sum(tokens_lst) / len(tokens_lst))
        print(f'avg_tokens={sum(total_tokens) / len(total_tokens)}')


def calc_answer_time():
    pred_lst = [predictd_LLMOnly_dict, predictd_WOC_dict, predictd_WC_dict, predictd_LLMGNN_dict]
    time_lst = [predictd_LLMOnly_time_dict, predictd_WOC_time_dict, predictd_WC_time_dict, predictd_LLMGNN_time_dict]
    for idx, time_pipline in enumerate(time_lst):
        time_lst = []
        for k in time_pipline:
            print(k)
            time_lst.append(time_pipline[k] / len(pred_lst[idx][k][0]))
            print(time_pipline[k], len(pred_lst[idx][k][0]))
        print(sum(time_lst) / len(time_lst))
        print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>")
if __name__ == '__main__':

    ground_truth_dict,dict_pred=generate_biokg_targets(targets_count=100,offset=0,filter_year=2006,dict_pred=drug_dict_pred,class_dict=drug_dict_pred_class)
    ground_truth_context_dict=generate_biokg_target_context(ground_truth_dict,dict_pred)
    Runs=args.runs
    WOC_runs_lst,WC_runs_lst,LLMOnly_runs_lst=[],[],[]
    for i in range(Runs):
        predictd_LLMOnly_dict,predictd_LLMOnly_time_dict=Answer_LLM_WOC_QPerPrompt(ground_truth_dict,None)
        predictd_WOC_dict,predictd_WOC_time_dict=Answer_LLM_WOC_QPerPrompt(ground_truth_dict,drug_dict_pred_class)
        predictd_WC_dict,predictd_WC_time_dict=Answer_LLM_WC_QPerPrompt(ground_truth_dict,ground_truth_context_dict,drug_dict_pred_class)

        LLMOnly_acc_res,merged_LLMOnly_df=eval_predictions_Exact(ground_truth_dict,predictd_LLMOnly_dict)
        LLMOnly_runs_lst.append([LLMOnly_acc_res,predictd_LLMOnly_time_dict])
        WOC_acc_res,merged_WOC_df=eval_predictions_Exact(ground_truth_dict,predictd_WOC_dict)
        WOC_runs_lst.append([WOC_acc_res,predictd_WOC_time_dict])
        WC_acc_res,merged_WC_df=eval_predictions_Exact(ground_truth_dict,predictd_WC_dict)
        WC_runs_lst.append([WC_acc_res,predictd_WC_time_dict])

        GNN_run_lst=[]
        for i in range(Runs):
          GNN_acc_res_dict={}
          GNN_answers_dict={}
          GNN_times_dict={}
          for k,v in ground_truth_dict.items():
              GNN_acc_res_dict[k], GNN_answers_dict[k],GNN_times_dict[k]=calc_GNN_predictions_acc(ground_truth_dict,drug_dict_pred_class,k=k)
          GNN_run_lst.append([GNN_acc_res_dict,GNN_times_dict,GNN_answers_dict])
          GNN_Materlized_answers_dict={}
          for k in ground_truth_dict.keys():
            ground_truth_dict[k]['GNN_pred']=ground_truth_dict[k]['target'].apply(lambda x:drug_dict_pred_class[k]['classes'][GNN_answers_dict[k][x]] if GNN_answers_dict[k][x] in drug_dict_pred_class[k]['classes'].keys() else None)
            GNN_Materlized_answers_dict[k]=list(ground_truth_dict[k]['GNN_pred'].values)
          GNN_Materlized_answers_dict


        GNNGRAG_run_lst=[]
        for i in range(Runs):
          predictd_LLMGNN_dict,predictd_LLMGNN_time_dict=Answer_LLM_WC_QPerPrompt(ground_truth_dict,ground_truth_context_dict,drug_dict_pred_class,GNN_Materlized_answers_dict)
          LLMGNN_acc_res,LLMGNN_merged_df=eval_predictions_Exact(ground_truth_dict,predictd_LLMGNN_dict)
          GNNGRAG_run_lst.append([LLMGNN_acc_res,predictd_LLMGNN_time_dict])

    final_results_dict={}
    for k in LLMOnly_acc_res:
        final_results_dict[k]=[]
        for res_dic in [LLMOnly_acc_res,WOC_acc_res,WC_acc_res,GNN_acc_res_dict,LLMGNN_acc_res]:
            final_results_dict[k].append(res_dic[k][0])
    pd.DataFrame(final_results_dict).transpose()
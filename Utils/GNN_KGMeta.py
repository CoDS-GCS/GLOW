from Utils.utils import executeSparqlQuery
from Utils.GLOW_Bench import KGMETA_SPARQLendpointUrl
from Utils.ollamaAPI import dopost
import json
import pandas as pd
model_targetEdge='KINGDOM'
def getGNN_Model(model_targetEdge):
    Kgnet_query=f"""select distinct ?mid ?targetEdge ?testAccuracy ?filters
    from <http://kgnet/>
    {{
    ?mid <kgnet:GMLModel/taskSubgraph/targetEdge> ?targetEdge.
    ?mid <kgnet:GMLModel/taskSubgraph/labelCount> ?labelCount.
    ?mid <kgnet:GMLModel/testAccuracy> ?testAccuracy.
    #?mid <kgnet:GMLModel/taskSubgraph/filters> ?filters.
    filter(contains(str(?targetEdge),'{model_targetEdge}')).
    #?mid ?p ?o.
    ?task <kgnet:GMLTask/modelID> ?mid .
    ?task <kgnet:GMLTask/targetNode>  <biokg:drug> .
    }}"""
    df_models=executeSparqlQuery(Kgnet_query,KGMETA_SPARQLendpointUrl)
    return df_models.values


def execute_gnn_inference_query( model_id="a8b68f681d8005dd63b675c2a758599eb3d93eef20ba16d5c992a04612c1a3aa",
  sparqlEndpointURL="http://206.12.98.118:8890/sparql",
  named_graph_uri="https://yago-knowledge.org",
  RDFEngine="OpenlinkVirtuoso",target_node_lst=None):
  if not target_node_lst:
    target_node_lst=["http://yago-knowledge.org/resource/Bethany_Firth",
    "http://yago-knowledge.org/resource/Nerea_Camacho",
    "http://yago-knowledge.org/resource/Arike_Ogunbowale",
    "http://yago-knowledge.org/resource/Braden_Mann",
    "http://yago-knowledge.org/resource/Bridget_Carleton",
    "http://yago-knowledge.org/resource/Chloë_Grace_Moretz",
    "http://yago-knowledge.org/resource/Daniel_Tobin",
    "http://yago-knowledge.org/resource/Dwayne_Haskins",
    "http://yago-knowledge.org/resource/Hayfa_Sdiri",
    "http://yago-knowledge.org/resource/Hiyori_Kon",
    "http://yago-knowledge.org/resource/Josh_Allen_(linebacker)",
    "http://yago-knowledge.org/resource/Kalani_Brown",
    "http://yago-knowledge.org/resource/Kelsey_Mitchell",
    "http://yago-knowledge.org/resource/Khadija_Shaw",
    "http://yago-knowledge.org/resource/Lydia_Ko",
    "http://yago-knowledge.org/resource/McKenzie_Milton",
    "http://yago-knowledge.org/resource/Michael_Dickson_(American_football)",
    "http://yago-knowledge.org/resource/Nicola_Tustain",
    "http://yago-knowledge.org/resource/Sabrina_Ionescu",
    "http://yago-knowledge.org/resource/Aleksandr_Pronkov",
    "http://yago-knowledge.org/resource/Alexander_Bolshunov",
    "http://yago-knowledge.org/resource/Alexey_Bugaev",
    "http://yago-knowledge.org/resource/Anastasia_Tatareva",
    "http://yago-knowledge.org/resource/Anna_Grimaldi",
    "http://yago-knowledge.org/resource/Auston_Matthews",
    "http://yago-knowledge.org/resource/Bella_Thorne",
    "http://yago-knowledge.org/resource/Brooke_Henderson",
    "http://yago-knowledge.org/resource/Caleb_Swanigan",
    "http://yago-knowledge.org/resource/Callum_Mills",
    "http://yago-knowledge.org/resource/Camila_Cabello"]

  api_url="http://206.12.102.12:64648/"
  if not model_id:
    model_id="a8b68f681d8005dd63b675c2a758599eb3d93eef20ba16d5c992a04612c1a3aa"
  model_url =api_url + "gml_inference/mid/" + str(model_id).replace('"', "")
  if not sparqlEndpointURL:
    sparqlEndpointURL="http://206.12.98.118:8890/sparql"
  if not named_graph_uri:
    named_graph_uri="https://yago-knowledge.org"
  if not RDFEngine:
    RDFEngine="OpenlinkVirtuoso"
  params = {"model_id": model_id,
              "named_graph_uri": named_graph_uri,
              "sparqlEndpointURL":sparqlEndpointURL,
              "RDFEngine":RDFEngine,
              "dataQuery": [],
              "targetNodesList":target_node_lst,
              "TOSG_Pattern": "d1h1",
              "topk": 1}
  print("GNN Inf Query:",params)
  inference_req_dic = json.loads(dopost(model_url, params).decode("utf-8"))
  Inference_Times=inference_req_dic['Inference_Times']
  del inference_req_dic['Inference_Times']
  return inference_req_dic,Inference_Times


def calc_acc_gnn(gnn_answers_dict,ground_truth_dict,ground_truth_col_name='given_award',ground_truth_k='person-occupation'):
  GNN_acc_res={}
  GNN_Answer_df = pd.DataFrame(list(gnn_answers_dict.items()), columns=['target', 'pred'])
  pred_col='pred'
  GNN_merged_df=pd.merge(ground_truth_dict[ground_truth_k], GNN_Answer_df, left_on='target',right_on='target', how='inner')
  ####################### LLM Judge ##########
  # print("GNN_merged_df:",GNN_merged_df.columns)

  true_count=0
  for idx, row in GNN_merged_df.iterrows():
    if str(row[pred_col]).strip().lower() in str(row[ground_truth_col_name]).strip().lower():
      true_count+=1


  # pairs=list(zip(list(GNN_merged_df[ground_truth_col_name].values),list(GNN_merged_df[pred_col].values)))
  # # res=llm_as_judge(pairs)
  # res,response,usage,full_response=llm_as_judge(pairs)
  # l1,l2=zip(*res)
  # l1=list(l1)
  # l2=list(l2)
  # for i in np.arange(0,len(GNN_merged_df)-len(l1)):
  #         l1.append(0)
  #         l2.append(0)
  # GNN_merged_df["is_true_pred"]=list(l1)
  # GNN_merged_df["pred_similarity_score"]=list(l2)
  ###############################
  # GNN_merged_df["is_true_pred"]= GNN_merged_df.apply(lambda row: compare_strings(row,ground_truth_col_name,pred_col),axis=1)
  # GNN_merged_df["pred_similarity_score"]= GNN_merged_df.apply(lambda row: get_similarity_score(row,ground_truth_col_name,pred_col),axis=1)
  # GNN_acc_res[ground_truth_k]=[sum(GNN_merged_df["is_true_pred"])/len(GNN_merged_df),sum(GNN_merged_df["pred_similarity_score"])/len(GNN_merged_df), sum(GNN_merged_df["is_true_pred"])]

  GNN_acc_res[ground_truth_k]=[true_count/len(GNN_merged_df),true_count/len(GNN_merged_df), true_count]
  return GNN_acc_res,GNN_merged_df

Infernce_query={
  "model_id": "a8b68f681d8005dd63b675c2a758599eb3d93eef20ba16d5c992a04612c1a3aa",
  "named_graph_uri": "https://yago-knowledge.org",
  "sparqlEndpointURL": "http://206.12.98.118:8890/sparql",
  "RDFEngine": "OpenlinkVirtuoso",
  "dataQuery": [ ],
  "targetNodesList":["http://yago-knowledge.org/resource/Bethany_Firth",
  "http://yago-knowledge.org/resource/Nerea_Camacho",
  "http://yago-knowledge.org/resource/Arike_Ogunbowale",
  "http://yago-knowledge.org/resource/Braden_Mann",
  "http://yago-knowledge.org/resource/Bridget_Carleton",
  "http://yago-knowledge.org/resource/Chloë_Grace_Moretz",
  "http://yago-knowledge.org/resource/Daniel_Tobin",
  "http://yago-knowledge.org/resource/Dwayne_Haskins",
  "http://yago-knowledge.org/resource/Hayfa_Sdiri",
  "http://yago-knowledge.org/resource/Hiyori_Kon",
  "http://yago-knowledge.org/resource/Josh_Allen_(linebacker)",
  "http://yago-knowledge.org/resource/Kalani_Brown",
  "http://yago-knowledge.org/resource/Kelsey_Mitchell",
  "http://yago-knowledge.org/resource/Khadija_Shaw",
  "http://yago-knowledge.org/resource/Lydia_Ko",
  "http://yago-knowledge.org/resource/McKenzie_Milton",
  "http://yago-knowledge.org/resource/Michael_Dickson_(American_football)",
  "http://yago-knowledge.org/resource/Nicola_Tustain",
  "http://yago-knowledge.org/resource/Sabrina_Ionescu",
  "http://yago-knowledge.org/resource/Aleksandr_Pronkov",
  "http://yago-knowledge.org/resource/Alexander_Bolshunov",
  "http://yago-knowledge.org/resource/Alexey_Bugaev",
  "http://yago-knowledge.org/resource/Anastasia_Tatareva",
  "http://yago-knowledge.org/resource/Anna_Grimaldi",
  "http://yago-knowledge.org/resource/Auston_Matthews",
  "http://yago-knowledge.org/resource/Bella_Thorne",
  "http://yago-knowledge.org/resource/Brooke_Henderson",
  "http://yago-knowledge.org/resource/Caleb_Swanigan",
  "http://yago-knowledge.org/resource/Callum_Mills",
  "http://yago-knowledge.org/resource/Camila_Cabello"],
  "TOSG_Pattern": "d1h1",
  "topk": 1
}


def calc_acc_gnn(gnn_answers_dict,ground_truth_dict,ground_truth_col_name='given_award',ground_truth_k='person-occupation'):
  GNN_acc_res={}
  GNN_Answer_df = pd.DataFrame(list(gnn_answers_dict.items()), columns=['target', 'pred'])
  pred_col='pred'
  GNN_merged_df=pd.merge(ground_truth_dict[ground_truth_k], GNN_Answer_df, left_on='target',right_on='target', how='inner')
  ####################### LLM Judge ##########
  # print("GNN_merged_df:",GNN_merged_df.columns)

  true_count=0
  for idx, row in GNN_merged_df.iterrows():
    if str(row[pred_col]).strip().lower() in str(row[ground_truth_col_name]).strip().lower():
      true_count+=1
  GNN_acc_res[ground_truth_k]=[true_count/len(GNN_merged_df),true_count/len(GNN_merged_df), true_count]
  return GNN_acc_res,GNN_merged_df

def calc_GNN_predictions_acc(ground_truth_dict,dict_pred_class,k=None):
  if not k:
    k='person-given_award'
  target_node_lst=list(ground_truth_dict[k]['target'].unique())
  if k in dict_pred_class.keys():
    model_id=dict_pred_class[k]['mid']
    if model_id is None:
        return [0,0,0]
    else:
      gnn_answers_dict,gnn_answers_time_dict=execute_gnn_inference_query(model_id=model_id,target_node_lst=target_node_lst)
      ground_truth_col_name=list(ground_truth_dict[k].columns)[1]
      print('ground_truth_col_name=',ground_truth_col_name)
      # print('k=',k)
      GNN_acc_res,GNN_merged_df=calc_acc_gnn(gnn_answers_dict,ground_truth_dict,ground_truth_col_name=ground_truth_col_name,ground_truth_k=k)
      return list(GNN_acc_res.values())[0],gnn_answers_dict,gnn_answers_time_dict
  else:
     return [0,0,0],None,None
import matplotlib.pyplot as plt
import numpy as np
def visualize_acc(WC_acc_res,WOC_acc_res,GNN_acc_res,LLMGNN_acc_res,ground_truth_dict):
  groups=list(WC_acc_res.keys())
  subgroups=['LLM_ExactMatch','LLM_SubCateogryMatch']
  KG_RAGs=["LLM","PQ-Graph_RAG","PQ-GNN_RAG","PQ-GRAPH_GNN_RAG"]
  data = {
      KG_RAGs[0]:[[v[0],v[1]-v[0]] for k,v in WOC_acc_res.items()],
      KG_RAGs[1]: [[v[0],v[1]-v[0]] for k,v in WC_acc_res.items()],
      KG_RAGs[2]:[[v[0],v[1]-v[0]] for k,v in GNN_acc_res.items()],
      KG_RAGs[3]:[[v[0],v[1]-v[0]] for k,v in LLMGNN_acc_res.items()],
  }
  # print(data)
  # Parameters for plotting
  bar_width = 0.20  # Width of each stacked bar group
  x = np.arange(len(groups))  # X positions for each group
  # Create figure and axis
  fig, ax = plt.subplots(figsize=(10, 6))
  colors=['#2b8ced','#b0d3f5','#13e873','#89f5b1','#ed8013','#f2b374','#de122a','#ed8c97']
  # Plot each category as a stacked bar for each group
  for i, (category, group_data) in enumerate(data.items()):
      print(category)
      # Offset each category position by the bar width
      position = x + i * bar_width
      # print(position)
      # Initialize bottom positions for stacking
      bottom = np.zeros(len(groups))
      # Plot each subgroup in the stack
      for j,subgroup in enumerate(subgroups):
          # Extract the subgroup data for each group
          values = [group[j] for group in group_data]
          values=[elem if elem >0 else 0 for elem in values]
          print(values)
          bars=ax.bar(position, values, bar_width, bottom=bottom ,color=colors[i*2+j])
          for baridx,bar in enumerate(bars):
            yval_pos = 0.03+bottom[baridx]+bar.get_height()/2
            yval = bar.get_height()
            if (yval==0 and j==0) or yval>0:
              plt.text(bar.get_x() + bar.get_width()/2, yval_pos, str(yval)[:4], ha='center', va='top')
          bottom += values  # Update bottom to stack the next subgroup

  # Configure the x-axis with the group labels
  ax.set_xticks(x + bar_width)
  ax.set_yticks(np.arange(0,1.1,0.1))
  ax.set_xticklabels(["\n".join(str(elem).split("-")) for elem in groups])
  ax.set_xlabel("Predictive Query Examples")
  ax.set_ylabel("Accuracy")
  # Custom legend
  custom_labels=[]
  for RAGP in KG_RAGs:
    custom_labels.append(RAGP)
    # custom_labels.append(RAGP+"_EM")
    # custom_labels.append(RAGP+"_HM")
  # print(custom_labels)
  handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10) for color in colors[0::2]]
  plt.legend(handles, custom_labels, loc='upper center', ncols=4,bbox_to_anchor=(0.5, 1.1))
  plt.title(f"""The Accuracy of ChatGPT-4o-mini answers for {len(ground_truth_dict)} predictive queries with {len(ground_truth_dict[list(ground_truth_dict.keys())[0]]["target"])} targets on BioKG""",y=1.13)
  plt.tight_layout()
  plt.show()
  return data

def visualize_time(WC_time_res,WOC_time_res,GNN_time_res,LLMGNN_time_res,ground_truth_dict):
  groups=list(WC_time_res.keys())
  subgroups=[0]
  KG_RAGs=["LLM","PQ-Graph_RAG","PQ-GNN_RAG","PQ-GRAPH_GNN_RAG"]
  data = {
      KG_RAGs[0]:list(WOC_time_res.values()),
      KG_RAGs[1]: list(WC_time_res.values()),
      KG_RAGs[2]:list(GNN_time_res.values()),
      KG_RAGs[3]:list(LLMGNN_time_res.values()) }
  # print(data)
  # Parameters for plotting
  bar_width = 0.20  # Width of each stacked bar group
  x = np.arange(len(groups))  # X positions for each group
  # Create figure and axis
  fig, ax = plt.subplots(figsize=(10, 6))
  colors=['#2b8ced','#b0d3f5','#13e873','#89f5b1','#ed8013','#f2b374','#de122a','#ed8c97']
  # Plot each category as a stacked bar for each group
  max_time=0
  for i, (category, group_data) in enumerate(data.items()):
      print(category)
      # Offset each category position by the bar width
      position = x + i * bar_width
      # print(position)
      # Initialize bottom positions for stacking
      bottom = np.zeros(len(groups))
      # Plot each subgroup in the stack
      for j,subgroup in enumerate(subgroups):
          # Extract the subgroup data for each group
          values = [group for group in group_data]
          max_time=max(max_time,max(values))
          values=[elem if elem >0 else 0 for elem in values]
          print(values)
          bars=ax.bar(position, values, bar_width, bottom=bottom ,color=colors[i*2+j])
          for baridx,bar in enumerate(bars):
            yval_pos = 0.03+bottom[baridx]+bar.get_height()/2
            yval = bar.get_height()
            if (yval==0 and j==0) or yval>0:
              plt.text(bar.get_x() + bar.get_width()/2, yval_pos, str(yval)[:4], ha='center', va='top')
          bottom += values  # Update bottom to stack the next subgroup

  # Configure the x-axis with the group labels
  ax.set_xticks(x + bar_width)
  ax.set_yticks(np.arange(0,max_time+10,10))
  ax.set_xticklabels(["\n".join(str(elem).split("-")) for elem in groups])
  ax.set_xlabel("Predictive Query Examples")
  ax.set_ylabel("QA Time in Seconds")
  # Custom legend
  custom_labels=[]
  for RAGP in KG_RAGs:
    custom_labels.append(RAGP)
    # custom_labels.append(RAGP+"_EM")
    # custom_labels.append(RAGP+"_HM")
  # print(custom_labels)
  handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10) for color in colors[0::2]]
  plt.legend(handles, custom_labels, loc='upper center', ncols=4,bbox_to_anchor=(0.5, 1.1))
  plt.title(f"""The Query Time of ChatGPT-4o-mini answers for {len(ground_truth_dict)} predictive queries with {len(ground_truth_dict[list(ground_truth_dict.keys())[0]]["target"])} targets on BioKG""",y=1.13)
  plt.tight_layout()
  plt.show()
  return data

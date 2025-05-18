import argparse
parser = argparse.ArgumentParser(description='GLOW-QA')
parser.add_argument('--llm_model', type=str, default='qwen3:8b',choices=["gpt-4o-mini","deepseek-chat","deepseek-r1","granite3.3","gemini-1.5-flash","llama3.2:3b","qwen3:8b","phi4-mini"])
parser.add_argument('--dataset_name', type=str, default='biokg', choices=['biokg','linkedIMDB','YAGO4','crunchBase','arxiv2023','ogbn_arxiv','amazon-product'])
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--top-k', type=int, default=3)
parser.add_argument('--glow-v', type=str, default='All',choices=['L','GN','G','N','LLM','All'])
args = parser.parse_args()


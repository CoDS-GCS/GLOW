import argparse
parser = argparse.ArgumentParser(description='GLOW-QA')
parser.add_argument('--llm_model', type=str, default='qwen3:8b',choices=["gpt-4o-mini","deepseek-chat","deepseek-r1","granite3.3","gemini-1.5-flash","llama3.2:3b","qwen3:8b","phi4-mini"])
parser.add_argument('--dataset_name', type=str, default='ogbnArxiv', choices=['biokg','linkedIMDB','yago4-person','yago4-creativwork','crunchbase','arxiv2023','ogbnArxiv','ogbnProduct'])
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--glow_v', type=str, default='All',choices=['LLMOnly','L','GN','G','N','All'])
parser.add_argument('--api_key', type=str, default=None)
args = parser.parse_args()


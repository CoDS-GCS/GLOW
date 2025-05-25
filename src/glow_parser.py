import argparse
parser = argparse.ArgumentParser(description='GLOW-QA')
parser.add_argument('--llm_model', type=str, default='hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_M',choices=["gpt-4o-mini","deepseek-chat","deepseek-r1","granite3.3","gemini-1.5-flash","llama3.2:3b","qwen3:8b","phi4-mini","hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_M"], help="The LLM version of the model")
parser.add_argument('--dataset_name', type=str, default='biokg', choices=['All','biokg','linkedIMDB','yago4-person','yago4-creativwork','crunchbase','arxiv2023','ogbnArxiv','ogbnProduct'], help="The GLOW-QA subet of questions")
parser.add_argument('--runs', type=int, default=1, help='number of runs')
parser.add_argument('--top_k', type=int, default=3 , help="The GNN top-k answers")
parser.add_argument('--glow_v', type=str, default='All',choices=['LLMOnly','L','GN','G','N','All'])
parser.add_argument('--api_key', type=str, default=None, help="The LLM API key")
parser.add_argument('--generate_GlowBench', type=bool, default=False,help="Genarte the dataset fro KG or load the pre-extrated vesrion")
args = parser.parse_args()


import argparse
parser = argparse.ArgumentParser(description='GLOW-QA')
parser.add_argument('--llm_model', type=str, default='qwen')
parser.add_argument('--dataset_name', type=str, default='arxiv_2023')
parser.add_argument('--runs', type=int, default=3)
args = parser.parse_args()


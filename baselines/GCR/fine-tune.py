import sys
import os
import torch
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from peft import AutoPeftModelForCausalLM
import torch.utils
import torch.utils.data
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
import logging
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
import datasets
# from src import utils

datasets.disable_progress_bar()
# import dotenv
from accelerate import Accelerator
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

def path_to_string(path: list) -> str:
    # print("path=",path)
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"

    return result.strip()

tokenizer=None
def input_formatter(example):
        global tokenizer
        chunks = []
        for i in range(len(example["q_entity"])):
            question = example["question"][i]
            start_node = example["q_entity"][i]
            anser_node = example["a_entity"][i]
            ground_paths = example["ground_truth_paths"][i]
            if not question.endswith("?"):
                question += "?"
            raw_input = ZERO_SHOT_PROMPT.format(
                question=question, entities=",".join(start_node)
            )
            # Split ground paths into multiple samples
            if len(ground_paths) > 0:
                for path in ground_paths:
                    if len(path) == 0:
                        continue
                    ground_path_string= f"{PATH_START_TOKEN}{path_to_string(path)}{PATH_END_TOKEN}"
                    # ground_path_string= f"{PATH_START_TOKEN}{'->'.join(path)}{PATH_END_TOKEN}"
                    # the last entity in the path is always the answer
                    path_answer = path[-1][-1].strip()
                    response = ANS_TEMPLATE.format(
                        reasoning_path=ground_path_string, answer=path_answer
                    )
                    chat = [
                        {"role": "user", "content": raw_input},
                        {"role": "assistant", "content": response},
                    ]
                    final_input = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=False
                    )
                    chunks.append(final_input)
        return {"text": chunks}


PATH_START_TOKEN = "<PATH>"
PATH_END_TOKEN = "</PATH>"

HF_TOKEN = os.getenv("HF_TOKEN")
N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)
ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question:
{question}
# Topic entities:
{entities}
"""
ANS_TEMPLATE = """# Reasoning Path:
{reasoning_path}
# Answer:
{answer}"""

g
@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    n_path_per_sample: int = field(
        default=10, metadata={"help": "Number of paths to sample"}
    )
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4bit"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8bit"})
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "attn implementation"})
    response_template: Optional[str] = field(default="[/INST]", metadata={"help": "Response template"})


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="saved_models/llama2_align",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=N_CPUS)


def train(args):
    # parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    # script_args, training_args = parser.parse_args_into_dataclasses()
    script_args=args
    global tokenizer
    global model
    global trainer
    # Load models
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        # attn_implementation=script_args.attn_implementation,
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=script_args.load_in_8bit,
        # quantization_config=quantization_config,
        # device=torch.device("cuda:1"),
        # device_map={"": Accelerator().local_process_index},
    )
    model.to('cuda')
    model.config.use_cache = False
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        token=HF_TOKEN,
    )

    # Add new tokens
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token # tokenizer.unk_token for LLAMA-2-7b-chat-hf
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    # Add new tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<PAD>'
    special_tokens_dict['additional_special_tokens'] = [PATH_START_TOKEN, PATH_END_TOKEN]
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    data_list = [
        datasets.load_from_disk(data_path) for data_path in script_args.data_path_list
    ]
    dataset = datasets.concatenate_datasets(data_list)
    train_dataset = dataset.map(
        input_formatter,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=N_CPUS,
    )

    print(train_dataset[0])

    # Prepare instruct tuning
    response_template = script_args.response_template
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer, mlm=False
    )
    sft_cfg = SFTConfig(
        # args.to_dict(),
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
        max_steps = args.max_steps ,
        report_to="none",
        # callbacks=[TrainOnStartCallback()],
    )


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        # processing_class=tokenizer,
        args=sft_cfg,
        data_collator=data_collator,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    # trainer.train()

    trainer.save_model(args.output_dir+"_PEFT")
    tokenizer.save_pretrained(args.output_dir+"_PEFT")

def save_full_model(args):
    model2 = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir+"_PEFT",
        # "/content/graph-constrained-reasoning/trainer_output/checkpoint-500",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16
        )
    model2 = model2.merge_and_unload()
    tokenizer2 = AutoTokenizer.from_pretrained(args.output_dir+"_PEFT")
    # tokenizer2 = AutoTokenizer.from_pretrained("/content/graph-constrained-reasoning/trainer_output/checkpoint-500")
    model2.save_pretrained(args.output_dir)
    tokenizer2.save_pretrained(args.output_dir)

if __name__ == "__main__":
    class Args():
        def __init__(self):
          self.DATASET_LIST=["../data/shortest_path_index/RoG-webqsp/train"]
          # Full
          self.BATCH_SIZE=10
          self.USE_PEFT=True
          self.EPOCH=3
          self.lora_r=8
          self.lora_alpha=16
          self.lora_dropout=0.05
          self.GRADIENT_CHECKPOINTING=False
          self.GRADIENT_ACCUMULATION_STEPS=16
          self.auto_find_batch_size=False
          self.overwrite_output_dir=True
          self.CONFIG="accelerate_configs/deepspeed_zero3.yaml"
          # self.MODEL_PATH="Qwen/Qwen3-8B"
          # self.MODEL_PATH="Qwen/Qwen3-1.7B"
          # self.ATTN_IMP="flash_attention_2"
          # self.RESPONSE_TEMPLATE="<|im_start|>assistant"
          # CONFIG="accelerate_configs/multi_gpu.yaml"

          self.MODEL_PATH="ibm-granite/granite-3.3-8b-instruct"
          # self.ATTN_IMP="flash_attention_2"
          self.RESPONSE_TEMPLATE="<|start_of_role|>assistant<|end_of_role|>"
          CONFIG="accelerate_configs/multi_gpu.yaml"

          # self.MODEL_PATH="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
          # # # self.ATTN_IMP="flash_attention_2"
          # self.RESPONSE_TEMPLATE="<｜Assistant｜>"

          # MODEL_PATH=Qwen/Qwen2-1.5B-Instruct
          # ATTN_IMP=flash_attention_2
          # RESPONSE_TEMPLATE="<|im_start|>assistant"
          # CONFIG="accelerate_configs/multi_gpu.yaml"

          # self.MODEL_PATH="Qwen/Qwen3-8B"
          # # ATTN_IMP=flash_attention_2
          # self.RESPONSE_TEMPLATE="<|im_start|>assistant"
          # # CONFIG="accelerate_configs/deepspeed_zero3.yaml"

          # MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
          self.ATTN_IMP="flash_attention_2"
          # RESPONSE_TEMPLATE="[/INST]"
          # CONFIG="accelerate_configs/deepspeed_zero3.yaml"

          self.SAVE_PATH="../save_models/GCR-"+self.MODEL_PATH.split("/")[-1]
          self.SAVE_NAME=self.MODEL_PATH.split("/")[-1]
          self.data_path_list= self.DATASET_LIST
          self.model_name_or_path=self.MODEL_PATH
          self.output_dir=self.SAVE_PATH
          self.use_peft=self.USE_PEFT
          self.bf16=True
          self.num_train_epochs=self.EPOCH
          self.per_device_train_batch_size=self.BATCH_SIZE
          self.per_device_eval_batch_size=1
          self.gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS
          self.eval_strategy="no"
          self.save_strategy="no"
          self.save_steps=100
          self.save_total_limit=1
          self.learning_rate=2e-5
          self.weight_decay=0.
          self.warmup_ratio=0.03
          self.lr_scheduler_type="cosine"
          self.logging_steps=1
          self.tf32=False
          self.report_to=None
          self.gradient_checkpointing=self.GRADIENT_CHECKPOINTING
          self.auto_find_batch_size=self.auto_find_batch_size
          self.neftune_noise_alpha=5
          self.attn_implementation=self.ATTN_IMP
          self.response_template=self.RESPONSE_TEMPLATE
          self.run_name=self.SAVE_NAME
          self.load_in_4bit=False
          self.load_in_8bit=False
          self.resume_from_checkpoint=False
          self.max_steps = 500
        def to_dict(self):
            return self.__dict__

    args=Args()
    print(args.__dict__)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    train(args)
    save_full_model(args)








import requests
import json


def dopost(url, json_body):
    try:
        response = requests.post(url, json=json_body)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        print('dopost request Error', e)

default_model_options= {
        'temperature': 1,  # Control creativity/randomness
        'top_p': 0.9,
        'top_k': 40,
        'num_predict': 2000,
        'repeat_penalty': 1.1
    }
models_dict={"deepseek-r1":default_model_options,
            "qwen2.5:7b":default_model_options,
            "qwen2.5:1.5b":default_model_options,
            "qwen2.5:3b":default_model_options,
            "llama3.2:3b":default_model_options,
            "qwen:7b":default_model_options,
            "qwen3:8b":default_model_options,
            "phi4-mini":default_model_options,
            "granite3.3":default_model_options,
            "GCR-Qwen3-1.7B_Q5_K_M":default_model_options,
            "GCR-DeepSeek-R1-Distill-Qwen-1.5B_Q5_K_M":default_model_options,
            "GCR-Llama-3.1-8B-Instruct_Q5_K_M":default_model_options,
            "GCR-granite-3.3-8b-instruct_Q5_K_M":default_model_options,
            "GCR-Meta-Llama-3.1-8B-Instruct_q8:latest":default_model_options,
            "GCR-Qwen2-0.5B-Instruct_f32:latest":default_model_options,
            "hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_M":default_model_options,
            "hf.co/unsloth/Qwen3-32B-GGUF:Q4_K_M":default_model_options,
            'hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_XL':{"temperature":0.6,"top_p":0.95,"min_p":0.01,"seed":3407,'num_predict': 2000}
             }

def query_ollama_dopost(model, prompt, system_prompt=None, temperature=1):
    # url = "http://192.168.41.218:11434/api/generate"
    # url = "http://206.12.96.43:11434/api/generate"
    dict_ollama_api = {"gpu8": "http://206.12.96.43:11434/api/generate",
                       "gpu16": "http://206.12.92.147:11434/api/generate"}

    url = dict_ollama_api["gpu16"]
    headers = {"Content-Type": "application/json"}
    if system_prompt:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}"
    print("ollama prompt", prompt)
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}


def query_ollama_client(model, prompt, system_prompt=None,assistant_prompt=None,temperature=None):
    from ollama import Client
    # url = "http://192.168.41.218:11434"
    # url = "http://206.12.96.43:11434"
    dict_ollama_api = {"gpu8": "http://206.12.96.43:11434",
                       "gpu16": "http://206.12.92.147:11434"}
    url = dict_ollama_api["gpu16"]
    headers = {"Content-Type": "application/json"}
    client = Client(
        host=url,
        headers=headers
    )
    messages = [
        {
            'role': 'system',
            'content': system_prompt if system_prompt else '',
        },
        {
            'role': 'user',
            'content': prompt,
        },
        {
            'role': 'assistant',
            'content': assistant_prompt if assistant_prompt else '',
        },
    ]
    print("\nmessages=", messages)
    options=models_dict[model]
    if temperature:
        options["temperature"]=temperature
    try:
        response = client.chat(model=model, messages=messages, options=options)
        return json.loads(response.model_dump_json())
    except client.ResponseError as e:
        return {"error": f"{client.ResponseError}"}

def open_ai_request(model, prompt, system_prompt=None,assistant_prompt=None,temperature=None,inference_api=None):
    import openai
    dict_ollama_api = {"gpu8": "http://206.12.96.43:11434",
                       "gpu16": "http://206.12.92.147:22101"}

    if inference_api:
        url=inference_api
    else:
        url = dict_ollama_api["gpu16"]

    client = openai.OpenAI(
        base_url=url,  # "http://<Your api-server IP>:port"
        api_key="sk-no-key-required"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': system_prompt if system_prompt else '',
            },
            {
                'role': 'user',
                'content': prompt,
            },
            {
                'role': 'assistant',
                'content': assistant_prompt if assistant_prompt else '',
            },
        ]
    )
    try:
        return json.loads(completion.model_dump_json())
    except client.ResponseError as e:
        return {"error": f"{client.ResponseError}"}

def chat(model="o1-mini", prompt_in="", key="", system_prompt=None,assistant_prompt=None,temperature=None,use_ollama=True):
    if model in ["o1-mini", "gpt-4o-mini"]:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_in
                        },
                    ],
                }
            ]
        )
        return response.choices[0].message.content, response.usage, response
    elif model in models_dict :

        if use_ollama:
            response = query_ollama_client(model, prompt_in, system_prompt,assistant_prompt,temperature)
            usage_keys = ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count',
                          'eval_duration']
            return response['message']['content'].split("</think>")[-1].replace("\n", ""), {key: response[key] for key in
                                                                                            usage_keys}, response
        else:
            response = open_ai_request(model, prompt_in, system_prompt, assistant_prompt, temperature, inference_api=None)
            usage_keys = ['completion_tokens', 'prompt_tokens', 'total_tokens']
            dict_res = {key: response["usage"][key] for key in usage_keys}
            timing_keys = ['prompt_n', 'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second', 'predicted_n',
                           'predicted_ms', 'predicted_per_token_ms', 'predicted_per_second']
            for key in timing_keys:
                dict_res[key] = response["timings"][key]
            return response["choices"][0]["message"]["content"].split("</think>")[-1].replace("\n", ""), dict_res, response
    # elif model=="gpt-4o-mini":
    #   import openai
    #   import os
    #   from llama_index.core import Settings
    #   from llama_index.llms.openai import OpenAI
    #   from llama_index.core.memory import ChatMemoryBuffer
    #   from llama_index.core import VectorStoreIndex
    #   from llama_index.core.schema import Document
    #   from llama_index.llms.openai import OpenAI
    #   os.environ["OPENAI_API_KEY"] = key
    #   openai.api_key = os.environ["OPENAI_API_KEY"]
    #   llm = OpenAI(temperature=0, model="gpt-4o-mini")
    #   Settings.llm = llm
    #   Settings.chunk_size = 512
    #   context_message=f""""""
    #   documents = [Document(text="context_message",embedding=None)]
    #   sparqlml_index = VectorStoreIndex.from_documents(documents)
    #   memory = ChatMemoryBuffer.from_defaults(token_limit=100000)
    #   chat_engine = sparqlml_index.as_chat_engine(  chat_mode="context",  memory=memory,   system_prompt=("You are a knoweldge reasoner system"))
    #   return chat_engine.chat(prompt_in).response,None
    elif model == "deepseek-chat":
        from openai import OpenAI
        llm = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        response = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a knoweldge reasoner system"},
                {"role": "user", "content": prompt_in},
            ],
            stream=False
        )
        return response.choices[0].message.content, response.usage, response
    elif model == "deepseek-reasoner":
        from openai import OpenAI
        llm = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        response = llm.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a knoweldge reasoner system"},
                {"role": "user", "content": prompt_in},
            ],
            stream=False
        )
        return response.choices[0].message.content, response.usage, response
    elif model == "gemini-1.5-flash":
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt_in)
        return response.text, response.usage_metadata, response
    else:
        return None, None, None

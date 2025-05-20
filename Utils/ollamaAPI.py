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
def query_ollama_dopost(model, prompt,system_prompt=None):
  # url = "http://192.168.41.218:11434/api/generate"
  # url = "http://206.12.96.43:11434/api/generate"
  dict_ollama_api={"gpu":"http://206.12.96.43:11434/api/generate"}
  url = dict_ollama_api["gpu"]
  headers = {"Content-Type": "application/json"}
  if system_prompt:
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}"
  print("ollama prompt", prompt)
  data = {
      "model": model,
      "prompt": prompt,
      "stream": False
  }
  response = requests.post(url, headers=headers, data=json.dumps(data))
  if response.status_code == 200:
          return response.json()
  else:
      return {"error": f"Request failed with status code {response.status_code}"}
def query_ollama_client(model, prompt,system_prompt=None):
    from ollama import Client
    # url = "http://192.168.41.218:11434"
    # url = "http://206.12.96.43:11434"
    dict_ollama_api = {"gpu": "http://206.12.96.43:11434"}
    url = dict_ollama_api["gpu"]
    headers = {"Content-Type": "application/json"}
    client = Client(
      host=url,
      headers=headers
    )
    messages=[
        {
            'role': 'system',
            'content': system_prompt if system_prompt else '' ,
        },
      {
        'role': 'user',
        'content': prompt,
      },
    ]
    print("messages=",messages)
    try:
        response = client.chat(model=model, messages=messages)
        return json.loads(response.model_dump_json())
    except client.ResponseError as e:
        return {"error": f"{client.ResponseError}"}

def chat(model="o1-mini",prompt_in="",key="",system_prompt=None):
  if model in ["o1-mini","gpt-4o-mini"]:
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
    return response.choices[0].message.content,response.usage,response
  elif model in ["deepseek-r1","qwen2.5:7b","qwen2.5:1.5b","qwen2.5:3b","llama3.2:3b","qwen:7b","qwen3:8b","phi4-mini","granite3.3"] :
    # response = query_ollama_dopost(model, prompt_in,system_prompt)
    response = query_ollama_client(model, prompt_in, system_prompt)
    usage_keys=['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']
    return response['message']['content'].split("</think>")[-1].replace("\n",""),{key:response[key] for key in usage_keys},response
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
  elif model=="deepseek-chat":
    from openai import OpenAI
    llm = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a knoweldge reasoner system"},
            {"role": "user", "content":prompt_in },
        ],
        stream=False
    )
    return response.choices[0].message.content,response.usage,response
  elif model=="deepseek-reasoner":
    from openai import OpenAI
    llm = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    response = llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a knoweldge reasoner system"},
            {"role": "user", "content":prompt_in },
        ],
        stream=False
    )
    return response.choices[0].message.content,response.usage,response
  elif model=="gemini-1.5-flash":
      import google.generativeai as genai
      genai.configure(api_key=key)
      model = genai.GenerativeModel(model)
      response = model.generate_content(prompt_in)
      return response.text,response.usage_metadata,response

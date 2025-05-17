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
def query_ollama(model, prompt):
  # url = "http://192.168.41.218:11434/api/generate"
  # url = "http://206.12.96.43:11434/api/generate"
  dict_ollama_api={"gpu":"http://206.12.96.43:11434/api/generate"}
  url = dict_ollama_api["gpu"]
  headers = {"Content-Type": "application/json"}
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

def chat(model="o1-mini",prompt_in="",key=""):
  if model in ["o1-mini","gpt-4o-mini"]:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
    model="o1-mini",
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
    response = query_ollama(model, prompt_in)
    usage_keys=['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']
    return response['response'].split("</think>")[-1].strip(),{key:response[key] for key in usage_keys},response
  elif "llama3.2:3b" in model:
    response = query_ollama("llama3.2:3b", prompt_in)
    usage_keys=['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']
    return response['response'].split("</think>")[-1].strip(),{key:response[key] for key in usage_keys},response
  elif model=="gpt-4o-mini":
    import openai
    import os
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import Document
    from llama_index.llms.openai import OpenAI
    os.environ["OPENAI_API_KEY"] = key
    openai.api_key = os.environ["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0, model="gpt-4o-mini")
    Settings.llm = llm
    Settings.chunk_size = 512
    llm = OpenAI(temperature=0, model="gpt-4o-mini")
    Settings.llm = llm
    Settings.chunk_size = 512
    context_message=f""""""
    documents = [Document(text="context_message",embedding=None)]
    sparqlml_index = VectorStoreIndex.from_documents(documents)
    memory = ChatMemoryBuffer.from_defaults(token_limit=100000)
    chat_engine = sparqlml_index.as_chat_engine(  chat_mode="context",  memory=memory,   system_prompt=("You are a knoweldge reasoner system"))
    return chat_engine.chat(prompt_in).response,None
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
      from google.colab import userdata
      # GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
      GEMINI_API_KEY="AIzaSyBNt-HnOrQ2FLWaH6leB9YGg1sbeKKFgRk"
      genai.configure(api_key=GEMINI_API_KEY)
      # for m in genai.list_models():
      #     if "generateContent" in m.supported_generation_methods:
      #         print(m.name)
      model = genai.GenerativeModel(model)
      response = model.generate_content(prompt_in)
      return response.text,response.usage_metadata,response

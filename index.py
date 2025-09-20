""" Flask 框架應用程式 """
import os
import sys
import json
import warnings
import logging
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
import transformers, datasets
from peft import PeftModel
from colorama import *
from flask import Flask, request, jsonify
from flask_cors import CORS # 處理跨域問題

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)
import requests
import chromadb
from chromadb.utils import embedding_functions

# --- Flask 應用程式初始化 ---
app = Flask(__name__)
CORS(app) # 允許跨域請求

# --- 模型載入與設置 ---
warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在載入 Qwen 模型至 {device}...")
model_name = "Qwen/Qwen2.5-3B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.to(device)
    print("模型載入成功。")
    print(f"修正後的模型設備: {model.device}")
except Exception as e:
    print(f"模型載入失敗：{e}")
    exit()

# 設置 GenerationConfig
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.do_sample = True
generation_config.temperature = 0.7
generation_config.top_p = 0.95
generation_config.repetition_penalty = 1.15
CUTOFF_LEN = 500

# --- ChromaDB 向量資料庫設置與載入 ---
try:
    with open("C:/Users/watch/Downloads/深度學習/AttractionList.json", "r", encoding="utf-8-sig") as f:
        attractions_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"載入景點資料失敗：{e}")
    exit()

all_attractions = attractions_data.get("Attractions", [])
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "taiwan_attractions"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

BATCH_SIZE = 5000
if collection.count() == 0:
    documents = []
    metadatas = []
    ids = []
    for attraction in all_attractions:
        doc_content = f"{attraction.get('AttractionName', '')}。地點：{attraction.get('PostalAddress', {}).get('City', '')}{attraction.get('PostalAddress', {}).get('Town', '')}。描述：{attraction.get('Description', '')}"
        documents.append(doc_content)
        
        simplified_metadata = {
            "AttractionName": attraction.get("AttractionName", ""),
            "City": attraction.get("PostalAddress", {}).get("City", ""),
            "AttractionID": attraction.get("AttractionID", ""),
            "Description": attraction.get("Description", "")
        }
        metadatas.append(simplified_metadata)
        
        ids.append(attraction.get("AttractionID"))

    print(f"正在將 {len(documents)} 個景點分批添加到 ChromaDB...")
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch_documents = documents[i:i + BATCH_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        
        try:
            collection.add(documents=batch_documents, metadatas=batch_metadatas, ids=batch_ids)
            print(f"已成功新增第 {i+1} 到 {i + len(batch_documents)} 個景點。")
        except Exception as e:
            print(f"新增批次資料時發生錯誤：{e}")
            break
            
    print("所有景點資料載入完成。")
else:
    print("ChromaDB 集合已包含資料，跳過載入。")

# --- 天氣 API 相關設定與函數 ---
CWB_API_AUTH_CODE = "CWA-63E5B3F1-053E-4787-8387-9962373B10E1" 

def get_taiwan_weather_forecast(locations_name: str):
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001?Authorization=CWA-63E5B3F1-053E-4787-8387-9962373B10E1"
    params = {
        "Authorization": CWB_API_AUTH_CODE,
        "format": "JSON",
        "locationName": locations_name,
        "elementName": "Wx,MinT,MaxT,PoP"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"查詢中央氣象署 API 時發生錯誤: {e}")
        return None

def parse_weather_data(weather_data, locations_name):
    try:
        location_data = weather_data['records']['location'][0]
        weather_elements = {
            element['elementName']: element['time'][0]['parameter']['parameterName']
            for element in location_data['weatherElement']
        }
        wx_element = weather_elements.get('Wx', '未知')
        min_temp_element = weather_elements.get('MinT', '未知')
        max_temp_element = weather_elements.get('MaxT', '未知')
        pop_element = weather_elements.get('PoP', '未知')

        weather_summary = (
            f"根據中央氣象署最新資料，{locations_name}目前的天氣預報是：\n"
            f"天氣狀況：{wx_element}。\n"
            f"氣溫範圍：約 {min_temp_element}°C 到 {max_temp_element}°C。\n"
            f"降雨機率：{pop_element}%。"
        )
        
        try:
            pop_value = int(pop_element)
            min_temp_value = int(min_temp_element)
            if pop_value >= 50:
                weather_summary += "天氣濕冷或有雨，建議攜帶雨具或安排室內活動喔！"
            elif pop_value < 20 and min_temp_value >= 20:
                weather_summary += "天氣晴朗舒適，非常適合戶外活動！"
        except ValueError:
            pass
            
        return weather_summary
    except (KeyError, IndexError) as e:
        print(f"解析天氣資料時發生錯誤：{e}")
        return "很抱歉，目前無法解析該地區的天氣資訊。"

# --- RAG 檢索函數 ---
def retrieve_attractions_from_db(query: str, location: str = None, top_k: int = 3) -> str:
    where_clause = {}
    if location:
        where_clause["City"] = location
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_clause
    )
    formatted_results = "[可用的景點資訊]\n"
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i]
            name = metadata.get("AttractionName", "未知景點")
            city = metadata.get("City", "")
            town = metadata.get("Town", "")
            description = metadata.get("Description", "無描述")
            formatted_results += (
                f"- {name}: 位於{city}{town}，描述：{description}\n"
            )
    else:
        formatted_results = f"在{location if location else '指定區域'}找不到符合'{query}'的相關景點資訊。"
    return formatted_results

# --- evaluate 函數 ---
def evaluate(query: str, generation_config, max_len: int, locations_name: str, verbose: bool = True):
    retrieved_attractions_info = retrieve_attractions_from_db(query, location=locations_name)
    
    if "找不到符合" in retrieved_attractions_info:
        no_results_response = (
            f"很抱歉，根據您的查詢：'{query}'，我們在{locations_name}的景點資料庫中找不到相關景點。\n"
            f"為了提供更精準的建議，您能否提供更具體的偏好呢？\n"
            f"例如：您對**戶外活動**、**文創市集**或是**美食**感興趣？"
        )
        return no_results_response

    raw_weather_data = get_taiwan_weather_forecast(locations_name)
    if raw_weather_data and raw_weather_data.get('success') == 'true':
        weather_summary_for_llm = parse_weather_data(raw_weather_data, locations_name)
    else:
        weather_summary_for_llm = "很抱歉，目前無法取得該地區的最新天氣資訊。"
    
    combined_input_data = (
        f"使用者查詢: {query}\n\n"
        f"[天氣資訊]\n{weather_summary_for_llm}\n\n"
        f"{retrieved_attractions_info}"
    )

    final_prompt = f"""\
[INST] <<SYS>>
你是一個專業的旅遊顧問和導遊，擅長根據使用者的偏好和需求提供個人化的旅遊建議、行程規劃和即時資訊。
**重要且嚴格的指令：**
1. 行程規劃中提及的所有景點**必須且只能**來自[可用的景點資訊]中提到的地點。
2. **絕對禁止**提及任何[可用的景點資訊]以外的景點、地點或設施。
3. **只能根據 [可用的景點資訊] 中明確提及的內容來生成。若資訊不完整，不得自行編造或補充細節，只需以概括性語言描述。**
4. 必須在回覆的開頭先告知天氣狀況，並在行程規劃中考慮天氣影響。
   - **如果天氣適合戶外活動（如晴朗、降雨機率低），請優先推薦適合戶外的景點。**
   - **如果天氣不適合戶外活動（如多雲有雨、降雨機率高），請優先推薦適合室內的景點，並避免推薦戶外景點。**
5. **嚴禁生成任何新的提示，例如 `[INST]`, `>>SYS` 或 `>>USER` 等標籤。回覆必須以最終的旅遊建議結尾。**\
6. 請使用繁體中文回答，並確保語句通順自然。
<</SYS>>
{combined_input_data}
[/INST]"""
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    generation_output = model.generate(
        input_ids=inputs["input_ids"],
        generation_config=generation_config,
        max_new_tokens=512,
        return_dict_in_generate=True,
        output_scores=True, 
    )
    decoded_output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=False)
    
    if "[/INST]" in decoded_output:
        output = decoded_output.split("[/INST]")[1].replace("</s>", "").strip()
    else:
        output = decoded_output.replace("</s>", "").strip()

    return output

cache_dir = "./cache"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 從指定的模型名稱或路徑載入預訓練的語言模型，添加 trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage = True,
    trust_remote_code=True
)

# 創建 tokenizer 並設定結束符號 (eos_token)，添加 trust_remote_code=True
logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

max_len = 512
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
    repetition_penalty=1.15,
    pad_token_id=tokenizer.eos_token_id, # 使用 tokenizer.eos_token_id
)

# --- API 端點 ---
@app.route('/generate_travel_advice', methods=['POST'])
def generate_travel_advice():
    data = request.json
    user_query = data.get('query')
    location = data.get('location')

    if not user_query or not location:
        return jsonify({"error": "請提供查詢內容和地點。"}), 400

    try:
        response = evaluate(
            query=user_query,
            generation_config=generation_config,
            max_len=512,
            locations_name=location,
            verbose=False
        )
        return jsonify({"response": response})
    except Exception as e:
        print(f"處理請求時發生錯誤: {e}")
        return jsonify({"error": "處理您的請求時發生內部錯誤。"}), 500

#if __name__ == '__main__':
    # 這裡的程式碼在運行時會被執行一次，並載入所有模型和資料
    # 然後，Flask 伺服器會開始運行
#   app.run(host='0.0.0.0', port=5000)
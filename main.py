import os
from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
import json
import pickle
import numpy as np
import string
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re
import logging
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, AdamW
from torch.utils.data import TensorDataset, DataLoader

from API_service import WordCloudService
from news_summary_model.news_summarization import transformer, predict, regex_column

import keyword_extract.key_extract_module as key_extract_module
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 장치 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()


# 키워드와 제목 리스트를 받는 모델 정의
class NewsRequest(BaseModel):
    keyword: str
    titles: List[str]

class NewsSummary(BaseModel):
    content: str


class NewsData(BaseModel):
    news: str

class ChatRequest(BaseModel):
    sender: str
    content: str

# 필요한 전역 변수 설정
SEN_MAX_LENGTH = 799
ABS_MAX_LENGTH = 149

model = None  # 모델 로드를 위한 변수
tokenizer = None  # 토크나이저 로드를 위한 변수
chatbot_model = None
chatbot_tokenizer = None
lang_model = None

# 모델 및 토크나이저 로드 함수 (앱 시작 시 실행)
def load_model_and_tokenizer():
    global model, tokenizer, chatbot_model, chatbot_tokenizer, lang_model
    # 모델과 토크나이저를 로드합니다.
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer')
    model = transformer(
        vocab_size=tokenizer.vocab_size + 2,  # 추가 토큰 크기
        num_layers=2,
        dff=256,
        d_model=128,
        num_heads=2,
        dropout=0.3
    )
    model.load_weights('transformer(202_0.89_0.22).h5')  # 미리 학습된 가중치 로드
    
    chatbot_tokenizer = AutoTokenizer.from_pretrained("/root/serving/ModelAPI/stock_chatbot_model")
    chatbot_model = BartForConditionalGeneration.from_pretrained("/root/serving/ModelAPI/stock_chatbot_model")
    chatbot_model.to(device)
    
    lang_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    

# API 엔드포인트 구성
@app.post("/summarizer")
def summarize_news(news: NewsSummary):
    clean_content = regex_column(news.content)  # 정규식으로 텍스트 정리
    # Transformer 모델을 통한 요약 수행
    summary = predict(clean_content)
    return {'news_content': summary}


@app.post("/keyword")
def keyword_extract(string: NewsData):
    key_text = string.news
    key_text = key_extract_module.preprocessing_article(key_text)
    (article_embedding, n_gram_embeddings, n_gram_words) = key_extract_module.article_embedding(key_text)
    news_keywords = key_extract_module.max_sum_sim(article_embedding, n_gram_embeddings, n_gram_words, top_n=6,
                                                   variety=10)
    
    (key_embedding, keys_list) = key_extract_module.key_extract(lang_model)
    top_n = 5

    distances = cosine_similarity(article_embedding, key_embedding)
    cosine_recommand = [keys_list[index] for index in distances.argsort()[0][-top_n:]]

    news_keywords = ', '.join(news_keywords)
    cosine_recommand = ', '.join(cosine_recommand)

    return {"keywords" : news_keywords, "recommand_keywords" : cosine_recommand}

# 앱 시작 시 모델과 토크나이저 로드``
@app.on_event("startup")
def startup_event():
    load_model_and_tokenizer()

@app.post("/news")
def generate_cloud(news: NewsRequest):
    keyword = news.keyword
    titles = news.titles

    news_list = WordCloudService.preprocessing_title(titles)

    # 워드 클라우드 생성
    WordCloudService.to_wordcloud_client(news_list, keyword)

@app.get("/download")
def download_file():
    wordcloud_image_path = "API_service/static/word_cloud/wordcloud.png"
    if os.path.exists(wordcloud_image_path):
        return FileResponse(wordcloud_image_path, media_type="image/png")
    return {"error": "word cloud image not found"}


@app.post("/chatbot")

    
    prompt = f"질문: {question}\n답변: "

    input_ids = chatbot_tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output_ids = chatbot_model.generate(
            input_ids,
            max_length=128,  # 적절한 max_length 설정
            pad_token_id=chatbot_tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.6,
            repetition_penalty=1.2,
            eos_token_id=chatbot_tokenizer.eos_token_id,
            use_cache=True
        )
    output = chatbot_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True  # 경고를 제거하기 위해 추가
    )
    
    if "답변:" in output:
        response = output.rsplit("답변:", 1)[-1].strip()
    else:
        response = "죄송합니다. 이해하지 못했어요."

    return json.dumps({"response":response}, ensure_ascii=False).encode('utf8')


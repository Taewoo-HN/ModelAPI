import os
from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re
import logging

from API_service import WordCloudService
from news_summary_model.news_summarization import transformer, predict

import keyword_extract.key_extract_module as key_extract_module
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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
    question: str

# 필요한 전역 변수 설정
SEN_MAX_LENGTH = 799
ABS_MAX_LENGTH = 149

model = None  # 모델 로드를 위한 변수
tokenizer = None  # 토크나이저 로드를 위한 변수


# 모델 및 토크나이저 로드 함수 (앱 시작 시 실행)
def load_model_and_tokenizer():
    global model, tokenizer, lang_model
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
    lang_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    logging.info("모델 및 토크나이저 로드 완료!")
    

# API 엔드포인트 구성
@app.post("/summarizer")
def summarize_news(news: NewsSummary):

    content = news.content
    clean_content = regex_column(content)  # 정규식으로 텍스트 정리

    if len(clean_content) > SEN_MAX_LENGTH:
        return clean_content[:SEN_MAX_LENGTH]

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

# regex_column 함수 정의
def regex_column(columnList):
    if not isinstance(columnList, str):
        return ''
    columnList = re.sub(r'\S+@\S+\.\S+', '', columnList)
    columnList = columnList.replace('\n', '')
    columnList = re.sub(r'\[.*?\]|\{.*?\}|\(.*?\)', '', columnList)
    columnList = re.sub(r'[^가-힣a-zA-Z0-9\u4e00-\u9fff\s.,!?\'\"~]', ' ', columnList)
    columnList = re.sub(r'\s+', ' ', columnList).strip()
    return columnList


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

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
from news_summary_model.summary_model import transformer

import keyword_extract.key_extract_module as key_extract_module
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI()


# 키워드와 제목 리스트를 받는 모델 정의
class NewsRequest(BaseModel):
    keyword: str
    titles: List[str]


class NewsSummary(BaseModel):
    content: List[str]


class NewsData(BaseModel):
    news: str


# 필요한 전역 변수 설정
SEN_MAX_LENGTH = 799
ABS_MAX_LENGTH = 149
START_TOKEN = [1]  # 토크나이저에 맞는 시작 토큰
END_TOKEN = [2]  # 토크나이저에 맞는 종료 토큰
model = None  # 모델 로드를 위한 변수
tokenizer = None  # 토크나이저 로드를 위한 변수


# 모델 및 토크나이저 로드 함수 (앱 시작 시 실행)
def load_model_and_tokenizer():
    global model, tokenizer
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


# 텍스트 요약 함수 (evaluate & predict 활용)
def evaluate(sentence, model, tokenizer, start_token, end_token, max_length):
    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)
    output = tf.expand_dims(start_token, 0)

    for i in range(max_length):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, end_token[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence, model, tokenizer, start_token, end_token, max_length):
    prediction = evaluate(sentence, model, tokenizer, start_token, end_token, max_length)
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    return predicted_sentence


# API 엔드포인트 구성
@app.post("/summarizer")
def summarize_news(news: NewsSummary):
    # 요청에서 받은 뉴스 내용 정제
    content = news.content
    clean_content = regex_column(content)  # 정규식으로 텍스트 정리
    logging.info("정제된 뉴스 내용: ", clean_content)

    if len(clean_content) > SEN_MAX_LENGTH:
        return clean_content[:SEN_MAX_LENGTH]

    # Transformer 모델을 통한 요약 수행
    summary = predict(clean_content, model, tokenizer, START_TOKEN, END_TOKEN, ABS_MAX_LENGTH)

    regex_news = regex_column(summary)
    return {'news_content': regex_news }

@app.post("/keyword")
def keyword_extract(string: NewsSummary):
    key_text = string.news
    key_text = key_extract_module.preprocessing_article(key_text)
    (article_embedding, n_gram_embeddings, n_gram_words) = key_extract_module.article_embedding(key_text)
    news_keywords = key_extract_module.max_sum_sim(article_embedding, n_gram_embeddings, n_gram_words, top_n=6,
                                                   variety=10)

    lang_model = SentenceTransformer('sentnece-transformers//xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    (key_embedding, keys_list) = key_extract_module.key_extract(lang_model)
    top_n = 5

    distances = cosine_similarity(article_embedding, key_embedding)
    cosine_recommand = [keys_list[index] for index in distances.argsort()[0][-top_n:]]


    logging.info("키워드: ", news_keywords, "추천 키워드: ", cosine_recommand)

    return { "keywords": news_keywords, "recommand_keywords": cosine_recommand}


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

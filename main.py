import os
from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re

from API_service import WordCloudService
from news_summary_model.summary_model import transformer

app = FastAPI()

class NewsSummary(BaseModel):
    content: str

# 필요한 전역 변수 설정
SEN_MAX_LENGTH = 799
ABS_MAX_LENGTH = 149
START_TOKEN = [1]  # 토크나이저에 맞는 시작 토큰
END_TOKEN = [2]    # 토크나이저에 맞는 종료 토큰
model = None       # 모델 로드를 위한 변수
tokenizer = None   # 토크나이저 로드를 위한 변수

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

    if len(clean_content) > SEN_MAX_LENGTH:
        return clean_content[:SEN_MAX_LENGTH]
    # Transformer 모델을 통한 요약 수행
    summary = predict(clean_content, model, tokenizer, START_TOKEN, END_TOKEN, ABS_MAX_LENGTH)

    # 요약 결과를 다시 정규화 (필요 시)

    return {"news_summary": summary}

# 앱 시작 시 모델과 토크나이저 로드
@app.on_event("startup")
def startup_event():
    load_model_and_tokenizer()

# regex_column 함수 정의 (이전 코드에 정의된 대로 사용)
def regex_column(columnList):
    if not isinstance(columnList, str):
        return ''
    columnList = re.sub(r'\S+@\S+\.\S+', '', columnList)
    columnList = columnList.replace('\n', '')
    columnList = re.sub(r'\[.*?\]|\{.*?\}|\(.*?\)', '', columnList)
    columnList = re.sub(r'[^가-힣a-zA-Z0-9\u4e00-\u9fff\s.,!?\'\"~]', ' ', columnList)
    columnList = re.sub(r'\s+', ' ', columnList).strip()
    return columnList

# 키워드와 제목 리스트를 받는 모델 정의
class NewsRequest(BaseModel):
    keyword: str
    titles: List[str]

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

import os
from typing import List

import tensorflow_datasets as tfds
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from API_service import WordCloudService
from news_summary_model.summary_model import transformer, predict, encoder
from news_summary_model.utils import regex_column


tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer')

model = transformer(vocab_size=tokenizer.vocab_size + 2, num_layers=2, dff=256, d_model=128, num_heads=2, dropout=0.3)
model.load_weights('transformer(202_0.89_0.22).h5')

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
SEN_MAX_LENGTH = 799  # Sentence max length 설정


app = FastAPI()

# 키워드와 제목 리스트를 받는 모델 정의
class NewsRequest(BaseModel):
    keyword: str
    titles: List[str]

class NewsSummary(BaseModel):
    content: List[str]
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

@app.post("/summarizer")
def summerize_news(news: NewsSummary):
    data = regex_column(news.content)
    news_content = tokenizer.encode(data)
    encoded_sentence = START_TOKEN + news_content + END_TOKEN
    if len(encoded_sentence) > SEN_MAX_LENGTH:
        encoded_sentence = encoded_sentence[:SEN_MAX_LENGTH]
    else:
        encoded_sentence += [0] * (SEN_MAX_LENGTH - len(encoded_sentence))
    print(encoded_sentence)
    tar = model.predict(encoded_sentence)
    redata = tokenizer.decode(tar)
    tea = regex_column(redata)
    return {"news_summery": tea}
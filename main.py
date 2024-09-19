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
    sender: str
    content: str

# 필요한 전역 변수 설정
SEN_MAX_LENGTH = 799
ABS_MAX_LENGTH = 149

model = None  # 모델 로드를 위한 변수
tokenizer = None  # 토크나이저 로드를 위한 변수


# 모델 및 토크나이저 로드 함수 (앱 시작 시 실행)
def load_model_and_tokenizer():
    global model, tokenizer, lang_model, lstm_model, lstm_tokenizer, keyword_dict, keyword_list, seq2seq_model, seq2seq_tokenizer, START_TOKEN, END_TOKEN, pre_token, prepre_token
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
    
    lstm_model = load_model('/usr/local/etc/ModelAPI/Financial_Chatbot/Model/bidirectional_LSTM.h5')
    lstm_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('/usr/local/etc/ModelAPI/Financial_Chatbot/LSTM/Data/tokenizer')

    ##      사전 데이터 로드
    with open('/usr/local/etc/ModelAPI/Financial_Chatbot/keyword_dict.pkl', 'rb') as file:
        keyword_dict = pickle.load(file)
    keyword_list = list(keyword_dict.keys())

    ##      챗봇 데이터 로드
    seq2seq_model = load_model('/usr/local/etc/ModelAPI/Financial_Chatbot/chatbot(1.19-0.81).h5')
    seq2seq_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('/usr/local/etc/ModelAPI/Financial_Chatbot/Final_Tokenizer')
    START_TOKEN, END_TOKEN = [seq2seq_tokenizer.vocab_size], [seq2seq_tokenizer.vocab_size+1]
    pre_token = -1
    prepre_token = -1
    
    model.load_weights('transformer(202_0.89_0.22).h5')  # 미리 학습된 가중치 로드
    lang_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    logging.info("모델 및 토크나이저 로드 완료!")
    

# API 엔드포인트 구성
@app.post("/summarizer")
def summarize_news(news: NewsSummary):

    clean_content = regex_column(news.content)  # 정규식으로 텍스트 정리
    
    if clean_content == '':
        return {'news_content': '내용이 없습니다.'}

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


@app.post("/chatbot")
def chatbot_response(question: ChatRequest):
    content = question.content
    logging.info(f"사용자 질문: {content}")
    if len(content) > 50:
        return json.dumps({'message': '질문이 너무 깁니다! 조금만 줄여주세요!'}, ensure_ascii=False).encode('utf-8')
    else:
        tokenied_question = []
        tokenied_question.append(lstm_tokenizer.encode(content))
        final_question = tf.keras.preprocessing.sequence.pad_sequences(
            tokenied_question, maxlen=77, padding='post'
        )
        score = float(lstm_model.predict(final_question))
        
        if score > 0.5:  # 사전 질문이 들어왔을 때
            try:
                question_no_space = content.replace(" ", "")
                included_keywords = [keyword for keyword in keyword_list if keyword in question_no_space]
                
                if not included_keywords:
                    raise ValueError
                
                if len(included_keywords) > 1:
                    keyword = max(included_keywords, key=len)
                else:
                    keyword = included_keywords[0]
                
                response = f"'{keyword}'의 뜻을 알려드릴께요\n{keyword}(이)란 '{keyword_dict.get(keyword)}'을(를) 뜻합니다."
                return json.dumps({'response': response}, ensure_ascii=False).encode('utf-8')
            except ValueError:
                return json.dumps({'response': '죄송합니다. 잘 모르겠습니다.'}, ensure_ascii=False).encode('utf-8')
        else:  # 지식인 질문이 들어왔을 때
            question_list = []
            result_list = []
            final_sentence = ''
            
            question_list.append(seq2seq_tokenizer.encode(content))
            tokenized_input_sentence = tf.keras.preprocessing.sequence.pad_sequences(
                question_list, maxlen=59, padding='post'
            )
            
            temparary_decoder_input = [START_TOKEN[0]]
            pre_token = -1
            prepre_token = -1
            
            for _ in range(142):
                pred = seq2seq_model.predict(
                    [tokenized_input_sentence, np.array(temparary_decoder_input).reshape(1, -1)],
                    verbose=0
                )
                last_pred = pred[:, -1, :]
                sorted_indices = np.argsort(last_pred, axis=-1)[:, ::-1]
                next_token = sorted_indices[:, 0][0]
                
                if next_token >= 16251:
                    break
                
                temparary_decoder_input.append(next_token)
                
                if pre_token != next_token and prepre_token != next_token:
                    result_list.append(seq2seq_tokenizer.decode([next_token]))
                
                prepre_token = pre_token
                pre_token = next_token
            
            final_sentence = ''.join(result_list)
            special_characters = string.punctuation
            final_sentence = final_sentence.lstrip(special_characters)
            
            if not final_sentence.endswith(('.', '!')):
                final_sentence += '.'
            
            response = final_sentence
            return json.dumps({'response': response}, ensure_ascii=False).encode('utf-8')
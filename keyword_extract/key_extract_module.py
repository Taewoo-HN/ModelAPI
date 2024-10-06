import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import re
import itertools


def preprocessing_article(article: str):
    article = article.replace('\n', '')
    split_article = re.split(r'(\s*[a-zA-Z]+(?:[^\w\s]+[a-zA-Z]+)*\s*)', article)
    split_article = [part.strip() for part in split_article if part.strip()]

    okt = Okt()
    nouns = []
    for sentence in split_article:
        sen = okt.nouns(sentence)
        if len(sen) != 0:
            nouns.append(sen)
        else:
            nouns.append(sentence)
    text = ' '.join([' '.join(item) if isinstance(item, list) else item for item in nouns])
    return text

def article_embedding(text: str, model):
    n_gram_range = (1, 2)
    word_vectorizer = CountVectorizer(ngram_range=n_gram_range).fit([text])
    n_gram_words = word_vectorizer.get_feature_names_out()

    article_embedding = model.encode([text])
    n_gram_embedding = model.encode(n_gram_words)

    return article_embedding, n_gram_embedding, n_gram_words


def max_sum_sim(article_embedding, n_gram_embeddings, n_gram_words, top_n, variety):
    #   뉴스 기사와 N-Gram의 유사도
    distances = cosine_similarity(article_embedding, n_gram_embeddings)
    #   N-Gram들 사이의 유사도
    distances_candidates = cosine_similarity(n_gram_embeddings, n_gram_embeddings)

    #   뉴스 기사와 유사도가 높은 N-Gram을 variety개 가져오기
    words_idx = list(distances.argsort()[0][-variety:])
    words_vals = [n_gram_words[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    #   그 중 유사도가 가장 낮은 조합 찾기
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def key_extract(model):
    with open('keyword_extract/data/theme_dict.pkl', 'rb') as f:
        loading_dict = pickle.load(f)
    keys = loading_dict.keys()
    keys_list = list(keys)

    new_keys_list = [re.sub(r'[^a-zA-Z0-9가-힣\s]', ' ', key) for key in keys_list]
    key_embedding = model.encode(new_keys_list)

    return key_embedding, new_keys_list



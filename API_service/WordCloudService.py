import re
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



def preprocessing_title(title_list) -> list:
    '''
    :param query: str
    :return: list
    DB에서 관련 뉴스 추출하여 전처리
    '''
    # 정규식으로 제목 전처리
    pattern = r'[^a-zA-Z0-9\u4e00-\u9fff가-힣\s.%]'
    regex_title = [re.sub(pattern, '', title) for title in title_list]

    return regex_title

def to_wordcloud_client(news_list: list, query:str):
    okt = Okt()

    nouns_list = []
    for title in news_list:
        nouns = okt.nouns(title)
        english_nouns = re.findall(r'[a-zA-Z]+', title)

        nouns_list.extend(nouns)
        nouns_list.extend(english_nouns)

    nouns_list = [noun for noun in nouns_list if len(noun) >= 2 and noun != query]

    text = ' '.join(nouns_list)
    image_mask = np.array(Image.open("API_service/static/cloud_mask.png"))
    image_mask = 255 - image_mask

    wordcloud = WordCloud(font_path='API_service/static/IBMPlexSansKR-Medium.ttf',
                      width=1600, height=1200,
                      background_color='white',
                      colormap='coolwarm',
                      mask=image_mask,
                      max_words=100,
                      scale=4.0).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    wordcloud.to_file("API_service/static/word_cloud/wordcloud.png")

import re

def regex_column(columnList):
    if not isinstance(columnList, str):
        return ''
    columnList = re.sub(r'\S+@\S+\.\S+', '', columnList)
    columnList = columnList.replace('\n', '')
    columnList = re.sub(r'\[.*?\]|\{.*?\}|\(.*?\)', '', columnList)
    columnList = re.sub(r'[^가-힣a-zA-Z0-9\u4e00-\u9fff\s.,!?\'\"~]', ' ', columnList)
    columnList = re.sub(r'\s+', ' ', columnList).strip()
    return columnList
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import Counter
from langid.langid import LanguageIdentifier, model
import re
import time

def remove_numbers_and_symbols(text):
    split_text = re.split(r'[-/_+]', str(text))
    
    # 각 분리된 문자열에서 숫자와 특정 특수 문자 제거
    return [
        re.sub(r'[\d.,=!@#$%^&*(){}\[\];:"\\|<>/?~`\'-]', '', segment) 
        for segment in split_text
        ]
    

def word_tokenzation(text):
    try:
        text = text.lower()
        words = word_tokenize(text)
    except:
        words = ['tokenerror']
    return words    


def get_word_freq(series):
    print('remove numbers...')
    series = series.map(remove_numbers_and_symbols)
    
    print('exploding series...')
    series = series.explode()
    
    print('text counting...')
    text_freq = series.value_counts().to_dict()
    
    word_freq = {}
    print('word counting ...')
    
    for k, v in text_freq.items():
        words = word_tokenzation(k)
        for word in words:
            if word in word_freq:
                word_freq[word] += v
            else:
                word_freq[word] = v
    word_freq = {
        k: v for k, v in word_freq.items() 
        if not is_float(k) and len(k) > 1
    }
    return word_freq


def detect_language(word_freq, ident):
    # word_freq: dict {word: count}
    # returns: dict {word: (lang, prob, count)}
    result = {}
    for k, v in word_freq.items():
        try: 
            id_lang = ident.classify(k)
        except:
            id_lang = ('unknown', 0)
        result[k] = (id_lang[0], round(id_lang[1], 3), v)
    return result


def langid_language(text):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    return identifier.classify(text)    


def plot_language_distribution(texts):  
    languages = [detect_language(text) for text in texts]
    counter = Counter(languages)
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df.columns = ['language', 'count']
    df.plot(kind='bar', x='language', y='count')
    plt.show()


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
    
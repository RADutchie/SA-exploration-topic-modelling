
import string
import spacy 
import nltk
import en_core_web_sm
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata
from autocorrect import Speller

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

nlp = en_core_web_sm.load(disable = ['ner', 'parser'])
nlp.max_length = 10000000


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()


def spell_check(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    spell = Speller()
    corrected_text = [spell(token) for token in tokens]
    return ' '.join(corrected_text)


def remove_single_chatacters(text):
    pattern = r'\b[a-zA-Z]\b'
    return re.sub(pattern, '', text)


def preprocess_doc(doc: str, accented_char_removal=True, text_lower_case=True, check_spelling=True,
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_single_char=True, remove_digits=True):
    """ 
    Text document preprocessor for cleaning and normalising input text for NLP processes.

    Text preprocessing includes steps for:
            1. convert to utf-8 and remove accented characters
            2. lower case all characters
            3. check word spelling (using autocorrect) - WARNING slow for large documents 
            4. remove extra new lines, whitespace and tabs
            5. lemmatization (using SpaCy)
            6. remove special characters and punctuation
            7. remove all digits
            8. remove single characters
            9. remove stopwords (uses NLTK stopword list)

    Parameters
    -------------------
        doc (str): Input text as a string
        accented_char_removal (bool): default = True, active if true
        text_lower_case (bool): default = True, active if true
        check_spelling (bool): default = True, active if true
        text_lemmatization (bool): default = True, active if true
        special_char_removal (bool): default = True, active if true
        stopword_removal (bool): default = True, active if true
        remove single characters (bool): default = True, active if true
        remove_digits (bool): default = True, active if true
    
    Returns
    ------------------
        doc (str): processed document

    """
    if accented_char_removal:
        doc = remove_accented_chars(doc)
    if text_lower_case:
        doc = doc.lower()
    # remove extra new lines
    doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
    if check_spelling:
        doc = spell_check(doc)
    if text_lemmatization:
        doc = lemmatize_text(doc)
    if special_char_removal:
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = remove_special_characters(doc, remove_digits = remove_digits)
    if remove_single_char:
        doc = remove_single_chatacters(doc)
    doc = remove_extra_whitespace_tabs(doc)
    #doc = re.sub(' +', ' ', doc)
    if stopword_removal:
        doc = remove_stopwords(doc)
    

    return doc

if __name__ == "__main__":
    preprocess_doc()
# REQUIREMENTS
# !python -m spacy download en_core_web_sm
# !pip install yake
# !pip install newsapi-python
# !pip install laserembeddings python-Levenshtein utils faiss-cpu
# !python -m laserembeddings download-models
# !pip install googletrans==3.1.0a0
# !pip install streamlit

import spacy
import yake
from newsapi.newsapi_client import NewsApiClient

import numpy as np
import faiss

from string import punctuation
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import PCA
from Levenshtein import distance as levenshtein_distance
import requests
from bs4 import BeautifulSoup

import nltk
from sentence_transformers import SentenceTransformer

from googletrans import Translator
import streamlit as st

import os
@st.cache(allow_output_mutation=True) 
# os.popen("python -m spacy download en_core_web_sm")
# os.popen("python -m laserembeddings download-models")



def get_keywords_spacy(text):
    doc = nlp(text)
    query = " ".join(map(str, doc.ents[:5]))
    return query

def get_keywords_yake(text, dup_threshold = 0.2, numKeywords = 5):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = dup_threshold
    numOfKeywords = numKeywords
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    query = ' '.join([kw[0] for kw in keywords])
    return query

def get_headlines(query):
    top_headlines = newsapi.get_top_headlines(
        q=query,
        language='en',
    )
    return top_headlines

def get_articles(query):
    all_articles = newsapi.get_everything(
        q=query,
        language='en',   
    )
    return all_articles

def get_page_text(url): 
        html_page = requests.get(url). content
        soup = BeautifulSoup(html_page, 'lxml')
        whitelist = ['p','strong','em','b','u','i','h1','h2','h3']
        out = ""
        for t in soup.find_all(text=True):
            if t.parent.name in whitelist:
                out += '{} '.format(t)
        escape = ['\r','\n','\t','\xa0']
        for e in escape:
            out = out.replace(e,'')
        return out

def get_related_articles(all_articles, maxArticles = 10):
    doc = ''
    for article in all_articles['articles'][:maxArticles]: 
        
        text = get_page_text(article['url'])
        doc = doc + text
    return doc

def doc_to_sentences(doc):
    """
    Splits a document into sentences.
    """
    doc = doc.replace('\n', ' ').replace('\t', ' ').replace('\x00', ' ')
    return sent_tokenize(doc)

def compute_embeddings(sequences, dim=512):
    """
    Computes the embeddings for a list of sequences.
    """
    laser = Laser()
    embeddings = laser.embed_sentences(sequences, lang='en')
    # embeddings is a N*1024 (N = number of sentences) NumPy array

    pca = PCA(n_components=dim)
    embeddings = pca.fit_transform(embeddings)
    return embeddings

def index_embeddings(embeddings):
    """
    Indexes a list of embeddings using a FAISS index.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def get_nearest_neighbors(index, embeddings, k=8):
    """
    Returns the k nearest neighbors of each embedding in a list of embeddings.
    """
    D, I = index.search(embeddings, k)
    return D, I


def filter_nearest_neighbors(D, I, max_L2_dist=0.05):
    """
    Filters the nearest neighbors to remove those which are too far from the queries.
    """
    filtered_neighbors = np.ones(I.shape) * (-1)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if D[i,j] <= max_L2_dist:
                filtered_neighbors[i,j] = I[i,j]
    
    filtered_neighbors = filtered_neighbors.astype(int)
    return filtered_neighbors


def filter_paraphrases(I, sequences, min_l_dist=0.2):
    """
    Removes almost identical pp with character level Levenshtein distance <= 20%
	or pp from coming same document         ** (need to implement this) **
	or pp where one sequence is contained in other
    """
    for i in range(I.shape[0]):
        cur_seq = sequences[i]
        for j in range(I.shape[1]):
            if I[i,j] == -1:
                continue
            
            target_seq = sequences[I[i,j]]
            dist = levenshtein_distance(cur_seq, target_seq)
            if dist <= min_l_dist:
                I[i,j] = -1
                continue
            
            if cur_seq in target_seq or target_seq in cur_seq:
                I[i,j] = -1

    return I

def find_l2_distance(text):
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("model") 
    translator = Translator()
    newsapi = NewsApiClient(api_key='e24dd3440d0443f48b53a0f8bb7cf97b')
    nltk.download('punkt')
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    translation = translator.translate(text)
    source_article = translation.text
    source_article_sentences = doc_to_sentences(source_article)
    source_length = len(source_article_sentences)

    keys = get_keywords_yake(source_article)
    keywords = ''
    count = 0
    for word in keys.split():
        if keywords.count(word) == 0:
            keywords += word + ' '
            count += 1
        if count > 5:
            break

    all_articles = get_articles(keywords)
    rel_articles = get_related_articles(all_articles, 15)
    rel_article_sentences = doc_to_sentences(rel_articles)

    sentence_embeddings = np.ascontiguousarray(sbert_model.encode(source_article_sentences + rel_article_sentences))
    index = index_embeddings(sentence_embeddings)
    D,I = get_nearest_neighbors(index, sentence_embeddings, k=8)
    total_l2_distance = 0
    for i in range(source_length):
        for j in range(8):
            if I[i][j] > source_length:
                total_l2_distance += D[i][j]
                break
    avg_l2_distance = total_l2_distance / source_length

    return length_rel_article_sentences, avg_l2_distance

def generate_output(text):
    length_sentences, avg_l2_distance = find_l2_distance(text)
    if (length_sentences != 0 and avg_l2_distance < 100):
        st.markdown("<h1><span style='color:green'>This is a real news article!</span></h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1><span style='color:red'>This is a fake news article!</span></h1>", unsafe_allow_html=True)   

desc = "This web app detects fake news.\
        You can either enter the URL of a news article, or paste the text directly."

st.title("Fake News Detector")
st.markdown(desc)

st.subheader("Enter the URL address/text of a news article")
select_input = st.radio("Select Input:", ["URL", "Text"])

if select_input == "URL":
    url = st.text_input("URL")   
    if st.button("Run"):
        text = get_page_text(url)  
        generate_output(text)
        
else:
    text = st.text_area("Text", height=300)
    if st.button("Run"):
        generate_output(text)


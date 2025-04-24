import streamlit as st
import joblib
import re
import os
import requests
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Load necessary NLTK data
try:
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
regex = re.compile('[^a-zA-Z]')

@st.cache_resource
def load_transformer_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

model1 = load_transformer_model()
nlp = load_spacy_model()

try:
    model = joblib.load('model/fake_news_model.pkl')
    vectorizer = joblib.load('model/tfid_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check your file paths.")

st.title("Fake News Predictor")
st.subheader("Enter News Title")

# Input for title only
title = st.text_input("Title")

def fetch_news(query):
    """Fetch news articles based on a query"""
    one_month = datetime.today() - relativedelta(months=1)
    date_str = one_month.strftime('%Y-%m-%d')
    api_key = os.getenv("NEWS_API_KEY")  # Replace with a safer environment variable in practice
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&from={date_str}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        with open('api_news/requests.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        st.write("Please wait for a few moments...")
    except requests.exceptions.RequestException as e:
        st.error(f"Error in API request: {e}")

def fetch_entity(text):
    """Extract essential entities from text using SpaCy"""
    doc = nlp(text)
    entity_set = set()
    for entity in doc.ents:
        essential_entities = [
            "PERSON", "FAC", "GPE", "LOC", "NORP", "EVENT", "LAW", "PRODUCT", "WORK_OF_ART"
        ]
        if entity.label_ in essential_entities:
            entity_set.add(entity.text)
    
    entity_text = " ".join(entity_set)
    return entity_text if entity_text else "world"

def show_related_news(entity_text):
    """Fetch and display related news articles if the news is real"""
    fetch_news(entity_text)
    
    with open("api_news/requests.json", "r") as f:
        t = json.load(f)
        articles = t["articles"]
        st.write("Here are some related news articles:")
        for article in articles:
            st.write(f"**Title**: {article['title']}")
            st.write(f"**Description**: {article['description']}")
            st.write(f"[Read more]({article['url']})")
            st.write("----")

if st.button("Predict"):
    entity_text = fetch_entity(title)
    fetch_news(entity_text)

    highest_cosine = 0
    with open("api_news/requests.json", "r") as f:
        t = json.load(f)
        articles = t["articles"]
        for article in articles:
            article1 = article["title"] + article["description"]
            article2 = title
            embeddings = model1.encode([article1, article2])
            cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
            if cosine_sim[0][0] > highest_cosine:
                highest_cosine = cosine_sim[0][0]

    input_text = regex.sub(' ', title).lower()
    input_text_tfidf = vectorizer.transform([input_text])
    
    try:
        prediction = model.predict(input_text_tfidf)
        predict = ""
        if prediction[0] == 0 and highest_cosine >= 0.3:
            predict = "Verified News"
        elif prediction[0] == 0 and highest_cosine <= 0.1:
            predict = "Questionable"
        elif prediction[0] == 1 and highest_cosine >= 0.5:
            predict = "Likely True"
        elif prediction[0] == 1 and highest_cosine >= 0.2:
            predict = "Potentially Misleading"
        else:
            predict = "❌ FAKE News"
            
        
        st.write("Prediction:", f"{predict}")
        
        # If the prediction is real news, show related articles
        if predict in ["Verified News", "Likely True"]:
            show_related_news(entity_text)
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")

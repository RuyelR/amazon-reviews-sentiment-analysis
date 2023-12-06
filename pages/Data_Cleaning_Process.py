# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import streamlit as st
from nltk import word_tokenize
from wordcloud import STOPWORDS
from transformers import pipeline
from utils import show_code


# Use Reviews_250.csv and NLTK to get a cleaned dataset 
# Word Tokenize/Sentence Tokenize/2 word tokenize etc.
# Stopword collection
# Stemming the tokens

data_df = pd.read_csv('pages/Reviews_250.csv')
sentiment_pipeline = pipeline("sentiment-analysis")

def clean_data():

    show_code(stopwords, 'Stopwords')
    show_code(tokenization, 'Tokenization')
    show_code(dataset_stats, 'Dataset Stats')
    show_code(test_sentiment, 'Sentiment Test')
    test_sentiment()
    dataset_stats()


def stopwords():
    # Stopwords
    my_stopwords = set(STOPWORDS)
    custom_input = st.text_input(label="Custom stopwords: ", value="", max_chars=50, help='Write the words comma seperated', placeholder="film, movie, cinema, theatre, ...")
    if custom_input:
        custom_stopwords = list(custom_input.split(', '))
        my_stopwords.update(custom_stopwords)
        st.write(custom_stopwords)
    return my_stopwords

def tokenization():
    # Tokenization
    word_tokens = [word_tokenize(review) for review in data_df.Text]
    cleaned_tokens = [[word for word in item if word.isalpha()] for item in word_tokens]
    # cleaned_tokens has all the text tokenized
    st.write(f"All reviews has been tokenzied. Number of texts tokenized: {len(cleaned_tokens)}")
    # st.write("Example: ", cleaned_tokens[random.randint(0,len(cleaned_tokens))])
    return cleaned_tokens

def dataset_stats():
    # Quick Stats
    len_review = data_df.Text.str.len()
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Stats: ")
        st.write("Number of reviews: ", len(data_df))
        st.write("Longest review: ", max(len_review), "characters", sep=" ")
        st.write("Shortest review: ", min(len_review), "characters", sep=" ")
        st.write("All reviews are in: :blue[English]")
        
    with col2:
        st.write("Number of reviews per stars: ")
        st.write(data_df.Stars.value_counts())
    

def test_sentiment():
    var = 4
    reivew = data_df.Text.iloc[var]
    st.write(reivew)
    st.write(sentiment_pipeline(reivew))
    seperator = ' '
    cleaned_tokens = tokenization()
    reivew_token = seperator.join(cleaned_tokens[var])
    st.write(reivew_token)
    st.write(sentiment_pipeline(reivew_token))


st.set_page_config(page_title="Data cleaning process", page_icon=":shark:")
st.markdown("# :green[Data cleaning process]")
st.sidebar.header("Data cleaning process")
st.write(
    """
    This page illustrates the process of cleaning the data. 
    """
)
clean_data()
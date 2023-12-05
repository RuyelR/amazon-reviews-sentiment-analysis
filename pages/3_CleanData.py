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
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from wordcloud import STOPWORDS
from transformers import pipeline
# Use Reviews_250.csv and NLTK to get a cleaned dataset 
# Tokenize and stem the words
# Save the dataset for other python files to use

data_df = pd.read_csv('pages/Reviews_250.csv')
def read_file():
    st.write(data_df.Score.value_counts())
    len_review = data_df.Text.str.len()

    if st.toggle(label="Longest vs Shortest review", value=False):
        st.write("Longest review: ", max(len_review), "characters", sep=" ")
        st.write("Shortest review: ", min(len_review), "characters", sep=" ")

    # Stopwords
    my_stopwords = set(STOPWORDS)
    cstpwds = st.text_input(label="Custom stopwords: ", value="", max_chars=50, help='Write the words comma seperated', placeholder="film, movie, cinema, theatre, ...")
    if cstpwds:
        custom_stopwords = list(cstpwds.split(', '))
        st.write(custom_stopwords)
        my_stopwords.update(custom_stopwords)
    
    sentiment_pipeline = pipeline("sentiment-analysis")
    data = ["I love you", "I hate you"]
    sentiment_pipeline(data)




    # Tokenization
    word_tokens = [word_tokenize(review) for review in data_df.Text]
    cleaned_tokens = [[word for word in item if word.isalpha()] for item in word_tokens]
    st.write(len(cleaned_tokens))
    # st.write(cleaned_tokens[6])



st.set_page_config(page_title="Clean Data", page_icon="📊")
st.markdown("# :green[Clean Data]")
st.sidebar.header("Clean Data")
st.write(
    """
    This demo illustrates the process of cleaning the data. 
    """
)
# st.write(    
#     "##### :rainbow[Sentiment Distribution Plot, Word Cloud, Topic Modeling]"
# )
read_file()
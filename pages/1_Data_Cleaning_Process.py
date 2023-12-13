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
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from wordcloud import STOPWORDS
from transformers import pipeline
from utils import show_code
from nltk.stem import WordNetLemmatizer

# Use Reviews_250.csv and NLTK to get a cleaned dataset 
# Word Tokenize/Sentence Tokenize/2 word tokenize etc.
# Stopword collection
# Stemming the tokens

# read csv for pandas dataframe
data_df = pd.read_csv('pages/Reviews_2622.csv')
# bring in sentiment-analysis pretrained model from huggingface
sentiment_pipeline = pipeline("sentiment-analysis")

cleaned_tokens = pd.read_csv('pages/tokens.csv')

def data_processing():
    # Shorten the texts down to 500 char for sentiment analysis limit, create tokens.csv
    # data_df['Text'] = data_df['Text'].apply(shorten_text)
    ###     Create new token.csv
    # pd.DataFrame(clean_tokens()).to_csv(path_or_buf='/workspaces/amazon-reviews-sentiment-analysis/pages/tokens.csv',index=False)
    ###

    test_num = st.number_input(
            label=f'Enter which review to check. Out of {len(data_df)}', 
            min_value=0, max_value=len(data_df),
            value=4, help="Enter an integer index from 0 - 250 to check that review's sentiment"
            )
    test_sentiment(test_num)
    dataset_stats()

    show_code(stopwords, 'Stopwords')
    show_code(tokenization, 'Tokenization')
    show_code(dataset_stats, 'Dataset Stats')
    show_code(lemmatization, 'Lemmatization')
    show_code(test_sentiment, 'Sentiment Test')

def stopwords():
    # Stopwords
    my_stopwords = set(STOPWORDS)
    if custom_input:
        custom_stopwords = list(custom_input.split(', '))
        my_stopwords.update(custom_stopwords)
        st.write(custom_stopwords)
    return my_stopwords

def shorten_text(text):
    # Reduces text size to under 500 length
    if len(text) > 500:
        return text[:500]
    else:
        return text

def tokenization():
    # Tokenization
    word_tokens = [word_tokenize(review) for review in data_df.Text]
    tokenized_text = [[word for word in item if word.isalpha()] for item in word_tokens]
    # cleaned_tokens has all the text tokenized
    return tokenized_text

def lemmatization(tokenized_text):
    # Lemmatization
    WNlemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [[WNlemmatizer.lemmatize(word) for word in text_tokens] for text_tokens in tokenized_text]
    return lemmatized_tokens

def clean_tokens():
    stopwords_set = stopwords()
    lemmatized_tokens = lemmatization(tokenization())
    my_bar = st.progress(value=0)
    clean_tokens=[]
    size=(len(lemmatized_tokens)/100)
    for i, text_tokens in enumerate(lemmatized_tokens):
        my_bar.progress(value=int(i//size))
        clean_tokens.append([word for word in text_tokens if word not in stopwords_set])
    st.success(body='Cleaned tokens have been created.')
    return clean_tokens

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
        st.write(data_df.Score.value_counts())
    
def test_sentiment(var = 4):
    reivew = data_df.Text.iloc[var]
    st.subheader(':violet[The base review being tested:]')
    st.write(reivew)
    X = sentiment_pipeline(reivew)
    st.write(X)
    score_non_tk =  X[0]['score']
    typical_tokens = tokenization()
    typical_test = ' '.join(typical_tokens[var])
    st.subheader(":blue[Tokenized output:]")
    st.write(":blue[Removed anything that doesn't contain alpha-numeric values.]")
    st.write(typical_test)
    
    # since cleaned_token is now csv with text in cell and None in empty cells
    # below: i drop None values from row[var] and change to list. Then its joined back into str. 
    
    cl_tk_list = cleaned_tokens.iloc[var].dropna().to_list()
    review_token = ' '.join(cl_tk_list)
    st.subheader(":green[Cleaned output:] ")
    st.write(':green[All words were tokenized, then lemmatized, and finally stopwords were removed.]')
    st.write(review_token)
    Y = sentiment_pipeline(review_token)
    st.write(Y)
    score_tk =  Y[0]['score']
    st.metric('Confidence score difference', value=score_tk, delta=score_tk-score_non_tk)




st.set_page_config(page_title="Data cleaning", page_icon=":shark:")
st.markdown("# :green[Data cleaning]")
st.sidebar.header("Data cleaning")
st.write(
    """
    This page illustrates the process of cleaning the data for our purposes. 
    The side panel will show checkboxes for each action we took in code blocks. 

    Below shows a custom stopwords input where we can remove certain words from all the reviews. 
    After that, is an example of sentiment analysis done on the base text and then tokenized text.
    There is a small difference between using raw texts compared to tokenized text however it is 
    a small fraction. 
    
    A cleaned output can mitigate some issues where symbols or suffixes can 
    interfere with the model, giving us poor results. Thus it is always optimal to clean the data 
    before is is used by the model.  
    """
)
custom_input = st.text_input(
label="Custom stopwords: ", 
max_chars=50, help='Write the words comma seperated', 
placeholder="film, movie, cinema, theatre, ...",
)

data_processing()
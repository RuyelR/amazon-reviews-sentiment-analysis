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
from transformers import pipeline

# read csv for pandas dataframe
data_df = pd.read_csv('pages/Reviews_250.csv')
# bring in sentiment-analysis pretrained model from huggingface
sentiment_pipeline = pipeline("sentiment-analysis")

cleaned_tokens = pd.read_csv('pages/tokens.csv')

def sa_application():
    create_label('dataframe')

def create_label(choice):
    if choice is 'dataframe':
        df_labels = []
        for text in data_df.Text[25:30].values:
            X = sentiment_pipeline(text)
            df_labels.append(X[0]['label'])
        st.write(df_labels[:])
    elif choice is 'token':
        tk_sentences = []
        for i in range(len(cleaned_tokens)):
            temp_list = cleaned_tokens.iloc[i].dropna().to_list()
            temp_sentence = ' '.join(temp_list)
            tk_sentences.append(temp_sentence)
        predictions = sentiment_pipeline(tk_sentences)
        tk_labels = [prediction['label'] for prediction in predictions]
        st.write(tk_labels[1:4])


def text_sentiment():
    pass
    

st.set_page_config(page_title="Sentiment Analysis Application", page_icon="📈")
st.markdown("# :blue[Sentiment Analysis Application]")
st.sidebar.header("Sentiment Analysis Application")
st.write(
    """This demo illustrates a combination of Quantitative Evaluations. 
    The Algorithms performance can be measured using the following quantitative metrics:
    """)

sa_application()
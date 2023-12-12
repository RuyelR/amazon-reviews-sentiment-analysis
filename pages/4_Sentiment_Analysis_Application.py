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
data_df = pd.read_csv('pages/Reviews_2622.csv')
# bring in sentiment-analysis pretrained model from huggingface
sentiment_pipeline = pipeline("sentiment-analysis")

cleaned_tokens = pd.read_csv('pages/tokens.csv')

def sa_application():
    user_text_app()
    pass

def create_label():
    all_sentiments = []
    for i in range(len(data_df)):
        st.write(i)
        cl_tk_list = cleaned_tokens.iloc[i].dropna().to_list()
        review_tk_str = ' '.join(cl_tk_list)
        label = sentiment_pipeline(review_tk_str)
        all_sentiments.append(label[0]['label'])
    data_df['Label'] = all_sentiments
    data_df.to_csv('Reviews_2622.csv', index=False)


def user_text_app():
    st.header('Sentiment analysis on custom review ')
    review_txt = st.text_input('Write your review:', 'I love this product')
    'Your Review:'
    review_txt
    custom_df = st.dataframe(data=sentiment_pipeline(review_txt), use_container_width=True)
    

st.set_page_config(page_title="Sentiment Analysis Application", page_icon="📈")
st.markdown("# :blue[Sentiment Analysis Application]")
st.sidebar.header("Sentiment Analysis Application")
st.write(
    """This illustrates our application in action. 
    You can experience how the artificial intelligence model performs sentiment anlysis to hundreds of reviews using your own custom text.
    You can also explore the models performance on the Amazon Reviews dataset. Using selected products and multiple reviews to choose from. 
    """)

sa_application()
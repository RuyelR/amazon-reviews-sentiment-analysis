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
import numpy as np
from urllib.error import URLError
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

# Sentiment Distribution Plots: sentiment balance in datasheet  - bar graph with all products
# Word_clouds: (+,-)catagory. prominent factors driving sentiment
# Topic Modeling: LDA to id key topics in each sentiment category

data_df = pd.read_csv('pages/Reviews_2622.csv')

text = open('pages/randomtext.txt').read()

def qualitative_evals():
    sentiment_dist_plots()
    text_input = st.text_input('Enter a review to turn into wordcloud: ')
    if len(text_input) > 5:
        word_cloud_eval(text_input)
    word_cloud_eval()
    topic_modeling_eval()

def sentiment_dist_plots():
    st.header('Sentiment Distribution Plot')
    st.write("""We can see the ratio of positive to negative reviews per product giving 
             us an insight on how the sentiment is distributed across all products.""")
    product_df = data_df.loc[:, ['ProductId', 'Label']].copy()
    grouped = product_df.groupby('ProductId')
    positive_counts = grouped['Label'].apply(lambda x: (x == 'POSITIVE').sum()).tolist()
    negative_counts = grouped['Label'].apply(lambda x: (x == 'NEGATIVE').sum()).tolist()
    negative_counts_arr = np.array(negative_counts)
    negative_counts_arr *= -1       # make them negative so that it shows below 0
    product_list = product_df['ProductId'].unique()
    # st.write([len(x) for x in [positive_counts, negative_counts_arr, product_list]])
    chart_data = pd.DataFrame(
   {"Products": product_list, "Positives": positive_counts, "Negatives": negative_counts_arr}
    )
    st.bar_chart(
        chart_data, x="Products", y=["Positives", "Negatives"], color=["#FF0000", "#0000FF"]
    )


def wc_text_by_product():
    pass

def wc_colormap():
    pass

def word_cloud_eval(text=text):
    st.header("Word Cloud: ")
    st.write("""
             Word clouds can demonstrate prominent factors that are driving a 
             sentiment from a given user review. It shows what the algorithm is 
             most likely concentrating on when determining sentiment.
             """)
    wc_stopwords = set(STOPWORDS)
    # Set list needs set to update so use []
    wc_stopwords.update(['yet', 'sentiments'])
    wordcloud = WordCloud(
        max_words=50, margin=10,
        random_state=1, stopwords=wc_stopwords,
        ).generate(text)
    # wordcloud.stopwords
    st.image(image=wordcloud.to_image(), caption="Word Cloud from User Review", use_column_width=True)

def topic_modeling_eval():
    pass

st.set_page_config(page_title="Qualitative Evaluations", page_icon="📊")
st.markdown("# :green[Qualitative Evaluations]")
st.sidebar.header("Qualitative Evaluations")
st.write(
    """
    This demo illustrates a combination of Quantitative Evaluations. 
    The Algorithms performance can be measured using the following qualitative metrics:
    """
)
st.write(    
    "##### :rainbow[Sentiment Distribution Plot, Word Cloud, Topic Modeling]"
)

qualitative_evals()
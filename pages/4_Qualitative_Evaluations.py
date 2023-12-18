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
import streamlit as st
from utils import show_code
from wordcloud import WordCloud, STOPWORDS

# Sentiment Distribution Plots: sentiment balance in datasheet  - bar graph with all products
# Word_clouds: (+,-)catagory. prominent factors driving sentiment
# Topic Modeling: LDA to id key topics in each sentiment category

data_df = pd.read_csv('pages/Reviews_2622.csv')

def qualitative_evals():
    sentiment_dist_plots()
    positive_text, negative_text = wc_text_by_product()
    st.header("🌦️ Word Cloud: ")
    st.write("""
             Word clouds can demonstrate prominent factors that are driving a 
             sentiment from a given user review. It shows what the algorithm is 
             most likely concentrating on when determining sentiment.

             For evaluation purposes we are using the first product (B000CQBZV0) with an even 50/50 positive to negative review.
             """)
    col1, col2 = st.columns(2)
    with col1:
        word_cloud_eval(positive_text, 'Blues', 'Positive')
    with col2:
        word_cloud_eval(negative_text, 'Reds', 'Negative')
    # topic_modeling_eval()
    
    show_code(sentiment_dist_plots, 'Sentiment Distribution Plot')
    show_code(word_cloud_eval, 'Word Cloud Evaluation')
    show_code(wc_text_by_product, 'Text used in wordcloud')


def sentiment_dist_plots():
    st.header('📊 Sentiment Distribution Plot')
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
    text_df = data_df.loc[0:100,['ProductId','Text','Label']].copy()
    grouped = text_df.groupby('Label')
    # Join the text strings for each group into long strings
    positive_text = ' '.join(grouped.get_group('POSITIVE')['Text'])
    negative_text = ' '.join(grouped.get_group('NEGATIVE')['Text'])
    return positive_text, negative_text


def word_cloud_eval(text, colors_pick=None, caption_txt='Dataset'):
    wc_stopwords = set(STOPWORDS)
    # Set list needs set to update so use []
    wc_stopwords.update(['br', '<', '>', 'will'])
    wordcloud = WordCloud(
        max_words=500, margin=10, max_font_size=40,
        random_state=1, stopwords=wc_stopwords, colormap=colors_pick,
        ).generate(text)
    # wordcloud.stopwords
    st.image(image=wordcloud.to_image(),caption=caption_txt+' reviews only', use_column_width=True)

def topic_modeling_eval():
    st.header('Topic Modeling')
    st.write('The Latent Dirichlet Allocation (LDA) for Topic Modeling ')
    pass

st.set_page_config(page_title="Qualitative Evaluations", page_icon="📀")
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
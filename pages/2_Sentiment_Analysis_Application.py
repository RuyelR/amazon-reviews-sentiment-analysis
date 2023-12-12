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
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from transformers import pipeline

# read csv for pandas dataframe
data_df = pd.read_csv('pages/Reviews_2622.csv')
# bring in sentiment-analysis pretrained model from huggingface
sentiment_pipeline = pipeline("sentiment-analysis")

cleaned_tokens = pd.read_csv('pages/tokens.csv')

def sa_application():
    user_text_app()
    product_review_stats()
    st.header("Sentiment analysis from Amazon customer reviews")
    product_dict = find_product_indeces()
    pick_options = list(product_dict.keys())
    f'There are :green[{len(pick_options)}] products.'
    picked = st.radio(label='Pick a product ID', options=pick_options, horizontal=True)
    product_review_stats(picked, product_dict)
    wordcloud_tagging(picked, product_dict)
    pass

# Don't use create_label(). Bulk operation causes 100% CPU usage, then fails.
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
    

def product_review_stats(picked=None, product_dict=None):
    if picked:
        start, end = product_dict[picked]
        col1, col2 = st.columns(2)
        with col1:
            st.header(picked)
            average_stars = round(data_df.loc[start:end, 'Score'].mean())
            median_stars = round(data_df.loc[start:end, 'Score'].median())
            ':star:'* average_stars + f'  {average_stars} / 5 stars on avarage'
            ':star:'* median_stars + f'  {median_stars} / 5 stars in median'
            median_comment_time = round(data_df.loc[start:end, 'Time'].median())
            # Turning epoch time to timestamp
            '## Comment date in median'
            dt_median = dt.datetime.fromtimestamp(median_comment_time)
            f'### Date: {dt_median.month} / {dt_median.day} / {dt_median.year}'
            # f'Time: {dt_median.hour} hour : {dt_median.minute} min : {dt_median.second} sec'

        with col2:
            st.write('## Sentiment count')
            labels = data_df.loc[start:end, 'Label'].value_counts()
            st.dataframe(labels, use_container_width=True)
            ratio = labels['POSITIVE'] / labels['NEGATIVE']
            if ratio >2:
                color = 'green'
            elif ratio >1:
                color = 'orange'
            else:
                color = 'red'
            '### Ratio is: '+ f':{color}[{ratio:.2f}]  positive reviews per negative review'


def wordcloud_tagging(picked=None, product_dict=None):
    # Build a word cloud generator from all the comments
    start, end = product_dict[picked]
    st.header("Tags from wordcloud: ")
    # Create one long text from all reviews
    text = data_df.loc[start:end, 'Text']
    text = ''.join([sent for sent in text])
    # Form stopwords list for the wordcloud to use
    stopwords = set(STOPWORDS)
    custom_input = 'br, will'
    custom_stopwords = list(custom_input.split(', '))
    stopwords.update(custom_stopwords)
    # Extract image as array
    prime_boxes = np.array(Image.open('pages/prime-boxes.png'))
    # Wordcloud generation
    wordcloud = WordCloud(
        background_color='white',
        mask=prime_boxes,
        stopwords=stopwords,
        max_words=200, margin=10,
        random_state=1,
        ).generate(text)
    
    image_colors = ImageColorGenerator(image=prime_boxes,default_color=(200,10,100))
    image_colors2 = wordcloud.random_color_func()
    # st.write(image_colors.image)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(wordcloud, interpolation="bilinear", )
    # # recolor wordcloud and show
    # # we could also give color_func=image_colors directly in the constructor
    axes[1].imshow(wordcloud.recolor(color_func=image_colors2), interpolation="bilinear")
    for ax in axes:
        ax.axis('off')
    st.pyplot(fig=fig,use_container_width=True)
    # st.image(image=wordcloud.to_image(), caption="Word Cloud from Dataset Reviews", use_column_width=True)
    pass


def single_review_analysis():
    # Build a word cloud generator from all the comments
    pass


def find_product_indeces():
    # Create a dictionary of product name and indices
    product_ranges = {}
    # iterate through unique IDs
    for product_id in data_df['ProductId'].unique():
        # using boolean masks to check if product IDs match. Then we make all index tolist
        indices = data_df.index[data_df['ProductId'] == product_id].tolist()
        if indices:
            # use the first and last item in indices list and product_id to make a dict entry
            product_ranges[product_id] = (indices[0], indices[-1])
    return product_ranges

st.set_page_config(page_title="Sentiment Analysis Application", page_icon="📈")
st.markdown("# :blue[Sentiment Analysis Application]")
st.sidebar.header("Sentiment Analysis Application")
st.write(
    """This illustrates our application in action. 
    You can experience how the artificial intelligence model performs sentiment anlysis to hundreds of reviews using your own custom text.
    You can also explore the models performance on the Amazon Reviews dataset. Using selected products and multiple reviews to choose from. 
    """)

sa_application()
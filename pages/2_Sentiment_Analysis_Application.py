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
    st.header('Sentiment analysis on custom review ')
    user_text_app()
    st.header("Sentiment analysis from Amazon customer reviews")
    product_dict = find_product_indeces()
    pick_options = list(product_dict.keys())
    st.write(f'There are :green[{len(pick_options)}] products.')
    picked = st.radio(label='Pick a product ID', options=pick_options, horizontal=True)
    product_review_stats(picked, product_dict)
    
    with st.sidebar:
        default = False
        wc_checkbox = st.checkbox(label='Wordcloud', value=default, help='Wordcloud of selected products reviews')
        timeline_checkbox = st.checkbox(label='Timeline', value=default, help='Timeline of selected products reviews')
        alert_checkbox = st.checkbox(label='Alert system', value=default, help='Sentiment alert system of selected products reviews')
    if wc_checkbox: wordcloud_tagging(picked, product_dict)
    if timeline_checkbox: products_timeline(picked, product_dict)
    if alert_checkbox: sentiment_alert(picked, product_dict)

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


def user_text_app(custom_key=None):
    review_txt = st.text_input(label='Write your review:', value='I love this product', key=custom_key)
    sentiment = sentiment_pipeline(review_txt)
    st.dataframe(data=sentiment, use_container_width=True)
    return sentiment

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
            '### Comment date in median'
            dt_median = dt.datetime.fromtimestamp(median_comment_time)
            f' Date: {dt_median.month} / {dt_median.day} / {dt_median.year}'
            # f'Time: {dt_median.hour} hour : {dt_median.minute} min : {dt_median.second} sec'

        with col2:
            st.write('## Sentiment analysis')
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
    st.header("Wordcloud of all reviews: ")
    # Create one long text from all reviews
    text = data_df.loc[start:end, 'Text']
    text = ''.join([sent for sent in text])
    # Form stopwords list for the wordcloud to use
    stopwords = set(STOPWORDS)
    custom_input = 'br, will'
    custom_stopwords = list(custom_input.split(', '))
    stopwords.update(custom_stopwords)
    # Extract image as array
    prime_boxes = np.array(Image.open('pages/amazon-boxes.png'))
    # Wordcloud generation
    rev_wordcloud = WordCloud(
        background_color=(245, 242, 233),
        mask=prime_boxes,
        stopwords=stopwords,
        max_font_size=40,
        max_words=2000, margin=10,
        random_state=42,
        ).generate(text)
    image_colors = ImageColorGenerator(image=prime_boxes)
    # Make the figure with color option given by the user or default.
    fig, axes = plt.subplots()
    if st.toggle(label='Color', help='Randomize color for clarity'):
        axes.imshow(rev_wordcloud, interpolation="bilinear", )
    else:
        axes.imshow(rev_wordcloud.recolor(color_func=image_colors), interpolation="bilinear",)
    axes.axis('off')
    st.pyplot(fig=fig,use_container_width=True)
    st.caption("Word Cloud from Product Reviews")
    # Test only raw worldcloud
    # st.image(image=wordcloud.to_image(), caption="Word Cloud from Dataset Reviews", use_column_width=True)


def calculate_middle_date(group):
    # Need this function to select middle dates from grouped time_dataframe.
    # Tried and failed: matching median(timestamps) and getting date,
    # sorting timestamps and getting date, 
    # sorting date and using median timestamps to get median date, etc...
    middle_index = len(group) // 2  # Calculate the index of the middle row
    middle_date = group.iloc[middle_index]['Date']  # Extract the Date value from the middle row
    return middle_date


def products_timeline(picked=None, product_dict=None):
    # Build a timeline for the product
    st.header("Time Series Analysis")
    st.write('We can plot sentiment trends to identify shifts in sentiment over different time periods')
    start, end = product_dict[picked]
    time_df = data_df.loc[start:end, ['Time', 'Score', 'Label']].copy()

    # The following uses datetime to extract m/d/y values.using strftime(string format time) we get string representation.
    formatted_dates = [dt.datetime.fromtimestamp(epoch).strftime('%m/%Y') for epoch in time_df['Time']]
    time_df['Date'] = formatted_dates
    time_df = time_df.sort_values(by='Date', ignore_index=True,ascending=False)

    # Group the DataFrame
    grouped = time_df.groupby(np.arange(len(time_df))//5)
    middle_dates = grouped.apply(calculate_middle_date).tolist()
    positive_counts = grouped['Label'].apply(lambda x: (x == 'POSITIVE').sum()).tolist()
    negative_counts = grouped['Label'].apply(lambda x: (x == 'NEGATIVE').sum()).tolist()
    negative_counts_arr = np.array(negative_counts)
    negative_counts_arr *= -1       # make them negative so that it shows below 0
    
    chart_data = pd.DataFrame(
   {"Date": middle_dates, "Positives": positive_counts, "Negatives": negative_counts_arr}
    )

    st.bar_chart(
        chart_data, x="Date", y=["Positives", "Negatives"], color=["#FF0000", "#0000FF"]  # Optional
    )
    

def sentiment_alert(picked=None, product_dict=None):
    start, end = product_dict[picked]
    labels = data_df.loc[start:end, 'Label'].value_counts()
    ratio = labels['POSITIVE'] / labels['NEGATIVE']
    ratio
    st.header('Sentiment alert system')
    st.write("""
             Based on differing sentiment between the products already existing 
             sentiment ratio and any new reviews sentiment, we can create an alert 
             when opinions are actively changing. 
             
             Try to write a review for the selected product with opposing sentiment 
             to what the reviews currently show
             """)
    sentiment = user_text_app('sentimentAlert')
    sentiment = sentiment[0]['label']
    neutral_case = 'No difference in sentiment.'
    if ratio < 1:
        if sentiment == 'POSITIVE':
            st.success('New reviews are glowing!')
        else:
            st.info(neutral_case)
    else:
        if sentiment == 'NEGATIVE':
            st.error(body='New reviews are declining.')
        else:
            st.info(neutral_case)


def bar_graph_test():
    ## Test bar graph creation  ###
    my_arr = np.arange(1,11)
    ## [::-1] is a slicing notation used to reverse the order of the array.
    my_arr_inv = (my_arr*-1)[::-1]
    chart_data = pd.DataFrame(
   {"col1": list(range(10)), "col2": my_arr, "col3": my_arr_inv}
    )
    st.bar_chart(
        chart_data, x="col1", y=["col2", "col3"], color=["#0000FF", "#FF0000"]  # Optional
    )


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
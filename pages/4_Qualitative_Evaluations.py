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

from urllib.error import URLError
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# Sentiment Distribution Plots: sentiment balance in datasheet
# Word_clouds: (+,-,~)catagory. prominent factors driving sentiment
# Topic Modeling: LDA to id key topics in each sentiment category
clf = svm.SVC(kernel='linear', C=1, random_state=42)
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

text = open('pages/randomtext.txt').read()

def sentiment_dist_plots():
    pass

def word_cloud_eval(text):
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

sentiment_dist_plots()
word_cloud_eval(text)
topic_modeling_eval()
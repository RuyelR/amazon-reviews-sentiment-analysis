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

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Sentiment Analysis",
        page_icon=":👋:",
    )

    st.write("# :rainbow[Welcome to our Sentiment Analysis project!]")

    st.sidebar.success("Select a page above.")

    st.markdown("""TLDR; 
                
                You can use the Sentiment Analysis Application 
                page on the sidebar to start exploring the app concept.""")
    st.markdown(
        """
        The project demonstrates a detailed and thorough process of building a Machine Learning(ML) 
        application from scratch. To get started please pick a page from the side panel.

        - :green[Data Cleaning]
        - :blue[Sentiment Analysis Application]
        - :orange[Quantitative Evaluations]
        - :violet[Qualitative Evaluations]

        
        Starting at the Data cleaning process, the dataset is organized 
        and manipulated in ways that best fit our requirements. The process can 
        involve removing unneeded or missing(N/A) rows and columns, curating 
        stopwords to remove from texts, tokenizing the texts,lemmatization 
        of those tokens, and even limiting the number of charecters per text 
        to fit the models parameter.

        For our application, we used the base sentiment analysis model from 🤗 
        Huggingface model pipeline. A pipeline leverages pre-trained models, can handle 
        all the complexities of text processing and provide a simple interface for NLP 
        applications. We used the general sentiment analysis model to determine the 
        sentiment of our unlabled data to create a labeled dataset.

        Next step for our project was to build the sentiment analysis model itself 
        using the tokens and the labels created previously. We created a custom 
        transformer that produced TF-IDF based feature representation of the texts.
        The transformed text is then fed into our Sentiment analysis model. 
        We demonstrate our custom model in the Quantitative Evaluations page 
        comparing its performance to the 🤗 Huggingface models output.
        """)


if __name__ == "__main__":
    run()

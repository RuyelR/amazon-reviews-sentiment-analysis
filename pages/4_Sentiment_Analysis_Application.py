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
from pages.Data_Cleaning_Process import tokenization, stopwords


def sa_application():
    all_tokens = tokenization()
    my_stopwords = stopwords()
    clean_tokens = [[word for word in text_tokens if word not in my_stopwords] for text_tokens in all_tokens]
    st.write(clean_tokens[58])

    

st.set_page_config(page_title="Sentiment Analysis Application", page_icon="📈")
st.markdown("# :blue[Sentiment Analysis Application]")
st.sidebar.header("Sentiment Analysis Application")
st.write(
    """This demo illustrates a combination of Quantitative Evaluations. 
    The Algorithms performance can be measured using the following quantitative metrics:
    """)

sa_application()
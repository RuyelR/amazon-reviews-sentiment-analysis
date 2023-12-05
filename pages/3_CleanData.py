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
from sklearn.model_selection import train_test_split

# Use Reviews_250.csv and NLTK to get a cleaned dataset 
# Tokenize and stem the words
# Save the dataset for other python files to use

data_df = pd.read_csv('Reviews_250.csv')
data_df.value_counts()







st.set_page_config(page_title="Clean Data", page_icon="📊")
st.markdown("# :green[Clean Data]")
st.sidebar.header("Clean Data")
st.write(
    """
    This demo illustrates the process of cleaning the data. 
    """
)
# st.write(    
#     "##### :rainbow[Sentiment Distribution Plot, Word Cloud, Topic Modeling]"
# )

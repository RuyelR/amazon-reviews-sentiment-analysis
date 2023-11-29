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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Sentiment Analysis",
        page_icon=":handbag:",
    )

    st.write("# Welcome to Streamlit!")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        """
        )
    

def example_ml():
    X, y = datasets.load_iris(return_X_y=True)
    # st.write("X and y shapes: ", X.shape,", ", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return [X, y]


if __name__ == "__main__":
    run()

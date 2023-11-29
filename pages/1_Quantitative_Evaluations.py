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

import time

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code
'''
Accuracy, Precision, Recall, F1 Score: how well the model classifies sentiments
Confusion Matrix: misclassifying sentiments and which sentiments are often confused
Cross-Validation: robust and not overfitting.estimate the model's generalization performance.
Statistical Significance Testing:differences in sentiment are statistically significant
'''


def confusion_matrix_eval():

    pass

def statistical_significance_eval():
    pass

def cross_validation_eval():
    
    pass

def multi_metric_eval():
    pass


st.set_page_config(page_title="Quantitative Evaluations", page_icon="📈")
st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)


# show_code()

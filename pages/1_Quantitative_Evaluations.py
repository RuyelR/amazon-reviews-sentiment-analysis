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
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# Accuracy, Precision, Recall, F1 Score: how well the model classifies sentiments
# Confusion Matrix: misclassifying sentiments and which sentiments are often confused
# Cross-Validation: robust and not overfitting.estimate the model's generalization performance.
# Statistical Significance Testing:differences in sentiment are statistically significant
clf = svm.SVC(kernel='linear', C=1, random_state=42)
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def round_percent(num):
    return round(num*100, 2)

def cross_validation_eval(clf, X, y):
    st.header('Cross Validation Score:')
    cv_option=[2,3,4,5,6,]
    cv_pick = st.select_slider(label='Number of CV count',options=cv_option,value=4)
    scores = cross_val_score(clf, X, y, cv=cv_pick)
    st.write(f"### :green[{round_percent(scores.mean())}%] accuracy with a standard deviation of :red[{scores.std():0.2f}]")

def statistical_significance_eval():
    # here check if the difference between sentiment is significant (+, -, ~)
    st.header("Statistical Significance:")

def confusion_matrix_eval(clf, X, y, cls_names):
    st.header('Confusion Matrix:')
    disp = ConfusionMatrixDisplay.from_estimator(clf,X,y,display_labels=cls_names,cmap='Blues')
    st.pyplot(disp.figure_)

def multi_metric_eval(y_test, predictions):
    st.header("Multiple Metrics:")
    avarage_param = ['micro', 'macro', 'weighted']
    accuracy = accuracy_score(y_test, predictions)
    st.write("#### Accuracy score: ", round_percent(accuracy)) # Closer to 1(100) is better
    data = {'F1': [], 'Precision':[], 'Recall':[]}
    for param in avarage_param:
        F1score = round_percent(f1_score(y_test, predictions, average=param))
        precision = round_percent(precision_score(y_test, predictions, average=param))
        recall = round_percent(recall_score(y_test, predictions, average=param))
        data['F1'].append(F1score)
        data['Precision'].append(precision)
        data['Recall'].append(recall)
        # st.write("### Precision score for ", param,": ", round(precision*100 , 3)) # Closer to 1(100) is better
        # st.write("### F1 score for ", param,": ", round(F1score*100 , 3)) # Closer to 1(100) is better
        # st.write("### Recall score for ", param,": ", round(recall*100 , 3)) # Closer to 1(100) is better
    
    st.write("#### F1 score, Precision score, and Recall score \n by different avarage parameter: micro, macro, weighted ")
    data_df = pd.DataFrame(data=data, index=avarage_param)
    min_num = data_df.min()
    column_config = {}

    for title in data_df.columns.values:
        column_config[title] = st.column_config.ProgressColumn(
            title,
            format="    %f",
            min_value=min_num[title]-10,
            max_value=100,
            width='medium'
        )

    st.data_editor(
        data_df,
        column_config=column_config,
        hide_index=False,
    )


st.set_page_config(page_title="Quantitative Evaluations", page_icon="📈")
st.markdown("# Quantitative Evaluations")
st.sidebar.header("Quantitative Evaluations")
st.write(
    """This demo illustrates a combination of Quantitative Evaluations. 
    The Algorithms performance can be measured using the following qualitative metrics:
    """)
st.write(    
    "##### :rainbow[Accuracy, Precision, Recall, F1 Score, Cross-Validation, Confusion Matrix, and Statistical Significance Testing]"
)


try:
    cross_validation_eval(clf, X, y)
    statistical_significance_eval()
    confusion_matrix_eval(clf, X, y, class_names)
    multi_metric_eval(y_test, predictions)
except TypeError:
    st.error('Something went wrong!')

# show_code()

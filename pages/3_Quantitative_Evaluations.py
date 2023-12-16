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
import logging
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Accuracy, Precision, Recall, F1 Score: how well the model classifies sentiments
# Confusion Matrix: misclassifying sentiments and which sentiments are often confused
# Cross-Validation: robust and not overfitting.estimate the model's generalization performance.
# Statistical Significance Testing:differences in sentiment are statistically significant


df = pd.read_csv('pages/Reviews_2622.csv')

model_path="../models/model.pkl"
transformer_path="../models/transformer.pkl"

# Use Loaded Model
loaded_model = pickle.load(open(model_path, 'rb'))
loaded_transformer = pickle.load(open(transformer_path, 'rb'))


def quantitative_eval():
    # cross_validation_eval(clf, X, y)
    # statistical_significance_eval()
    # confusion_matrix_eval(clf, X, y, class_names)
    # multi_metric_eval(y_test, predictions)

    model_save()
    # print("\nAccuracy={0}; MRR={1}".format(accuracy,mrr_at_k))

    test_features=loaded_transformer.transform(["President Trump AND THE impeachment story !!!"])
    get_top_k_predictions(loaded_model,test_features,2)

def model_save():
    field = 'Text'
    top_k=3

    model,transformer=train_model(df,field=field,top_k=top_k)
    # we need to save both the transformer -> to encode a document and the model itself to make predictions based on the weight vectors 
    pickle.dump(model,open(model_path, 'wb'))
    pickle.dump(transformer,open(transformer_path,'wb'))



def extract_features(df,field,training_data,testing_data):
    # TF-IDF BASED FEATURE REPRESENTATION
    tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
    tfidf_vectorizer.fit_transform(training_data[field].values)
    
    train_feature_set=tfidf_vectorizer.transform(training_data[field].values)
    test_feature_set=tfidf_vectorizer.transform(testing_data[field].values)
    
    return train_feature_set,test_feature_set,tfidf_vectorizer

def get_top_k_predictions(model,X_test,k):
    
    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]
    
    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
    
    preds=[ item[::-1] for item in preds]
    
    return preds
   
    
def train_model(df,field="Text",top_k=3):
    
    logging.info("Starting model training...")
    
    # GET A TRAIN TEST SPLIT (set seed for consistent results)
    training_data, testing_data = train_test_split(df,random_state = 2000,)

    # GET LABELS
    Y_train=training_data['Label'].values
    Y_test=testing_data['Label'].values
     
    # GET FEATURES
    X_train,X_test,feature_transformer=extract_features(df,field,training_data,testing_data)

    # INIT LOGISTIC REGRESSION CLASSIFIER
    logging.info("Training a Logistic Regression Model...")
    scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
    model=scikit_log_reg.fit(X_train,Y_train)

    # GET TOP K PREDICTIONS
    preds=get_top_k_predictions(model,X_test,top_k)
    
    # GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS - for ease of evaluation
    eval_items=collect_preds(Y_test,preds)

    logging.info("Done training and evaluation.")

    return model,feature_transformer


def collect_preds(Y_test,Y_preds):
    """Collect all predictions and ground truth"""
    
    pred_gold_list=[[[Y_test[idx]],pred] for idx,pred in enumerate(Y_preds)]
    return pred_gold_list




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
st.markdown("# :blue[Quantitative Evaluations]")
st.sidebar.header("Quantitative Evaluations")
st.write(
    """This demo illustrates a combination of Quantitative Evaluations. 
    The Algorithms performance can be measured using the following quantitative metrics:
    """)
st.write(    
    "##### :rainbow[Accuracy, Precision, Recall, F1 Score, Cross-Validation, Confusion Matrix, and Statistical Significance Testing]"
)

quantitative_eval()

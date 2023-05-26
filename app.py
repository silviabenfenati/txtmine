from sklearn.base import accuracy_score
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

st.title('fake news classification')

uploaded_model = joblib.load('fakenews.pkl')

def user_input_features():
    user_input_text = st.text_input("Enter your text here")
    return user_input_text

df_user_input = user_input_features()

st.subheader('User Input')
st.write(df_user_input)

prediction = uploaded_model.predict([df_user_input])

st.subheader('Classification')
st.write(prediction[0])

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    X = data['text'] #feature
    y = data['class'] #target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=667)

    bow = CountVectorizer(max_features=27000, min_df=5, max_df=0.7)
    tfidf = TfidfTransformer()
    clf = MultinomialNB(alpha=0.1)

    pipe = Pipeline([
                    ('bow',bow),
                    ('tfidf',tfidf),
                    ('clf',clf),
                    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    st.subheader('Model Accuracy')
    st.write(accuracy_score(y_test, y_pred))

    scores = cross_val_score(pipe, X, y, scoring = 'f1_micro', cv = 8)

    st.subheader('Cross Validation Scores')
    st.write(f'scores={scores}')
    st.write(f'mean={np.mean(scores)}')
    st.write(f'std={np.std(scores)}')

    joblib.dump(pipe,'fakenews.pkl')

    #non ho neanche provato a runnarlo lol
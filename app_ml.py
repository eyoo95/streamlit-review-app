import streamlit as st
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
my_stopwords = stopwords.words('english')

import joblib
import numpy as np

def message_cleaning(sentence) :
  # 1. 구두점 제거
  Test_punc_removed = [char for char in sentence if char not in string.punctuation ]
  # 2. 각 글자들을 하나의 문자열로 합친다.
  Test_punc_removed_join = ''.join(Test_punc_removed)
  # 3. 문자열에 불용어가 포함되어있는지 확인해서, 불용어 제거한다.
  Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in my_stopwords ]
  # 4. 결과로 남은 단어들만 리턴한다.
  return Test_punc_removed_join_clean

def run_ml():
    st.text('문장을 입력하면 긍정, 부정을 예측한다.')

    sentence = st.text_input('문장입력')

    # 유저가 버튼을 누르면 예측하도록 만든다.

    if st.button('예측 실행'):

        classifier = joblib.load('data/classifier1.pkl')
        vec = joblib.load('data/vec.pkl')
        new_data = np.array([sentence]) # 함수에 바로 넣으면 안된다.
        X_new = vec.transform(new_data)
        X_new = X_new.toarray()
        y_pred = classifier.predict(X_new)
        
        if y_pred[0] == 5:
            st.text('입력하신 문장은 긍정입니다.')
        else:
            st.text('입력하신 문장은 부정입니다.')




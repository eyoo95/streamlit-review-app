import streamlit as st
from app_home import run_home
from app_eda import run_eda
from app_ml import run_ml
from app_ml import message_cleaning

def main():
    st.title('리뷰 긍정 부정 예측 앱')

    menu = ['Home','EDA','ML']
    choice = st.sidebar.selectbox('메뉴를 선택하세요.',menu)

    if choice == menu[0]:
        run_home()
    elif choice == menu[1]:
        run_eda()
    elif choice == menu[2]:
        run_ml()

if __name__ == '__main__':
    main()
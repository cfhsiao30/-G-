import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="G電商展示平台", page_icon="🌍", layout="wide")
st.title("🌍G電商展示平台")

st.write("歡迎來到G電商，這裡提供各式各樣的服飾及配件，歡迎您隨意探索")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_推薦商品.py", label="**👉 去『推薦商品』頁**")
with col2:
    st.page_link("pages/2_常見問題.py", label="**👉 去『常見問題』頁**")




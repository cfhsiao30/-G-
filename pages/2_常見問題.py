import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


st.subheader("❓ 智慧搜尋 × 購物FAQ")

st.divider()

DEFAULT_FAQ = pd.DataFrame(
    [
        {"question":"可以用自然語言輸入需求嗎？","answer":"可以，例如「夏天通勤的白鞋」或「防潑水輕外套」，系統會理解語意並推薦相關商品。"},
        {"question":"出貨後多久會送達？","answer":"約 1–3 個工作天內送達，依物流狀況可能不同。"},

    ]
)

if "faq_df" not in st.session_state:
    st.session_state.faq_df = DEFAULT_FAQ.copy()
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None

# 步驟一：上傳知識庫
st.write("##### 上傳QA知識庫")
file=st.file_uploader("上傳 FAQ 檔案 (限csv檔) ", type=["csv"])
if file is not None: #若有上傳，內容不是空的
    df=pd.read_csv(file)
    #st.write(df)
    #取代前面的DEFAULT_FAQ，去空白紀錄
    st.session_state.faq_df = df.dropna().reset_index(drop=True)
    st.success(f"已成功上傳{len(df)}筆資料")

with st.expander("檢視資料", expanded=False):
    st.dataframe(st.session_state.faq_df)
    #'''
    #try:
    #    st.session_state.faq_df = pd.read_csv(file)
    #    st.success("知識庫已更新！")
    #except Exception as e:
    #    st.error(f"檔案讀取錯誤: {e}")
    #'''

# 步驟二：建立索引
do_index=st.button("建立/重設索引")
def jieba_tokenize(text:str):
    return list(jieba.cut(text)) #將輸入的句子分詞，一個個列出來

if do_index or (st.session_state.vectorizer is None):
    corpus=(st.session_state.faq_df["question"].astype(str)+
            " "+
            st.session_state.faq_df["answer"].astype(str)).tolist()
    v=TfidfVectorizer(tokenizer=jieba_tokenize)
    tfidf=v.fit_transform(corpus)
    st.session_state.vectorizer=v
    st.session_state.tfidf=tfidf
    st.success("建立索引完成")

# 步驟三：詢問客服
st.write("##### 客服詢問平台")
q=st.text_input("請輸入問題",placeholder=("例如：如何申請退貨？"))
top_k=st.slider("取得前k筆回答",1,3)
c=st.slider("信心門檻",0.0,1.0,key="c")

#若按下按鈕且有輸入問題
if st.button("送出") and q.strip():
    #以防萬一雙重保障
    if (st.session_state.vectorizer is None) or (st.session_state.tfidf is None):
        #出現警告
        st.warning("尚未建立索引，會自動建立")
        #自動建立
        corpus=(st.session_state.faq_df["question"].astype(str)+
            " "+
            st.session_state.faq_df["answer"].astype(str)).tolist()
        
        v=TfidfVectorizer(tokenizer=jieba_tokenize)
        tfidf=v.fit_transform(corpus)
        st.session_state.vectorizer=v
        st.session_state.tfidf=tfidf
        st.success("建立索引完成")
    
    vec=st.session_state.vectorizer.transform([q])
    sims=linear_kernel(vec,st.session_state.tfidf).flatten()
    idxc=sims.argsort()[::-1][::top_k]
    rows=st.session_state.faq_df.iloc[idxc].copy()
    rows['score']=sims[idxc]
    
    best_ans=None
    best_score=float(rows['score'].iloc[0]) if len(rows) else 0.0
    if best_score>=c:
        best_ans=rows['answer'].iloc[0]

    if best_ans:
        st.success(best_ans)
    else:
        st.info("找不到適合的答案...請電洽")


    #展開可能的回答
    with st.expander("檢索結果：",expanded=False):
        st.dataframe(rows[["question","answer",'score']],use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="推薦商品", layout="wide")

@st.cache_resource
def load_model():
    # 多語模型，支援中英文查詢
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_products():
    df = pd.read_csv(Path("data") / "products.csv")
    df["text"] = (df["name"].astype(str) + "。"
                  + df["category"].astype(str) + "。"
                  + df["desc"].astype(str))
    return df

@st.cache_data(show_spinner=False)
def embed_texts(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = load_model()
    emb = model.encode(list(texts), normalize_embeddings=True)
    return emb

def search(query, prod_df, prod_emb, topk=6, category=None, price_min=None, price_max=None):
    # 過濾條件（可選）
    mask = np.ones(len(prod_df), dtype=bool)
    if category:
        mask &= (prod_df["category"] == category)
    if price_min is not None:
        mask &= (prod_df["price"] >= price_min)
    if price_max is not None:
        mask &= (prod_df["price"] <= price_max)

    sub_df = prod_df[mask].reset_index(drop=True)
    sub_emb = prod_emb[mask]

    if sub_df.empty:
        return sub_df, np.array([])

    q_emb = embed_texts([query])[0].reshape(1, -1)
    sims = cosine_similarity(q_emb, sub_emb)[0]
    idx = np.argsort(-sims)[:topk]
    return sub_df.iloc[idx].assign(score=sims[idx]), sims[idx]

def show_cards(df):
    cols_per_row = 3
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for c, (_, row) in enumerate(df.iloc[i:i+cols_per_row].iterrows()):
            with cols[c]:
                st.image(row.get("image_url", ""), use_container_width=True)
                st.markdown(f"**{row['name']}**｜${int(row['price'])}")
                st.caption(f"{row['category']}｜{row['desc']}")
                st.progress(float(row.get("score", 0)), text=f"相似度：{row.get('score', 0):.2f}")

def main():
    st.subheader("🔎 語意搜尋 × 商品推薦")
    st.write("請隨意填寫您想尋找的商品，例如：**適合夏天通勤、輕量、白鞋、好清洗**")
    st.divider()

    df = load_products()
    emb = embed_texts(df["text"])

    with st.sidebar:
        st.write("##### 篩選")
        category = st.selectbox("類別（可選）", options=["（全部）"] + sorted(df["category"].unique().tolist()))
        category = None if category == "（全部）" else category
        c1, c2 = st.columns(2)
        with c1:
            price_min = st.number_input("最低價（可選）", min_value=0, value=0, step=100)
        with c2:
            price_max = st.number_input("最高價（可選）", min_value=0, value=0, step=100)
            price_max = None if price_max == 0 else price_max
        topk = st.slider("顯示數量", 3, 12, 6)

    query = st.text_input("輸入你的需求 / 風格 / 場景", value="")
    if st.button("找找看"):
        with st.spinner("比對向量中…"):
            res_df, _ = search(query, df, emb, topk=topk, category=category,
                               price_min=price_min or None, price_max=price_max)
        if res_df.empty:
            st.warning("沒有符合條件的結果，換個描述或放寬篩選吧。")
        else:
            st.subheader("推薦結果")
            show_cards(res_df)

#    with st.expander("如何運作？", expanded=False):
#        st.markdown("""
#- 使用 **Hugging Face 多語句向量模型** 產生商品描述與查詢的向量。
#- 以 **餘弦相似度** 找出最接近你需求的商品。
#- 支援中文；可同時搭配類別與價格範圍過濾。
#        """)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import gensim

@st.cache_resource
def cache_list():
    list=[]
    return list

# 映画情報の読み込み
movies = pd.read_csv("data/movies.tsv", sep="\t")

# 学習済みのitem2vecモデルの読み込み
model = gensim.models.word2vec.Word2Vec.load("item2vec.model")

# 映画IDとタイトルを辞書型に変換
movie_titles = movies["title"].tolist()
movie_ids = movies["movie_id"].tolist()
movie_genres = movies["genre"].tolist()
moview_reviews = [0.5*(i+1) for i in range(10)]
movie_id_to_title = dict(zip(movie_ids, movie_titles))
movie_title_to_id = dict(zip(movie_titles, movie_ids))
movie_id_to_genre = dict(zip(movie_ids, movie_genres))

st.markdown("## 複数の映画を選んでおすすめの映画を表示する")

selected_movies = st.multiselect("映画を複数選んでください", movie_titles)
selected_movie_ids = [movie_title_to_id[movie] for movie in selected_movies]
vectors = [model.wv.get_vector(movie_id) for movie_id in selected_movie_ids]
if len(selected_movies) > 0:
    user_vector = np.mean(vectors, axis=0)
    st.markdown(f"### おすすめの映画")
    recommend_results = []
    for movie_id, score in model.wv.most_similar(positive=user_vector,topn=10+len(selected_movies)):
        title = movie_id_to_title[movie_id]
        genre = movie_id_to_genre[movie_id]
        recommend_results.append({"movie_id":movie_id, "title": title, "genre": genre, "score": score})
    recommend_results=[result for result in recommend_results if result["title"] not in selected_movies]
    recommend_results = pd.DataFrame(recommend_results)
    st.write(len(recommend_results),"件")
    st.write(recommend_results)
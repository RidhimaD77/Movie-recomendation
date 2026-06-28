import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# sabse pehle data set laaoo
df = pd.read_csv("movieData.csv") # abhi source laana hai
# fir dataset clean karoo


df = df[["title","overview"]]
df.dropna(inplace= True)
# print(df.isnull().sum())
# ab tfidf wala vectorize karo

tfidf = TfidfVectorizer(stop_words= "english")
tfidf_matrix = tfidf.fit_transform(df["overview"])
# cosine similarity karo
similarity = cosine_similarity(tfidf_matrix)


# print(df.head())
# print(tfidf_matrix.shape)
# printing similarity score and shape
# print(similarity.shape)
# index mapping
df = df.reset_index()
indices = pd.Series(df.index,index = df['title'])

def recommend(movie_title):
    index = indices[movie_title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key = lambda x: x[1],reverse = True)
    top_movies = similarity_scores[1:6]
    for i in top_movies:
        print(df.iloc[i[0]]["title"])

recommend("Rio")
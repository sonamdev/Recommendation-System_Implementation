# %%
import pandas as pd
import numpy as np

ratings = pd.read_csv('ratings.csv')

ratings.head()

# %%
ratings.isnull().sum()

# %%
movies = pd.read_csv('movies.csv')

movies.head()

# %%
movies.isnull().sum()

# %%
data = pd.merge(movies,ratings,how='inner',on='movieId')
data.head()

# %%
data = data.drop(['genres','timestamp'],axis=1)
data.head()

# %%
userRatings = data.pivot_table(index=['userId'],columns=['title'],values='rating')
userRatings.head()

# %%
userRatings = userRatings.dropna(thresh=10, axis=1)
userRatings = userRatings.fillna(0,axis=1)

userRatings.head()

# %%
%time corrMatrix = userRatings.corr(method='pearson')
corrMatrix.head()

# %%
def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*rating
    similar_ratings = similar_ratings.sort_values(ascending=False)
    similar_50_movies = similar_ratings.index[:50]
    similar_50_movies_ratings = similar_ratings.values[:50]
    dic = {}
    for sim_50_m , sim_50_m_r in zip(similar_50_movies , similar_50_movies_ratings):
        dic[sim_50_m] = sim_50_m_r
    return dic

# %%
get_similar("Interstellar (2014)",5)

# %%
def update_recommendations(rec , par_rec , rec_count):
    for movie_name , pred_rating in par_rec.items():
        if movie_name not in rec:
            rec[movie_name] = pred_rating 
        else :
            rec[movie_name] += pred_rating
            if movie_name not in rec_count :
                rec_count[movie_name] = 2 
            else :
                rec_count[movie_name] += 1
    return rec , rec_count

# %%
my_self = [("Interstellar (2014)",5),("Avengers, The (2012)",4),("Inception (2010)",5)]

final_recommendations = {}
recommendation_count = {}

for movie,rating in my_self:
    par_rec = get_similar(movie,rating)
    final_recommendations , recommendation_count = update_recommendations(final_recommendations , par_rec , recommendation_count)

for movie_name, _ in my_self :
    del final_recommendations[movie_name]
    del recommendation_count[movie_name]

# %%
for movie_name , count in recommendation_count.items():
    final_recommendations[movie_name] /= count

# %%
final_recommendations

# %%

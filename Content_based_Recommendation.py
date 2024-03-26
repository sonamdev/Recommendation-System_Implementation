# %%
"""
## Loading Dependencies
"""

# %%
#Reading all the required libraries

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("movie_dataset.csv")

# %%
data.head()

# %%
#Checking the shape of the dataset
data.shape

# %%
#Checking for the null values
data.isnull().sum()

# %%
data = data[~data['release_date'].isnull()]

# %%
data["release_year"] = data["release_date"].apply(lambda x : int(x.split("-")[0]))

# %%
def get_title_from_index(data , index):
    return data[data.index == index]["title"].values[0]

def get_index_from_title(data , title):
    return data[data.title == title]["index"].values[0]

def get_year_from_index(data , index):
    return data[data.index == index]["release_year"].values[0]

def get_year_from_title(data , title):
    return data[data.title == title]["release_year"].values[0]

# %%
features = ['cast','genres','director']

for feature in features:
    data[feature] = data[feature].fillna('')

# %%
def combine_features(row):
    try:
        return row["cast"]+" "+row["genres"]+" "+row["director"]
    except:
        pass

data["combined_features"] = data.apply(combine_features,axis=1)

# %%
cv = CountVectorizer()

count_matrix = cv.fit_transform(data["combined_features"])

# %%
count_matrix.toarray().shape

# %%
%time cosine_sim = cosine_similarity(count_matrix) 

# %%
movie_user_likes = "Interstellar"

movie_index = get_index_from_title(data , movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))

# %%
similar_movies

# %%

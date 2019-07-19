import pandas as pd
import numpy as np


r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('/home/snehamishra/Data Science/DataScience-Python3/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('/home/snehamishra/Data Science/DataScience-Python3/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)
print (ratings.head())

movierat = ratings.pivot_table( index=['user_id'] , columns=['title'] , values='rating')
starWarsRatings = movierat['Star Wars (1977)']
similarMovies = movierat.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()

movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
popularMovies = movieStats['rating']['size'] >= 100
pop = movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)

df = pop.join(pd.DataFrame(similarMovies, columns=['similarity']))
af = df.sort_values(['similarity'], ascending=False)
print(af.head(15))

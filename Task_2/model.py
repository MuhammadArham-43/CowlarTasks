import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import Union

df = pd.read_csv('data/movies_dataset.csv')

ratings_columns = df.columns[1:]

df[ratings_columns] = df[ratings_columns].fillna(0)

scaler = StandardScaler()
df[ratings_columns] = scaler.fit_transform(df[ratings_columns])

def get_recommendation(input_movie:str, num_recommendations:int=1) -> Union[list, str]:
    """
    Takes an movie and fetches its recommendations from the dataset.
    Uses Cosine Similarity metric to compare it with other movie recommendations in the dataset.
    Returns the top-k mathces from the dataset.

    Args:
        input_movie (str): name of movie
        num_recommendations (int, optional): Number of recommendations to return. Defaults to 1.

    Returns:
        Union[list, str]: Returns a list of movies to watch or a error string if the input movie is not found in the dataset.
    """
    
    input_movie_row = df[df['title'] == input_movie]

    if input_movie_row.empty:
        return "Movie not found in the dataset."

    input_movie_ratings = input_movie_row[ratings_columns].values

    similarity = cosine_similarity(df[ratings_columns], input_movie_ratings)

    similar_movies_indices = similarity[:, 0].argsort()[::-1][1:]

    recommended_movies = df.iloc[similar_movies_indices][:num_recommendations]['title']

    return recommended_movies.tolist()
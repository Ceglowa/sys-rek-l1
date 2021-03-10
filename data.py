import pandas as pd
import surprise
from surprise import Dataset
from typing import Tuple


def get_movielens_1m() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movielens = Dataset.load_builtin("ml-1m", prompt=False)

    movies_file = f"{surprise.get_dataset_dir()}/ml-1m/ml-1m/movies.dat"
    users_file = f"{surprise.get_dataset_dir()}/ml-1m/ml-1m/users.dat"
    ratings_file = f"{surprise.get_dataset_dir()}/ml-1m/ml-1m/ratings.dat"

    movies_df = pd.read_csv(
        movies_file, sep="::", names=["MovieID", "Title", "Genres"], engine="python"
    )

    users_df = pd.read_csv(
        users_file,
        sep="::",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        engine="python",
    )
    ratings_df = pd.read_csv(
        ratings_file,
        sep="::",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
    )

    return movies_df, users_df, ratings_df

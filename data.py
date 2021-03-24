import pandas as pd
import surprise
import torch
from surprise import Dataset
from typing import Tuple
import numpy as np
import torch.utils.data


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


class MovieLens1MDataset(torch.utils.data.Dataset):

    def __init__(self):
        _, _, ratings_df = get_movielens_1m()

        self.items = ratings_df.iloc[:, :2].astype(np.longlong) - 1  # -1 because ID begins from 1
        self.ratings = ratings_df.iloc[:, 2]

        self.dims = np.max(self.items, axis=0) + 1

        self.user_field_idx = np.array((0,), dtype=np.longlong)
        self.item_field_idx = np.array((1,), dtype=np.longlong)

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, index):
        return np.array(self.items.iloc[index]), np.array(self.ratings.iloc[index])
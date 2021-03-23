from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from data import get_movielens_1m
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# based on https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e


class MovielensDataModule(LightningDataModule):
    def prepare_data(self, *args, **kwargs):
        pass

    def __init__(self):
        super().__init__()
        self.user_train = None
        self.user_test = None
        self.movie_train = None
        self.movie_test = None
        self.rating_train = None
        self.rating_test = None
        self.movies = None
        self.users = None
        self.ratings = None

    def setup(self, **kwargs):
        movies_df, users_df, ratings_df = get_movielens_1m()
        self.movies = movies_df
        self.users = users_df
        self.ratings = ratings_df
        self.movies['m_id'] = self.movies.index
        self.users['u_id'] = self.users.index

        self.ratings = self.ratings.merge(self.movies[['MovieID', 'm_id']], left_on='MovieID', right_on='MovieID')
        self.ratings = self.ratings.merge(self.users[['UserID', 'u_id']], left_on='UserID', right_on='UserID')
        self.ratings['rating_scaled'] = MinMaxScaler().fit_transform(ratings_df['Rating'].values[:, None])
        self.ratings.sort_values(by='Timestamp', inplace=True)

        user_ids = self.ratings['u_id'].values[:, None]
        movie_ids = self.ratings['m_id'].values[:, None]
        rating = self.ratings['rating_scaled'].values[:, None]

        user_train, user_test, movie_train, movie_test, rating_train, rating_test = train_test_split(user_ids, movie_ids, rating, train_size=0.8, stratify=user_ids)

        self.user_train = user_train
        self.user_test = user_test
        self.movie_train = movie_train
        self.movie_test = movie_test
        self.rating_train = rating_train
        self.rating_test = rating_test

    def train_dataloader(self):
        train_split = MovieLensDataset(self.user_train, self.movie_train, self.rating_train)
        return DataLoader(train_split, batch_size=512)

    def val_dataloader(self):
        val_split = MovieLensDataset(self.user_test, self.movie_test, self.rating_test)
        return DataLoader(val_split, batch_size=512)

    def test_dataloader(self):
        val_split = MovieLensDataset(self.user_test, self.movie_test, self.rating_test)
        return DataLoader(val_split, batch_size=512)

    def n_users(self) -> int:
        return len(self.users)

    def n_movies(self) -> int:
        return len(self.movies)


class MovieLensDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

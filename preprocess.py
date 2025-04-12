import pandas as pd
import numpy as np
import random


def process_data():
    # Config
    holdouts_per_user = 2
    random.seed(42)
    np.random.seed(42)

    # Load Excel, skip the first column (num_rated)
    df = pd.read_excel('jester-data-1.xls', header=None)
    df.index = [f'user_{i}' for i in range(len(df))]
    df.columns = ['total'] + [f'J{i}' for i in range(1, 101)]

    # Adjust for zero indexing
    df = df.replace(99.0, np.nan)
    sub_df = df.iloc[:, [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]]

    # Prepare training matrix and holdouts
    train_matrix = sub_df.copy()
    holdout_list = []

    for user_id in train_matrix.index:
        user_ratings = train_matrix.loc[user_id]

        # Valid jokes are the ones this user rated (not NaN)
        valid_jokes = user_ratings.dropna().index.tolist()

        if len(valid_jokes) <= holdouts_per_user:
            continue

        # Randomly choose holdouts
        holdout_jokes = np.random.choice(valid_jokes, size=holdouts_per_user, replace=False)

        for joke_id in holdout_jokes:
            true_rating = train_matrix.at[user_id, joke_id]
            holdout_list.append((user_id, joke_id, true_rating))
            train_matrix.at[user_id, joke_id] = np.nan  # Mask the rating

    return sub_df, train_matrix, holdout_list

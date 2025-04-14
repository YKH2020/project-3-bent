from preprocess import process_data
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

def mean_model():
    # Load data
    original_df, train_df, val, holdout_list = process_data()
    # Compute mean ratings for each joke
    joke_means = train_df.mean(skipna=True)
    return joke_means, train_df, holdout_list

def evaluate(joke_means, holdout_list):
# Group holdouts by user
    user_to_holdouts = {}
    for user_id, joke_id, true_rating in holdout_list:
        user_to_holdouts.setdefault(user_id, []).append((joke_id, true_rating))

    ndcg_scores = []

    for user_id, holdouts in user_to_holdouts.items():
        if len(holdouts) < 3:
            continue  # Need at least 3 for NDCG@3

        # Get joke_ids and normalized relevance scores
        joke_ids = [jid for jid, _ in holdouts]
        raw_ratings = [rating for _, rating in holdouts]

        # Normalize ratings from [-10, 10] to [0, 1]
        normalized_relevance = [(r + 10) / 20 for r in raw_ratings]

        # Get predicted joke scores from mean model
        predicted_scores = [joke_means.get(jid, 0) for jid in joke_ids]

        # Format for ndcg_score (expects 2D)
        y_true = np.array([normalized_relevance])
        y_score = np.array([predicted_scores])

        score = ndcg_score(y_true, y_score, k=3)
        ndcg_scores.append(score)

    return ndcg_scores

def recommend_top_jokes(user): # for demo, pass in array of user ratings.
    joke_means, data, _ = mean_model()
    user_ratings = list(data.loc[user])
    joke_columns = ['J5', 'J7', 'J8', 'J13', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20']

    unrated_indices = [i for i, r in enumerate(user_ratings) if pd.isna(r)]
    
    if len(unrated_indices) < 3:
        raise ValueError("At least 3 NaNs (unrated jokes) are required to make recommendations.")
    
    # Gather predictions for unrated jokes using mean model
    preds = [(joke_columns[i], joke_means.get(joke_columns[i], 0)) for i in unrated_indices]
    
    # Sort by predicted mean rating, descending
    sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)
    
    # Return top 3 joke column names
    return [joke_id for joke_id, _ in sorted_preds[:3]]


if __name__ == "__main__":
    model_data, _, holdouts = mean_model()
    ndcg_scores = evaluate(model_data, holdouts)

    print(f"Average NDCG@3 (with normalized ratings): {np.mean(ndcg_scores):.3f}")
    print(recommend_top_jokes('user_1'))  # Example user for recommendation

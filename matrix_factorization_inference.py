import pandas as pd
import numpy as np
from preprocess import process_data

def inference(user_index,k=3):

    user_features = np.load('traditional_ml_model.npz')['user_features']
    item_features = np.load('traditional_ml_model.npz')['item_features']
    original_matrix, _, _, test_holdout_list = process_data()

    # Reconstruct the matrix using user and item features
    reconstructed_matrix = np.dot(user_features, item_features.T)

    # Convert to DataFrame (same index & columns as original)
    reconstructed_df = pd.DataFrame(
        reconstructed_matrix,
        index=original_matrix.index,
        columns=original_matrix.columns
    )

    predicted_scores = {}

    for user_id, joke_id, true_rating in test_holdout_list:
        if f"user_{user_index}" == user_id:
            predicted_score = reconstructed_df.at[user_id, joke_id]
            predicted_scores[joke_id] = predicted_score

    sorted_jokes = [key for key, value in sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)]
    top_k_jokes = sorted_jokes[:k]

    print(f"Top {k} jokes for user_{user_index}: {top_k_jokes}")
    return top_k_jokes

    
if __name__ == "__main__":
    inference(1)




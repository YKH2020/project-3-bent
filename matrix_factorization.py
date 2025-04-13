from preprocess import process_data
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD

def train(train_matrix,val_holdout_list):
    """
    Train a matrix factorization model using stochastic gradient descent (SGD) on the training matrix.
    Parameters:
    - train_matrix: DataFrame containing the training data.
    - val_holdout_list: List of tuples (user_id, item_id, true_rating) for validation.
    Returns:
    - user_features: User latent features matrix.
    - item_features: Item latent features matrix.
    - final_training_loss: Final training loss.
    - final_val_loss: Final validation loss.
    """
    
    # Convert DataFrame to numpy array
    train_matrix_converted = train_matrix.to_numpy()

    # L1 nomalization

    # Row-wise min and max, ignoring NaNs
    row_min = np.nanmin(train_matrix_converted, axis=1, keepdims=True)
    row_max = np.nanmax(train_matrix_converted, axis=1, keepdims=True)

    # Normalize: (x - min) / (max - min), ignoring NaNs
    normalized_train_matrix = (train_matrix_converted - row_min) / (row_max - row_min + 1e-8)

    # Set hyperparameters
    num_users, num_items = train_matrix.shape
    num_features = 10
    learning_rate = 0.01
    regularization = 0.02
    num_epochs = 10000

    # Initialize user and item latent feature matrices

    user_features = np.random.normal(scale=1./np.sqrt(num_features), size=(num_users, num_features))
    item_features = np.random.normal(scale=1./np.sqrt(num_features), size=(num_items, num_features))

    # Training loop
    for epoch in range(num_epochs):
        for user_id in range(num_users):
            for item_id in range(num_items):
                if not np.isnan(normalized_train_matrix[user_id][item_id]):
                    # Calculate the prediction error
                    prediction = np.dot(user_features[user_id], item_features[item_id])
                    error = normalized_train_matrix[user_id][item_id] - prediction

                    # Update user and item features
                    user_features[user_id] += learning_rate * (error * item_features[item_id] - regularization * user_features[user_id])
                    item_features[item_id] += learning_rate * (error * user_features[user_id] - regularization * item_features[item_id])

        # Optional: Print loss every 100 epochs
        if epoch % 100 == 0:
            reconstruction_loss = np.nansum((normalized_train_matrix - np.dot(user_features, item_features.T)) ** 2)
            regularization_loss = regularization * (np.sum(user_features**2) + np.sum(item_features**2))
            training_loss = reconstruction_loss + regularization_loss

            val_loss = cal_l2_loss(user_features, item_features, val_holdout_list, train_matrix)
            val_ndcg = cal_ndcg(user_features, item_features, val_holdout_list, train_matrix)
            print(f'Epoch {epoch}, Training Loss: {training_loss}', 
                  f'Validation Loss: {val_loss}',
                  f"Validation NDCG: {val_ndcg}")

    # Final loss
    final_rec_loss = np.nansum((normalized_train_matrix - np.dot(user_features, item_features.T)) ** 2)
    final_reg_loss = regularization * (np.sum(user_features**2) + np.sum(item_features**2))
    final_training_loss = final_rec_loss + final_reg_loss
    final_val_loss = cal_l2_loss(user_features, item_features, val_holdout_list, train_matrix)
    final_val_ndcg = cal_ndcg(user_features, item_features, val_holdout_list, train_matrix)

    return user_features, item_features, final_training_loss, final_val_loss, final_val_ndcg

def cal_l2_loss(user_features, item_features, holdout_list, original_matrix):

    """
    Calculate the L2 loss for the holdout set.
    Parameters:
    - user_features: User latent features matrix.
    - item_features: Item latent features matrix.
    - holdout_list: List of tuples (user_id, item_id, true_rating) for evaluation.
    - original_matrix: The original DataFrame containing the ratings.
    Returns:
    - average_rmse: The average RMSE across all users.
    """

    # reconstruct the matrix to DataFrame format
    reconstructed_matrix = np.dot(user_features, item_features.T)

    # Convert to DataFrame
    reconstructed_df = pd.DataFrame(reconstructed_matrix, index=original_matrix.index, columns=original_matrix.columns)

    # Calculate RMSE for holdout set
    rmse_list = []
    for user_id, joke_id, true_rating in holdout_list:
        predicted_rating = reconstructed_df.at[user_id, joke_id]
        rmse = np.sqrt((predicted_rating - true_rating) ** 2)
        rmse_list.append(rmse)
    
    # Calculate average RMSE
    average_rmse = np.mean(rmse_list)

    return average_rmse



def cal_ndcg(user_features, item_features, holdout_list, original_matrix, k=3):
    """
    Calculate the average NDCG@k for a recommendation system
    using matrix factorization.

    Parameters:
    - user_features: User latent features matrix.
    - item_features: Item latent features matrix.
    - holdout_list: List of tuples (user_id, item_id, true_rating) for evaluation.
    - original_matrix: The original DataFrame containing the ratings.
    - k: The rank at which to compute NDCG (default is 3).
    Returns:
    - average_ndcg: The average NDCG@k score across all users.

    """

    # 1) Reconstruct the matrix: U * V^T
    reconstructed_matrix = np.dot(user_features, item_features.T)

    # 2) Convert to DataFrame (same index & columns as original)
    reconstructed_df = pd.DataFrame(
        reconstructed_matrix,
        index=original_matrix.index,
        columns=original_matrix.columns
    )

    # ---------- Helper functions for DCG/NDCG ----------
    def dcg_at_k(relevances, k):
        """
        Compute Discounted Cumulative Gain (DCG) at rank k.
        relevances: array-like of ground truth relevance scores in ranked order.
        """
        relevances = np.array(relevances, dtype=float)[:k]
        if relevances.size:
            discounts = np.log2(np.arange(2, relevances.size + 2))
            return np.sum(relevances / discounts)
        return 0.0

    def ndcg_at_k(ranked_relevances, k):
        """
        Compute Normalized DCG at rank k, given the ground-truth relevances
        in the order of predicted ranking.
        """
        dcg = dcg_at_k(ranked_relevances, k)
        # Sort relevances in descending order to get ideal DCG
        ideal_relevances = np.sort(ranked_relevances)[::-1]
        idcg = dcg_at_k(ideal_relevances, k)
        if idcg == 0.0:
            return 0.0
        return dcg / idcg
    # ---------------------------------------------------

    # 3) Group holdout items by user: user_id -> [(item_id, true_rating), ...]
    user2holdouts = defaultdict(list)
    for user_id, item_id, true_rating in holdout_list:
        user2holdouts[user_id].append((item_id, true_rating))

    ndcg_scores = []

    # 4) For each user, rank ONLY their holdout items by predicted score
    for user_id, holdouts in user2holdouts.items():
        if not holdouts:
            continue
        
        # The list of holdout items for this user
        holdout_items = [t[0] for t in holdouts]
        # The true ratings for those items
        holdout_true_ratings = [t[1] for t in holdouts]
        
        # Get the predicted scores for these holdout items only
        predicted_scores = []
        for item in holdout_items:
            pred_score = reconstructed_df.at[user_id, item]
            predicted_scores.append(pred_score)
        
        # Sort the holdout items by predicted score (descending)
        # We'll get the indices that sort predicted_scores in descending order
        sorted_indices = np.argsort(predicted_scores)[::-1]
        
        # Re-order the true ratings based on the predicted order
        ranked_relevances = [holdout_true_ratings[i] for i in sorted_indices]

        # 4. Compute NDCG@k for this user's holdout-based ranking
        user_ndcg = ndcg_at_k(ranked_relevances, k)
        ndcg_scores.append(user_ndcg)

    # 5) Compute the average NDCG across all users
    average_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return average_ndcg


def main():
    # Load and process data
    original_matrix, train_matrix, val_holdout_list, test_holdout_list = process_data()

    # Train the model
    user_features, item_features, train_rmse, val_rmse, final_val_ndcg = train(train_matrix,val_holdout_list)

    calculated_ndcg = cal_ndcg(user_features, item_features, test_holdout_list, original_matrix)
    
    print(f'Train RMSE: {train_rmse}')
    print(f'Val RMSE: {val_rmse}')
    print(f'Val NDCG: {final_val_ndcg}')
    print(f'Test NDCG: {calculated_ndcg}')

if __name__ == "__main__":
    main()
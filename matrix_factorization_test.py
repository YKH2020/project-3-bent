import numpy as np
from matrix_factorization_train import cal_ndcg
from preprocess import process_data

def test():
    """
    Test the matrix factorization model with loaded user and item features
    """
    # load user_features, item_features
    checkpoint = np.load('checkpoint_epoch_6000.npz')

    # Extract user and item features from the checkpoint
    user_features = checkpoint['user_features']
    item_features = checkpoint['item_features']

    # Process the data
    original_matrix, train_matrix, val_holdout_list, test_holdout_list = process_data()
    calculated_ndcg = cal_ndcg(user_features, item_features, test_holdout_list, original_matrix,3)

    print(f'Test NDCG: {calculated_ndcg}')

if __name__ == "__main__":
    test()

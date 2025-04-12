from preprocess import process_data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

original_df, train_df, holdout_list = process_data()
# Mean model: compute mean rating for each joke (column-wise mean)
joke_means = train_df.mean(skipna=True)


# Step 1: Compute mean rating for each joke (ignoring NaNs)
joke_means = train_df.mean(skipna=True)

# Step 2: Predict for each holdout and compare
predictions = []
actuals = []

for user_id, joke_id, true_rating in holdout_list:
    pred = joke_means.get(joke_id, 0)  # fallback to 0 if joke_id is missing for any reason
    predictions.append(pred)
    actuals.append(true_rating)

# Step 3: Evaluate
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")


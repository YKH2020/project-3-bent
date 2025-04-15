# project-3-bent

## Motivation
It is often hard to gauge a sense of humor and find jokes that align with what people like, so we decided to create a recommendation model that attempts to judge opinions on jokes and gives recommendations.

## Data
Our data is from the Jesters Datasets found here: https://eigentaste.berkeley.edu/dataset/. It is composed of many users who rated many jokes from -10 to 10.

## Preprocessing
We took a dense subset of this data, including 10 of the most commonly rated jokes. We then incorporated validation and test holdouts to the dense subset, taking 2 jokes for validation and 3 for testing per user.

## Naive
The naive approach used a mean model. Jokes were recommended based on the mean score across all other users who rated that joke. Jokes with the highest mean were recommended.

## Non-Deep Learning

We implemented a traditional machine learning-based recommendation system using Matrix Factorization with Stochastic Gradient Descent (SGD). Each user and joke is represented in a shared latent space, enabling personalized joke predictions based on the dot product of learned feature vectors. For the processing of the input, we used Z-scored normalization to account for rating scale differences across users. We used SGD optimizer with L2 regularization with a value of 0.02 as the regularization parameter. In the training process, we set the learning rate as 0.01 to make sure that the model parametes have a decent update in each step. We used periodic evaluation using MSE and NDCG@k on validation sets to monitor the training process. In the training process, we noticed that the model reached its best performance on validation set at about 6000 epochs and we stopped training at that point to prevent overfitting. We saved the trained model as traditional_ml_model.npz for downstream test and inference.

## Deep Learning

## Eval
The naive approach had an accuracy of .422. 

## Demo

## Ethics
This dataset is for open source but the original source should be listed for credit. Our joke recommender models are for free use as well. The data and models should not be used for malicious intent and should be used to further research.

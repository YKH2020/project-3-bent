# project-3-bent

## Prior Efforts
1. [RBM Based Joke Recommendation System and Joke Reader Segmentation](https://www.researchgate.net/publication/337489351_RBM_Based_Joke_Recommendation_System_and_Joke_Reader_Segmentation):

The above source describes the development of a joke recommender system that, alternatively to our implementation, uses using a Bernoulli Restricted Boltzmann Machine (RBM) for collaborative filtering on the jester dataset. They segmented users using k-means clustering and was trained based on preference similarities. Their proposed methodology aimed to help recommend jokes and assist joke writers in creating content for specific reader groups.

2. [Joke Recommender System using Humor Theory](https://hammer.purdue.edu/articles/thesis/JOKE_RECOMMENDER_SYSTEM_USING_HUMOR_THEORY/12735302?file=24105254)

The above source employed the use of _Attardo and Raskin's (1991) General Theory of Verbal Humor (GTVH)_ to simply analyze joke text from the jester dataset, and perform data annotation on it. The annotations were done based on the 6 knowledge resources (KRs) of Language, Narrative Strategy, Target, Situation, Logical Mechanism, and Script Opposition. To compute similarity for any one of these target categories, Lin's similarity metric and word2vec similarity were employed. Finally, they used hierarchical clustering to first group similar jokes together and aimed to use the humor theory ideologies to potentially improve performance over the eigentaste algorithm of the original jester system.

## Motivation
It is often hard to gauge a sense of humor and find jokes that align with what people like, so we decided to create a recommendation model that attempts to judge opinions on jokes and gives recommendations.

## Data
Our data is from the Jesters Datasets found here: https://eigentaste.berkeley.edu/dataset/. It is composed of many users who rated many jokes from -10 to 10.

## Preprocessing
We took a dense subset of this data, including 10 of the most commonly rated jokes. We then incorporated validation and test holdouts to the dense subset, taking 2 jokes for validation and 3 for testing per user.

## Naive
The naive approach used a mean model. Jokes were recommended based on the mean score across all other users who rated that joke. Jokes with the highest mean were recommended.

## Non-Deep Learning
We implemented a traditional machine learning-based recommendation system using Matrix Factorization with Stochastic Gradient Descent (SGD). Each user and joke is represented in a shared latent space, enabling personalized joke predictions based on the dot product of learned feature vectors.     
    
To process the data, we used Z-scored normalization to account for rating scale differences across users. This transformation helps account for differences in individual rating scales and ensures that the model learns patterns that are independent of users' rating biases.  
    
We used SGD optimizer with L2 regularization with a value of 0.02 as the regularization parameter. In the training process, we set the learning rate as 0.01 to balance the convergence speed and stability, ensuring that model parameters receive meaningful updates at each step.   
    
We used periodic evaluation using MSE and NDCG@k on validation sets to monitor the training process. In the training process, we noticed that the model reached its best performance on validation set at about 6000 epochs and we stopped training at that point to prevent overfitting.   
       
We saved the trained model as traditional_ml_model.npz for downstream test and inference.

## Deep Learning
We implemented a custom autoencoder that took our dense sub matrix of jokes, where all users in the dataset had 10 jokes associated with them. The final layer uses tanh to further squash the latent space after dimensionality reduction in the hidden layers. The model recommends the best joke for a given user based on the accuracy of the user's jokes' reconstructions after the decoder output. The joke with the lowest loss in the test set is chosen as the best joke for our user.

## Eval
We chose the accuracy of recommending the best joke for each user in the corresponding holdout list. In naive approach, jokes were recommended based on the mean score across all other users who rated that joke. Jokes with the highest mean were recommended. In traditional machine learning and deep learning approaches, We got the ratings for each joke in the holdout list and we picked the joke with highest predicted rating for each user. The accuracy of recommending the best joke in the holdout list was used as our evaluation metrics.
      
The naive approach had an accuracy of .422.    
       
The traditional ml approach (matrix factorization) had an accuracy of .403.

The deep learning ml approach (custom autoencoder) had an accuracy of .401.

Despite using a supposedly dense subset of the total sparse dataset, we ran into overfitting issues with our results. In truth, it was particularly difficult to manage the number of holdouts for our dataset, as having fewer holdouts meant that fewer test points were evaluated, which would lead to randomness. Having too many holdouts would in theory minimize this, but it meant that there was far less data to learn from. However, the results were certainly more stable (albeit low) with no data leakage.

## Demo

The demo is available on the following [link](https://laxman-22-recommendation-system-app-4nvsmh.streamlit.app/)
and it is deployed on streamlit cloud.

## Ethics
This dataset is for open source but the original source should be listed for credit. Our joke recommender models are for free use as well. The data and models should not be used for malicious intent and should be used to further research.

## Run the model

### Preprocess:  
`python preprocess.py ` 

### Naive Approach:   
`python naive.py`   

### Traditional ML Approach:  
#### Train   
`python matrix_factorization_train.py`   

#### Test   
`python matrix_factorization_test.py`  

#### Inference    
`python matrix_factorization_inference.py`   

### DL Approach:
`python dl.py`


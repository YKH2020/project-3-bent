# project-3-bent

## Data
Our data is from the Jesters Datasets found here: https://eigentaste.berkeley.edu/dataset/. It is composed of many users who rated many jokes from -10 to 10.

## Preprocessing
We took a dense subset of this data, including 10 of the most commonly rated jokes. We then incorporated validation and test holdouts to the dense subset, taking 2 jokes for validation and 3 for testing per user.

## Naive
The naive approach used a mean model. Jokes were recommended based on the mean score across all other users who rated that joke. Jokes with the highest mean were recommended.

## Non-Deep Learning

## Deep Learning

## Eval
The naive approach had an accuracy of .422. 

## Demo

## Ethics
This dataset is for open source but the original source should be listed for credit. Our joke recommender models are for free use as well. The data and models should not be used for malicious intent and should be used to further research.

__Understanding Yelp Text and Star Reviews for Business Success__
## Background:
User reviews can provide useful feedback for business growth and establishment. With the ease of online review apps like Yelp, consumers have added an ever growing amount of text and “star” review data for businesses they visit. Machine learning techniques in NLP can be used to decipher the importance of  text features of a business for its success. Specifically, NLP can be used to analyze the actual text of the reviews for the data and perform sentiment analysis. This sentiment analysis can be powerful in gauging public tone. It can be used to understand and respond to customer criticisms of business marketing and products in real time. 

## Problem: 
There is a large amount of text review data that is connected to businesses on Yelp. Can we build a model to accurately predict sentiment of reviews (using the star rating as labels)? Can we find meaningful words in reviews that correlate highly with success in star rating? 

## Data:
**Note: data files, including cleaned data and hyperparameter data files, are not available due to size restrictions.**
The data that will be used comes from the Yelp dataset on Kaggle: https://www.kaggle.com/yelp-dataset/yelp-dataset 
The project specifically uses two of the json files in this dataset: 
yelp_academic_dataset_business.json
Yelp_academic_dataset_review.json
The dataset contains data from businesses in 11 metropolitan areas located in 4 countries. There are 5,200,000 user reviews and information on 174,000 businesses.

Additionally, a recurrent neural network will be trained in tensorflow.keras framework. This model will use byte-pair encoded data that has been generated and is available through tensorflow datasets: https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews#yelp_polarity_reviewsplain_text_default_config
This dataset has 560,000 highly polar yelp reviews in its training set, and 38,000 highly polar yelp reviews in its test set. Though this dataset is also from Yelp, it is not the exact same dataset and therefore a 1:1 comparison between the RNN model trained on this dataset and the other ML models trained on the Kaggle dataset cannot be performed. However, it is a useful model to demonstrate and quantify the performance of a RNN model for sentiment analysis. 

## Data Cleaning:
Jupyter Notebook: [Data Sampling & Joining notebook](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/data_sampling_joining.ipynb)
[Data Cleaning and NLP Notebook](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/data_cleaning_NLP.ipynb)

## Exploratory Data Analysis:
Jupyter Notebook: [Exploratory data analysis notebook](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/exploratory_data_analysis.ipynb)

## Statistical Data Analysis:
Jupyter notebook: [Statistical data analysis notebook](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/statistical_analysis.ipynb)

## Machine Learning Models:
Jupyter Notebooks: 
- [Bag of Words Method](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/sentiment_analysis_bow.ipynb)
- [BOW + Singular Value Decomposition Method](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/sentiment_analysis_svd.ipynb)
- [Word Embedding Method](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/sentiment_analysis_embeddings.ipynb)
- [RNN](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/sentiment_analysis_RNN.ipynb)

## Final Model Analysis:
Jupyter Notebook: [Final model comparison notebook](https://github.com/gksullan/yelp_review_sentiment_analysis/blob/master/final_model_analysis.ipynb)





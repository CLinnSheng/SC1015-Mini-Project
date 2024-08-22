# X (formerly Twitter) Sentiment Analysis

>School of Computer Science and Computer Engineering \
Nanyang Technological University \
Lab: FCSE \
Team: 9

## Our Team
| Name | Parts Done | Github ID |
|---|:---:|---|
| Chua Tze Wu | Exploratory Data Analysis (EDA), Data Preprocessing, K-nearest neighbours model, Github Repo and Report | [ctzewu](https://github.com/ctzewu) |
| Chin Linn Sheng | Exploratory Data Analysis (EDA), Data Preprocessing, Naive Bays Classifier | [linnsheng](https://github.com/CLinnSheng) |

## About

This is our mini project for our module **SC1015 - Introduction to Data Science and Artificial Intelligence**.

We want to gain insights into people's emotional state as they tweet based on the text in the tweet. 

## Dataset

The [dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) we used was taken from Kaggle.
Each entry in this dataset consists of a text segment representing a Twitter message and a corresponding label indicating the predominant emotion conveyed. The emotions are classified into six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5). 

Included in the dataset are 2 columns:
- text: `A string feature representing the content of the Twitter message`
- label: `A classification label indicating the primary emotion, with values ranging from 0 to 5`

## Problem Statement

To accurately predict and classify the sentiment of tweets into the six categories: **`sadness`**, **`joy`**, **`love`**, **`anger`**, **`fear`** or **`surprise`** using different models.

## Preprocessing Methods

  We'll explore some `Data Preprocessing` techniques including some of the following:
  | Data Preprocessing Techniques | Definition |
  |---|---|
  | Removing duplicates | Removal of duplicate datapoints from dataset | 
  | Deleting Stopwords | Filter words you want to ignore of your text when processing it | 
  | Removal of links | Removes links such as hyperlinks from the data | 
  | Removing frequent and rare words | Removal of words which may not be relevant to the dataset. | 
  | Stemming | A text processing task in which you reduce words to their root, which is the core part of a word. | 
  | Lemmatization | Reduce words to their core meaning, but it will give you a complete English word that make sense on its own instead of just a fragment of a word like 'discoveri'. | 
  | Handling of chat words | Converts common chat abbreviation to full words . | 
  | Spell checking | Converts typos or stemmed words back to proper spelling. | 

  ### Data Preprocessing
  
  1. Data Cleaning
      - Removal of links: observing the data points, the dataset has already been cleaned of links prior to use, hence removal of links is not needed.
      - Stemming vs Lemmatization:
        - Stemming:  reducing inflected (or sometimes derived) words to their word stem, base or root form.
        - Lemmanization: similar to stemming in reducing inflected words to their word stem but differs in the way that it makes sure the root word (also called as lemma) belongs to the language.

        Ultimately, despite lemmanization being slower, we choose to lemmanize the text data as it reduces them to a more accurate and correct form. 

  2. Feature Engineering
      - Added a `label_string` column for more legibility to reading the data.
    
  ## Exploratory Data Analysis 

  ### Seaborn heatmap
  
  **6 types of sentiment:**
  1. `joy` (Count: 135030)
  2. `sadness` (Count: 118511)
  3. `anger` (Count: 54777)
  4. `fear` (Count: 43629)
  5. `love` (Count: 29468)
  6. `surprise` (Count: 12407)
  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download.png" width="500">

  ### Seaborn count plot
  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(1).png" width="500">  

  ### Word cloud for the 6 emotions
  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(2).png" width="500">  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(3).png" width="500">  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(4).png" width="500">  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(5).png" width="500">  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(6).png" width="500">  
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(7).png" width="500">  

  ### 20 of the most common words per emotion

  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(8).png" width="800"> 
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(9).png" width="800"> 
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(10).png" width="800"> 
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(11).png" width="800"> 
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(12).png" width="800"> 
  <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/download%20(13).png" width="800"> 

  ## Model Training and Evaluation
  Two models with different approach was used to accurately predict and classify the sentiment of tweets into the six categories: **`sadness`**, **`joy`**, **`love`**, **`anger`**, **`fear`** or **`surprise`**

  1. K-Nearest Neighbor Classifier (KNN)
  2. Naive Bayes Classifier

  We will utilize the __Weighted F1 score__ from classification_report function. 
  This metric consider the most appropriate because it takes into account that class imbalance by weighting the F1-score of each class according to its frequency in the data.
  
  This metric is more preferred for imbalanced datasets like the X's tweets due to the reason there are many more neutral tweets than those positive or negative ones. Other metrices can be misleading in such cases. 
A model might simply predict 'neutral' for every tweet and achieve high accuracy, but it wouldn't be useful for sentiment analysis.

  The data was split into 75, 25 for training and testing respectively.
  
  ## K-Nearest Neighbor Classifier (KNN)
  The k-nearest neighbors (KNN) algorithm is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. It is one of the popular and simplest classification and regression classifiers used in machine learning today.

KNN takes in data points, and uniquely categorizes them into clusters. When using KNN we first feed into it known data, to train and classify the algorithm. Afterwards we will be able to feed in a new tweet to test, KNN places this new tweet on a graph, and sees which other data points on the plot it is most closely associated with.

Count_Vectorizer transforms the text data of tweets into numerical data such that the KNN Classifer can plot the data on a graph.

**Confusion Matrix of KNN**
-  Legend: `sadness (0)`, `joy (1)`, `love (2)`, `anger (3)`, `fear (4)`, and `surprise (5)`

<img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/KNN_cm.png" width="800"> 

**F1 Score of KNN**

<img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/f1_KNN.png" width="800"> 

- The `Weighted F1 Score` of KNN model is `0.84`

> **The K Value**

> The k value in the KNN algorithm defines how many neighbors will be checked to determine the classification of a specific query point. For example, if k=1, the instance will be assigned to the same class as its single nearest neighbor. Lower values of k can have high variance, but low bias, and larger values of k may lead to high bias and lower variance.
> Here we train and test the KNN Classifier with k values from 1 to 40, to see which k value will be the best suited for our model.
> <img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/error.png" width="800">

> Hence from this graph, we can tell that the most suitable for K-value for KNN model according to our dataset is between 10 to 15 as it has the lowest error rate
## Naive Bayes Classifier
Naive Bayes model is a widely used algorithm in machine learning for classification tasks. It leverages Bayes' theorem, a powerful tool in probability, to predict the likelihood of an instance belonging to a particular class. 

Naive Bayes shines when dealing with text data classification, which is exactly where the X's tweet data comes in. Naive Bayes performs well on large datasets. It can efficiently handle the massive amounts of text data. 

> Represent tweet as a collection of features where these features are individual words, presence or absence of specific words. 
> In the sentiment analysis for the data we have, the classes are the 6 classes that we had mentioned above. 
> Naive Bayes assumes independence between words in a tweet given the class. 
> For a new tweet, this model calculates the probability of each word appearing in each class. Using Bayes' theorem, it predicts the class with the highest probability based on the tweet's features.

We use __MultinomialNB__ function from Scikit-learn library. 
> __MultinomialNB__ implements the Naive Bayes algorithm specifically designed for text data. This makes it easy to integrate Naive Bayes algorithm into our text classification workflow.

**Confusion Matrix of Naive Bayes**
-  Legend: `sadness (0)`, `joy (1)`, `love (2)`, `anger (3)`, `fear (4)`, and `surprise (5)`
<img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/output.png" width="800"> 

**F1 Score of Naive Bays**

<img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/f1_NB.png" width="800"> 

- The `Weighted F1 Score` of Naive Bays model is `0.85`

## Conclusion
- In conclusion, the __Naive Bayes model__ is a better model than the __K-nearest Neighbor__ when analyzing sentiment in our X's tweet dataset. Even though their F1 scores are almost the same, where Naive Bayes achieved an F1 score of 0.85 and KNN achieved an F1 score of 0.84. 
- However, the Naive Bayes model exhibited a significant advantage in training time, making it computationally faster for processing large volumes of tweets. 
- These findings suggest that Naive Bayes is a more favorable option for sentiment analysis on X's tweet dataset, which has a large dataset, due to its efficiency and effectiveness.

<img src="https://github.com/ctzewu/SC1015-mini-project/blob/main/images/conclu.png" width="800"> 

## References
1. [https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data)
2. [https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing](https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing)
3. [https://www.kaggle.com/code/nebilekodaz/knn-notebook](https://www.kaggle.com/code/nebilekodaz/knn-notebook)
4. [https://www.datacamp.com/tutorial/wordcloud-python](https://www.datacamp.com/tutorial/wordcloud-python)
5. [https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)


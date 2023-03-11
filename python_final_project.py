import os
from os import path
import pandas as pd
import numpy as np
import scipy
from nltk.corpus import stopwords
from wordcloud import STOPWORDS, WordCloud
import matplotlib.pyplot as plt
import re
import string
import nltk as nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report






# Import Data
book_details = pd.read_csv("E:\\TSOM\MODULE 3\\pythonFinalProject\\books_data-1.csv")
ratings = pd.read_csv("E:\\TSOM\\MODULE 3\\pythonFinalProject\\Books_rating.csv")

# Exploring the data, Initail data exploration
print("The top rows of book_details table: ")
print(book_details.head())
print("The top rows of rating table: ")
print(ratings.head())
print("The rows and columns of rating table: ")
print(ratings.shape) # 300000 rows ,10 columns when imported
print("Printing the summary statictics of Rating table:")
print(ratings.describe(include= 'all'))
print("Printing the summary statictics of book_details table:")
print(book_details.describe(include= 'all'))


# The data in categories is stored as a list and some data also is of float type
# Hence the function takes categories column as input if its a float converts it into str
# Then replace is used to remove []
def converListIntoText(categories):
    if type(categories) == float:
        return str(categories)
    text = "".join(str(cateogry) for cateogry in categories)
    return text.replace("[", "").replace("]", "")
book_details["categories"] = book_details["categories"].apply(converListIntoText)


# function for changing Juvenile Fiction to Fiction . Since the Categories have alot of unique values
def convert(categories):
    return categories.replace('Juvenile Fiction','Fiction')
book_details["categories"]= book_details["categories"].apply(convert)



# Now we want to get the categories from the book_details table into new_ratings. So we are merging it a temporay table with title and categories

book_details_temp = book_details[["Title", "categories"]]
new_ratings = ratings.merge(book_details_temp, how='left', on='Title')

# Duplicates Handling
print(new_ratings[new_ratings.duplicated(subset = ['Id','Title'])])

# The Id and Title columns have dupilcates . But we will not handle them as a customer might give different review to the same book on diffrent
# occasions.

# NA handling
print(new_ratings.isnull().sum())
# The title, 'review/summary', 'review/text', 'categories' have NAs . Since these are categorical data we cant replace the Nulls with mean or median
# As we are analyzing the text data. Dropping rows with Nulls will not make a considerable difference as the volume of the Text data is huge

new_ratings.dropna(subset=['Title', 'review/summary', 'review/text', 'categories'], inplace= True)

# The price column has Nulls which is handled by mean price grouped by categories as these hold a categorical relationship.
new_ratings['Price'] = new_ratings['Price'].fillna(new_ratings.groupby('categories')['Price'].transform('mean'))




# Handling the nan values in categories
new_ratings["categories"] = new_ratings["categories"].replace('nan','others')
new_ratings['categories'] = new_ratings['categories'].fillna('others')
count1 = book_details["categories"].value_counts(ascending = False)
print("Printing the count of each category: ")
print(count1)
category10 = count1[count1 <= 2500]
print("printing the categoires which have count less than 2500 ")
print(category10)

# function to keep the top categories by making the rest of categories less than 2500 in count as others
def removes(value):
    if value in category10:
        return 'others'
    else:
        return value
new_ratings.categories = new_ratings.categories.apply(removes)


print("printing unique categories: ")
print(new_ratings.categories.unique())
print("printing the count of each category")
print(new_ratings["categories"].value_counts(ascending = False))




# Histgram
#Importing plotly for visulizations

#plotting histogram for review/score
import plotly.express as px
fig = px.histogram(new_ratings, x="review/score")
fig.update_traces(marker_color='steelblue',marker_line_color='rgb(80,15,11)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Rating')
fig.show()


# plotting pie chart for categories and review/text
sum = new_ratings.groupby('categories').agg(['count'])[['review/text']].reset_index()
sum.columns = ['Category','Count']
print(sum)
fig = px.pie(sum, values='Count', names='Category', title='Review Summary')
fig.show()




# splitting the data as train set and test set
new_ratings_train = new_ratings.sample(frac =.85)
print(new_ratings_train.head(10))
new_ratings_test = new_ratings.sample(frac =.15)
print(new_ratings_test.head(10))


# Adding a new column sentiment for  Label 'positive' or 'negative' for Sentimental Analysis
new_ratings_train['sentiment'] = ['positive' if x > 3 else 'negative' for x in new_ratings_train["review/score"]]
print(new_ratings_train.head(10))




#Create stopword list
stops= set(STOPWORDS)

stops.update(["book", "books","read","make","think","walk","I","it","take","made","find","because","one","even","sort","paint"])

print(new_ratings_train.columns)
print(new_ratings_train.head())

# cearting a single text on all the data from review/ text comumn
textt1 = ' '.join(new_ratings_train["review/text"].astype(str))
print(textt1)

# Generating a word cloud image for positive reviews
reviews_train_p = new_ratings_train[new_ratings_train.sentiment == 'positive'][1:50000]
textt_p = " ".join(x for x in reviews_train_p["review/text"])
wordcloud = WordCloud(stopwords=stops).generate(textt_p)
wordcloud.to_file("E:\\first_review1_n.png")


# Generating a word cloud image for negative reviews
reviews_train_n = new_ratings_train[new_ratings_train.sentiment == 'negative'][1:50000]
textt_n = " ".join(x for x in reviews_train_n["review/text"])
wordcloud = WordCloud(stopwords=stops).generate(textt_n)
wordcloud.to_file("E:\\first_review1_n.png")

# Regression
# selecting the columns  'review/Text',  'review/score', and 'Title' variables that would matter to the
# analysis. Creating a new DataFrame called new_reviews_sub
new_ratings_sub = new_ratings[['review/text', 'review/summary', 'review/score']]
print(new_ratings_sub.head())
print()

# function is created to remove punctuations from the text.
def remove_punc_stopwords(text):
    text = re.sub(f"[{string.punctuation}]"," ",text)
    text_tokens = set(text.split())
    stops = set(STOPWORDS)
    text_tokens = text_tokens.difference(stops)
    return " ".join(text_tokens)



""" Remove punctuations and stopwords from the text data in review/text and review/summary"""
new_ratings_sub["review/summary"] = new_ratings_sub["review/summary"].apply(remove_punc_stopwords)
new_ratings_sub["review/text"] = new_ratings_sub["review/text"].apply(remove_punc_stopwords)

""" Add a new variable called sentiment; if review/score is greater than 3, then sentiment = 1, else sentiment = -1 """
new_ratings_sub['sentiment'] = [1 if x>3 else -1 for x in new_ratings_sub['review/score']]

import random
""" split the dataset into two: train (85% of the obs.) and test (15% of the obs.)"""
new_ratings_sub['random_index'] = [random.uniform(0,1) for x in range(len(new_ratings_sub))]

new_ratings_sub_train = new_ratings_sub[new_ratings_sub.random_index < 0.85]
new_ratings_sub_test = new_ratings_sub[new_ratings_sub.random_index >= 0.85]

print(new_ratings_sub_train.head())
print(new_ratings_sub_test.head())


vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(new_ratings_sub_train['review/summary'])
test_matrix = vectorizer.transform(new_ratings_sub_test['review/summary'])
train_matrix2 = vectorizer.fit_transform(new_ratings_sub_train['review/text'])
test_matrix2 = vectorizer.transform(new_ratings_sub_test['review/text'])



"""Perform binomial Logistic Regression"""

lr = LogisticRegression()

X_train = train_matrix
X_test = test_matrix
y_train = new_ratings_sub_train['sentiment']
y_test = new_ratings_sub_test['sentiment']


lr.fit(X_train,y_train)

""" Generate the predictions for the test dataset"""
predictions = lr.predict(X_test)
new_ratings_sub_test['predictions'] = predictions
print(new_ratings_sub_test.head(10))

"""Calculate the prediction accuracy"""
new_ratings_sub_test['match'] = new_ratings_sub_test['sentiment'] == new_ratings_sub_test['predictions']
print(classification_report(y_test,predictions))


lr2 = LogisticRegression()

X_train = train_matrix2
X_test = test_matrix2
y_train2 = new_ratings_sub_train['review/score']
y_test2 = new_ratings_sub_test['review/score']


lr2.fit(X_train,y_train2)

""" Generate the predictions for the test dataset"""
predictions2 = lr2.predict(X_test)
new_ratings_sub_test['predictions_rating'] = predictions2
print(new_ratings_sub_test.head(10))

"""Calculate the prediction accuracy"""
new_ratings_sub_test['match_score'] = new_ratings_sub_test['review/score'] == new_ratings_sub_test['predictions_rating']
print(classification_report(y_test2, predictions2))

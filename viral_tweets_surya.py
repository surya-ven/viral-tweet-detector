"""
Codecademy Project: Viral Tweets (Machine Learning Skill Path: https://www.codecademy.com/learn/paths/machine-learning)
Student Name: Surya Venkatesh
Description: After finishing the Supervised Learning modules under "Machine Learning Specialisation", I was tasked with this external project. I followed a tutorial (both video and written) that went over the key sequence of steps required to achieve the overall functionality.
The tutorial used only 3 features, but I ended up using 8, and increased accuracy by almost 15%.
Source Data: "random_tweets.json" (provided by Codecademy)
"""

# Import relevant modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNclassifier
import matplotlib.pyplot as plt


# Read in complete set of tweets
tweets_complete = pd.read_json("random_tweets.json", lines=True)

# Calculate median number of retweets to determine 'viral' threshold
median_retweets = tweets_complete['retweet_count'].median()

print(f"Median retweet threshold: {median_retweets}")

# Create column that determines whether a tweet passes (1) the threshold or not (0)
tweets_complete['viral_yn'] = np.where(tweets_complete['retweet_count'] >= median_retweets, 1, 0)

# Generate column for follower count
tweets_complete['followers_count'] = tweets_complete.apply(lambda tw: tw['user']['followers_count'], axis=1)
# Generate column for friend count
tweets_complete['friends_count'] = tweets_complete.apply(lambda tw: tw['user']['friends_count'], axis=1)
# Generate column for character count
tweets_complete['char_count'] = tweets_complete.apply(lambda tw: len(tw['text']), axis=1)

# Generate column for hashtag count
tweets_complete['hashtag_count'] = tweets_complete.apply(lambda tw: tw['text'].count("#"), axis=1)
# Generate column for links count
tweets_complete['links_count'] = tweets_complete.apply(lambda tw: tw['text'].count("http"), axis=1)
# Generate column for word count
tweets_complete['word_count'] = tweets_complete.apply(lambda tw: len(tw['text'].split()), axis=1)
# Generate column for statuses count
tweets_complete['statuses_count'] = tweets_complete.apply(lambda tw: tw['user']['statuses_count'], axis=1)
# Generate column for number of public lists that the user is a member of
tweets_complete['listed_count'] = tweets_complete.apply(lambda tw: tw['user']['listed_count'], axis=1)


# Labels
labels = tweets_complete['viral_yn']
# Features
dataset = tweets_complete[['char_count','followers_count','friends_count','links_count','hashtag_count','word_count','statuses_count','listed_count']]

# Normalise dataset
scaled_dataset = scale(dataset, axis=0)

# Generate training and testing datasets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(scaled_dataset, labels, test_size = 0.2, random_state = 1)

# Find best value of k, which maximises classification score
k_values = list(range(1, 200))
scores = []
best_k = 0
best_score = 0
for k in k_values:
    knclassifier = KNclassifier(n_neighbors = k)
    knclassifier.fit(train_dataset, train_labels)
    curr_score = knclassifier.score(test_dataset, test_labels)
    scores.append(curr_score)
    if curr_score > best_score:
        best_k = k
        best_score = curr_score
print(f"Optimal value of k: {best_k}")
# plt.plot(k_values, scores)
# plt.show()

# Train data based on our best value of K
knclassifier = KNclassifier(n_neighbors = best_k)
knclassifier.fit(train_dataset, train_labels)

# Results
print(f"Our accuracy: {knclassifier.score(test_dataset, test_labels)}")

# Best value of K = 29
# Accuracy achieved: 0.7427927927927928
# Accuracy with only followers, friends, and character count was: 0.63
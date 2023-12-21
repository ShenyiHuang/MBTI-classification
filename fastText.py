import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from gensim.models import FastText
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def vectorize(words):
    v = [ft.wv[word] for word in words if word in ft.wv]
    if len(v) == 0:
        return np.zeros(100)
    v = np.array(v)
    return v.mean(axis=0)

fp = 'cleaned_posts.json'

with open(fp, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

df = pd.DataFrame(data)

# x is the posts, and y is the personality type
X_train, X_test, y_train0, y_test0 = train_test_split(df['posts'], df['type'], test_size=0.2, random_state=42)

ft_train = [s for s in X_train]
ft = FastText(ft_train, vector_size=100)

X_train = np.array([vectorize(post) for post in X_train])
X_test = np.array([vectorize(post) for post in X_test])

log_accuracy = []
labels = ['E/I', 'N/S', 'T/F', 'J/P']

# train the model independently based on each personality type
# y_train0.str[i] get each letter from the type
for i in range(4):
    logReg = LogisticRegression(max_iter=200)
    logReg.fit(X_train, y_train0.str[i])

    y_pred = logReg.predict(X_test)
    s = round(accuracy_score(y_test0.str[i], y_pred),4)
    log_accuracy.append(s)
    print('Logistic regression with l2 penalty has an accuracy ', s, ' for ', labels[i])


'''
TODO pick another model and get accuracy scores
TODO visualize the result using a bar chart
l and m stands for the bar for log regression and the model you pick

accuracy    l m l m l m l m
            ---------------
            e/i n/s f/t j/p
'''
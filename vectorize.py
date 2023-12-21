import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
def vectorize(words):
    v = [w2v.wv[word] for word in words if word in w2v.wv]
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

w2v_train = [s for s in X_train]
w2v = Word2Vec(w2v_train)

X_train = np.array([vectorize(post) for post in X_train])
X_test = np.array([vectorize(post) for post in X_test])

log_accuracy = []
labels = ['E/I', 'N/S', 'T/F', 'J/P']

# train the model independently based on each personality type
# y_train0.str[i] get each letter from the type
# for i in range(4):
#     logReg = LogisticRegression(max_iter=200)
#     logReg.fit(X_train, y_train0.str[i])

#     y_pred = logReg.predict(X_test)
#     s = round(accuracy_score(y_test0.str[i], y_pred),4)
#     log_accuracy.append(s)
#     print('Logistic regression with l2 penalty has an accuracy ', s, ' for ', labels[i])


'''
TODO pick another model and get accuracy scores

TODO visualize the result using a bar chart
l and m stands for the bar for log regression and the model you pick

accuracy    l m l m l m l m
            ---------------
            e/i n/s f/t j/p
'''
# sgd_accuracy = []
# for i in range(4):
#     sgdClass = SGDClassifier(n_jobs=-1, early_stopping=True)
#     sgdClass.fit(X_train, y_train0.str[i])

#     y_pred = sgdClass.predict(X_test)
#     score = round(accuracy_score(y_test0.str[i], y_pred),4)
#     sgd_accuracy.append(score)
#     print('SGD with l2 penalty has an accuracy ', score, ' for ', labels[i])

mlp_accuracy = []
for i in range(4):
    mlpClass = MLPClassifier(solver='adam', max_iter=10000, learning_rate_init=0.0001
                             , beta_1=.9, beta_2=.999)
    mlpClass.fit(X_train, y_train0.str[i])

    y_pred = mlpClass.predict(X_test)
    score = round(accuracy_score(y_test0.str[i], y_pred),4)
    mlp_accuracy.append(score)
    print('MLP with maxiter10000 learningrate0.0001 beta_1=.9 beta_2.=.999 has an accuracy '
          , score, ' for ', labels[i])
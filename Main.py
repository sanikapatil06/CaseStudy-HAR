import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm


logr = LogisticRegression(random_state=0)

data = pd.read_csv('train.csv')

x = data.drop(['Activity','subject'], axis=1)
y = data['Activity'].astype(object)

le = LabelEncoder()
y = le.fit_transform(y)


scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)

logr.fit(x_train, y_train)

ylogr_predict = logr.predict(x_test)


print('Logistic:', accuracy_score(y_test, ylogr_predict))

'''
Accuracy Scores:
Logistic: 0.9714396735962697
'''

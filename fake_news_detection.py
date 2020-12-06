import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier

#นำเข้าไฟล์ที่ล้างแล้วซึ่งมี text และ label
df = pd.read_csv('https://raw.githubusercontent.com/ravidahiya74/Fake-news-detection/master/news.csv')
X = df['text']
y = df['label']

#แบ่งข้อมูลเป็น train 80 & test 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#สร้างชุดคำสั่งที่ใช้แปรตัว text เป็น ตัวเลข แล้วเอาเข้า model 
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('pac', PassiveAggressiveClassifier())])
pipeline1 = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('nbmodel', MultinomialNB())])

# Training
pipeline.fit(X_train, y_train)
pipeline1.fit(X_train, y_train)

#Predict ค่า
pred = pipeline.predict(X_test)
pred1 = pipeline1.predict(X_test)

# ตรวจสอบความถูกต้อง
print(f'PassiveAggressiveClassifier Accuracy  \n',classification_report(y_test, pred))
print("---------------------------------------------------------------------------------")
print(f'MultinomialNB Accuracy                \n',classification_report(y_test, pred1))
print("---------------------------------------------------------------------------------")
print(confusion_matrix(y_test,pred))
print("-------------")
print(confusion_matrix(y_test, pred1))

#Serialising the file
with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
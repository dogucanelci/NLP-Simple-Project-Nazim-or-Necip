
import numpy
import io
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
import sys
from time import time
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import FunctionTransformer
# x[3:6] = [''.join(x[3:6])]
y = ""
nazim_list = []
open1 = open("nazim.txt","r",encoding="utf-8")
for i in range(1001):
    a = open1.readline()
    if a == "*\n":
        nazim_list.append(y)
        y = ""
    else:
        y=y.replace("\n","")
        y += a

y1 = ""
necip_list = []
open2 = open("necip.txt","r",encoding="utf-8")
for i in range(1000):
    a = open2.readline()
    if a == "*\n":
        necip_list.append(y1)
        y1 = ""
    else:
        y1 = y1.replace("\n", " ")
        y1 += a

nazim_necip_features=[]
for i in necip_list:
    nazim_necip_features.append(i)
for i in nazim_list:
    nazim_necip_features.append(i)

nazim_necip_labels=[]
for i in range(12):
    nazim_necip_labels.append(0)
for i in range(12):
    nazim_necip_labels.append(1)

x_train,x_test,y_train,y_test=train_test_split(nazim_necip_features,nazim_necip_labels,test_size=0.1,random_state=15)
#print(type(x_train))
#Hem kelimeleri ayırıyor hem de onlara özel numara atıyor .
vectorizer=TfidfVectorizer(sublinear_tf=True, min_df=1,stop_words='english')
features_train_transformed = vectorizer.fit_transform(x_train)
# print vectorizer.get_feature_names()
features_test_transformed = vectorizer.transform(x_test)
#print(vectorizer.get_feature_names())

#feature selection yapılır .
"""
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, y_train)
#print(vectorizer.get_feature_names())
feature_selected_transformed_x_train = selector.transform(features_train_transformed).toarray()
feature_selected_transformed_x_test = selector.transform(features_test_transformed).toarray()
#print(features_train_transformed)
"""

clf=GaussianNB()
features_train_transformed=features_train_transformed.todense()
clf.fit(features_train_transformed,y_train)
input_poem=input("şiiri giriniz...")
input_list=[]
input_list.append(input_poem)
features_test_poem = vectorizer.transform(input_list)
#feature_selected_transformed_poem_test = selector.transform(features_test_poem).toarray()
#print(feature_selected_transformed_poem_test)
pred=clf.predict(features_test_poem.todense())
if pred==0:
    print("Şiir Necip Fazıl a aittir .")
if pred==1:
    print("Şiir Nazım Hikmet e aittir .")
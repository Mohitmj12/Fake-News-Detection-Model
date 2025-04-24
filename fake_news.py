
import pandas as pd
import numpy as np
import re
import nltk
import swifter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
# from sklearn.naive_bayes import MultinomialNB

# import textwrap
import joblib




news_df=pd.read_csv("./dataset/news_data.csv")



news_df.head()



news_df.shape




news_df.isna().sum()


# ### filling null with empty string




news_df = news_df.fillna(' ')





news_df.isna().sum()



ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # Load stopwords once
regex = re.compile('[^a-zA-Z]')               # Compile regex once

def stemming(content):
    stemmed_content = regex.sub(' ', content).lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)





nltk.download('stopwords')




news_df['content'] = news_df['content'].swifter.apply(stemming)





X=news_df['content'].values
y=news_df['label'].values





X
# checking if X got expected values




vector= TfidfVectorizer()
vector.fit(X)
X=vector.transform(X)



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)




X_train.shape




X_test.shape


# ### SGDClassifier Algo



model = SGDClassifier(loss='log_loss', max_iter=1000)





model.partial_fit(X_train, y_train, classes=[0, 1]) 




train_y_pred = model.predict(X_train)
print("Train accuracy:", accuracy_score(train_y_pred, y_train))




test_y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(test_y_pred, y_test))


# ### Logistic Regression



# model=LogisticRegression()
# model.fit(X_train,y_train)





# train_y_pred=model.predict(X_train)
# print("train accuracy : ",accuracy_score(train_y_pred,y_train))



# test_y_pred=model.predict(X_test)
# print("test accuracy : ",accuracy_score(test_y_pred,y_test))




# y_proba = model.predict_proba(X_test)[:]
# y_proba 


# ### Naive Bayes Classifier



# model2=MultinomialNB()
# model2.fit(X_train,y_train)



# y_pred=model2.predict(X_test)




# print("naive bayes accuracy : ",accuracy_score(y_test,y_pred))




# classification_report(y_test, y_pred)



input_data=X_test[10]
prediction=model.predict(input_data)
if prediction[0]==1:
    print("fake news")
else :
    print('real news')




joblib.dump(model,'model/fake_news_model.pkl')




joblib.dump(vector,'model/tfid_vectorizer.pkl')
print('model saved')







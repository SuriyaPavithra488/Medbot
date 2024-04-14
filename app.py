from flask import Flask, render_template, request, jsonify
from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
import torch
import random


df = pd.read_excel('C:\\Users\\raksh\\Downloads\\LearnMate-main\\intents.xlsx')
df.ffill(axis=0, inplace= True)
def step1(x):
  for i in x:
    a = str(i).lower()
    p = re.sub(r'[^a-z0-9]',' ',a)

step1(df['context'])



def text_normalization(text):
  text = str(text).lower()
  spl_char_text = re.sub(r'[^ a-z]','',text)
  tokens = nltk.word_tokenize(spl_char_text)
  lema = wordnet.WordNetLemmatizer()
  tags_list = pos_tag(tokens, tagset=None)
  lema_words = []
  for token, pos_token in tags_list:
    if pos_token.startswith('V'):
      pos_val = 'v'
    elif pos_token.startswith('J'):
      pos_val = 'a'
    elif pos_token.startswith('R'):
      pos_val = 'r'
    else:
      pos_val = 'n'
    lema_token = lema.lemmatize(token, pos_val)
    lema_words.append(lema_token)

  return " ".join(lema_words)

df['lemmatized_text'] = df['context'].apply(text_normalization)
cv = CountVectorizer()
X = cv.fit_transform(df['lemmatized_text']).toarray()
features = cv.get_feature_names_out()
df_bow = pd.DataFrame(X, columns= features)
stop = stopwords.words('english')
Question = 'Will you help me and tell me about yourself more'
#checking for stop words
Q = []
a = Question.split()
for i in a:
  if i in stop:
    continue
  else:
    Q.append(i)
  b= " ".join(Q)


Question_lemma = text_normalization(b)
Question_bow = cv.transform([Question_lemma]).toarray()
cosine_value = 1- pairwise_distances(df_bow, Question_bow, metric = 'cosine')
tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(df['lemmatized_text']).toarray()
df_tfidf = pd.DataFrame(x_tfidf, columns=tfidf.get_feature_names_out())
Question_tfidf = tfidf.transform([Question_lemma]).toarray()
cos=1-pairwise_distances(df_tfidf,Question_tfidf,metric='cosine')
index_value1 = cos.argmax()
df['text response'].loc[index_value1]

greet_inputs = ('hello','hi','whassup','how are you?')
greet_responses = ('hi','Hey','Hey There!','There there!!')
def greet(sentence):
  for word in sentence.split():
    if word.lower() in greet_inputs:
      return random.choice(greet_responses)
# ###

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    user_response = str(text)
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thank you' or user_response == 'thanks'): 
            return "You are welcome"
        else:
            if(greet(user_response) != None):
               return greet(user_response)
            else:
              lemma = text_normalization(user_response)
              tf = tfidf.transform([lemma]).toarray()
              if(tf.any() == 0):
                 return "Sorry Unable to Understand"
              else:
                cos=1-pairwise_distances(df_tfidf,tf,metric='cosine')
                index_value=cos.argmax()
                return df['text response'].loc[index_value]
    else:
        return "Good bye"

if __name__ == '__main__':
    app.run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:04:54 2018

@author: root
"""
import pandas as pd 
import warnings

import re    #for regex
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import TweetTokenizer   


train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

APPO = {
"aren't" : "are not", "can't" : "cannot", "couldn't" : "could not", "didn't" : "did not",
"doesn't" : "does not", "don't" : "do not", "hadn't" : "had not","hasn't" : "has not",
"haven't" : "have not", "he'd" : "he would", "he'll" : "he will", "he's" : "he is",
"i'd" : "I", "i'll" : "I will", "i'm" : "I am", "isn't" : "is not", "it's" : "it is",
"it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not",
"mustn't" : "must not", "shan't" : "shall not", "she'd" : "she would", "she'll" : "she will",
"she's" : "she is", "shouldn't" : "should not", "that's" : "that is","there's" : "there is", 
"they'd" : "they would", "they'll" : "they will", "they're" : "they are", "they've" : "they have",
"we'd" : "we would", "we're" : "we are","weren't" : "were not", "we've" : "we have", "what'll" : "what will",
"what're" : "what are", "what's" : "what is", "what've" : "what have", "where's" : "where is",
"who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is", "who've" : "who have",
"won't" : "will not","wouldn't" : "would not", "you'd" : "you would", "you'll" : "you will",
"you're" : "you are", "you've" : "you have", "'re": " are", "wasn't": "was not", "we'll":" will", "didn't": "did not",
"tryin'":"trying"
}


def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    #remove \n
    comment=re.sub("\\n","",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    #remove links
    comment = re.sub("http://.*com", "", comment)
    
    #remove article ids 
    comment = re.sub("\d:\d\d\s{0,5}$","", comment)
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    clean_sent=re.sub("\W+"," ",clean_sent)
    clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)

train["clean_text"] = train["comment_text"].apply(lambda x :clean(x))
test["clean_text"] = test["comment_text"].apply(lambda x :clean(x))

train.to_csv("input/clean_train.csv", columns=['id', 'clean_text', 'toxic', 'severe_toxic', 'obscene', 
                                               'threat','insult', 'identity_hate'], index=False)
test.to_csv("input/clean_test.csv", columns=["id", "clean_text"], index=False)

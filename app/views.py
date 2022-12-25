from django.shortcuts import render
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import re
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
from django.contrib import messages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request,'index.html')

def hashtag_analysis(request):
    return render(request,'analysis_index.html')

def tweet_analysis(request):
    return render(request,'sentiment_index.html')

def analysis(request):
    if request.method == "POST":
        hashtag = request.POST['hashtag']
        limit = 20

    else:
        messages.error(request,"Please enter a hashtag")
        return render(request,'index.html')

    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(hashtag).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date,tweet.user.username,tweet.user.location,tweet.lang,tweet.content])
    df = pd.DataFrame(tweets,columns = ['Date','User','Location','Language','Tweet'])


    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    df = df[df.Language == 'en']

    def preprocess_tweet_text(tweet):
        # Converting all text into lowercase
        tweet = tweet.lower()
        
        # Removing any urls
        tweet = re.sub(r"http\S+|www\S+|https\S+","",tweet, flags=re.MULTILINE)
        
        # Removing punctuations
        tweet = tweet.translate(str.maketrans("","",string.punctuation))
        
        # Removing user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#',"",tweet)
        
        # Removing stopwords
        tweet_tokens = word_tokenize(tweet)
        filtered_words = [word for word in tweet_tokens if word not in stop_words]
        
        # Stemming
        ps = PorterStemmer()
        stemmed_words = [ps.stem(w) for w in filtered_words]
        
        # Lemmatizing
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(w,pos='a') for w in stemmed_words]
        
        return " ".join(lemma_words)

    df.Tweet = df.Tweet.apply(preprocess_tweet_text)

    def find_sentiment(tweet):
        tweet = TextBlob(tweet)
        return tweet.sentiment.polarity

    df['Sentiment'] = df.Tweet.apply(find_sentiment)

    def classify(n):
        if n == 0:
            return "neutral"
        elif n > 0 :
            return "positive"
        else:
            return "negative"

    df['Sentiment'] = df.Sentiment.apply(classify)



    positive = df.loc[df['Sentiment'].str.contains('positive')]
    negative = df.loc[df['Sentiment'].str.contains('negative')]
    neutral = df.loc[df['Sentiment'].str.contains('neutral')]

    positive_per = round((positive.shape[0]/df.shape[0])*100, 1)
    negative_per = round((negative.shape[0]/df.shape[0])*100, 1)
    neutral_per = round((neutral.shape[0]/df.shape[0])*100, 1)

    context = {'hashtag':hashtag,'positive':positive_per,'negative':negative_per,'neutral':neutral_per}

    return render(request,'analysis.html', context)


def sentiment(request):
    model = pickle.load(open('model.pkl','rb'))

    if request.method == 'POST':
        tweet = request.POST['tweet']
    else:
        messages.error(request,"Please enter a tweet")
        return render(request,'sentiment_index.html')

    df = pd.read_pickle('data.pkl')

    X = df['Tweet']
    y = df['Sentiment']

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.25)

    vectorizer = CountVectorizer()
    vectorizer.fit(X_train,X_test)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    vecttweet = vectorizer.transform(np.array([tweet]))
    prediction = model.predict(vecttweet)
    sentiment = 0

    if prediction == 0:
        sentiment = "neutral"
    elif prediction == 1:
        sentiment = "positive"
    else:
        sentiment = "negative"

    context = {"tweet": tweet,"sentiment": sentiment}
    
    return render(request,'sentiment.html',context)



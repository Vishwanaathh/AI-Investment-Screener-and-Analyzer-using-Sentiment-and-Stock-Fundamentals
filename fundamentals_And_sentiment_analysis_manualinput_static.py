import pandas as pd
import re
import numpy as np
import joblib
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#'market_cap','current_value','high_52week','low_52week','book_value','price_earnings','dividend_yield','roce','roe','sales_growth_3yr
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          
    text = re.sub(r"@\w+", "", text)             
    text = re.sub(r"#\w+", "", text)             
    text = re.sub(r"[^a-z ]", "", text)          
    return text.strip()

nltk.download('punkt')

from nltk.tokenize import word_tokenize
print("Welcome to Stock Fundamentals and sentiment analysis")
print("This model uses NLP techniques and ML Random Forest to analyze stocks")
print("This script is designed for manual input")
print("Loading Fundamentals model...")
fund=joblib.load('./fundamentals_stock_model.joblib')
print("Finished loading")
print("Loading Sentiment Analyzer")
analyzer=SentimentIntensityAnalyzer()
print("Finished loading")
while True:
    n=input("Enter y to continue q to quit")
    if n=="y" or n=="Y":
        name=input("Enter name ")
        mkcap=int(input("Enter market cap "))
        cval=float(input("Enter current value of stock "))
        high=float(input("Enter 52 week high "))
        low=float(input("Enter 52 week low "))
        bval=float(input("Enter book value "))
        pe=float(input("Enter P/E Ratio "))
        div=float(input("Enter dividend percentage "))
        roce=float(input("Enter roce "))
        roe=float(input("Enter roe "))
        sal=float(input("Enter sales growth in last 3yrs "))
        
        
        tweet=input("Enter the latest tweet")
        tweet=clean_tweet(tweet)
        s=sentiment=analyzer.polarity_scores(tweet)["compound"]
        f=fund.predict([[mkcap,cval,high,low,bval,pe,div,roce,roe,sal]])
        if f==0 and s<-0.2:
            print("bad fundamentals and bearish sentiment.")
        elif f==0 and (s<0.2 and s>-0.2):
            print("bad fundamentals and neutral sentiment")
        elif f==0 and s>0.2:
            print("bad fundamentals but bullish sentiment")
        elif f==1 and s<-0.2:
            print("good fundamentals but bearish sentiment")
        elif f==1 and s>-0.2:
            print("good fundamentals but neutral sentiment")
        else:
            print("good fundamentals and bullish sentiment")
    else:
        print("bye bye")
        break
            
    

# -*- coding: utf-8 -*-


import re
import nltk
from sklearn.externals import joblib
import tweepy
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
import datetime
 
class TwitterClient(object):
    
    #Generic Twitter Class for sentiment analysis.
    
    def __init__(self):
        
        #Class constructor or initialization method.
        
        # keys and tokens from the Twitter Dev Console
        consumer_key = '1qRm35j3kskUyITp8FquUk3Sj'
        consumer_secret = 'bdzrMnivVpi5ku4i1Dd4Dpmxdyo1oWjsnQNUvHPAZWRaKuAroi'
        access_token = '158240218-M7DsUlvQKmxOtjfnKxFNKBTEmheuvNn4vi0MM6BP'
        access_token_secret = 'oWY5G9sTxnH81tFbaicN5DKs1AjkD2WsWM5oCyoSh8NoR'
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    #Processing Tweets

    def preprocessTweets(self,tweet):
        
        #Convert www.* or https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
        
        #Convert @username to __HANDLE
        tweet = re.sub('@[^\s]+','__HANDLE',tweet)  
        
        #Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        
        #trim
        tweet = tweet.strip('\'"')
        
        # Repeating words like happyyyyyyyy
        rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
        tweet = rpt_regex.sub(r"\1\1", tweet)
        
        #Emoticons
        emoticons = \
        [
         ('__positive__',[ ':-)', ':)', '(:', '(-:', ':-D', ':D', 'X-D', 'XD', 'xD', '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),\
         ('__negative__', [':-(', ':(', '(:', '(-:', ':,(', ':\'(', ':"(', ':((', ] ),\
        ]
    
        def replace_parenth(arr):
           return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]
        
        def regex_join(arr):
            return '(' + '|'.join( arr ) + ')'
    
        emoticons_regex = [ (repl, re.compile(regex_join(replace_parenth(regx))) ) for (repl, regx) in emoticons ]
        
        for (repl, regx) in emoticons_regex :
            tweet = re.sub(regx, ' '+repl+' ', tweet)
    
         #Convert to lower case
        tweet = tweet.lower()
        
        return tweet
    

    
    #Stemming of Tweets
    
    def stem(self,tweet):
        stemmer = nltk.stem.PorterStemmer()
        tweet_stem = ''
        words = [word if(word[0:2]=='__') else word.lower() \
                 for word in tweet.split() \
                 if len(word) >= 3]
        words = [stemmer.stem(w) for w in words] 
        tweet_stem = ' '.join(words)
        return tweet_stem
    
    
    #Predict the sentiment
    
    def predict(self, tweet,classifier):
        
        #Utility function to classify sentiment of passed tweet
            
        tweet_processed = self.stem(self.preprocessTweets(tweet))
                 
        if ( ('__positive__') in (tweet_processed)):
             sentiment  = 1
             return sentiment
            
        elif ( ('__negative__') in (tweet_processed)):
             sentiment  = 0
             return sentiment       
        else:  
            X =  [tweet_processed]
            sentiment = classifier.predict(X)
            return (sentiment[0])        

 
    def get_tweets(self,classifier, query, count = 1000):
            '''
            Main function to fetch tweets and parse them.
            '''
            # empty list to store parsed tweets
            tweets = []
     
            try:
                # call twitter api to fetch tweets
                fetched_tweets = self.api.search(q = query, count = count)
     
                # parsing tweets one by one
                for tweet in fetched_tweets:
                    # empty dictionary to store required params of a tweet
                    parsed_tweet = {}
     
                    # saving text of tweet
                    parsed_tweet['text'] = tweet.text
                    # saving sentiment of tweet
                    parsed_tweet['sentiment'] = self.predict(tweet.text,classifier)
                    # appending parsed tweet to tweets list
                    if tweet.retweet_count > 0:
                        # if tweet has retweets, ensure that it is appended only once
                        if parsed_tweet not in tweets:
                            tweets.append(parsed_tweet)
                    else:
                        tweets.append(parsed_tweet)
     
                # return parsed tweets
                return tweets
     
            except tweepy.TweepError as e:
                # print error (if any)
                print("Error : " + str(e))  
                
                
                

                
# Main function

def main():
    print('Loading the Classifier, please wait....')
    classifier = joblib.load('svmClassifier.pkl')
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    q = 0
    while (q == 0):    
        query = input("Enter the Topic for Opinion Mining: ")
        tweets = api.get_tweets(classifier, query, count = 1000)
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 0]
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 1]
        neg=(100*len(ntweets)/len(tweets))
        pos=(100*len(ptweets)/len(tweets))
        
        # console output of sentiment
        print("Opinion Mining on ",query)
        
        # plotting graph
        ax1 = plt.axes()
        ax1.clear()
        xar = []
        yar = []
        x = 0
        y = 0
        for tweet in tweets:
            x += 1
            if tweet['sentiment'] == 1 :
                y += 1
            elif tweet['sentiment'] == 0 :
                y -= 1
            xar.append(x)
            yar.append(y)
              
    
        ax1.plot(xar,yar)
        ax1.arrow(x, y, 0.5, 0.5, head_width=1.5, head_length=4, fc='k', ec='k')
        plt.title('Graph')
        plt.xlabel('Time')
        plt.ylabel('Opinion')
        plt.show()    
        
       
        # plotting piechart
        labels = 'Positive Tweets', 'Negative Tweets'
        sizes = [pos,neg]
         # exploding Negative
        explode = (0, 0.1) 
        fig1, ax2 = plt.subplots()
        ax2.pie(sizes, explode=explode, labels=labels, autopct='%2.3f%%', shadow=False, startangle=180)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2.axis('equal')  
        plt.title('Pie Chart')
        plt.show()
        
        
        
        # percentage of negative tweets
        print("Negative tweets percentage: ",neg)
        # percentage of positive tweets
        print("Positive tweets percentage: ",pos)
        
           
        now = datetime.datetime.now()
        print ("Date and Time analysed: ",str(now)) 
        
        q = input("Do you want to continue[Press 1 for Yes/ 0 for No]? ")
        
        if(q == 0):
            break
        
        
     
if __name__ == "__main__":
    main()
        
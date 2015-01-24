import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
nltk.download() 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text() 

    #  Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    #  In Python, searching a set is much faster than searching
    stops = set(stopwords.words("english"))                  
    # 
    #  Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # Join the words back into one string separated by space, 
   
    stemmer.stem('shopping')
    meaningful_words = [stemmer.stem(w) for w in words]   
    return( " ".join( meaningful_words ))



# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_reviews ):
    # clean reviews
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_train_reviews.append( review_to_words( train["review"][i] ) )

vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000) 
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train["sentiment"] )



#test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )
num_reviews = len(test["review"])
clean_test_reviews = [] 
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#predict
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
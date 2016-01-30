import pandas as pd
import numpy as np
import sys
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVR 
from sklearn import cross_validation
from nltk.stem.snowball import SnowballStemmer


# import_datasets imports datasets
# first clean and import datasets
def import_datasets(lemmatize = True, load_all = False):

	#train = pd.DataFrame.from_csv('train.csv', encoding = "ISO-8859-1")
	#test = pd.DataFrame.from_csv('train.csv', encoding = "ISO-8859-1")
	train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
	test = pd.read_csv('train.csv', encoding = "ISO-8859-1")

	#if load_all is true, load all datasets and merge on product_uid
	if load_all == True:
		# attributes = pd.DataFrame.from_csv('attributes.csv')
		# product_descriptions = pd.DataFrame.from_csv('product_descriptions.csv')
		attributes = pd.read_csv('attributes.csv')
		product_descriptions = pd.read_csv('product_descriptions.csv')

		train = pd.merge(train,product_descriptions,how='left',on='product_uid')
		test = pd.merge(test,product_descriptions,how='left',on='product_uid')
	# attach attributes and product_descriptions on product_id

	# make all words lower case in train product titles
	if lemmatize == True:
		lmtzr = WordNetLemmatizer()
		lowercase_titles = []
		for title in train.product_title:
			title = title.lower()
			lemmatized_phrase = ""
			for word in title:
				lemmatized_phrase += lmtzr.lemmatize(title) + " "
			lowercase_titles.append(lemmatized_phrase)
		train.product_titles = lowercase_titles

	return train, test 

# linear regression
# y = mx + b
# y ~ relevance
# m = coefficent, x = keyword matches
def simple_linear_predictor(X, y):

	lr = LinearRegression(normalize = True)
	lr.fit(X,y)
	# if one match relevance equals what?
	#test = pd.DataFrame.from_csv('test.csv')
	return lr

# SVM -> SVR() for continous variables
# X is feature based on number of matching words
# matching words on "search_term" and "title"
# y is relevance
def simple_SVM_model(X,y):

	clf = SVR()
	clf.fit(X,y)
	return clf

# first bit of analysis is to match search -> title -> relevance

def match_words(train):

	reload(sys)
	sys.setdefaultencoding('utf-8')

	# num_matching_words1 is the number of matching words on search and product title
	# num_matching_words2 is the number of matching words on search and description
	num_matching_words1 = []
	num_matching_words2 = []
	matching_words = []
	for index, row in train.iterrows():
		#title = row['product_title']
		# search = row['search_term']
		title = row['product_title']
		search = row['search_term']
		#descrip = row['product_description']
		descrip = ""
		title = title.split()
		search = search.split()
		for word, pos in nltk.pos_tag(descrip.split()):
			if pos == 'NN' or pos == 'VBZ':
				descrip += word + " "
		num_match1 = 0
		num_match2 = 0

		for word in title:
			for word2 in search:
				if str(word).lower() in str(word2).lower():
					num_match1 += 1
				elif str(word2).lower() in str(word).lower():
					num_match1 += 1
				for word3 in descrip:				
					if str(word2).lower() in str(word3).lower():
						num_match2 += 1
		num_matching_words1.append(num_match1)
		num_matching_words2.append(num_match2)

	return num_matching_words1, num_matching_words2

def cross_validate(X, train):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
		X, train['relevance'], test_size = 0.4, random_state = 0)

	return X_train, X_test, y_train, y_test

# import datasets
train, test = import_datasets(lemmatize = True, load_all = True)

# find number of matched words
num_matched_words1, num_matching_words2 = match_words(train)
#print float(sum(num_matched_wrds) / len(num_matched_words))
X = np.array(num_matched_words1)
X = np.hstack(num_matching_words2)
#convert to X
#X = X[:,np.newaxis]

# Cross validation on Linear Regression and Linear Regression
#print len(X) 
#print len(train)
X_train, X_test, y_train, y_test = cross_validate(X,train)
	
# based on linear regression
def linear_submission():	
	lr = simple_linear_predictor(X_train,y_train)
	print lr.coef_, lr.intercept_
	print lr.score(X_test, y_test)

	# load the testing data, put the testing data into the model
	# append the results and then send to a csv
	test_data = pd.DataFrame.from_csv('test.csv')
	num_matching_words = []
	num_matching_words = match_words(test_data)
	X_test = np.array(num_matching_words)
	X = X_test[:,np.newaxis]
	results = lr.predict(X)

	# need to clean results for values greater than 3
	np.place(results, results>3, 3)

	test_data['results'] = results

	submission_df = test_data['results']
	print submission_df
	submission_df.to_csv('kaggle_home_depot_sub.csv', sep = ',', encoding='utf-8')
 
# SVM testing
# for the Linear SVM model
# X has to be the data
# y has to be the groups 1,2,3

def SVM_train():
	#num_matching_words = match_words(train)
	#X_train = np.insert(X_train,1,num_matching_words,axis=1)
	# print X_train
	#train['matches'] = num_matching_words
	#training_df = train.drop(['product_uid', 'product_title', 'search_term', 'relevance'],axis=1).values
	clf = simple_SVM_model(X_train, y_train)
	clf.score(X_test,y_test)
	print clf.score(X_test,y_test)
	test_data = pd.DataFrame.from_csv('test.csv')
	num_matching_words = []
	num_matching_words = match_words(test_data)
	X_test = np.array(num_matching_words)
	X = X_test[:,np.newaxis]
	results = clf.predict(X)

	# need to clean results for values greater than 3
	np.place(results, results>3, 3)

	test_data['results'] = results

	submission_df = test_data['results']
	print submission_df
	submission_df.to_csv('kaggle_home_depot_sub.csv', sep = ',', encoding='utf-8')


SVM_train()

import numpy as np
import pandas as pd
import gzip
import json
import preprocessing as pre

#Read json.gz (gzip)
#Chunk reading needed
def parse(path, lower, limit):
  g = gzip.open(path, 'rb')
  i=0
  for l in g:
    i = i+1
    if i < lower:
        continue
    if i > limit:
        break
    yield json.loads(l)

def getDF(path, lower, limit):
  i = 0
  df = {}
  for d in parse(path, lower, limit):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#No chunk reading needed
def parseSmall(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)
    
def getDFSmall(path):
  i = 0
  df = {}
  for d in parseSmall(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#Transforms big dataset
def transform(df):
    df['author'] = df.authors.str[:1]
    df['author'] = df['author'].astype(str)
    df['author_id'] = df['author'].str.slice(start= 13, stop=26).astype(str)
    df['author_id'] = df['author'].str.extract('(\d+)', expand=False)
    df['author_id'] = df['author_id'].astype(float)
    return df

#Setup external data to fuzzy match to items.csv
#Books1: http://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data
books = pd.read_csv('books.csv', delimiter=',', error_bad_lines=False)

#Books2: https://data.world/divyanshj/users-books-dataset
books2 = pd.read_csv('Dataworld\BX-Books.csv', delimiter=';', quotechar='"', encoding = 'unicode_escape',error_bad_lines=False)
books2Rat = pd.read_csv('Dataworld\BX-Book-Ratings.csv', delimiter=';', quotechar='"', encoding = 'unicode_escape',error_bad_lines=False)

#Transformation
books2Rat = books2Rat.drop(columns=['User-ID'])
books2Rat = books2Rat.groupby(['ISBN']).mean()
books2Rat = books2Rat.reset_index()

#Merge with ratings
books2 = books2.merge(books2Rat, how='left', on='ISBN')

#Adapt scale to other datasets (max = 5)
scaleMin = 0
scaleMax = 5
books2['Book-Rating'] = (scaleMax-scaleMin) * ((books2['Book-Rating']-min(books2['Book-Rating']))/(max(books2['Book-Rating'])-min(books2['Book-Rating']))) + scaleMin
#Other transformation
books2['Year-Of-Publication'] = books2['Year-Of-Publication'].str.strip()
books2['Year-Of-Publication'] = books2['Year-Of-Publication'].replace('[^0-9]+', np.nan, regex=True)
books2['Year-Of-Publication'] = books2['Year-Of-Publication'].str.strip()
books2['Year-Of-Publication'] = books2['Year-Of-Publication'].replace(' ', '', regex=True)
books2['Year-Of-Publication']= books2['Year-Of-Publication'] .fillna('0')
#Put in 1.1.xxxx as publication_date (only year was given for this dataset)
books2['Year-Of-Publication'] = "1/1/" + books2['Year-Of-Publication']
books2.rename(columns={'Book-Title': 'title', 'Book-Author': 'authors', 'Publisher': 'publisher', 'ISBN': 'isbn', 
                       'Year-Of-Publication': 'publication_date', 'Book-Rating': 'average_rating'}, inplace=True)

#Decided to append all (instead of joining), even if some isbn might be duplicates, because other way of writing title/author, other publisher etc.
#Merge books and books2
booksMerge = books2.append(books)
booksMerge = booksMerge.rename(columns = {'  num_pages': 'num_pages'})

import gc
del books
del books2
del books2Rat
gc.collect()

#Books3: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
#Wasn't able to load the file all in once, so did chunks instead, but wasn't able to load all at once. So I don't know how many missing authors can be found
#Needs to be uncommented to load all parts
books3Part1 = getDF('Good Reads\Books\goodreads_books.json.gz', 0, 300000) 
books3Part1 = transform(books3Part1)
books3Part2 = getDF('Good Reads\Books\goodreads_books.json.gz', 300001, 600000)
books3Part2 = transform(books3Part2)
books3 = pd.concat([books3Part1, books3Part2])
del books3Part1
del books3Part2
gc.collect()

books3Part3 = getDF('Good Reads\Books\goodreads_books.json.gz', 600001, 900000)
books3Part3 = transform(books3Part3)
books3 = pd.concat([books3, books3Part3])
del books3Part3
gc.collect()

books3Part4 = getDF('Good Reads\Books\goodreads_books.json.gz', 900001, 1200000)
books3Part4 = transform(books3Part4)
books3 = pd.concat([books3, books3Part4])
del books3Part4
gc.collect()

books3Part5 = getDF('Good Reads\Books\goodreads_books.json.gz', 1200001, 1500000)
books3Part5 = transform(books3Part5)
books3 = pd.concat([books3, books3Part5])
del books3Part5
gc.collect()

books3Part6 = getDF('Good Reads\Books\goodreads_books.json.gz', 1500001, 1800000)
books3Part6 = transform(books3Part6)
books3 = pd.concat([books3, books3Part6])
del books3Part6
gc.collect()

books3Part7 = getDF('Good Reads\Books\goodreads_books.json.gz', 1800001, 2100000)
books3Part7 = transform(books3Part7)
books3 = pd.concat([books3, books3Part7])
del books3Part7
gc.collect()

books3Part8 = getDF('Good Reads\Books\goodreads_books.json.gz', 2100001, 2400000)
books3Part8 = transform(books3Part8)
books3 = pd.concat([books3, books3Part8])
del books3Part8
gc.collect()

#Author info for dataset
books3Authors = getDFSmall('Good Reads\Books\goodreads_book_authors.json.gz') 

#Transformation
books3Authors = books3Authors[['author_id', 'name']]
books3Authors.author_id = books3Authors.author_id.astype(float)
books3 = books3.merge(books3Authors, how = 'left', on = 'author_id')
books3 = books3.rename(columns = {'authors': 'authors_list'})
books3 = books3.rename(columns = {'name': 'authors'})
books3ToDrop = list(filter(lambda value: value.startswith(r'count (#' ) or value.startswith(r'name (#' ) or value.startswith(r'author_id (#' ) or value.startswith(r'role (#' ), list(books3.columns)))
books3 = books3.drop(columns = books3ToDrop)
books3 = books3.rename(columns = {'image_url': 'Image-URL-S', 'name': 'name_2'})
books3['publication_date'] = books3.publication_month.astype(str) + r'/' + books3.publication_day.astype(str) + r'/' + books3.publication_year.astype(str)
books3 = books3.drop(columns = ['publication_day', 'publication_month', 'publication_year', 'author'])

#Append to other 2 datasets
booksMerge = booksMerge.append(books3)

del books3
del books3Authors
gc.collect()


#Fuzzy Script adapted from TowardsDataScience from https://drive.google.com/file/d/1Z4-cEabpx7HM1pOi49Mdwv7WBOBhn2cl/view

## load libraries and set-up:
pd.set_option('display.max_colwidth', -1)
import re
#pip install ftfy #  text cleaning for decode issues..
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
#!pip install nmslib
import nmslib

'''
The below script creates a cleaning function and turns both the master data (the 'clean' data) and the items to be matched against into vectors for matching.

df This is our 'clean' list of company names
df_CF This is the messy raw data that we want to join to the clean list
output1 tf_idf_matrix produced from clean data
output2 messy_tf_idf_matrix produced from the raw data
'''

df = booksMerge
input1_column = 'title'
#Uses preprocessing script of Haowen
df_CF = pre.preprocessing('../items.csv', 'transactions.csv', False)[0]
input2_column = 'title'

del booksMerge
gc.collect()

#transforms company names with assumptions taken from: http://www.legislation.gov.uk/uksi/2015/17/regulation/2/made
def ngrams(string, n=3):
    """Takes an input string, cleans it and converts to ngrams. 
    This script is focussed on cleaning UK company names but can be made generic by removing lines below"""
    string = str(string)
    string = string.lower() # lower case
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    chars_to_remove = [")","(",".","|","[","]","{","}","'","-"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']' #remove punc, brackets etc...
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

#See how n-grams works
# print('All 3-grams in "Department":')
# print(ngrams('Depar-tment &, Ltd'))

###FIRST TIME RUN - used to build the matching table
##### Create a list of items to match here:
org_names = list(df[input1_column].unique()) #unique org names from company watch file
#Building the TFIDF off the clean dataset - takes about 5 min
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(org_names)

##### Create a list of messy items to match here:
# file containing messy supplier names to match against
messy_names = list(df_CF[input2_column].unique()) #unique list of names
#Creation of vectors for the messy names
messy_tf_idf_matrix = vectorizer.transform(messy_names)


'''
Matching

This script takes the two sets of vectors and matches them to each other. It uses the NMSLIB library https://github.com/nmslib/nmslib as this is the fastest python library avaliable for this matching

Input1 - 'tf_idf_matrix' created from scripts above from the Company Watch bulk data
Input2 - 'messy_tf_idf_matrix' created from the scripts above (from the data set to match against. eg Contracts Finder)
Output - 'nbrs' which contains the index matches across the two inputs alongside a confidence score (lower is better)
'''

# create a random matrix to index
data_matrix = tf_idf_matrix#[0:1000000]

# Set index parameters
# These are the most important ones
M = 80
efC = 1000

num_threads = 4 # adjust for the number of threads
# Intitialize the library, specify the space, the type of the vector and add data points 
index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 

index.addDataPointBatch(data_matrix)
# Create an index
index.createIndex() 

#K-Nearest Neighbour
# Number of neighbors 
num_threads = 4
K=1
query_matrix = messy_tf_idf_matrix
query_qty = query_matrix.shape[0]
nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)


'''Script for joining matches back to the data set'''
mts =[]
for i in range(len(nbrs)):
  origional_nm = messy_names[i]
  try:
    matched_nm   = org_names[nbrs[i][0][0]]
    conf         = nbrs[i][1][0]
  except:
    matched_nm   = "no match found"
    conf         = None
  mts.append([origional_nm,matched_nm,conf])

mts = pd.DataFrame(mts,columns=['title','title_match','conf'])
results = df_CF.merge(mts,how='left',on='title') 

del df_CF
del org_names
del messy_names
del mts
del nbrs
gc.collect()

results.conf.hist()
#Profile of matches - lower is higher confidence
###Last step of algorithm

#Further Transformation
#Only take certain confidence
confTreshold = -0.7
results.title_match[results.conf > confTreshold] = np.nan
results = results.drop_duplicates()

#Match with other columns
#Afterwards there will be some duplicates item-wise, but it is because booksMerge has listet the same book 
#(same title) with different isbn etc. (like it is in items, too) > Keep for now, decide later what to do
results = results.merge(df, how = 'left', left_on = 'title_match', right_on = "title", suffixes = ('', '_y'))

del df
gc.collect()

results = results.drop(columns=['title_y'])

#######Do second checky by fuzzy matching authors: Fuzzy Match Row by Row
from fuzzywuzzy import fuzz

results['author'] = results['author'].astype(str)
results['authors'] = results['authors'].astype(str)
results['confAuthor'] = results.apply(lambda x : fuzz.ratio(x.author, x.authors),axis=1)

confAuthorsThreshold = 50
results.authors[(results.confAuthor < confAuthorsThreshold) & (results.author != 'Unknown')] = 'Unknown'
results.isbn[(results.confAuthor < confAuthorsThreshold) & (results.author != 'Unknown')] = 0
results.average_rating[(results.confAuthor < confAuthorsThreshold) & (results.author != 'Unknown')] = -1

results.authors[results.authors == 'nan'] = 'Unknown'
results.authors[results.authors == 'NaN'] = 'Unknown'

#True: There is additional data available that can be used! False: Don't use joined data (conf too low)
results['AdditionalData'] = np.where((results['authors'] == 'Unknown') & (results['isbn'] == 0) & (results['average_rating'] == -1), False, True)

results = results.drop(columns = ['asin', 'format', 'is_ebook', 'kindle_asin', 'edition_information', 'author_id', 'work_id', 'book_id'])

missingValuesAuthorBefore = results[results['author'] == 'Unknown'].iloc[:,0:6].drop_duplicates().shape[0]
#Missing authors are filled in
results.author[(results['author'] == 'Unknown') & (results['authors'] != 'Unknown')] = results['authors']

#Could find?
columns = list(results[results['author'] == 'Unknown'].iloc[:,0:6].columns)
columns.append('authors')
missingValuesAuthorNow = results[(results['author'] == 'Unknown') & (results.authors == 'Unknown')][columns].drop_duplicates().shape[0]
#Was able to find:
print('Was able to find: ' + str(missingValuesAuthorBefore - missingValuesAuthorNow))

results.title_match.isnull().sum()

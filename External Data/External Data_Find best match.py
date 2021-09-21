import pandas as pd
from fuzzywuzzy import fuzz
import preprocessing as pre
import re

def remove_umlaut(string):
    """
    Removes umlauts from strings and replaces them with the letter+e convention
    :param string: string to remove umlauts from
    :return: unumlauted string
    """
    u = 'ü'.encode()
    U = 'Ü'.encode()
    a = 'ä'.encode()
    A = 'Ä'.encode()
    o = 'ö'.encode()
    O = 'Ö'.encode()
    ss = 'ß'.encode()

    string = string.encode()
    string = string.replace(u, b'ue')
    string = string.replace(U, b'Ue')
    string = string.replace(a, b'ae')
    string = string.replace(A, b'Ae')
    string = string.replace(o, b'oe')
    string = string.replace(O, b'Oe')
    string = string.replace(ss, b'ss')

    string = string.decode('utf-8')
    return string

def tryconvert(x):
    for value in x:
        try:
            if value['type'] == 'ISBN_10':
                isbn = value['identifier']
                return isbn
        except Exception as e:
            return 'Unknown'
    return 'Unknown'

itemsOrig = pre.preprocessing(r'../items.csv', r'transactions.csv', False)[0]

#This is the preprocessing that was used for the API queries, so apply same preprocessing to title, so that the titles can function as keys again
items = itemsOrig.copy()
items.title = items.title.apply(lambda x: remove_umlaut(x))
items.publisher = items.publisher.apply(lambda x: remove_umlaut(x))
items.title = items.title.replace(':', '', regex = True)
items.publisher = items.publisher.replace(':', '', regex = True)
items.publisher = items.publisher.apply(lambda x: re.sub("[^a-zA-Z\d\s:]+", "", x))
items.title = items.title.apply(lambda x: re.sub("[^a-zA-Z\d\s:]+", "", x))
items['titleOrig'] = items.title
items.title = items.title.replace(' ', '%20', regex = True)
items['publisherOrig'] = items.publisher
items.publisher = items.publisher.replace(' ', '%20', regex = True)


#External data
api = pd.read_pickle(r'ISBNDB_All.pkl')
google = pd.read_pickle(r'Google Books_All.pkl')
fuzzy = pd.read_pickle(r'Fuzzy Matching_All.pkl')

google.columns

#Some preprocessing
#fuzzy = fuzzy.rename(columns = {'author': 'authorOrig', 'publisher': 'publisherOrig', 'title': 'titleOrig', 'title_match': 'title', 'publisher_y': 'publisher'})
google = google.explode('authors')
google = google.reset_index().drop(columns = 'index')
google['industryIdentifiers'] = google['industryIdentifiers'].apply(lambda d: d if isinstance(d, list) else [])
#Get isbns out of list of dicts in industryIdentifiers
google['industryIdentifiers'] = google['industryIdentifiers'].apply(lambda x: list(x))
google['isbn'] = google['industryIdentifiers'].apply(lambda x: tryconvert(x))


#Process external data to combine as a next step
columns = ['itemID', 'title', 'titleOrig', 'authors', 'authorOrig', 'publisher','publisherOrig', 'isbn']
a = api[columns]
g = google[columns]
f = fuzzy[columns]

#Authors are a list in google dataset, so expand into several rows so that later the one best match can be found (partial_ratio)
g = g.explode('authors') 

a = a.drop_duplicates()
g = g.drop_duplicates()
f = f.drop_duplicates()

a['source'] = 'ISBNDB'
g['source'] = 'Google'
f['source'] = 'Fuzzy Matching'

#Combine all external data
final = pd.DataFrame()
final = pd.concat([a, g, f])
#Preprocess authors
final = final.fillna('Unknown')
final.isbn = final.isbn.replace(0, 'Unknown')
final.isbn = final.isbn.replace('', 'Unknown')
final.isbn = final.isbn.replace(' ', 'Unknown')
unknownAuthorNames = ['Various', 'n/a', 'n/c', 'N/a', 'Na', 'NA NA', 'N', 'NONE, NONE, NONE']
for value in unknownAuthorNames:
    final['authors'] = final['authors'].replace(value, 'Unknown')

#Do fuzzy matching, for title normal ratio, for author and publisher partial, so e.g. one author can be identified among a list of several
#Install pip install python-Levenshtein
final['title'] = final['title'].astype(str)
final['titleOrig'] = final['titleOrig'].astype(str)
final['confTitle'] = final.apply(lambda x : fuzz.ratio(x.title.lower(), x.titleOrig.lower()),axis=1)

final['publisher'] = final['publisher'].astype(str)
final['publisherOrig'] = final['publisherOrig'].astype(str)
final['confPublisher'] = final.apply(lambda x : fuzz.partial_ratio(x.publisher.lower(), x.publisherOrig.lower()),axis=1)

final['authorOrig'] = final['authorOrig'].astype(str)
final['authors'] = final['authors'].astype(str)
final['confAuthor'] = final.apply(lambda x : fuzz.partial_ratio(x.authorOrig.lower(), x.authors.lower()),axis=1)


final = final[final.isbn != 'Unknown']

#Now filter out too low confidences according to agreed rules (Weekly 27.05.)
final = final[((final.confTitle >= 45) & (final.confAuthor >= 45)) | ((final.confTitle >= 95) & ((final.confAuthor >= 30) | (final.authors == 'Unknown')))]

#Sort also by source (descending), because we have the following preference: ISBNDB, Google, Fuzzy Matching
final = final.sort_values(by = ['itemID', 'confTitle', 'confAuthor', 'confPublisher', 'source'], ascending = False)

#Now we drop all except the first (we have sorted, so the one we want per group is the first)
final = final.drop_duplicates('itemID')

#Some measures for the final dataset
measures = {
    'unknownISBNs': final[final.isbn == 'Unknown'].shape[0],
    'uniqueIDs': final.itemID.nunique(),
    'percentIDs': final.itemID.nunique() / itemsOrig.itemID.nunique(),
    'missingIDs': itemsOrig.itemID.nunique() - final.itemID.nunique(),
    'distributionSources': final.source.value_counts()/final.shape[0]
    }

#For the final result we want back original titles of items.csv not preprocessed ones (see row 47)
final = final.rename(columns = {'title': 'titleMatch', 'authors': 'authorMatch', 'publisher': 'publisherMatch'})
result = final[['itemID', 'titleMatch', 'authorMatch', 'publisherMatch', 'isbn', 'confTitle', 'confAuthor', 'confPublisher']].merge(itemsOrig, on = 'itemID', how = 'left').sort_values(by = 'itemID')
#Reorder columns
result = result[list(itemsOrig.columns) + list(['titleMatch', 'authorMatch', 'publisherMatch', 'isbn', 'confTitle', 'confAuthor', 'confPublisher'])]

#Export data
#pd.to_pickle(result, r'External Data_final.pkl')

#Test how many matches we have for evaluation dataset
evaluation = pd.read_csv(r'../clustering model/evaluation.csv')
evaluation = evaluation.merge(result, how = 'left', on = 'itemID')
#How many items of evaluation do not have a match?
missingItemIDsEvaluation = evaluation[evaluation.titleMatch.isnull()].shape[0]

import pandas as pd
import preprocessing as pre
import re
from urllib.request import urlopen
import json

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


items = pre.preprocessing('../items.csv', 'transactions.csv', False)[0]

#Some preprocessing needed?
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

key = 'AIzaSyBp8SR44Q144Ua_SoQnY5vGIzM1x0teo-A'

#api = googlebooks.Api()
api = "https://www.googleapis.com/books/v1/volumes?q="

final = pd.DataFrame()
df = pd.DataFrame()
result = pd.DataFrame()

countFails = 0
for title, publisher, titleOrig, publisherOrig, i in zip(items.title, items.publisher, items.titleOrig, items.publisherOrig, range(0, items.shape[0]+1)):
    if i > 100000:
        break
    if i < 51000:
        continue
    url = api + title + '%20' +publisher + '&key=' + key
    print(i)
    result = result.iloc[0:0]
    try:
        # send a request and get a JSON response
        resp = urlopen(url)
        # parse JSON into Python as a dictionary
        request = json.load(resp)
        request = request["items"]
        length = len(request)
        if length > 1:
            #Only want first 5 results max
            if length > 5:
                request = request[:5]
            else:
                request = request[:length]
        else:
            request = request[0]
        #book_data = book_data["items"][0]["volumeInfo"]
    except Exception as e:
        print('URL for title No. ' + str(i) + ' - ' + title + ' von ' + publisher + ' - not working')
        print(e)
        countFails = countFails + 1
        continue
    try:
        for item in request:
            df = df.iloc[0:0]
            for subItem in item:
                length = df.shape[0]
                if length == 0:
                    try:
                        itemToAdd = pd.DataFrame(request)[subItem]
                        df = pd.json_normalize(itemToAdd)
                    except:
                        pass
                else:
                    try:
                        df = pd.concat([df, pd.json_normalize(pd.DataFrame(request)[subItem])], axis = 1)
                    except:
                        pass
            #Rename duplicate columns
            cols=pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique(): 
                cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
                df.columns=cols
        result = result.append(df)
        result['items_title'] = titleOrig
        result['items_publisher'] = publisherOrig
    except Exception as e:
        print(e)
    final = pd.concat([final, result])

final = final[ ['items_title'] + [ col for col in final.columns if col != 'items_title' ] ]     

final.to_pickle(r'GoogleBooks.pkl')



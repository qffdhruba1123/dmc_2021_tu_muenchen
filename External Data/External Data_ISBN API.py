import requests as req
import pandas as pd
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


items = pre.preprocessing('../items.csv', 'transactions.csv', False)[0]

#Some preprocessing needed?
items.title = items.title.apply(lambda x: remove_umlaut(x))
items.publisher = items.publisher.apply(lambda x: remove_umlaut(x))
items.publisher = items.publisher.apply(lambda x: re.sub("[^a-zA-Z\d\s:]+", "", x))
items.title = items.title.apply(lambda x: re.sub("[^a-zA-Z\d\s:]+", "", x))

items['titleOrig'] = items.title
items.title = items.title.replace(' ', '%20', regex = True)
items['publisherOrig'] = items.publisher
items.publisher = items.publisher.replace(' ', '%20', regex = True)

#Example URL
#https://api2.isbndb.com/search/books?page=1&pageSize=1&text=The%20Maze%20Runner%201&publisher=Scholastic%20Ltd

h = {'Authorization': '46057_91e34ec9a04bbd6085696baa0b9ba4be'}

#Test
#items = items[items.titleOrig.str.contains('Red Queen')]

final = pd.DataFrame()
df = pd.DataFrame()

for title, publisher, titleOrig, publisherOrig, i in zip(items.title, items.publisher, items.titleOrig, items.publisherOrig, range(0, items.shape[0]+1)):
    if i < 72287:
        continue
    url = "https://api.pro.isbndb.com/search/books?page=1&pageSize=1&text=" + title + "&publisher=" + publisher
    print(i)
    try:
        resp = req.get(url, headers=h)
        data = resp.json()
        data = data['data']
        df = pd.DataFrame.from_dict(data, orient='columns')
    except:
        df = df.iloc[0:0]
        print('URL for title No. ' + str(i) + ' - ' + titleOrig + ' von ' + publisherOrig + ' - not working')
    df['items_title'] = titleOrig
    df['items_publisher'] = publisherOrig
    final = final.append(df)
    
final = final[ ['items_title'] + [ col for col in df.columns if col != 'items_title' ] ]    
finalCopy = final.copy()

finalCopy.to_pickle(r'isbn-API_including publisher3.pkl', index = False)


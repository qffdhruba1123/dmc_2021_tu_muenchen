import pandas as pd
import numpy as np
import os
import csv



def find_topic(subtopic):
    subtopics = subtopic[1:-1].split(",")

    assumed_val = None
    for sub in subtopics:
        if sub and sub[0].isalpha():
            assumed_val = sub
            break

    if assumed_val is None:
        assumed_val = subtopics[0]

    return assumed_val


def preprocessing(item_path=os.path.join(os.getcwd(), 'items.csv'),
                  transaction_path=os.path.join(os.getcwd(), 'transactions.csv'),
                  remove_duplicates=True):
    """load both csv files (by default direct form the current working space) and clean items data by
            - filling NaN with data collected from Amazon, Apple Books
            - filling item's missing main topic with its first subtopic
            - replacing dirty inputs and missing inputs with 'Unknown'
            - remove duplicates in the items (optional)

            :param remove_duplicates: specifies whether duplicates should be removed or not
            :param transaction_path: path to transactions.csv file
            :param item_path: path to items.csv file
        Usage:


    """

    # print(item_path)
    # print(transaction_path)
    items = pd.read_csv(item_path, header=0, encoding='utf-8', sep='|',
                        quoting=csv.QUOTE_NONE)
    transactions = pd.read_csv(transaction_path, header=0, encoding='utf-8', sep='|',
                               quoting=csv.QUOTE_NONE)
    eval = pd.read_csv('clustering model/evaluation.csv', header=0, encoding='utf-8', sep='|',
                       quoting=csv.QUOTE_NONE)


    ## fill missing publisher with data collected from Amazon, Apple Books
    missing_publisher_data = {
        'itemID': [58383, 41902, 59119, 66096, 53966, 45452, 65060, 26648, 44964],
        'publisher': [
            'Tektime', 'LitRes', 'Tektime', 'Tektime', 'Tektime', 'Tektime', 'Tektime', 'Tektime', 'Tektime'
        ]
    }
    missing_publisher_data = pd.DataFrame(missing_publisher_data)
    # put the values
    target_item_ids = missing_publisher_data['itemID'].tolist()
    items.loc[
        items['itemID'].isin(target_item_ids),
        'publisher'
    ] = missing_publisher_data['publisher'].tolist()

    # ##
    # items['publisher'] = items['publisher'].str.lower()
    # items['publisher'] = items['publisher'].replace({'^[^\w]*$': ''}, regex=True)
    # items['publisher'] = items['publisher'].replace(np.nan, 'NaN')

    ## fill item's missing main topic with its first subtopic
    # putting values
    missing_mains = items['main topic'].isna()
    missing_sub = items[missing_mains]
    items.loc[
        missing_mains,
        'main topic'
    ] = missing_sub['subtopics'].apply(find_topic)

    ## cleaning author
    # items.set_index('itemID')
    items['author'] = items['author'].replace({'^,': ''}, regex=True)
    items['author'] = items['author'].replace({'^-': ''}, regex=True)
    items['author'] = items['author'].replace({'^-': ''}, regex=True)
    items['author'] = items['author'].replace({'^[^\w]*$'}, 'Unknown', regex=True)
    items['author'] = items['author'].replace(np.nan, 'Unknown')

    if remove_duplicates:
        subset = list(set(items.columns) - {'itemID'})
        duplicated = items.duplicated(keep=False, subset=subset)
        duplicated_ids = items[duplicated]['itemID']

        # not remove those that occur in transactions data and evaluation data
        _re = set(transactions.itemID).union(set(eval.itemID))
        removable_ids = pd.Series(list((set(duplicated_ids) - _re)))
        duplicated_items = items[items['itemID'].isin(removable_ids)]
        still_duplicated = duplicated_items.duplicated(subset=subset)
        duplicated_items = duplicated_items[still_duplicated]
        del_ids = duplicated_items['itemID']
        items = items.drop(items[items['itemID'].isin(del_ids)].index)
        items = items.reset_index(drop=True)
        # items[~items['itemID'].isin(del_ids)]
        # df.drop(df[ < some
        # boolean
        # condition >].index)

    items['subtopics'] = items['subtopics'].replace({'^\[': ''}, regex=True)
    items['subtopics'] = items['subtopics'].replace({'\]$': ''}, regex=True)

    return items, transactions, eval

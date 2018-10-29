import zerorpc
import pdb
#Zotero Library
from pyzotero import zotero
import pandas as pd
import json
from pathlib import Path

class Zotero(object):
    def __init__(self, id, type, token):
        self.name = '2476068'
        self.type = 'user'
        self.token = 'ravDnfy0bMKyuDrKq5kNz5Rh'
        self.zot = zotero.Zotero(self.name, self.type, self.token)
        # self.folder = 'sessData'
    def getColletions(self):
        return self.zot.collections()
    def getCollectionItems(self,collection_key):
        # pdb.set_trace()
        results = []
        #Get items
        collection_items = self.zot.collection_items(collection_key)
        #Retrieve data pieces from each item
        for each in collection_items:
            results.append(each['data'])
        #Data is not agregated in order to agregate
        #Create a pandas object
        df = pd.DataFrame(results)
        #Separate PDFS and data
        pdfs = df.loc[df['contentType']== 'application/pdf']
        no_pdfs = df.loc[df['contentType']!= 'application/pdf']
        #We need to do the following merging in order to centralize all in one table.
        #Merge both into a single table
        pdfs = pdfs[['key','parentItem','contentType', 'charset', 'filename', 'md5', 'mtime', 'tags', 'relations','note','linkMode']]
        #Removed some keys to avoid errors: 1 something (dont remember what key),  2 'publicationTitle'
        no_pdfs = no_pdfs[['key', 'version', 'parentItem', 'itemType', 'title', 'accessDate', 'url', 'note', 'tags', 'relations', 'dateAdded', 'dateModified','creators','date']]
        items = pd.merge(pdfs,no_pdfs,left_on='parentItem',right_on='key',how='outer')
        #We rename the columns to show which is pdf file and wich is main key
        items = items.rename(columns={'key_x': 'pdf_file','title': 'name','key_y': 'key'})
        #For now we filter the elements that do not have a title.
        items = items[items.name.notnull()]
        return items.to_json(orient='records')
    def downloadItems(self,itemKeys,collection_items,folder):
        # pdb.set_trace()
        items_path = []
        for eachItem in itemKeys:
            if eachItem != None:
                items_path.append(folder + '/documents/' + eachItem + '.pdf')
                if Path(folder + '/documents/' + eachItem + '.pdf').is_file():
                    print ("file already Downloaded do not need to download")
                else:
                    print ("Downloading file : " + folder + '/documents/' + eachItem + '.pdf')
                    self.zot.dump(eachItem,folder + '/documents/' + eachItem + '.pdf')
        return items_path
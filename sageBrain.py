#!/usr/bin/python
#I THINK THIS ARE UNUSED IMPORTS THEY MIGHT BE USEFUL IN SESSION OR DOCUMENTS CLASSES FOR CERTAIN FUNTIONS THAT I AM NOT
#USING NOW THEY COME FROM WHERE ALL CLASSESS WHERE TOGETHER
 
#lda
# from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
# from nltk.stem.snowball import SnowballStemmer
# from gensim import corpora, models
# import gensim
#Bag of Words
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
#TfIdf
# import gensim
#Word2Vect Extra
# from gensim.models.doc2vec import TaggedDocument
#Corpus Test
# from nltk.corpus import PlaintextCorpusReader
# from nltk.corpus import wordnet as wn
# import nltk as nltk

#HIndex
# from scholarmetrics import hindex
#otrher imports
# import operator
# from collections import Counter

#END OF UNUSED IMPORTS

#Server stuff
import zerorpc
import logging
logging.basicConfig()
#Data Science Imports
import pandas as pd
#Debugg
import pdb
#json parser
import json

#Import Brain Classes
from classes.session_class import Session
from classes.document_class import Document
from classes.topic_class import Topic #Not using this now
from classes.zotero_class import Zotero

#Interaface Class to talk with Node
class SageBrain(object):
    """ Sage Brain Class"""
    def __init__(self, session_id):
        self.id = "brainInterface"
        self.sessions = []
        self.session_counter = -1
        self.actualSession = -1
        self.addSession(session_id)
        self.zotero = Zotero('2476068','user','ravDnfy0bMKyuDrKq5kNz5Rh')
        self.sess_folder = './sessData/' + session_id

    def Zotero(self,function_name,collection_key,itemKeys,collection_items):
        value_to_return = False
        if function_name == 'getCollections':
            value_to_return = self.zotero.getColletions()
        elif function_name == 'getCollectionItems':
            value_to_return = self.zotero.getCollectionItems(collection_key)
        elif function_name == 'downloadItems':
            value_to_return = self.zotero.downloadItems(itemKeys,collection_items,self.sess_folder)
        
        return value_to_return

    def DocInterface(self, fileName, doc_id,dataType,metadata):
        """ Brain Interface """
        if dataType == 'zoteroCollection':
            for index,each_doc in enumerate(metadata):
                doc_in_sess = self.actualSession.docInSess(metadata[index]['key'])
                if doc_in_sess == False:
                    doc = Document(self.actualSession, metadata[index]['name'],  metadata[index]['key'], 'zotero', "user", each_doc)
                    self.actualSession.addDoc(doc)
                else:
                    print "doc in sess"
                    doc_from_sess = self.actualSession.returnDoc(metadata[index]['key'])
                    doc = Document(self.actualSession, metadata[index]['name'],  metadata[index]['key'], 'inSession', "user", doc_from_sess)
        else:
            doc = Document(self.actualSession, fileName, doc_id, "doc", "user", False)
            self.actualSession.addDoc(doc)

        #Store the Session Documents in CSV file
        self.actualSession.documents.to_csv(self.sess_folder + '/documents.csv',header=True,encoding='utf-8',index_label='index')
        #Get authors test
        test = self.actualSession.returnDocsBy('author')
        test2 = self.actualSession.get_topics_by(test,'author')
        pdb.set_trace()


        #We get the topic and words Using Umap NSA algorithm and we include them into session
        self.actualSession.get_topics(self.actualSession.documents)
        #We get the years
        years = self.actualSession.get_years()
        #We store the authors in session
        # pdb.set_trace()
        self.actualSession.authorList.to_csv(self.sess_folder + '/authors.csv',header=True,encoding='utf-8',index='Author',index_label='index')
        return {"documents":self.actualSession.documents.to_json(),"years":years.to_json(),"authors":self.actualSession.authorList.to_json(),"doc_topics":{'topics':self.actualSession.topics.to_json(), 'order':json.dumps(self.actualSession.topics.columns.values.tolist())}}
    
    def addCitations(self):
        documents_msg = []
        for each_doc in self.actualSession.documents:
            print each_doc.title
            each_doc.GetScholarInfo()
            documents_msg.append(each_doc.create_document_msg())
        # pdb.set_trace()
        return {"documents":documents_msg}


    def addSession(self, session_id):
        """ Function to add new sessions """
        self.session_counter = self.session_counter + 1
        self.sessions.append(Session(session_id))
        self.actualSession = self.sessions[self.session_counter]


#Internal Classes declarations

#Main Function starts the brain
def main():
    s = zerorpc.Server(SageBrain('sess1'),heartbeat=None)
    s.bind("tcp://0.0.0.0:9000")
    s.run()


#start process
if __name__ == '__main__':
    main()
#!/usr/bin/python
#Server stuff
import zerorpc
#rest of imports
import sys
import pdb
#Read Pdf imports
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
#To acccess table of contents
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
#json parser
import json
#lda
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
import gensim
#Bag of Words
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#TfIdf
import gensim
#Word2Vect Extra
from gensim.models.doc2vec import TaggedDocument
#Corpus Test
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import wordnet as wn
import nltk as nltk
#GoogleScholar
import scholar
#HIndex
from scholarmetrics import hindex
#otrher imports
import operatorfrom collections import Counter

#Interaface Class to talk with Node
class SageBrain(object):
    """ Sage Brain Class"""
    def __init__(self, session_id):
        self.id = "brainInterface"
        self.sessions = []
        self.session_counter = -1
        self.actualSession = -1
        self.addSession(session_id)

    def DocInterface(self, fileName, doc_id):
        """ Brain Interface """
        doc = Document(self.actualSession, fileName, doc_id, "doc", "user", False)
        self.actualSession.addDoc(doc)
        # Add citations as documents.
        for citation in self.actualSession.documents[0].citationArticles:
            # I need to send the object to send the information
            doc = Document(self.actualSession, citation['Title'], citation["ID"], "doc", "user", citation)
            self.actualSession.addDoc(doc)
        pdb.set_trace()
        msg = doc.create_document_msg()
        #We return the document to Node
        return msg

    def addSession(self, session_id):
        """ Function to add new sessions """
        self.session_counter = self.session_counter + 1
        self.sessions.append(Session(session_id))
        self.actualSession = self.sessions[self.session_counter]


#Internal Classes declarations
#Session
class Session:
    """Session Class"""
    def __init__(self, session_id):
        self.id = session_id
        self.documents = []
        self.topics = []
        self.authorList = []
        self.lda = False
        self.textAnalysis = []

    def addDoc(self, doc):
        """ Function to add a new document to the session """
        self.documents.append(doc)
        self.textAnalysis.append(doc.textAnalysis)
        # self.addDocTopics(doc)
        # self.lda = doc.lda;

    def addDocTopics(self, doc):
        for topic in doc.topics:
            self.topics.append(topic)
             
    def addAuthor(self, author):
        self.authorList.append(author)
    
    def searchAuthor(self, author):
        """Create function to search for authors that returns the author or False"""
        try:
            index = self.authorList.index(author)
            return self.authorList[index]
        except ValueError:
            return "Null"

    def get_lda(self):
        """Calculates a returns an LDA model of the Session"""
        tokenizer = RegexpTokenizer(r'\w+')

        # create English stop words list
        en_stop = get_stop_words('en')

        # Create p_stemmer of class PorterStemmer
        p_stemmer = SnowballStemmer('english')
        # create sample documents
        texts = []
        for eachDocument in textAnalytics:
            texts.append(eachDocument[parameter])

        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        #Store to visualize
        # pickle.dump(corpus,open('corpus.pkl','wb'))
        # dictionary.save('dictionary.gensim')

        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=9, id2word = dictionary, passes=100)
        #Store to visualize
        ldamodel.save('20top_100Tok.gensim')

        return ldamodel

    def get_topics(self, num_topics, num_words):
        doc_topics = []
        topics = self.lda.print_topics(num_topics=num_topics, num_words=num_words)
        for topic in topics:
            topicObject = Topic(self.name)
            topicObject.extract_words(topic)
            doc_topics.append(topicObject)
        return doc_topics

    def calculateTFidf(self,parameter):
        #We store the tokenized text in a vector
        tokText = []
        for each in self.textAnalysis:
            tokText.append(each[parameter])
        #We map each word to a number
        dictionary = gensim.corpora.Dictionary(tokText)
        num_words = len(dictionary)
        #Create a corpus: List of bag of words
        corpus = [dictionary.doc2bow(eachtokText) for eachtokText in tokText]
        tf_idf = gensim.models.TfidfModel(corpus)
        #Create the simiarity measure Object
        sims = gensim.similarities.Similarity('./similarityStorage',tf_idf[corpus],num_features=num_words)
        return {"sims":sims,"dict":dictionary,"tf_idf":tf_idf}

    def word2VectTraining(self,parameter):
        """Training Word Vectors"""
        txt = []
        for each in self.textAnalysis:
            txt.append(each[parameter])
        model = gensim.models.word2vec.Word2Vec(txt,min_count=10,size=300)
        return model

    def word2VectDocumentTraining(self,parameter):
        """Training Document Word Vectors
        Needs to run with raw Text for expected results"""
        tagged_documents = []
        for i, doc in enumerate(self.textAnalysis):
            tagged_documents.append(TaggedDocument(doc[parameter],["doc_{}".format(self.documents[i].title)]))
        d2v_model = gensim.models.doc2vec.Doc2Vec(tagged_documents,size=300)
        return d2v_model

#Topic Class
class Topic:
    """Topic Class"""
    def __init__(self, owner):
        self.id = "id"
        self.owner = owner
        self.words = []
        self.text =[]

    def extract_words(self, topic):
        """ Function to extract the words and probabilities of the topics"""
        for wordPlusProbality in topic[1].split('+'):
            word = {"prob":wordPlusProbality.split('*')[0], "word":wordPlusProbality.split('*')[1] }
            self.words.append(word)
            self.text.append(word['word'])

    def create_topic_msg(self):
        """ Function to create a topic message"""
        topic = {
            "id": self.id,
            "owner": self.owner,
            "words": self.words,
            "text":self.text
        }
        return topic

#Document Class
class Document:
    """ Document Class """
    def __init__(self, session, name, doc_id, doc_type, doc_user, metadata):
        self.session = session
        if metadata != False:
            self.metadata = metadata
            self.parseMetadata()
        else:
            self.metadata = False
            self.title = name.split('.')[0] #Delete the extension
            self.id = doc_id
            self.type = doc_type
            self.user = doc_user
            self.text = unicode(self.get_text(name), "utf-8")
            self.toc = self.get_toc(name)
            if self.toc:
                self.sections = self.get_sections()
            else:
                self.sections = False
            self.GetScholarInfo()
        #H-index
        self.hindex = self.calculateHIndex()
        self.authorHindex = self.calculateAuthorHindex()
        if self.text:
            self.textAnalysis = self.runTextAnalysis()
        else:
            self.textAnalysis = "False"
        # self.topics = self.get_topics(2,4)


    #Implement Function to Loop accross all clusterIDs.

    def GetScholarInfo(self):
        """Get Article information from Google Scholar"""
        def SearchScholar(options):
            """Send Google Scholar Query"""
            querier = scholar.ScholarQuerier()
            settings = scholar.ScholarSettings()
            querier.apply_settings(settings)

            if options['cluster_id']:
                query = scholar.ClusterScholarQuery(cluster=options.cluster_id)
            else:
                query = scholar.SearchScholarQuery()
                if options['author']:
                    query.set_author(options['author'])
                if options['allw']:
                    query.set_words(options['allw'])
                if options['some']:
                    query.set_words_some(options['some'])
                if options['none']:
                    query.set_words_none(options['none'])
                if options['phrase']:
                    query.set_phrase(options['phrase'])
                if options['title_only']:
                    query.set_scope(True)
                if options['pub']:
                    query.set_pub(options['pub'])
                if options['after'] or options['before']:
                    query.set_timeframe(options.after, options.before)
                if options['no_patents']:
                    query.set_include_patents(False)

            query.get_url()
            querier.send_query(query)
            return scholar.get_results_objects(querier)

        """Define Options for the Scholar search"""
        self.SetScholarOptions = {
            'author': False,
            'cluster_id': False,
            'allw': False,
            'some':False,
            'none':False,
            'phrase': self.title,
            'title_only': False,
            'pub': False,
            'after': False,
            'before': False,
            'no_patents': False
        }
        self.scholarInfo = SearchScholar(self.SetScholarOptions)[0]
        self.parseScholarInfo()

    def parseMetadata(self):
        self.title = self.metadata["Title"]
        self.url = self.metadata["URL"]
        self.year = self.metadata["Year"]
        self.citations = self.metadata["Citations"]
        self.versions = self.metadata["Versions"]
        self.clusterID = self.metadata["Cluster ID"]
        self.citations_list = self.metadata["Citations list"]
        self.excerpt = self.metadata["Excerpt"]
        self.author = []
        self.abstract = self.metadata["Abstract"]
        self.type = self.metadata["Type"]
        self.globalID = self.metadata["ID"]
        self.conference = self.metadata["Conference"]
        self.organization = self.metadata["Organization"]
        self.pages = self.metadata["Pages"]
        self.citationArticles = self.metadata["Citation articles"]
        self.text = unicode(self.metadata["Text"], "utf-8")
        ##We create the Authors here too
        ##I might need to do this later
        for eachAuthor in self.metadata["Author"].split("and"):
            if self.session.searchAuthor(eachAuthor) == "Null":
                self.author.append(scholar.get_author_data(eachAuthor))
            else:
                self.author.append(self.session.searchAuthor(eachAuthor))

    def parseScholarInfo(self):
        self.title = self.scholarInfo["Title"]
        self.url = self.scholarInfo["URL"]
        self.year = self.scholarInfo["Year"]
        self.citations = self.scholarInfo["Citations"]
        self.versions = self.scholarInfo["Versions"]
        self.clusterID = self.scholarInfo["Cluster ID"]
        self.citations_list = self.scholarInfo["Citations list"]
        self.excerpt = self.scholarInfo["Excerpt"]
        self.author = []
        self.abstract = self.scholarInfo["Abstract"]
        self.type = self.scholarInfo["Type"]
        self.globalID = self.scholarInfo["ID"]
        self.conference = self.scholarInfo["Conference"]
        self.organization = self.scholarInfo["Organization"]
        self.pages = self.scholarInfo["Pages"]
        self.citationArticles = self.scholarInfo["Citation articles"]
        ##We create the Authors here too
        for eachAuthor in self.scholarInfo["Author"].split("and"):
            if self.session.searchAuthor(eachAuthor) == "Null":
                self.author.append(scholar.get_author_data(eachAuthor))
            else:
                self.author.append(self.session.searchAuthor(eachAuthor))

    def calculateHIndex(self):
        authors_with_hIndex = 0
        paperCitations = []
        for eachCitation in self.citationArticles:
            paperCitations.append(eachCitation["Citations"])
        return hindex(paperCitations)

    def calculateAuthorHindex(self):
        hIndex = 0
        authors_with_hIndex = 0
        for eachAuthor in self.author:
            if eachAuthor['Hindex'] != False:
                hIndex = hIndex + eachAuthor['Hindex']
                authors_with_hIndex = authors_with_hIndex + 1
        return hIndex/authors_with_hIndex

    def get_sections(self):
        sections = [];
        temp_text = self.text
        for section in self.toc:
            if len(temp_text.split(section[1].upper())) > 0:
                sections.append({"section":section[1],"id":section[0],"text":temp_text.split(section[1].upper())[0]})
                if len(temp_text.split(section[1].upper())) > 1:
                    temp_text = temp_text.split(section[1].upper())[1]
            elif len(temp_text.split(section[1].title())) > 0:
                sections.append({"section":section[1],"id":section[0],"text":temp_text.split(section[1].title())[0]})
                if len(temp_text.split(section[1].title())) > 1:
                    temp_text = temp_text.split(section[1].title())[1]
            elif len(temp_text.split(section[1]) > 0):
                sections.append({"section":section[1],"id":section[0],"text":temp_text.split(section[1])[0]})
                if len(temp_text.split(section[1]) > 1):
                    temp_text = temp_text.split(section[1])[1]
            else:
                sections.append("Could not Parse")
        return sections

    def get_text(self,fname, pages=None):
        if not pages:
            pagenums = set()
        else:
            pagenums = set(pages)
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)
        infile = file(fname, 'rb')
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close
        return text

    def get_toc(self, pdf_path):
        infile = open(pdf_path, 'rb')
        parser = PDFParser(infile)
        document = PDFDocument(parser)
        toc = list()
        try:
            for (level, title, dest, a, structelem) in document.get_outlines():
                toc.append((level, title))
            return toc
        except Exception:
            return False
    
    def runTextAnalysis(self):
        # create English stop words list
        en_stop = get_stop_words('en')
        # Create p_stemmer of class PorterStemmer
        p_stemmer = SnowballStemmer('english')
        
        wnl = nltk.WordNetLemmatizer()
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        txtAnalysis = {
            'rawText':self.text,
            'abstract': self.text.split('introduction')[0].split('abstract'),
            'tokens': [i for i in nltk.word_tokenize(self.text) if not i.isdigit()],
            'stopped_tokens':[i for i in nltk.word_tokenize(self.text) if not i.isdigit() if not i in en_stop],
            'stemmed_tokens':0,
            'num_words': 0,
            'words': 0,
            'vocab': 0,
            'lemmatizedVocab': 0,
            'senteces': 0
        }  
        txtAnalysis['stemmed_tokens'] = [p_stemmer.stem(i) for i in txtAnalysis['stopped_tokens']]
        txtAnalysis['num_words'] = len(txtAnalysis['tokens'])
        txtAnalysis['words'] = [w.lower() for w in txtAnalysis['stopped_tokens']]
        txtAnalysis['vocab'] = sorted(set(txtAnalysis['words']))
        txtAnalysis['lemmatizedVocab'] = [wnl.lemmatize(t) for t in txtAnalysis['vocab']]
        txtAnalysis['sentences'] = sent_tokenizer.tokenize(self.text)
        #convert to nltk Text
        text = nltk.Text(txtAnalysis['tokens'])
        #Collocations are very interesting but just prints text to screen need to retrieve this somehow.
        #collocations = text.collocations()
        return txtAnalysis

    def create_document_msg(self):
        doc_topics = []
        doc_sections =[]
        # for section in self.sections:
        #     #pdb.set_trace()
        #     doc_sections.append(json.dumps(section))
        for topic in self.topics:
            doc_topics.append(topic.create_topic_msg())
        msg = {
            'name':self.name,
            'id':self.id,
            'user':self.user,
            'text': self.text,
            'toc': self.toc,
            #Problems parsing in Json
            #'sections':json.loads(doc_sections),
            'topics':doc_topics
        }
        #pdb.set_trace()
        return msg

#Main Function starts the brain
def main():
    # s = zerorpc.Server(SageBrain(),heartbeat=None)
    # s.bind("tcp://0.0.0.0:4242")
    # s.run()
    a = SageBrain("sess1312")
    a.DocInterface("Multimodal Deep Learning.pdf", 0)
    #a.docInterface("test1.pdf","id")


#start process
if __name__ == '__main__':
    main()
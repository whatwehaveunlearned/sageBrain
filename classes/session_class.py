import pandas as pd

import pdb

#Functions from NSA Are called from shell because they run in Python3
import subprocess
import os

#Session
class Session:
    """Session Class"""
    def __init__(self, session_id):
        self.id = session_id
        self.sess_folder = './sessData/' + session_id
        #We initialize fixed now with this session in future will pass sess_folder as a parameter.
        try:
            self.documents = pd.read_csv(self.sess_folder + '/documents.csv',encoding='utf-8',index_col='index')
        except IOError:
            self.documents = pd.DataFrame()
        try:
            self.topics = pd.read_csv(self.sess_folder + '/topics.csv',encoding='utf-8')
        except IOError:
            self.topics = pd.DataFrame()
        try:
            self.authorList = pd.read_csv(self.sess_folder + '/authors.csv',encoding='utf-8',index_col='index')
        except IOError:
            self.authorList = pd.DataFrame()
        try:
            self.words = pd.read_csv(self.sess_folder + '/words.csv',encoding='utf-8')
        except IOError:
            self.words = pd.DataFrame()
        self.documents_df = []
        self.lda = False
        self.textAnalysis = []

    def addDoc(self, doc):
        """ Function to add a new document to the session """
        doc_to_add = doc.create_document_msg()
        document = pd.DataFrame([doc_to_add],columns=doc_to_add.keys(),index=[doc_to_add['globalID']])
        self.documents = self.documents.append(document)
    
    def returnDoc(self,doc):
        return self.documents.loc[doc]
    
    def returnDocsBy(self, type):
        docs_by_array = []
        if type == 'author':
            if isinstance(self.authorList.index,list):
                for each_author in self.authorList.index:
                    element = {
                    'author':each_author,
                    'Paper_Ids': self.authorList.loc[each_author]['Paper_Ids']   
                    }
                    docs_by_array.append(element)
            else: #Only one paper
                element = {
                'author':self.authorList.index,
                'Paper_Ids': self.authorList.loc[self.authorList.index]['Paper_Ids']   
                }
                docs_by_array.append(element)
        return docs_by_array

    
    def docInSess(self,doc):
        try:
            is_doc_in_sess = self.documents['globalID'].isin([doc]).any()
        except KeyError:
            is_doc_in_sess = False
        return is_doc_in_sess

    def addDocTopics(self, doc):
        for topic in doc.topics:
            self.topics.append(topic)
             
    def addAuthor(self, author,doc_id):
        """Function to add author to session authorList"""
        try:
            papers_in_collection = author['Papers_in_collection']
            self.authorList.loc[author.Author, 'Papers_in_collection'] = papers_in_collection + 1
            pdb.set_trace()
            paper_id_array = []
            paper_id_array.append(self.authorList.loc[author.Author, 'Papers_Ids'])
            paper_id_array.append(doc_id)
            self.authorList.loc[author.Author, 'Papers_Ids'] = paper_id_array.append(doc_id)

        except KeyError:
            author['Papers_in_collection'] = 1
            author['Paper_Ids'] = [doc_id]
            self.authorList = self.authorList.append(author)
    
    def searchAuthor(self, author):
        """Function to search for an author in the session returns True if in Session and False if not"""
        # pdb.set_trace()
        try:
            self.authorList.loc[author]
            is_author = True
        except KeyError:
            is_author = False
        return is_author

    def returnAuthor(self, author):
        author_name = author['firstName'] + ' ' +  author['lastName']
        return self.authorList.loc[author_name]
        

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
    
    def get_topics_by(self,data,organized_by):
        """Calculates the topics of the session organized by authors or years"""
        if organized_by == 'author':
            df = pd.DataFrame()
            #Get papers for each author
            # pdb.set_trace()
            for each_author in data:
                for each_paper in each_author['Paper_Ids']:
                    # pdb.set_trace()
                    df = df.append(self.returnDoc(each_paper))
        return self.get_topics(df)


    def get_topics(self,doc_dictionary):
        """Returns Topics object and Words Object from documents df passed"""
        doc_dictionary.to_csv(self.sess_folder + '/temp_sess_topics.csv',header=True,encoding='utf-8',index_label='index')
        #Get Topics NSA function
        wd = os.getcwd()
        os.chdir("../brain/.")
        activate_this_file = "./vizlit/bin/activate_this.py"
        execfile(activate_this_file, dict(__file__=activate_this_file))
        python_bin="./vizlit/bin/python"
        script_file="topic_extractor.py"
        subprocess.call([python_bin,script_file,'session','../sageBrain/sessData/sess1/documents.csv'])
        os.chdir(wd)
        self.topics = pd.read_csv(self.sess_folder + '/topics.csv',encoding='utf-8').dropna()
        self.words = pd.read_csv(self.sess_folder + '/words.csv',encoding='utf-8')
        return {'topics':self.topics,'words':self.words}
    
    def get_years(self):
        return self.documents.groupby('year')['year'].count()

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
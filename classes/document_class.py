import pandas as pd
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
#Date Parsing
from dateutil.parser import parse as dateParser
#GoogleScholar
import scholar

#Functions from NSA Are called from shell because they run in Python3
import subprocess
import os

#Document Class
class Document:
    """ Document Class """
    def __init__(self, session, name, doc_id, doc_type, doc_user, metadata):
        self.session = session
        self.topics = False
        self.folder = self.session.sess_folder
        self.metadata = metadata
        print metadata['pdf_file']
        if doc_type == 'zotero':
            if metadata['pdf_file'] == None:
                print 'No File'
                self.text = 'No File'
            try:
                self.text = unicode(self.get_text(metadata['pdf_file'] + '.pdf',self.folder), "utf-8")
            except TypeError:
                print 'Parsing Error'
                self.text = 'Parsing Error'
            self.parseMetadata()
            if self.text != 'Parsing Error':
                self.topics = self.get_topics()
        elif doc_type == 'inSession':
            # pdb.set_trace()
            self.parseFromSession()
        else:
            self.metadata = False
            self.title = name.split('.')[0] #Delete the extension
            self.id = doc_id
            self.type = doc_type
            self.user = doc_user
            self.text = unicode(self.get_text(name,self.folder), "utf-8")
            self.toc = self.get_toc(self.folder + name)
            if self.toc:
                self.sections = self.get_sections()
            else:
                self.sections = False

    #Implement Function to Loop accross all clusterIDs.

    def get_topics(self):
        """Function to calculate the topics from a document, returns the topics in order of importance and the words"""
        doc_dictionary = self.create_document_msg()
        #Create a csv file with the text to read by topic_extractor
        pd.DataFrame.from_dict(doc_dictionary).to_csv(self.session.sess_folder + '/temp_text_for_individual_doc_topics.csv',header=True,encoding='utf-8',index_label='index')
        
        #Get Topics NSA function
        wd = os.getcwd()
        os.chdir("../brain/.")
        activate_this_file = "./vizlit/bin/activate_this.py"
        execfile(activate_this_file, dict(__file__=activate_this_file))
        python_bin="./vizlit/bin/python"
        script_file="topic_extractor.py"
        subprocess.call([python_bin,script_file,'document','../sageBrain/sessData/sess1/temp_text_for_individual_doc_topics.csv'])
        os.chdir(wd)
        self.topics = pd.read_csv(self.session.sess_folder + '/temp_topics_for_individual_doc.csv',encoding='utf-8').dropna()
        self.words = pd.read_csv(self.session.sess_folder + '/temp_words_for_individual_doc.csv',encoding='utf-8')
        return {'topics':self.topics,'words':self.words}

    #Parse from Session
    def parseFromSession(self):
        # pdb.set_trace()
        self.title = self.metadata.title
        self.pdf_file = self.metadata.pdf_file
        self.parent_item = self.metadata.parent_item
        self.tags = self.metadata.tags
        self.url = self.metadata.url
        self.year = self.metadata.year
        self.citations = self.metadata.citations
        self.versions = self.metadata.versions
        self.clusterID = self.metadata.clusterID
        self.citations_list = self.metadata.citationsList
        self.notes = self.metadata.notes
        self.abstract = self.metadata.abstract
        self.type = self.metadata.type
        self.globalID = self.metadata.globalID
        self.scholarID = False
        # self.conference = self.metadata.conference
        self.organization = self.metadata.organization
        self.pages = self.metadata.pages
        self.text = self.metadata.text
        self.citationArticles = self.metadata.citationArticles
        parsed_authors = pd.read_json(self.metadata.author)
        self.authors = parsed_authors
        pdb.set_trace()
        self.topics =  pd.read_json(self.metadata.topics)
        

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
        try:
            self.scholarInfo = SearchScholar(self.SetScholarOptions)[0]
            self.parseScholarInfo()
        except IndexError:
            print "No Information Found in Google Scholar too much Queries or No Info"

    def parseMetadata(self): #I have 2 parent items and 2 notes, 2 tags I have to put them together in the near future only reading x now
        self.title = self.metadata["name"]
        self.pdf_file = self.metadata['pdf_file']
        self.parent_item = self.metadata['parentItem_x']
        self.tags = self.metadata['tags_x']
        self.url = self.metadata["url"]
        self.year = dateParser(self.metadata["date"]).year
        self.citations = False
        self.versions = False
        self.clusterID = False
        self.citations_list = False
        self.notes = self.metadata["note_x"]
        zotero_authors = self.metadata["creators"]
        self.abstract = False
        self.type = self.metadata["itemType"]
        self.globalID = self.metadata['key']
        self.scholarID = False
        # self.conference = self.metadata["publicationTitle"]
        self.organization = False
        self.pages = False
        self.citationArticles = False
        self.authors = pd.DataFrame()
        # ##We create the Authors here too
        for eachAuthor in zotero_authors:
            author = eachAuthor['firstName'] + " " +  eachAuthor['lastName']
            # pdb.set_trace()
            if self.session.searchAuthor(author) == False:
                # self.author.append(scholar.get_author_data(author))
                author_data = scholar.fast_get_author_data(author)
                # Need to change arrays to strings to save as PD
                author_data['Interests'] = str(author_data['Interests'])
                author_data['Paper_Ids'] = [self.globalID] 
                author_data_df = pd.DataFrame.from_records(author_data,index=[author_data['Author']])
                self.authors = self.authors.append(author_data_df)
                self.session.addAuthor(author_data_df,self.globalID)
            else:
                print "Author was in Session"
                #Get the author from session
                author_frame = self.session.returnAuthor(eachAuthor)
                #Append to the document
                self.authors.append(author_frame)
                #Append to the session
                self.session.addAuthor(author_frame,self.globalID)

    def authors_to_json(self):
        author_array = []
        for each in self.authors:
            pdb.set_trace()
            author_array.append(each.to_json())
        return author_array

    def parseScholarInfo(self):
        # self.title = self.scholarInfo["Title"]
        # self.pdf_file = self.metadata['pdf_file']
        # self.url = self.scholarInfo["URL"]
        # self.year = self.scholarInfo["Year"]
        self.citations = self.scholarInfo["Citations"]
        self.versions = self.scholarInfo["Versions"]
        self.clusterID = self.scholarInfo["Cluster ID"]
        self.citations_list = self.scholarInfo["Citations list"]
        # self.excerpt = self.scholarInfo["Excerpt"]
        # self.author = []
        self.abstract = self.scholarInfo["Abstract"]
        # self.type = self.scholarInfo["Type"]
        self.scholarID = self.scholarInfo["ID"]
        self.conference = self.scholarInfo["Conference"]
        # self.organization = self.scholarInfo["Organization"]
        self.pages = self.scholarInfo["Pages"]
        self.citationArticles = self.scholarInfo["Citation articles"]
        # ##We create the Authors here too
        # for eachAuthor in self.scholarInfo["Author"].split("and"):
        #     if self.session.searchAuthor(eachAuthor) == "Null":
        #         self.author.append(scholar.get_author_data(eachAuthor))
        #     else:
        #         self.author.append(self.session.searchAuthor(eachAuthor))

    def calculateHIndex(self):
        authors_with_hIndex = 0
        paperCitations = []
        for eachCitation in self.citationArticles:
            paperCitations.append(eachCitation["Citations"])
        return hindex(paperCitations)

    def calculateAuthorHindex(self):
        hIndex = 0
        authors_with_hIndex = 0
        for eachAuthor in self.authors:
            if eachAuthor['Hindex'] != False:
                hIndex = hIndex + eachAuthor['Hindex']
                authors_with_hIndex = authors_with_hIndex + 1
        return hIndex/authors_with_hIndex

    def get_sections(self):
        sections = []
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

    def get_text(self,fname,folder, pages=None):
        if not pages:
            pagenums = set()
        else:
            pagenums = set(pages)
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)
        infile = file(folder + '/documents/' + fname, 'rb')
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
        #transform author dataframe into json to searialize
        if self.topics != False:
            pdb.set_trace()
            self.topics = self.topics['topics'].to_json()
        msg = {
            'user':'self.user',
            'text': self.text,
            'title' : self.title,
            'pdf_file': self.pdf_file,
            'parent_item': self.parent_item,
            'tags': self.tags,
            'url': self.url,
            'year': self.year,
            'citations': self.citations,
            'versions': self.versions,
            'clusterID': self.clusterID,
            'citationsList':self.citations_list,
            'notes': self.notes,
            'author': self.authors.to_json(),
            'abstract':self.abstract,
            'type':self.type,
            'globalID':self.globalID,
            # 'conference':self.conference,
            'organization': self.organization,
            'pages':self.pages,
            'citationArticles': self.citationArticles,
            'topics':self.topics
        }
        return msg
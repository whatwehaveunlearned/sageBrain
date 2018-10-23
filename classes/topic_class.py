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
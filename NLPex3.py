
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import numpy as np
from sklearn.preprocessing import normalize
import random


class DialogueManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,1))

    def train(self, data):
        # build the dictionary of word with the dataset
        self.vectorizer.fit(data)            

    def predict(self, context, utterances):
        # Convert context and distractor into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)

    
    def evaluate_recall1(self,y,distractor,correct, k=1):
        """compute the number of correct predictions
           distractor is vector of size 20 including the correct answer and 19 wrong
        """
        num_examples = len(y)
        num_correct = 0
        for i in range(num_examples):
            #check if the prediction is the same as the correct
            if distractor[i][np.argmax(y[i])]==correct[i]:                
                num_correct+=1
        return num_correct/num_examples
    
    def load(self,path):
        with open(path,'rb') as f:
            self.vectorizer = pkl.load(f)


    def save(self,path):
        with open(path,'wb') as fout:
            pkl.dump(self.vectorizer,fout)




def loadData(path):
    """
        :param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)
        :return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue
            is a sequence of (utterance, answer, options)
    """
    with open(path) as f:
        descYou, descPartner = [], []
        dialogue = []
        for l in f:
            l=l.strip()
            lxx = l.split()
            idx = int(lxx[0])
            if idx == 1:
                if len(dialogue) != 0:
                    yield descYou,  descPartner, dialogue
                # reinit data structures
                descYou, descPartner = [], []
                dialogue = []

            if lxx[2] == 'persona:':
                # description of people involved
                if lxx[1] == 'your':
                    description = descYou
                elif lxx[1] == "partner's":
                    description = descPartner
                else:
                    assert 'Error, cannot recognize that persona ({}): {}'.format(lxx[1],l)
                description.append(lxx[3:])

            else:
                # the dialogue
                lxx = l.split('\t')
                utterance = ' '.join(lxx[0].split()[1:])
                answer = lxx[1]
                options = [o for o in lxx[-1].split('|')]
                dialogue.append( (idx, utterance, answer, options))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)
    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--gen', help='enters generative mode')

    opts = parser.parse_args()
    PATH = opts.text
    dm = DialogueManager()
    
    if opts.train:
        # initialise data structures
        distractor=[]
        context=[]
        correct=[]
        for you,other, dialogue in loadData(PATH):
            pers=[] 
            sentence=""
            for sent in you:
                pers.extend(sent)
            sentence1=' '.join(pers)
            pers=[]
            for sent in other:
                pers.extend(sent)
            sentence2=' '.join(pers)
            sentence=sentence1+sentence2
            
            for idx, utterance, answer,options in dialogue:
                distractor.extend(options)
                #combine personality and utterance
                context.append(str(sentence+utterance))
                correct.append(str(answer))
        # combine all sentences for training, in order to get all the owrds in a ditionary
        data=context+distractor
        dm.train(data)
        dm.save(opts.model)
    else:
        assert opts.test,opts.test
        dm.load(opts.model)
        distractor=[]
        context=[]
        correct=[]
        utterances=[]
        for you,other, dialogue in loadData(PATH):
            pers=[] 
            sentence=""
            for sent in you:
                pers.extend(sent)
            sentence1=' '.join(pers)
            pers=[]
            for sent in other:
                pers.extend(sent)
            sentence2=' '.join(pers)
            sentence=sentence1+sentence2
            
            for idx, utterance, answer,options in dialogue:
                distractor.extend(options)
                utterances.append(str(utterance))
                context.append(str(sentence+utterance))
                correct.append(str(answer))
                
        distractor=np.array(distractor).reshape((len(context),20))
        y = [dm.predict(context[x], distractor[x]) for x in range(len(context))]

        print("recall 1:",dm.evaluate_recall1(y,distractor,correct)) # print recall 1
        print("============show examples=========")
        # print 10 random examples, context, prediction and correct answer
        for i in [random.randint(0,len(context)-1) for x in range(10)]:
            print("person :",utterances[i])
            print("response :",distractor[i][np.argmax(y[i])])
            print("correct:",distractor[i][-1])
            print("______")
        

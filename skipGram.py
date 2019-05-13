from __future__ import division
import argparse
import pandas as pd

import numpy as np
from numpy import random
from scipy.special import expit
from sklearn.preprocessing import normalize
import string
from six import iteritems
import pickle




###### tokenization###########################

def text2sentences(path):
    sentences = []
    punctuation = set(string.punctuation)
    with open(path, encoding = "utf-8") as f:
        for l in f:
            sentences.append( l.lower().split() ) #Tokenization
        for i in range(len(sentences)):
            #We remove the punctuation:
            sentences[i] = [w for w in sentences[i] if w not in punctuation] 
            #We remove the numeric values:
            sentences[i] = [w for w in sentences[i] if w.isalpha()] 
    return sentences


#sentences=text2sentences(path)

###### load test set #####################################

def loadPairs(path):
    
    data = pd.read_csv(path,delimiter='\t',header = None) 
    data.columns = ['word1','word2','similarity']
    pairs = zip(data['word1'],data['word2'],data['similarity'])

    return pairs

#pairs2 = loadPairs(path2)

class SkipGram:
    
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 2, minCount = 5):
        self.sentences = sentences
              
        #The number of component in each word vector
        self.nEmbed = nEmbed 
        
        #The number of negative samples
        self.negativeRate = negativeRate 
        
        #The minimum number of occurence in the initial dataset that a word 
        #must have to be used in the model
        self.minCount = minCount 
        
        #The learning rate
        self.lr = 0.025
        
        ##### Define the vocabulary and count the number of occurence of 
        # each word in the document #####
        self.vocabulary = []
        self.count={}
        for sentence in self.sentences:
            for word in sentence:
                if word not in  self.vocabulary:
                    self.vocabulary.append(word)
                    self.count[word]=1
                else:
                    self.count[word]+=1
        
        self.dictionary = {w: index for (index,w) in enumerate(self.vocabulary)}
        
        self.dictionary_new = {} #New dictionary with words that appear more than 5 times
        self.count_new = {}
        self.indices = [] #List with the word indices
        
        #Remove all the words that appear less than 5 times in the dataset
        i=0
        for word,occ in self.count.items():
            if int(occ) >= self.minCount:
                self.dictionary_new[word]=i
                self.indices.append(i)
                self.count_new[word]=occ
                i+=1
        
        
        ##### Computation of the unigram distribution #####
        
        #We raise to the power 3/4 as it has been shown empiricaly from 
        #T.Mikolov’s team to have better results
        power=0.75
        
        #Number of words in our vocabulary
        self.n_voc = len(self.count_new)       
        
        #Initialization of unigram
        self.unigram=np.zeros(self.n_voc) 
        
        #We define the denominator 
        total_power=sum([self.count_new[w]**power for w in self.count_new.keys()])

        #We apply the formula
        for word,occ in self.count_new.items():
            self.unigram[self.dictionary_new[word]]=self.count_new[word]**power/total_power
        

        
        ##### We define the pair of words #####
        
        # The size of the window is equal to winSize + winSize + 1 = 5
        self.window_size = winSize
        
        #Initialization of list of pairs of words
        self.idx_pairs = []

        # Cycle through for each word in the position of center word
        for center_word_pos in range(len(self.indices)):
            
        # Cycle through for each word in the window position
            for w in range(-self.window_size, self.window_size + 1):
                context_word_pos = center_word_pos + w
                
            # Make sure not to jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(self.indices) or center_word_pos == context_word_pos:
                    continue
                
                context_word_idx = self.indices[context_word_pos]
                self.idx_pairs.append((self.indices[center_word_pos], context_word_idx))

        #Transform list into array
        self.idx_pairs = np.array(self.idx_pairs)
        
        
        
        ##### We initialize the embedding and context matrices with random 
        # numbers selected from a normal distribution with mean 0 and standard 
        # deviation 1 #####
        self.weight_matrix = np.random.normal(0, 1, (self.nEmbed, self.n_voc))
        self.weight_matrix2 = np.random.normal(0, 1, size = (self.n_voc, self.nEmbed))
    
    
    
    ##### One Hot encoding #####
    def input_layer(self, word_idx):
        x = np.zeros(self.n_voc)
        x[word_idx] = 1.0
        return x     
    
    
    ##### Sigmoid function #####
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    
    ##### Forward pass #####
    def architecture(self, idx):
        one_hot = self.input_layer(idx) 
        self.hidden_layer = np.matmul(self.weight_matrix, one_hot)
        self.output_layer = np.matmul(self.weight_matrix2,self.hidden_layer)       
        return self.hidden_layer, self.output_layer
        
    
    
    def train(self, epochs = 10):
        #Cycle through each epoch
        for e in range(epochs):
            
            #Cycle through in pair of words
            for center, context in self.idx_pairs: 
                
                delta1 = np.zeros((self.nEmbed,))
                
                ##### Forward pass #####
                (hidden_layer, output_layer) = self.architecture(center)
                
                #Initialize the list of samples (context word + negative samples)
                samples = []
                
                #Append the context word with assigned value 1
                samples.append((context, 1))
                
                ##### Negative sampling #####
                #Selecting 5 negative samples randomly using the unigram distribution
                #We notice here that there is a slight probability that a 
                #negative sample will be the same as the context word. 
                Nidx = (np.random.choice(len(self.unigram), 5, p = self.unigram))
                
                for idx in Nidx:
                    #Append the negative samples with assigned value 0
                    samples.append((idx,0))          
                

                #Cycle trough each sample word
                for i, label in samples:
                    # Compute the weighted sum value at the ouput layer
                    x = np.dot(self.input_layer(i).T, output_layer)
                    
                    #Activation score at the output layer
                    x = self.sigmoid(x) 
                    
                    ##### Backpropagation #####
                    delta = (label - x) * self.lr
                    delta1 += delta * self.weight_matrix2[i,:]
                    
                    #We update the weight of the ouput matrix
                    self.weight_matrix2[i,:] += delta * hidden_layer
                
                #We update the weight of the input matrix
                self.weight_matrix[:,center] += delta1
                    



    def save(self, path):
        #Saving the embedding matrix and the words in descending order
        Embeddings = {}
        
        for word,index in self.dictionary_new.items():
            Embeddings[word] = self.weight_matrix[:,index]
        
        with open(path, 'wb') as file:
            pickle.dump(Embeddings, file)  
    
    def load(self,path):
        with open(path, 'rb') as file:
            embeddings=pickle.load(file)

            self.weight_matrix =list(embeddings.values())
            self.dictionary_new=list(embeddings.keys())



    def similarity(self,word1,word2):

        
        if(word1 in self.dictionary_new) and (word2 in self.dictionary_new):
            index1 = self.dictionary_new.index(word1)
            index2 = self.dictionary_new.index(word2)
            #print(index1,index2)  
        ##### We compute the cosine similarity #####
            #print(len(self.weight_matrix[index1]))
            norm1=np.array(self.weight_matrix[index1])/(np.array(self.weight_matrix[index1])**2).sum()**0.5
            norm2=np.array(self.weight_matrix[index2])/(np.array(self.weight_matrix[index2])**2).sum()**0.5
            dot_product = np.dot(norm1.T,norm2)
            #dot_product = np.dot(self.weight_matrix[index1]).T,self.weight_matrix[index2])
            return (dot_product + 1)/2
        
        else:
            return 'NA'        


                         


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help= 'path containing training data', required=True)
    parser.add_argument('--model', help= 'path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()
    sentences = text2sentences(opts.text)
    sg = SkipGram(sentences)
    if not opts.test:

        #print(sg.dictionary)
        sg.train(10)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg.load(opts.model)
        #print(sg.weight_matrix[0])
        #print(sg.dictionary_new)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))


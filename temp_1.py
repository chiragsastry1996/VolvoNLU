import nltk
from nltk.stem.lancaster import LancasterStemmer
import string
import tensorflow as tf
import tflearn
import json
import numpy as np
import random
import pickle
import ssl

graph_def = tf.GraphDef()

# These are set to the default names from exported models, update as needed.
filename = "frozen_model.pb"

# Import the TF graph
with tf.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

stemmer = LancasterStemmer()

words = ['a', 'afternoon', 'ar', 'assist', 'bef', 'brok', 'bunk', 'can', 'check', 'day', 'diesel', 'dist', 'drink', 'eat', 'empty', 'ev', 'fam', 'far', 'farth', 'fil', 'food', 'for', 'fuel', 'gas', 'go', 'good', 'hello', 'help', 'hey', 'hi', 'how', 'hungry', 'ind', 'is', 'left', 'light', 'long', 'lot', 'me', 'morn', 'much', 'near', 'on', 'petrol', 'plac', 'pump', 'refil', 'remain', 'resta', 'right', 'sid', 'so', 'starv', 'stat', 'tank', 'thank', 'the', 'thirsty', 'to', 'ton', 'travel', 'turn', 'we', 'what', 'you']
test = []

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    
    test = np.array(bag)
    test = np.reshape(test,[1,65])
    return(test)

output_layer = 'FullyConnected_2/Softmax:0'
input_node = 'InputData/X:0'

with tf.Session() as sess:
    try:
        
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions, = sess.run(prob_tensor, {input_node: bow("Suggest a good place to grab a bite", words) })
        print(predictions)
        highest_probability_index = np.argmax(predictions)
        print(highest_probability_index)
    except KeyError:
        print ("Couldn't find classification output layer: " + output_layer + ".")
        print ("Verify this a model exported from an Object Detection project.")
        exit(-1)
### NLP - GTT Talking Truck

# imports
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



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

# create stemmer object
stemmer = LancasterStemmer()

# import the chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
# ignore_words = ['?']
ignore_words = [ch for ch in string.punctuation]

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']: # pattern is each word in the utterances
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))

print("\n", len(documents), "documents.")
print("\n", len(classes), "classes:\n", classes)
print("\n", len(words), "unique stemmed words:\n", words)

# train
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - make lower() and then stem words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    # append(1) if word is in pattern_words (i.e; the stemmed version of the words)
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

### ========================================================================
### TensorFlow model creation

# Reset underlying graph data
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
model.save('model.tflearn')


## ========================================================================
#Using the saved model
model_file = "saved_model.pb"

def load_graph(pbmodelFile):
    with tf.gfile.GFile(pbmodelFile, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'

    return graph, input_name, output_name

graph, inputName, outputName = load_graph(model_file)
input_x = graph.get_tensor_by_name(inputName)
output_y = graph.get_tensor_by_name(outputName)
print(input_x)
print(output_y)

pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

# restore all of our data structures
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# Load model (.tflearn)
model.load('./model.tflearn')

### Functions
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

    print(bag)

    return(np.array(bag))

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return print(random.choice(i['responses']))

            results.pop(0)

### ========================================================================
### TESTING
### Classifier class (Extra)
class classify_me():
    def classify(sentence):
        # generate probabilities from the model
        results = model.predict([bow(sentence, words)])[0]
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list, return_list[0][0]

class get_intention():
    def get_intent(string):
        return classify_me.classify(string)[1]


print(classify_me.classify("Left indicator on"))
# print(classify_me.classify("distance to go to destination?"))
# print(classify_me.classify("Famished"))
# print(classify_me.classify("Thank you so much!!"))
# print(classify_me.classify("Good day to you"))
# print(classify_me.classify("where's McDonalds?"))
# print(classify_me.classify("when will the truck stop?"))

# print(get_intention.get_intent("Almost outta fuel"))
# print(get_intention.get_intent("distance to go to destination?"))
# print(get_intention.get_intent("Famished"))
# print(get_intention.get_intent("Thank you so much!!"))
# print(get_intention.get_intent("Good day to you"))
# print(get_intention.get_intent("where's McDonalds?"))
# print(get_intention.get_intent("when will the truck stop?"))

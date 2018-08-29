#!/bin/python

import os
import tqdm
import codecs
import pickle
from optparse import OptionParser

import keras
import numpy as np
import pandas

# NLP
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import re


########## Global properties ##########
stemmer = SnowballStemmer("english")

# Add the list of negatation words
negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
		  "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
		  "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
		  "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
		  "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere", 
		  "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
		  "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",  
		  "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

zhpunc = "''``’_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-"

# Remove the negation words from stopwords list
stopWordsList = set(stopwords.words('english')) - set(stopwords.words('english')).intersection(set(negate))


########## Function definitions ##########
def loadWordEmbeddings(options, vocabulary):
	# Load embeddings from file
	print('Loading word embeddings from file: %s' % options.pretrainedEmbeddingsFile)
	embeddingsIndex = {}
	f = codecs.open(options.pretrainedEmbeddingsFile, encoding='utf-8')
	for line in tqdm.tqdm(f):
		values = line.rstrip().rsplit(' ')
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddingsIndex[word] = coefs
	f.close()
	print('Found %s word vectors' % len(embeddingsIndex))

	# Generate embedding matrix
	print('Preparing embedding matrix')
	wordsNotFound = []
	numWords = len(vocabulary.keys()) + 1
	embeddingMatrix = np.zeros((numWords, options.embeddingSize))
	for word, i in vocabulary.items():
	    if i >= numWords:
	        continue
	    embeddingVector = embeddingsIndex.get(word)
	    if (embeddingVector is not None) and len(embeddingVector) > 0:
	        # words not found in embedding index will be all-zeros.
	        embeddingMatrix[i] = embeddingVector
	    else:
	        wordsNotFound.append(word)
	print('Number of null word embeddings: %d' % np.sum(np.sum(embeddingMatrix, axis=1) == 0))

	return embeddingMatrix


def trainModel(options):
	if (not os.path.exists(options.dataFileName)) or options.createNewDataFile:
		# Load data into the system
		df = pandas.read_csv(options.trainingDataFileName)
		df['0'] = df['category'].values
		
		print(df.shape)
		#print(df)
		df['2'] = df['0'].values # Add extra column



		dropIndices = []
		vocabulary = {}
		maxVocabIndex = 1
		
		for idx, i in enumerate(df.index):
		#for i in range(0,3):
			text = df.at[i, '0']
			if text == "":
				print ("Error: Empty text at line %d" % i)

			# Update the words in the list
			try:
				tokenizedList = sent_tokenize(re.sub(r'[\d+]','', text))[0]
			except:
				dropIndices.append(i)
				continue

			wordList = word_tokenize(tokenizedList)
			filteredTokenList = [word.lower() for word in wordList if word.lower() not in stopWordsList \
										and word not in string.punctuation + zhpunc]
			#filteredTokenList = wordList
			#print(filteredTokenList)
			if options.trainEmbeddinglayer:
				# Reduce the words to the stem
				filteredTokenList = [stemmer.stem(word) for word in filteredTokenList]
			#print(filteredTokenList)

			# Add the words to the vocabulary as well
			for word in filteredTokenList:
				if word not in vocabulary:
					vocabulary[word] = maxVocabIndex
					maxVocabIndex += 1

			idxFilteredTokenList = [vocabulary[word] for word in filteredTokenList]

			# Add the updated word to the df
			if len(filteredTokenList) < options.maxSequenceLength:
				df.at[i, '0'] = filteredTokenList
				df.at[i, '2'] = idxFilteredTokenList
			else:
				df.at[i, '0'] = filteredTokenList[:options.maxSequenceLength]
				df.at[i, '2'] = idxFilteredTokenList[:options.maxSequenceLength]

		print ("Dropping unresolved indices")
		print ("Size before dropping: %d" % df.shape[0])
		df = df.drop(df.index[dropIndices])
		print ("Size after dropping: %d" % df.shape[0])
		print ("Maximum vocab index found to be: %d" % maxVocabIndex)

		#Load FastText embeddings
		print ("Loading FastText embedding matrix")
		embeddingMatrix = loadWordEmbeddings(options, vocabulary)

		# Save data
		print ("Saving data")
		with open(options.dataFileName, "wb") as f:
			pickle.dump([df, vocabulary, maxVocabIndex, embeddingMatrix], f)
		print ("Saving completed!")

	# else:
	# 	# Load data
	# 	print ("Loading data")
	# 	with open(options.dataFileName, "rb") as f:
	# 		df, vocabulary, maxVocabIndex, embeddingMatrix = pickle.load(f)
	# 	print ("Loading completed!")
	# 	print ("Maximum vocab index found to be: %d" % maxVocabIndex)

	# assert (len(vocabulary.keys()) == maxVocabIndex)
	# print(embeddingMatrix)
	# print (df)
	# print ("Initiating training!")

	# # Create the model
	# model = keras.models.Sequential()

	# if options.trainEmbeddinglayer:
	# 	model.add(keras.layers.Embedding(maxVocabIndex, options.embeddingSize, input_length=options.maxSequenceLength))
	# else:
	# 	print ("Creating embedding layer with FastText weights")
	# 	model.add(keras.layers.Embedding(maxVocabIndex, options.embeddingSize, weights=[embeddingMatrix], input_length=71, trainable=False))

	# if options.useLSTM:
	# 	# LSTM
	# 	model.add(keras.layers.LSTM(128))
	# 	model.add(keras.layers.Flatten())
	# 	model.add(keras.layers.Dense(units=1, activation='sigmoid'))

	# else:
	# 	# Convolutional Network
	# 	# Average the embeddings of all words in the document (fixed size vector)
	# 	model.add(keras.layers.GlobalMaxPooling1D())
	# 	model.add(keras.layers.Reshape((options.embeddingSize, 1)))

	# 	model.add(keras.layers.Conv1D(filters=64, kernel_size=3))
	# 	model.add(keras.layers.LeakyReLU(alpha=0.3))

	# 	model.add(keras.layers.Conv1D(filters=32, kernel_size=3))
	# 	model.add(keras.layers.LeakyReLU(alpha=0.3))

	# 	model.add(keras.layers.Flatten())
	# 	model.add(keras.layers.Dense(units=1, activation='sigmoid'))

	# # Train the model
	# X = keras.preprocessing.sequence.pad_sequences(df[2].values, padding='post')
	# # X = np.expand_dims(X, axis=-1)
	# y = df[0].values

	# y = np.reshape(y,(y.shape[0],1))
	# #print(df[0].values)
	# t = np.asarray(y[1])
	# print(t)
	# print(t.shape)
	# x = np.array(y[1]) > 0.0 # Convert to binary class labels
	# print ("Data shape | X: %s | y: %s" % (str(X.shape), str(y.shape)))
	# print (X[1], y[1])

	# sgd = keras.optimizers.SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True, clipnorm=4)
	# model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])
	# model.fit(X, y, epochs=options.numEpochs, batch_size=options.batchSize)

	# # Evaluate model performance
	# print ("Evaluating performance on the training data")
	# performance = model.evaluate(X, y)
	# print (performance)

	# # Save the final model
	# print ("Saving final model!")
	# model.save(options.modelName)


def testModel(options):
	# Load the models
	print ("Loading pretrained model")
	with open(options.dataFileName, "rb") as f:
		df, vocabulary, maxVocabIndex, embeddingMatrix = pickle.load(f)
	model = keras.models.load_model(options.modelName)
	print ("Model loaded successfully!")

	print (model.summary())

	if not os.path.exists(options.testDataFileName):
		print ("Error: Test file does not exist (%s)" % options.testDataFileName)
		exit (-1)

	# Perform inference on the test examples
	print ("Starting inference")
	with open(options.testDataFileName, "r") as f, open(options.testOutputFileName, "w") as of:
		for line in f:
			line = line.strip()

			# Update the words in the list
			try:
				tokenizedList = sent_tokenize(re.sub(r'[\d+]','', line))[0]
			except:
				print ("Error: Unable to tokenize")
				continue

			wordList = word_tokenize(tokenizedList)
			filteredTokenList = [word.lower() for word in wordList if word.lower() not in stopWordsList \
										and word not in string.punctuation + zhpunc]
			
			if options.trainEmbeddinglayer:
				# Reduce the words to the stem
				filteredTokenList = [stemmer.stem(word) for word in filteredTokenList]

			idxFilteredTokenList = []
			for word in filteredTokenList:
				if word in vocabulary:
					idxFilteredTokenList.append(vocabulary[word])
			# idxFilteredTokenList = [vocabulary[word] for word in filteredTokenList if word in vocabulary else -1]
			idxFilteredTokenList = np.expand_dims(np.array(idxFilteredTokenList)[:options.maxSequenceLength], 0)
			idxFilteredTokenList = keras.preprocessing.sequence.pad_sequences(idxFilteredTokenList, padding='post', maxlen=options.maxSequenceLength)

			prediction = model.predict(idxFilteredTokenList)
			print ("Sentence: %s | Prediction: %s" % (line, 'Positive' if prediction[0, 0] > 0.5 else 'Negative'))
			of.write ("%s\t%f\n" % (line, prediction[0, 0]))
	print ("Inference complete. Predictions written to file!")


if __name__ == "__main__":
	# Command line options
	parser = OptionParser()

	# Base options
	parser.add_option("--trainModel", action="store_true", dest="trainModel", default=True, help="Whether to train the model")
	parser.add_option("--testModel", action="store_true", dest="testModel", default=False, help="Whether to test the model")

	parser.add_option("--createNewDataFile", action="store_true", dest="createNewDataFile", default=True, help="Whether to create new data file")
	parser.add_option("--dataFileName", action="store", type="string", dest="dataFileName", default="category_data1.pkl", help="Data file name")
	parser.add_option("--trainingDataFileName", action="store", type="string", dest="trainingDataFileName", default="../preprocessing/50k_index_sorted.csv", help="Name of the file containing the training samples")
	parser.add_option("--testDataFileName", action="store", type="string", dest="testDataFileName", default="./assets/test.csv", help="Name of the file containing the test samples")
	parser.add_option("--testOutputFileName", action="store", type="string", dest="testOutputFileName", default="./assets/out.csv", help="Name of the file containing the predictions for the test samples")
	parser.add_option("--useLSTM", action="store_true", dest="useLSTM", default=False, help="Use LSTM network instead of the ConvNet")
	parser.add_option("--maxSequenceLength", action="store", type="int", dest="maxSequenceLength", default=10, help="Maximum sentence length to be used")
	parser.add_option("--modelName", action="store", type="string", dest="modelName", default="./sentiment-analyzer.h5", help="Name of the model to be saved")

	parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=32, help="Batch size to be used")
	parser.add_option("--numEpochs", action="store", type="int", dest="numEpochs", default=10, help="Number of training epochs")

	# Embedding options
	parser.add_option("--trainEmbeddinglayer", action="store_true", dest="trainEmbeddinglayer", default=False, help="Train embedding layer")
	parser.add_option("--pretrainedEmbeddingsFile", action="store", type="string", dest="pretrainedEmbeddingsFile", default="../fasttext/wiki.en.vec", help="File containing the pretrained embeddings")
	parser.add_option("--embeddingSize", action="store", type="int", dest="embeddingSize", default=300, help="Size of the embedding vector to be used")

	# Parse command line options
	(options, args) = parser.parse_args()
	print ("Options:", options)

	if options.trainModel:
		trainModel(options)

	if options.testModel:
		testModel(options)
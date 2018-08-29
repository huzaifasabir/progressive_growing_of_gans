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

	
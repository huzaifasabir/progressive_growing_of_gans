#!/bin/python

import os
import tqdm
import codecs
import pickle
from optparse import OptionParser

#import keras
import numpy as np
import pandas

# NLP
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import re
from matplotlib import pyplot as plt


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
		df['onehot'] = df['0'].values # Add extra column



		dropIndices = []
		vocabulary = {}
		vocabulary_frequency = {}
		maxVocabIndex = 1
		filteredTokenLists = []
		
		for idx, i in enumerate(df.index):
		#for i in range(0,3):
			text = df.at[i, '0']
			if text == "":
				print ("Error: Empty text at line %d" % i)

			# Update the words in the list
			try:
				tokenizedList = sent_tokenize(re.sub(r'[\d+]','', text))[0]
			except:
				#print(text)
				df.at[i, '2'] = []
				dropIndices.append(i)
				continue
			#print(tokenizedList)

			sub_categories = tokenizedList.split(',')
	
			wordList = []
			for word in sub_categories:
				wordList.append((word))
			
			
			filteredTokenList = [] 
			for word in wordList:
				if word.lower() not in stopWordsList and word not in string.punctuation + zhpunc:
					filteredTokenList.append(word.lower())
			
			for word in filteredTokenList:
				#print(word)
				if word not in vocabulary:
					vocabulary[word] = maxVocabIndex
					vocabulary_frequency[word] = 1
					maxVocabIndex += 1
				else:
					vocabulary_frequency[word] = vocabulary_frequency[word] + 1

			filteredTokenLists.append(filteredTokenList)

			#idxFilteredTokenList = [vocabulary[word] for word in filteredTokenList]
			#print(filteredTokenList)
			# idxFilteredTokenList = []
			# for sub_category in filteredTokenList:
			# 	idxFilteredTokenList1 = []
			# 	for word in sub_category:
			# 		idxFilteredTokenList1.append(vocabulary[word])
			# 	idxFilteredTokenList.append(idxFilteredTokenList1)
			# #print(idxFilteredTokenList)
			# # Add the updated word to the df
			# if len(filteredTokenList) < options.maxSequenceLength:
			# 	df.at[i, '0'] = filteredTokenList
			# 	df.at[i, '2'] = idxFilteredTokenList
			# else:
			# 	df.at[i, '0'] = filteredTokenList[:options.maxSequenceLength]
			# 	df.at[i, '2'] = idxFilteredTokenList[:options.maxSequenceLength]


		#print(dropIndices)


		# print ("Dropping unresolved indices")
		# print ("Size before dropping: %d" % df.shape[0])
		# df = df.drop(df.index[dropIndices])
		# print ("Size after dropping: %d" % df.shape[0])
		# print ("Maximum vocab index found to be: %d" % maxVocabIndex)

		# #Load FastText embeddings
		# print ("Loading FastText embedding matrix")
		# embeddingMatrix = loadWordEmbeddings(options, vocabulary)

		# # Save data
		# print ("Saving data")print(vocabulary)
		# with open(options.dataFileName, "wb") as f:
		# 	pickle.dump([df, vocabulary, maxVocabIndex, embeddingMatrix], f)
		# print ("Saving completed!")
	print(len(filteredTokenLists))
	occurrences = []
	#for it1 in range(0,50):
	p = 0 
	new_vocab = {}
	index = 1
	for word1 in vocabulary_frequency:
		if(vocabulary_frequency[word1] > 25):
			new_vocab[word1] = index
			index = index + 1
			p = p + 1
	


	for i, list1 in enumerate(filteredTokenLists):
		idxFilteredTokenList = [new_vocab[word] for word in list1 if word in new_vocab]
		filteredTokenList = [word for word in list1 if word in new_vocab]
		#print(idxFilteredTokenList)
		onehot = [0 for i in range(index)]
		df.at[i, '0'] = filteredTokenList
		df.at[i, '2'] = idxFilteredTokenList
		for ind in idxFilteredTokenList:
			onehot[ind - 1] = 1
		#print(onehot)
		df.at[i, 'onehot'] = onehot
	
	#print(len(onehot))
	#print(df)

	#occurrences.append(p)
	with open(options.dataFileName, "wb") as f:
		pickle.dump([df, new_vocab, index], f)



	#x = [i for i in range(50)] 
	#print(new_vocab)
	#plt.plot(x, occurrences)
	#plt.show()
	#print(occurrences)

if __name__ == "__main__":
	# Command line options
	parser = OptionParser()

	# Base options
	parser.add_option("--trainModel", action="store_true", dest="trainModel", default=True, help="Whether to train the model")
	parser.add_option("--testModel", action="store_true", dest="testModel", default=False, help="Whether to test the model")

	parser.add_option("--createNewDataFile", action="store_true", dest="createNewDataFile", default=True, help="Whether to create new data file")
	parser.add_option("--dataFileName", action="store", type="string", dest="dataFileName", default="multilabel.pkl", help="Data file name")
	parser.add_option("--trainingDataFileName", action="store", type="string", dest="trainingDataFileName", default="../preprocessing/50k_index_sorted.csv", help="Name of the file containing the training samples")
	parser.add_option("--testDataFileName", action="store", type="string", dest="testDataFileName", default="./assets/test.csv", help="Name of the file containing the test samples")
	parser.add_option("--testOutputFileName", action="store", type="string", dest="testOutputFileName", default="./assets/out.csv", help="Name of the file containing the predictions for the test samples")
	parser.add_option("--useLSTM", action="store_true", dest="useLSTM", default=False, help="Use LSTM network instead of the ConvNet")
	parser.add_option("--maxSequenceLength", action="store", type="int", dest="maxSequenceLength", default=30, help="Maximum sentence length to be used")
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

	
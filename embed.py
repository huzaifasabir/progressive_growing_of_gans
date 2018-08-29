import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
with open('data.pkl', "rb") as f:
	df, vocabulary, maxVocabIndex, embeddingMatrix = pickle.load(f)

X = tf.keras.preprocessing.sequence.pad_sequences(df[2].values, padding='post')
arg = tf.convert_to_tensor(X, dtype=tf.float32)

#Sequential = tf.keras.Sequential
#Embedding = tf.keras.layers.Embedding
#classes = list(range(0, 32))
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(maxVocabIndex, 300, weights=[embeddingMatrix], input_length=71, trainable=False))

model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3))
model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3))
model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=300, activation='sigmoid', input_shape=(71,)))
sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True, clipnorm=4)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])

output_array = model.apply(arg)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output_array[0]))
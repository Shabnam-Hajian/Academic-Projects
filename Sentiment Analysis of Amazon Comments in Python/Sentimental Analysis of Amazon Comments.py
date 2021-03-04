
# Assignment 4- Using keras
# Import Libraries
import warnings
import os
import sys
import re
import json
import numpy as np
import pandas as pd
import gensim
import tensorflow as tf
from keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Embedding
from tensorflow.keras import regularizers
# from keras.layers import Dropout


# To get rid of tenserflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Define functions:
def read_csv_data(data_path):
    # To read CSV files and clean it
    with open(data_path) as f:
        data = f.readlines()
        data = [' '.join(line.strip().split(',')) for line in data]
        data = [line.lower() for line in data]
        clean_data = [re.sub(r"[^a-z]+", ' ', x) for x in data]
    return clean_data


def word_counter(textlist):
    # return number of unique words in text
    counter = []
    for sent in textlist:
        for word in sent.split():
            counter.append(word)
    unique = set(counter)
    return len(unique)


def max_lenght(textlist):
    # return 95 percentile of length of sentences
    num_words = [len(sent.split()) for sent in textlist]
    return int(np.percentile(num_words, 95))


def vocab_maker(data, max_dic_size, batch_size):
    # Create a vocabulary of the recommended size-1 for pad and out of range
    vectorizer = TextVectorization(max_tokens=max_dic_size-1,
                                   output_mode='int')
    text_data = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    vectorizer.adapt(text_data)
    # index 0 and 1 are reserved values for padding and out of dic
    vocab = vectorizer.get_vocabulary()
    vocab = [x.decode('utf-8') for x in vocab]
    return vocab


def embedding_maker(vocab, embedding_dim, w2v_model):
    # Build the embedding matrix and 2 dictionaries as below:
    # token2word = build token-id -> word dict (will be used when decoding)
    # word2token = build word -> token-id dict (will be used when encoding)
    vocab_size = len(vocab)+2
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # There are two characters that doesnt have any vector in Word2Vec
    token2word = {0: '<pad>', 1: '<UKN>'}
    word2token = {'<pad>': 0, '<UKN>': 1}
    # Set random seed
    np.random.seed(0)
    embedding_matrix[0] = np.random.random((1, embedding_dim))
    np.random.seed(1)
    embedding_matrix[1] = np.random.random((1, embedding_dim))
    for i in range(len(vocab)):
        try:
            word = str(vocab[i])
            token2word[i+2] = word
            word2token[word] = i+2
            embedding_matrix[i+2] = w2v_model.wv[word]
        except KeyError:
            # skip it as we already replace it with <UKN> token
            continue
    with open('a4/data/token2word.json', 'w') as f:
        json.dump(token2word, f)
    with open('a4/data/word2token.json', 'w') as f:
        json.dump(word2token, f)
    return embedding_matrix, word2token, token2word, vocab_size


def text_encoder(data, word2token):
    # This is a personilize encoder to encode data based on word-> token_id
    encoded_data = []
    for i in range(len(data)):
        vectorize = []
        sent = data[i].lower().split()
        for word in sent:
            token_id = word2token.get(word)
            if token_id is None:
                vectorize.append(word2token['<UKN>'])
            else:
                vectorize.append(token_id)
        encoded_data.append(vectorize)
    return encoded_data


def Functional_Model_Maker(max_sent_lenght, vocab_size, embedding_dim,
                           batch_size, embedding_matrix, activation,
                           padded_train, ytrain,
                           padded_val, yval,
                           padded_test, ytest):
    # Make exact same model with differebt activation in hidden layer
    inputs = Input(shape=(max_sent_lenght,), name='Input')
    embedding = Embedding(vocab_size, embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_sent_lenght,
                          trainable=True)(inputs)
    flatten = Flatten(name='Flatten')(embedding)
    hidden = Dense(2, activation=activation, name='Hidden',
                   activity_regularizer=regularizers.l2(1e-4))(flatten)
    outputs = Dense(2, activation='softmax', name='Predictions')(hidden)
    model = Model(inputs=inputs, outputs=outputs, name=activation)
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(padded_train, ytrain,
              batch_size=batch_size,
              epochs=3,
              validation_data=(padded_val, yval))
    test_scores = model.evaluate(padded_test, ytest,
                                 batch_size=batch_size)
    accuracy = test_scores[1]
    loss = test_scores[0]
    return model, accuracy, loss


def main(data_dir):
    # import all data from Assignment 1
    train = read_csv_data(os.path.join(data_dir, 'train.csv'))
    test = read_csv_data(os.path.join(data_dir, 'test.csv'))
    val = read_csv_data(os.path.join(data_dir, 'val.csv'))
    with open(os.path.join(data_dir, 'label.csv'))as f:
        label = f.read()
    labels = [int(x) for x in label.splitlines()]
    y_train = to_categorical(np.array(labels[:len(train)]))
    y_test = to_categorical(np.array(
        labels[len(train): (len(train)+len(test))]))
    y_val = to_categorical(np.array(labels[(len(train)+len(test)):]))
    #
    # Set some of hyper parameters
    # Set maximum for sentence length based on 95% of our training data
    max_sent_lenght = max_lenght(train)
    print('max_sent_lenght:', max_sent_lenght)
    word_count = word_counter(train)
    print('word_count:', word_count)
    embedding_dim = 300  # Comes from Word2vector vector size
    Max_num_words = 20000  # Should set as "max_words" or an integer
    # Here set 20000 as studies shows in many cases this a good enough amount
    batch_size = 400  # Should be an integer
    #
    # Making a vocab with size of Max_num_words
    vocab = vocab_maker(train, Max_num_words, batch_size)
    print('len(vocab):', len(vocab))
    #
    # We Want to use w2v model from assignment 3 for embedding- Load model
    w2v = gensim.models.Word2Vec.load('a3/data/w2v.model')
    #
    # Create the embedding matrix and id-> word and word-> id dictionaries
    embedding_matrix, word2token, token2word, vocab_size = embedding_maker(
        vocab, embedding_dim, w2v)
    print('len(token2word):', len(token2word))
    print('len(word2token):', len(word2token))
    print('embedding_matrix.shape => {}'.format(embedding_matrix.shape))
    # lengths should be the same, if error, check below codes
    # problem = []
    # for words in vocab:
    #     if words not in list(word2token):
    #         problem.append(words)
    # print(problem)
    #
    # We want to build customize encoder with our dictionaries
    encoded_train = text_encoder(train, word2token)
    padded_train = pad_sequences(encoded_train, maxlen=max_sent_lenght,
                                 padding='post', truncating='post')
    encoded_val = text_encoder(val, word2token)
    padded_val = pad_sequences(encoded_val, maxlen=max_sent_lenght,
                               padding='post', truncating='post')
    encoded_test = text_encoder(test, word2token)
    padded_test = pad_sequences(encoded_test, maxlen=max_sent_lenght,
                                padding='post', truncating='post')
    #
    # Built a models
    print('**********Building models**********')
    nn_relu, relu_accuracy, relu_loss = Functional_Model_Maker(
        max_sent_lenght, vocab_size, embedding_dim,
        batch_size, embedding_matrix, 'relu',
        padded_train, y_train,
        padded_val, y_val,
        padded_test, y_test)
    nn_sigmoid, sigmoid_accuracy, sigmoid_loss = Functional_Model_Maker(
        max_sent_lenght, vocab_size, embedding_dim,
        batch_size, embedding_matrix, 'sigmoid',
        padded_train, y_train,
        padded_val, y_val,
        padded_test, y_test)
    nn_tanh, tanh_accuracy, tanh_loss = Functional_Model_Maker(
        max_sent_lenght, vocab_size, embedding_dim,
        batch_size, embedding_matrix, 'tanh',
        padded_train, y_train,
        padded_val, y_val,
        padded_test, y_test)
    #
    # Print summary of models accuracy
    print('***Models Accuracy Table***')
    index = ['nn_relu', 'nn_sigmoid', 'nn_tanh']
    acc = ({'Model accuracy': [relu_accuracy, sigmoid_accuracy, tanh_accuracy],
            'Model loss': [relu_loss, sigmoid_loss, tanh_loss]})
    acc_table = pd.DataFrame(data=acc, index=index)
    print(acc_table)
    #
    # Saving models for future uses
    print('***saving Models***')
    nn_relu.save('a4/data/nn_relu.model')
    nn_sigmoid.save('a4/data/nn_sigmoid.model')
    nn_tanh.save('a4/data/nn_tanh.model')


if __name__ == '__main__':
    main(sys.argv[1])

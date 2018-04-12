from __future__ import absolute_import
from __future__ import division
import sys
sys.path.append('../python_libraries')
import logging
import os
from sentiment_model import SentimentVanillaNeuralNetworkModel #,SentimentWordCNNNeuralNetwork,SentimentLSTMNeuralNetwork
import json
import tensorflow as tf
from nlp_functions.word_and_character_vectors import get_char,get_glove

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

tf.app.flags.DEFINE_integer("gpu", 1, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_integer("num_epochs",100, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("review_length", 300, "The maximum words in the review")
tf.app.flags.DEFINE_integer("word_embedding_size", 300, "Size of the pretrained word vectors. This needs to be 300")
tf.app.flags.DEFINE_integer("word_length", 15, "The maximum length of words of your model")
tf.app.flags.DEFINE_integer("char_embedding_size", 128, "Size of the pretrained character vectors. This needs to be 128")
tf.app.flags.DEFINE_bool("discard_long",False,"Discard lines longer than review_length")
tf.app.flags.DEFINE_float("test_size",0.10,"Dev set to split from training set")
tf.app.flags.DEFINE_integer("ratings_size",5, "Number of ratings")
tf.app.flags.DEFINE_integer("CONV_SHAPE", 128, "Size of convolutional layers")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "C:\\Users\\tihor\\Documents\\ml_data_files", "Path to glove .txt file.")
tf.app.flags.DEFINE_string("char_path", "C:\\Users\\tihor\\Documents\\ml_data_files", "Path to character .txt file.")
tf.app.flags.DEFINE_string("review_path","C:\\Users\\tihor\\Documents\\yelp_reviews\\", "Where to find yelp reviews")
tf.app.flags.DEFINE_string("rating_path","C:\\Users\\tihor\\Documents\\yelp_reviews\\", "Where to find yelp ratings")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint from. You need to specify this for official_eval mode.")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

emb_matrix_char, char2id, id2char=get_char('../ml_data_files')
emb_matrix_word, word2id, id2word=get_glove('../ml_data_files')

sentiment_model=SentimentVanillaNeuralNetworkModel(FLAGS,word2id,char2id,emb_matrix_word,emb_matrix_char)
#sentiment_model=SentimentWordCNNNeuralNetwork(FLAGS,word2id,char2id,emb_matrix_word,emb_matrix_char)
#sentiment_model=SentimentLSTMNeuralNetwork(FLAGS,word2id,char2id,emb_matrix_word,emb_matrix_char)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for epoch in range(FLAGS.num_epochs):
    validation_accuracy=sentiment_model.run_epoch(sess)
    print('validation_accuracy for epoch ' + str(epoch)+' => ' + str(validation_accuracy))

print('Final validation_accuracy => ' +str(sentiment_model.get_validation_accuracy(sess)))
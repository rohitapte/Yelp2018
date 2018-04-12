import sys
sys.path.append('../python_libraries')
from nlp_functions.word_and_character_vectors import PAD_ID,UNK_ID
from nlp_functions import word_and_character_vectors
from nlp_functions.sentence_operations import get_ids_and_vectors,sentence_to_word_and_char_token_ids,split_by_whitespace,pad_words,pad_characters,convert_ids_to_word_vectors,convert_ids_to_char_vectors
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import defaultdict
import os

class YelpDataFromFile(object):
    def __init__(self,word2id,char2id,word_embed_matrix,char_embed_matrix,review_path,rating_path,batch_size,review_length,word_length,discard_long=False,test_size=0.1):
        self.one_hot_rating={}
        self.one_hot_rating[1]=[0,0,0,0,1]
        self.one_hot_rating[2]=[0,0,0,1,0]
        self.one_hot_rating[3]=[0,0,1,0,0]
        self.one_hot_rating[4]=[0,1,0,0,0]
        self.one_hot_rating[5]=[1,0,0,0,0]
        vocab_size=5261669
        indices=[i for i in range(vocab_size)]
        train_indices,self.test_indices=train_test_split(indices,test_size=0.1)
        self.train_indices,self.dev_indices=train_test_split(train_indices,test_size=0.1)
        self.word2id=word2id
        self.char2id=char2id
        self.word_embed_matrix=word_embed_matrix
        self.char_embed_matrix=char_embed_matrix
        self.review_path=review_path
        self.rating_path=rating_path
        self.batch_size=batch_size
        self.review_length=review_length
        self.word_length=word_length
        self.discard_long=discard_long

    def get_lines_and_ratings_from_file(self,indices):
        file_dict=defaultdict(list)
        for item in indices:
            file_dict[item//100000].append(item)
        lines={}
        ratings={}
        for file in file_dict:
            with open(os.path.join(os.path.normpath(self.review_path),'reviews'+str(file).zfill(2)+'.txt'),'r',encoding='utf-8') as f:
                i=int(file)*100000
                for line in f:
                    if i in file_dict[file]:
                        lines[i]=line
                    i+=1
            with open(os.path.join(os.path.normpath(self.review_path),'ratings'+str(file).zfill(2)+'.txt'),'r',encoding='utf-8') as f:
                i=int(file)*100000
                for line in f:
                    if i in indices:
                        ratings[i]=line
                    i+=1
        lines_return=[lines[item] for item in indices]
        ratings_return=[int(ratings[item]) for item in indices]
        return lines_return,ratings_return

    def convert_data_to_embeddings(self,indices):
        review_words_for_mask=[]
        review_words = []
        review_chars = []
        ratings = []
        lines, rating_lines = self.get_lines_and_ratings_from_file(indices)
        for line in lines:
            word_ids, word_ids_to_vectors, char_ids_to_vectors = get_ids_and_vectors(line, self.word2id, self.char2id,
                                                                                     self.word_embed_matrix,
                                                                                     self.char_embed_matrix,
                                                                                     self.review_length,
                                                                                     self.word_length,
                                                                                     self.discard_long)
            review_words_for_mask.append(word_ids)
            review_words.append(word_ids_to_vectors)
            review_chars.append(char_ids_to_vectors)
        for line in rating_lines:
            ratings.append(self.one_hot_rating[line])
        review_words = np.array(review_words)
        review_chars = np.array(review_chars)
        ratings = np.array(ratings)
        review_words_for_mask = np.array(review_words_for_mask)
        review_mask = (review_words_for_mask != PAD_ID).astype(np.int32)
        return review_words,review_chars,review_mask,ratings

    def generate_one_epoch(self,batch_size=1000):
        num_batches=int(len(self.train_indices))//batch_size
        if batch_size*num_batches<len(self.train_indices): num_batches += 1
        random.shuffle(self.train_indices)
        for i in range(num_batches):
            indices=self.train_indices[i*batch_size:(i+1)*batch_size]
            review_words,review_chars,review_mask,ratings=self.convert_data_to_embeddings(indices)
            yield review_words,review_chars,review_mask,ratings

    def generate_test_data(self,batch_size=1000):
        num_batches=int(len(self.test_indices))//batch_size
        if batch_size*num_batches<len(self.test_indices): num_batches+=1
        for i in range(num_batches):
            indices=self.test_indices[i*batch_size:(i+1)*batch_size]
            review_words, review_chars, review_mask, ratings=self.convert_data_to_embeddings(indices)
            yield review_words,review_chars,review_mask,ratings

    def generate_dev_data(self,batch_size=1000):
        num_batches=int(len(self.dev_indices))//batch_size
        if batch_size*num_batches<len(self.dev_indices): num_batches+=1
        for i in range(num_batches):
            indices=self.test_indices[i*batch_size:(i+1)*batch_size]
            review_words, review_chars, review_mask, ratings=self.convert_data_to_embeddings(indices)
            yield review_words,review_chars,review_mask,ratings

#emb_matrix_char, char2id, id2char=word_and_character_vectors.get_char('C:\\Users\\tihor\\Documents\\ml_data_files')
#emb_matrix_word, word2id, id2word=word_and_character_vectors.get_glove('C:\\Users\\tihor\\Documents\\ml_data_files')
#zz=YelpDataFromFile(word2id=word2id,char2id=char2id,word_embed_matrix=emb_matrix_word,char_embed_matrix=emb_matrix_char,review_path='C:\\Users\\tihor\\Documents\\yelp_reviews\\',rating_path='C:\\Users\\tihor\\Documents\\yelp_reviews\\',batch_size=20,review_length=300,word_length=15,discard_long=False,test_size=0.1)

#for review_words,review_chars,review_mask,ratings in zz.generate_one_epoch():
#    print("words")
#    print(review_words.shape)
#    print("chars")
#    print(review_chars.shape)
#    print("mask")
#    print(review_mask.shape)
#    print("ratings")
#    print(ratings.shape)
#    break
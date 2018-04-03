from word_and_character_vectors import PAD_ID,UNK_ID
import word_and_character_vectors
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import defaultdict
import os

def split_by_whitespace(sentence):
    words=[]
    for word in sentence.strip().split():
        if word[-1] in '.,?!;':
            words.append(word[:-1])
            words.append(word[-1])
        else:
            words.append(word)
        #words.extend(re.split(" ",word_fragment))
    return [w for w in words if w]

def sentence_to_word_and_char_token_ids(sentence,word2id,char2id):
    """
    convert tokenized sentence into word indices
    any word not present gets converted to unknown
    """
    tokens=split_by_whitespace(sentence)
    word_ids=[]
    char_ids=[]
    for w in tokens:
        word_ids.append(word2id.get(w,UNK_ID))
        char_word_ids=[]
        for c in w:
            char_word_ids.append(char2id.get(c,UNK_ID))
        char_ids.append(char_word_ids)

    return tokens,word_ids,char_ids

def pad_words(word_array,pad_size):
    if len(word_array)<pad_size:
        word_array=word_array+[PAD_ID]*(pad_size-len(word_array))
    return word_array

def pad_characters(char_array,pad_size,word_pad_size):
    if len(char_array)<pad_size:
        char_array=char_array+[[PAD_ID]]*(pad_size-len(char_array))
    for i,item in enumerate(char_array):
        if len(item)<word_pad_size:
            char_array[i]=char_array[i]+[PAD_ID]*(word_pad_size-len(item))
        if len(item) > word_pad_size:
            char_array[i] = item[:word_pad_size]
    return char_array

def convert_ids_to_word_vectors(word_ids,emb_matrix_word):
    retval=[]
    for id in word_ids:
        retval.append(emb_matrix_word[id])
    return retval

def convert_ids_to_char_vectors(char_ids,embed_matrix_char):
    retval=[]
    for word_rows in char_ids:
        row_val=[]
        for c in word_rows:
            row_val.append(emb_matrix_char[c])
        retval.append(row_val)
    return retval

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
        self.train_indices,self.test_indices=train_test_split(indices)
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
        review_words = []
        review_chars = []
        ratings = []
        lines, rating_lines = self.get_lines_and_ratings_from_file(indices)
        for line in lines:
            tokens, word_ids, char_ids = sentence_to_word_and_char_token_ids(line, word2id, char2id)
            if len(tokens) > self.review_length:
                if self.discard_long:
                    continue
                else:
                    tokens = tokens[:self.review_length]
                    word_ids = word_ids[:self.review_length]
                    char_ids = char_ids[:self.review_length]
            word_ids = pad_words(word_ids, self.review_length)
            char_ids = pad_characters(char_ids, self.review_length, self.word_length)
            word_ids = convert_ids_to_word_vectors(word_ids, emb_matrix_word)
            char_ids = convert_ids_to_char_vectors(char_ids, emb_matrix_char)
            review_words.append(word_ids)
            review_chars.append(char_ids)
        for line in rating_lines:
            ratings.append(self.one_hot_rating[line])
        review_words = np.array(review_words)
        review_chars = np.array(review_chars)
        ratings = np.array(ratings)
        review_mask=(review_words!=PAD_ID).astype(np.int32)
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

class YelpData(object):
    """
        word2id - dict converting words to embedding_matrix id
        char2id - dict converting characters to embedding_char_matrix id
        review_path - location of reviews.txt
        rating_path - location of ratings.txt
        batch_size - batch size
        review_length - number of words in each review
        word_length - character length of each word
        discard_long - if true, discard reviews longer than review_length words. else truncate
    """
    def __init__(self,word2id,char2id,word_embed_matrix,char_embed_matrix,review_path,rating_path,batch_size,review_length,word_length,discard_long=False,test_size=0.1):
        vocab_size=5261669
        stop_at=500000
        review_words=[]
        review_chars=[]
        ratings=[]
        self.batch_size=batch_size
        self.batches=[]
        self.review_length=review_length
        self.word_length=word_length
        i=0
        with open(review_path,'r',encoding='utf-8') as fh:
            for line in tqdm(fh, total=stop_at):
                i+=1
                if i > stop_at: break
                tokens, word_ids, char_ids=sentence_to_word_and_char_token_ids(line,word2id,char2id)
                if len(tokens)>review_length:
                    if discard_long:
                        continue
                    else:
                        tokens=tokens[:review_length]
                        word_ids=word_ids[:review_length]
                        char_ids=char_ids[:review_length]
                word_ids=pad_words(word_ids,review_length)
                char_ids=pad_characters(char_ids,review_length,word_length)
                zz=convert_ids_to_word_vectors(word_ids,emb_matrix_word)
                aa=convert_ids_to_char_vectors(char_ids,emb_matrix_char)
                review_words.append(word_ids)
                review_chars.append(char_ids)
        i=0
        one_hot_rating={}
        one_hot_rating[1]=[0,0,0,0,1]
        one_hot_rating[2]=[0,0,0,1,0]
        one_hot_rating[3]=[0,0,1,0,0]
        one_hot_rating[4]=[0,1,0,0,0]
        one_hot_rating[5]=[1,0,0,0,0]
        with open(rating_path,'r',encoding='utf-8') as fh:
            for line in tqdm(fh, total=stop_at):
                i += 1
                if i > stop_at: break
                rating=int(line.strip())
                ratings.append(one_hot_rating[rating])
        indices=np.arange(len(review_words))
        self.review_words_train,self.review_words_test,self.ratings_train,self.ratings_test,idx_train,idx_test=train_test_split(np.array(review_words),np.array(ratings),indices,test_size=test_size)
        review_chars=np.array(review_chars)
        self.review_chars_train=review_chars[idx_train]
        self.review_chars_test=review_chars[idx_test]

    def generate_one_epoch(self,batch_size=1000):
        num_batches=int(self.ratings_train.shape[0])//batch_size
        if batch_size*num_batches<self.ratings_train.shape[0]: num_batches+=1
        perm=np.arange(self.review_words_train.shape[0])
        np.random.shuffle(perm)
        self.review_words_train=self.review_words_train[perm]
        self.review_chars_train=self.review_chars_train[perm]
        self.ratings_train=self.ratings_train[perm]
        for i in range(num_batches):
            review_words=self.review_words_train[i*batch_size:(i+1)*batch_size]
            review_chars=self.review_chars_train[i*batch_size:(i+1)*batch_size]
            ratings=self.ratings_train[i*batch_size:(i+1)*batch_size]
            yield review_words,review_chars,ratings


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
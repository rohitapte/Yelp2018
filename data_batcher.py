from word_and_character_vectors import PAD_ID,UNK_ID
import word_and_character_vectors
from tqdm import tqdm
import random
import numpy as np

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
    def __init__(self,word2id,char2id,review_path,rating_path,batch_size,review_length,word_length,discard_long=False,test_size=0.1,val_size=0.1):
        vocab_size=5261669
        stop_at=50000
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
        self.review_words = np.array(review_words)
        self.review_chars = np.array(review_chars)
        self.ratings = np.array(ratings)

    def generate_one_epoch(self,batch_size=100):
        print(self.review_words.shape)
        print(self.review_chars.shape)
        print(self.ratings.shape)
        num_batches=int(self.ratings.shape[0])//batch_size
        if batch_size*num_batches<self.ratings.shape[0]: num_batches+=1
        perm=np.arange(self.review_words.shape[0])
        np.random.shuffle(perm)
        self.review_words=self.review_words[perm]
        self.review_chars=self.review_chars[perm]
        self.ratings=self.ratings[perm]
        for i in range(num_batches):
            review_words=self.review_words[i*batch_size:(i+1)*batch_size]
            review_chars=self.review_chars[i*batch_size:(i+1)*batch_size]
            ratings=self.ratings[i*batch_size:(i+1)*batch_size]
            yield review_words,review_chars,ratings




emb_matrix_char, char2id, id2char=word_and_character_vectors.get_char('C:\\Users\\tihor\\Documents\\ml_data_files')
#emb_matrix_word, word2id, id2word=word_and_character_vectors.get_glove('C:\\Users\\tihor\\Documents\\ml_data_files')
emb_matrix_word, word2id, id2word=word_and_character_vectors.get_char('C:\\Users\\tihor\\Documents\\ml_data_files')
zz=YelpData(word2id=word2id,char2id=char2id,review_path='C:\\Users\\tihor\\Documents\\yelp_reviews\\reviews.txt',rating_path='C:\\Users\\tihor\\Documents\\yelp_reviews\\ratings.txt',batch_size=20,review_length=600,word_length=15,discard_long=False,test_size=0.1,val_size=0.1)

for review_words,review_chars,ratings in zz.generate_one_epoch():
    print("words")
    print(review_words.shape)
    print("chars")
    print(review_chars.shape)
    if review_chars.shape != (100,600,15):
        x=2
    print("ratings")
    print(ratings.shape)
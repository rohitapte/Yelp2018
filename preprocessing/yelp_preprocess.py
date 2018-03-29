import os
import json
import string

FILEPATH='C:/Users/tihor/Documents/yelp_reviews'
FILENAME='review.json'
reviews=[]
ratings=[]

count=0
f_out_reviews=open(os.path.join(os.path.normpath(FILEPATH),'reviews.txt'),'w',encoding='utf-8')
f_out_ratings=open(os.path.join(os.path.normpath(FILEPATH),'ratings.txt'),'w',encoding='utf-8')
with open(os.path.join(os.path.normpath(FILEPATH),FILENAME),'r',encoding='utf-8') as f_in:
    for line in f_in:
        value_dict=json.loads(line)
        review=value_dict['text'].replace('\r','').rstrip()
        if review[-1] not in string.punctuation: review+='.'
        if '\n' in review:
            reconstituted=''
            review_split=review.split('\n')
            for item in review_split:
                stripped=item.rstrip()
                if len(stripped)>0:
                    if stripped[-1] not in string.punctuation: stripped+='.'
                    reconstituted+=stripped+' '
            review=reconstituted.rstrip()
        temp=review.replace('\n','').replace('\r','').replace('\r\n','').replace('  ',' ')
        reviews.append(temp.lower())
        ratings.append(str(value_dict['stars']).strip())
        #f_out_reviews.write(review.replace('  ',' ')+'\n')
        #f_out_ratings.write(str(value_dict['stars']).strip()+'\n')

for item in reviews:
    f_out_reviews.write(item+'\n')
for item in ratings:
    f_out_ratings.write(item+'\n')
f_out_reviews.close()
f_out_ratings.close()

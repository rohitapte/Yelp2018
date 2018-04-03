import os
import json
import string
import re

def extract_reviews_and_ratings():
    """
    preprocessing of yelp data
    takes the reviews.json and spits out reviews in lower case stripping new lines (and adding periods to sentence endings).
    also writes rating file
    """
    re_patterns=[]
    re_patterns.append((re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),''))
    re_patterns.append((re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE),''))
    re_patterns.append((re.compile(r'[\?\.\!]+(?=[\?\.\!])'),''))
    re_patterns.append((re.compile(r'(?<=[.,])(?=[^\s])'),r' '))
    FILEPATH='C:/Users/tihor/Documents/yelp_reviews'
    FILENAME='review.json'
    reviews=[]
    ratings=[]
    replacement={}
    replacement['947503']='if you want to get a good meal in a reasonable amount of time then you should probably look elsewhere. this restaurant needs some serious help. there are dishes everywhere. servers do not pick up plates from tables. it takes forever to get food and when you do it is usually luke warm. i imagine if you just want to drink beers it would be fine but be prepared to wait forever to get your food because there is no sense of urgency here.'
    replacement['1963615']="mmm wow mmm wowwowowwow mmm wow so i have been seeing 5 star reviews for this place for so long i had to try it. mmm wow my friend kevin was trying to explain sweet republic's awesomeness and when he got to the point of twittering one of the owners to make sure he nabbed the last of a flavor i knew i had to find out what everyone is obsessing about. mmm wow mmm helen and jan were so nice mmm wow mmm wow mmm jan recommended the combination of mayan chocolate with avacado jalapeno. holy crap that was some kind of other worldly flavor mmm wow mmm we can't wait to go back mmm wow mmm i also had the espresso. it was great as well! mmm oh gawd mmm wow mmmm"
    #replacement[2895633]="I don't have very many choices for a drug store. Walgreens's is practically on every conner just like cvs who's is a smaller store."
    #replacement[2558896]="Good food but terrible service."
    #replacement[2558897]="Bad."
    #replacement[3008514]="Very grateful Michael Lin lawyer to help us to win my case, this Thursday my girlfriend and I come from California to Las Vegas to find Lin Law Group to help us to deal with my cases, Michael Lin is a super professional and nice lawyer, he is the most professional I have ever seen of lawyers in the United States, his experience is very rich, the most important thing that Michael Lin lawyer is very responsible, he took our case for a long time to help us to analyze this case, every detail is considered very considerate ,and also he is very patient of our problems to our losses to a minimum, we feel very reassuring and happy, but also we want to thank Michael Lin lawyer's assistant Miss Crystal，she is also a very responsible assistant, before she called to our case to do professional advice and  explained many times, each time to spend a long time to help us to analyze my case, Miss Crystal is very responsible and grateful for all us to do, once again thank Michael Lin lawyer and her assistant Miss Crystal, they are a very professional and responsible team of lawyers! ！Thank you"
    replacement['2675546']="I'm not certain, but I'm fairly positive Firewater is Lava reincarnated, but with a new name. Now, I never made it to Lava but even trying Firewater was a stretch. I don't why, but the exterior just doesn't appeal to me.\n\nOnce inside much of the same story. The decor is um..odd. Kinda contemporary American, with what I can only describe as miss-moshed Italian flair. Food was eh, just okay. Penne was used for the mac n cheese. The ribeye was a low, I mean \"looooooooooow down dirty shame\" low grade cut. I guess, at $16 or $18 whatever it was, I should have known. $16 isn't cheap by any means, but for a ribeye..that's criminal. Can we say USDA grade D?\n\nNow to the good stuff. \n\nThe NY strip:\nNow this is a steak. Far from Del Frisco's, but great cut and what a steak should look like. \n\nThe drinks: As the name implies, they seem to specialize in cocktails..Martinis.\n\n$8 Dirty Martini:\nYea buddy. This price might be the cheapest in town."
    replacement['5010195']="Wegen der absolut tollen Landschaft und der netten Leute muss man dort unbedingt mal gewesen sein. Die Berge, der Morgennebel, im Sommer ist es hell bis Mitternacht usw. \nMake your own Ness: Das Monster sieht man öfter mal nachgebildet, alles sehr witzig und kultig. Warum Nessi ausgerechnet lila sein soll, ist für uns allerdings ein Geheimnis geblieben. In einem Ort stehen solche farbenfrohen Nessis als Familie am Seeufer, östlich von Urquart Castle. In Fort Augustus gibt es eine hübsche Drahtskultur mit Kind. Den Vogel abgeschossen hatte im Juli 2013 die Mannschaft von einem Segelboot, die ein großes lila Aufblastier um Baum und Mast gewickelt hatte. Es wurde viel gegrüßt und gewunken."
    count=0
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
            if str(count) in replacement:
                review=replacement[count]
            temp=review.replace('\n','').replace('\r','').replace('\r\n','').replace('  ',' ').replace('*','').replace('=','').replace('/',' or ').replace('-',' - ')
            #temp=re.sub(r'(?<=[.,])(?=[^\s])', r' ',temp)
            temp=temp.replace('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS','SPORTS')
            temp=temp.replace('noooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo','no')
            temp=temp.lower()
            for pattern,replacement in re_patterns:
                temp=pattern.sub(replacement,temp)
            #temp=re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', temp)
            #temp=http_pattern.sub('',temp)
            #temp=chinese_pattern.sub('',temp)
            reviews.append(temp)
            ratings.append(str(value_dict['stars']).strip())
            count += 1

    i=0
    f_out_reviews = open(os.path.join(os.path.normpath(FILEPATH),'reviews'+str(i).zfill(2)+'.txt'),'w',encoding='utf-8')
    for item in reviews:
        if i%100000==0:
            temp=i//100000
            f_out_reviews.close()
            f_out_reviews=open(os.path.join(os.path.normpath(FILEPATH),'reviews'+str(temp).zfill(2)+'.txt'),'w',encoding='utf-8')
        f_out_reviews.write(item+'\n')
        i+=1
    f_out_reviews.close()
    i=0
    f_out_ratings=open(os.path.join(os.path.normpath(FILEPATH),'ratings'+str(i).zfill(2)+'.txt'),'w',encoding='utf-8')
    for item in ratings:
        if i%100000==0:
            temp=i//100000
            f_out_ratings.close()
            f_out_ratings = open(os.path.join(os.path.normpath(FILEPATH),'ratings'+str(temp).zfill(2)+'.txt'),'w',encoding='utf-8')
        f_out_ratings.write(item+'\n')
        i+=1
    f_out_ratings.close()


extract_reviews_and_ratings()
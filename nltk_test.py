

#
# nltk.download()  # Downdloads necessary packages (type all)


########################################################################################
#                       How to break into word or sentence
########################################################################################
# from nltk import sent_tokenize, word_tokenize



# example_text = "hello Mr. Smith. How are you doing? This is an example sentence. I've always wondered that. Don't eat poison..."

# print(sent_tokenize(example_text))

# print(word_tokenize(example_text))


########################################################################################
#                       Stop words
#               filtering stop words from word list
#
########################################################################################

# from nltk.corpus import stopwords
# from nltk import word_tokenize
# example_text = "hello Mr. Smith. How are you doing? This is an example sentence. I've always wondered that. Don't eat poison..."
#
# stop_words = set(stopwords.words("english"))
#
# words = word_tokenize(example_text)
#
# filtered_words = [ w for w in words if w not in stop_words]
#
# print(filtered_words)



########################################################################################
#                       Stemming:
#      Reduces number of different words by reducing redundant terminations
#
########################################################################################


# from nltk.stem import PorterStemmer
#
# from nltk.tokenize import word_tokenize
#
# ps = PorterStemmer()
#
# example_words = ["python", "pythoner", "pythoned", "pythoning", "pythonly"]
#
# for w in example_words:
#     print(ps.stem(w))
#


########################################################################################
#                       Speech Tagging & chunking
#
########################################################################################

import nltk

# State of the union addresses from past presidents
from nltk.corpus import state_union

# A different tokenizer
# unsupervized machine learning tokenizer - comes pre-trained but you can retrain
from nltk.tokenize import PunktSentenceTokenizer


train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# train the tokenizer on the previous state of the union
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try :
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)


            # Chunk looks for a possible adverb, then possible verb,
            # then Necessary propper noun, then possible noun
            chunkGram = r"""Chunk: {<RB.?>* <VB.?>* <NNP>+ <NN>?}"""

            # the parser we want to use is the chunkGram we just made
            chunkParser = nltk.RegexpParser(chunkGram)

            # run it on the tagged words
            chunked = chunkParser.parse(tagged)


            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
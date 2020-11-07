import aiml
from Witcher_Wiki_Parser import WitcherWikiParser
import numpy as np
from sklearn.feature_extraction import text
import math
import pandas

# TODO add more stop words
stop_words = ['a', 'an', 'is', 'in', 'at', 'and']

name = ''


def tf_idf(sentences):
    vectorizer = text.TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    bagOfWords = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    print(denselist)
    df = pandas.DataFrame(denselist, columns=bagOfWords)
    print(df)


def create_word_dict(senteces):
    vectorizer = text.TfidfVectorizer()

    x = vectorizer.fit_transform(senteces)

    print(vectorizer.get_feature_names())
    print(x.shape)

    allWords = vectorizer.get_feature_names()
    idf = {}
    bagOfWords = []

    # Gets the term frequency TF for each word in each sentence
    for sentence in senteces:
        bag = {}

        for word in allWords:
            bag[word] = sentence.lower().count(word)

        bagOfWords.append(bag)

    # gets the idf for each word
    for word in allWords:
        numOfOcurances = 0

        for bag in bagOfWords:
            if bag[word] > 0:
                numOfOcurances += 1

        print( math.log( (len(senteces) / numOfOcurances)  ) )
        idf[word] = np.log( (len(senteces) / numOfOcurances)  )

    for bag in bagOfWords:
        tempArrTf = []
        tempArrIdf = []
        for word in allWords:
            tempArrTf.append(bag[word])
            tempArrIdf.append(idf[word])

        tf = np.array(tempArrTf)
        idfArr = np.array(tempArrIdf)
        #print(tf)
        #print(idfArr)

        print(tf * idfArr)

        tempArr = []

    #print(bagOfWords)
    #print(idf)


kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

parser = WitcherWikiParser()

# create_word_dict(['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?'])
# tf_idf(['The sky is blue', 'The sky is not blue'])
# create_word_dict(['The sky is blue', 'The sky is not blue'])

tf_idf(['the man went out for a walk', 'the children sat around the fire'])

while True:
    userInput = input("> ")

    responseAgent = "aiml"

    if responseAgent == "aiml":
        answer = kernal.respond(userInput)

        if '#' == answer[0]:
            params = answer.split('$')
            phrase = params[-1]
            cmd = params[0]
            if cmd == "#0":
                print(phrase + " " + name + "!")
                break
            if cmd == "#1":
                name = params[1]
            if cmd == "#2":
                beast = params[1]
                weaknesses = parser.get_beast_weaknesses(beast)
                print("A " + beast + " is weak to:")
                for weakness in weaknesses:
                    print(weakness)
        else:
            print(answer)

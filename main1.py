import aiml
from Witcher_Wiki_Parser import WitcherWikiParser
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import math
import pandas
from csv import reader


# TODO add more stop words
stop_words = ['a', 'an', 'is', 'in', 'at', 'and']

qa_pairs = []

name = ''

def load_qa_pairs(dir='qapairs.csv'):
    file = open(dir, 'r')

    fileReader = reader(file)

    for row in fileReader:
        qa_pairs.append(row)


def tf_idf(sentences):
    vectorizer = text.TfidfVectorizer()
    # Calculates tf.idf
    vectors = vectorizer.fit_transform(sentences)
    # Gets list of all words used in each sentence
    bagOfWords = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    #print(denselist)
    df = pandas.DataFrame(denselist, columns=bagOfWords)
    #print(df)
    css = cosine_similarity(vectors)

    # Gets senetence with the highest similarity
    sentence1Simalerity = css[0][1:]

    highestSim = max(sentence1Simalerity)
    print(highestSim)
    ret = {}
    for i in range(0, len(sentences[1:])):
        if sentence1Simalerity[i] == highestSim:
            ret[sentences[i+1]] = highestSim
            break

    print(ret)
    print(css)
    #return sentences[max(css[0][1:])]


kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

parser = WitcherWikiParser()

tf_idf('Tell me about withcer')

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

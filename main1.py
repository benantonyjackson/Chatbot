import aiml
from Witcher_Wiki_Parser import WitcherWikiParser
import numpy as np
from sklearn.feature_extraction import text

# TODO add more stop words
stop_words = ['a', 'an', 'is', 'in', 'at', 'and']

name = ''


def create_word_dict(senteces):
    vectorizer = text.TfidfVectorizer(stop_words=stop_words)

    x = vectorizer.fit_transform(senteces)

    print(vectorizer.get_feature_names())
    print(x.shape)

    allWords = vectorizer.get_feature_names()

    bagOfWords = []

    for sentence in senteces:
        bag = {}

        for word in allWords:
            bag[word] = sentence.count(word)

        bagOfWords.append(bag)

    for word in allWords:
        numOfOcurances = 0

        for bag in bagOfWords:
            if bag[word] > 0:
                numOfOcurances += 1

        print(word)
        # This is the IDF for each word
        print(np.log( len(senteces) / numOfOcurances  ))

    print(bagOfWords)


kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

parser = WitcherWikiParser()

create_word_dict(['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?'])

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

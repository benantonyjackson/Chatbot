import aiml
from Witcher_Wiki_Parser import WitcherWikiParser
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import math
import pandas
from bs4 import BeautifulSoup
import string
from PyDictionary import PyDictionary
from spellchecker import spellchecker
from autocorrect import Speller

from qapairs import QApairs

ALL_CATEGORIES = []
CATEGORIES_WITHOUT_WILDCARDS = []

qapairs = QApairs()


def get_all_patterns():
    file = open("chatbot.xml")
    soup = BeautifulSoup(file.read(), "lxml")
    file.close()

    categories = soup.findAll('pattern')

    for category in categories:
        ALL_CATEGORIES.append(category.text)
        if '*' not in category:
            CATEGORIES_WITHOUT_WILDCARDS.append(category.text)


# Employs different techniques to find the most similar question
def find_most_similar_question(question):
    # tf_idf(questions + CATEGORIES_WITHOUT_WILDCARDS)
    response = find_most_similar_document(question, qapairs.questions + CATEGORIES_WITHOUT_WILDCARDS)

    # If the similarity is 0 then an appropriate match was not found
    if response['similarity'] == 0:
        print("I'm sorry, I'm not sure what you mean :( Can you rephrase it?")
    else:
        # Checks if the most similar response is a qa pair or a pattern without a wild card
        if response['index'] < len(qapairs.questions):
            # Prints most similar answer, adding a new line after each full stop to make it more readable
            print(qapairs.answers[response['index']].replace(". ", ".\n"))
        else:
            # If the most similar result is a pattern without a wild card, input the pattern into the chat bot
            process_input(ALL_CATEGORIES[response['index'] - len(qapairs.questions)])


def tf_idf(documents):
    vectorizer = text.TfidfVectorizer()
    # Calculates tf.idf
    vectors = vectorizer.fit_transform(documents)
    words = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pandas.DataFrame(denselist, columns=words)
    print(df)


# Uses cosine similarity to find the most similar document to the input
def find_most_similar_document(inp, allQuestions):
    # Adds input the the questions list to get full list of documents
    sentences = [inp] + allQuestions

    # Calculates tf-idf
    vectorizer = text.TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)

    css = cosine_similarity(vectors)

    # Gets the cosine similarity matrix for the input and strips the similarity score for its self
    sentence1Simalerity = css[0][1:]

    # Finds the document most similar to the input
    highestSim = max(sentence1Simalerity)
    ret = {}
    for i in range(0, len(sentences[1:])):
        if sentence1Simalerity[i] == highestSim:
            ret['question'] = sentences[i+1]
            ret['similarity'] = highestSim
            ret['index'] = i
            break

    return ret


def spell_check_sentence(inp):
    spell = Speller(lang='en')

    # Defines a list of valid in this context that would likely get picked up by the spell checker package
    wordsToIgnore = ['chatbot', 'witcher', 'geralt', 'rivia']

    wordsToIgnore += parser.ALL_BEASTS

    ret = ""

    for word in inp.split(' '):
        if word not in wordsToIgnore:
            ret += spell(word) + " "
        else:
            ret += word + " "

    return ret[:-1]


def process_input(userInput):
    responseAgent = "aiml"

    # Strips punctuation from input
    userInput = userInput.translate(str.maketrans('', '', string.punctuation))

    userInput = userInput.lower()

    userInput = spell_check_sentence(userInput)

    if responseAgent == "aiml":
        answer = kernel.respond(userInput)

        if '#' in answer[0]:
            params = answer.split('$')
            phrase = params[-1]
            cmd = params[0]
            if cmd == "#0":
                print(phrase)
                quit()
            if cmd == "#1":
                print(parser.get_summary(params[1]))
            if cmd == "#2":
                beast = params[1]
                weaknesses = parser.get_beast_weaknesses(beast)
                print("A " + beast + " is weak to:")
                for weakness in weaknesses:
                    print(weakness)
            if cmd == "#3":
                # QA pairs
                find_most_similar_question(params[1])

        else:
            print(answer)


if __name__ == "__main__":
    get_all_patterns()
    qapairs.load_qa_pairs()

    kernel = aiml.Kernel()

    kernel.setTextEncoding(None)

    kernel.bootstrap(learnFiles="chatbot.xml")

    parser = WitcherWikiParser()

    while True:
        process_input(input('> '))

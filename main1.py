import aiml
from Witcher_Wiki_Parser import WitcherWikiParser
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import math
import pandas
from csv import reader
from bs4 import BeautifulSoup
import string
from PyDictionary import PyDictionary
from spellchecker import spellchecker
from autocorrect import Speller

# TODO add more stop words
stop_words = ['a', 'an', 'is', 'in', 'at', 'and']

questions = []
answers = []

ALL_CATEGORIES = []
CATEGORIES_WITHOUT_WILDCARDS = []


def get_all_patterns():
    file = open("chatbot.xml")
    soup = BeautifulSoup(file.read(), "lxml")
    file.close()

    categories = soup.findAll('pattern')

    for category in categories:
        ALL_CATEGORIES.append(category.text)
        if '*' not in category:
            CATEGORIES_WITHOUT_WILDCARDS.append(category.text)


def handle_qa_pairs(question):
    # tf_idf(questions + CATEGORIES_WITHOUT_WILDCARDS)
    response = find_most_similar_document(question, questions + CATEGORIES_WITHOUT_WILDCARDS)

    if response['similarity'] == 0:
        print("I'm sorry, I'm not sure what you mean :( Can you rephrase it?")
    else:
        print(response)
        if response['index'] < len(questions):
            # Prints most similar answer, adding a new line after each full stop to make it more readable
            print(answers[response['index']].replace(". ", ".\n"))
        else:
            process_input(ALL_CATEGORIES[response['index'] - len(questions)])


def load_qa_pairs(dir='qapairs.csv'):
    file = open(dir, 'r')

    fileReader = reader(file)

    for row in fileReader:
        questions.append(row[0])
        answers.append(row[1])
    file.close()


def tf_idf(documents):
    vectorizer = text.TfidfVectorizer()
    # Calculates tf.idf
    vectors = vectorizer.fit_transform(documents)
    words = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pandas.DataFrame(denselist, columns=words)
    print(df)


def find_most_similar_document(inp, allQuestions):
    sentences = [inp] + allQuestions

    vectorizer = text.TfidfVectorizer()
    # Calculates tf*idf
    vectors = vectorizer.fit_transform(sentences)
    css = cosine_similarity(vectors)

    # Gets senetence with the highest similarity
    sentence1Simalerity = css[0][1:]

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

    wordsToIgnore = ['chatbot', 'witcher', 'Geralt']

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
    spell = spellchecker.SpellChecker()

    userInput = spell_check_sentence(userInput)

    print(userInput)

    if responseAgent == "aiml":
        answer = kernal.respond(userInput)

        if '#' in answer[0]:
            params = answer.split('$')
            phrase = params[-1]
            cmd = params[0]
            if cmd == "#0":
                print(phrase)
                quit()
            if cmd == "#1":
                # TODO remove this code and change the command to something else
                print(parser.get_summary(params[1]))
            if cmd == "#2":
                beast = params[1]
                weaknesses = parser.get_beast_weaknesses(beast)
                print("A " + beast + " is weak to:")
                for weakness in weaknesses:
                    print(weakness)
            if cmd == "#3":
                # QA pairs
                handle_qa_pairs(params[1])

        else:
            print(answer)


if __name__ == "__main__":
    get_all_patterns()
    load_qa_pairs()
    dictionary = PyDictionary()

    kernal = aiml.Kernel()

    kernal.setTextEncoding(None)

    kernal.bootstrap(learnFiles="chatbot.xml")

    parser = WitcherWikiParser()

    while True:
        process_input(input('> '))



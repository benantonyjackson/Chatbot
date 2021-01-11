import aiml
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

from Witcher_Wiki_Parser import WitcherWikiParser

from bs4 import BeautifulSoup
import string
from autocorrect import Speller

from qapairs import QApairs
from Trained_CNN_Wrapper import TrainedModel

import easygui



ALL_CATEGORIES = []
CATEGORIES_WITHOUT_WILDCARDS = []

tm = TrainedModel()

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
    response = find_most_similar_document(question, qapairs.questions + CATEGORIES_WITHOUT_WILDCARDS)

    # If the similarity is 0 then an appropriate match was not found
    if response['similarity'] == 0:
        print("I'm sorry, I'm not sure what you mean. Can you rephrase it?")
    else:
        # Checks if the most similar response is a qa pair or a pattern without a wild card
        if response['index'] < len(qapairs.questions):
            # Prints most similar answer, adding a new line after each full stop to make it more readable
            print(qapairs.answers[response['index']].replace(". ", ".\n"))
        else:
            # If the most similar result is a pattern without a wild card, input the pattern into the chat bot
            process_input(CATEGORIES_WITHOUT_WILDCARDS[response['index'] - len(qapairs.questions)])


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
    wordsToIgnore = ['chatbot', 'witcher', 'geralt', 'rivia', 'warg', 'fugas']

    wordsToIgnore += parser.ALL_ENEMIES

    ret = ""

    for word in inp.split(' '):
        if word not in wordsToIgnore:
            ret += spell(word) + " "
        else:
            ret += word + " "

    return ret[:-1]


def get_enemy_description(inp, parser):
    try:
        words = inp.split(' ')

        if words not in parser.ALL_ENEMIES and len(words) > 1:
            for word in words:
                if word in parser.ALL_ENEMIES:
                    inp = word
        print(parser.get_summary(inp))
    except (Exception):
        print("Sorry I'm not sure what a '" + inp + "' is. Try asking something else")


def get_enemy_weaknesses(inp, parser):
    try:
        words = inp.split(' ')

        if words not in parser.ALL_ENEMIES and len(words) > 1:
            for word in words:
                if word in parser.ALL_ENEMIES:
                    inp = word

        weaknesses = parser.get_enemy_weaknesses(inp)
        print("A " + inp + " is weak to:")
        for weakness in weaknesses:
            print(weakness)
    except (Exception):
        print("Sorry I'm not sure what a '" + inp + "' is. Try asking something else")


def classify_image(filepath=""):
    # responses = {
    #     'bears': 'This image appears to contain a bear. Bears are enemies that can be found in the witcher 3'
    #     'wolves'
    # }

    if filepath == "":
        filepath = easygui.fileopenbox()

    prediction = tm.predict_local_image(path=filepath)

    print("This image appears to contain " + prediction)
    print(parser.get_summary(prediction))


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
                get_enemy_description(params[1], parser)
            if cmd == "#2":
                get_enemy_weaknesses(params[1], parser)
            if cmd == "#3":
                # QA pairs
                find_most_similar_question(params[1])
            if cmd == "#4":
                classify_image()
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

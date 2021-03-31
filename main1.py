import aiml
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

from Witcher_Wiki_Parser import WitcherWikiParser

from bs4 import BeautifulSoup
import string
from autocorrect import Speller

import requests

import uuid

from qapairs import QApairs
from Trained_CNN_Wrapper import TrainedModel
from Trained_GAN_Wrapper import TrainedGan

import easygui

from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk.corpus import stopwords

from simpful import FuzzySystem
from simpful import FuzzySet
from simpful import Triangular_MF
from simpful import LinguisticVariable

import pandas

read_expr = Expression.fromstring

kb = []

ALL_CATEGORIES = []
CATEGORIES_WITHOUT_WILDCARDS = []

qapairs = QApairs()

FS = None

COG_KEY = '4ba57df8aa534bbca29b0b3f24a888f2'
COG_ENDPOINT = 'https://n0736563-cw.cognitiveservices.azure.com/'
COG_REGION = 'uksouth'


# Taken from https://github.com/MicrosoftDocs/ai-fundamentals
def translate_text(text, to_lang, cog_key=COG_KEY, cog_endpoint=COG_ENDPOINT, cog_region=COG_REGION):
    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    # params = '&from={}&to={}'.format(from_lang, to_lang)
    params = '&to={}'.format(to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region': cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    # return response
    print(response)
    return response[0]["translations"][0]["text"], response[0]['detectedLanguage']['language']


def init_fuzzy_system():
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.2), term="weightless")
    S_2 = FuzzySet(function=Triangular_MF(a=0.2, b=0.7, c=1), term="light")
    S_3 = FuzzySet(function=Triangular_MF(a=1, b=15, c=15), term="heavy")
    FS.add_linguistic_variable("Weight",
                               LinguisticVariable([S_1, S_2, S_3], concept="Weight", universe_of_discourse=[0, 15]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=50, c=100), term="cheap")
    F_2 = FuzzySet(function=Triangular_MF(a=100, b=10000, c=10000), term="expensive")
    FS.add_linguistic_variable("Value",
                               LinguisticVariable([F_1, F_2], concept="Gold Value", universe_of_discourse=[0, 10000]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="Sword")
    T_2 = FuzzySet(function=Triangular_MF(a=10, b=15, c=20), term="Junk")
    T_3 = FuzzySet(function=Triangular_MF(a=20, b=25, c=30), term="Food")
    T_4 = FuzzySet(function=Triangular_MF(a=30, b=35, c=40), term="Bomb or Potion")
    FS.add_linguistic_variable("Category", LinguisticVariable([T_1, T_2, T_3, T_4], universe_of_discourse=[0, 40]))

    # Define fuzzy rules
    R1 = "IF (Weight IS heavy) THEN (Category IS Sword)"
    R2 = "IF (Weight IS light) THEN (Category IS Junk)"
    R3 = "IF (Weight IS weightless) AND (Value IS cheap) THEN (Category IS Food)"
    R4 = "IF (Weight IS weightless) AND (Value IS expensive) THEN (Category IS Bomb or Potion)"
    FS.add_rules([R1, R2, R3, R4])

    return FS


def fuzzy_logic(gold_value, weight_value):
    FS.set_variable("Weight", weight_value)
    FS.set_variable("Value", gold_value)

    strengths = FS.get_firing_strengths()

    outputs = ['a Sword', 'Junk', 'Food or Drink', 'a Potion or Bomb']

    # Returns the name of the category which best fits the input
    return outputs[strengths.index(max(strengths))]


def check_integrity(dummy_subject, dummy_object):
    return evaluate_expression(dummy_object + '(' + dummy_subject + ')') == "I don't know"


def load_knowledge_base(filename='kb.csv'):
    data = pandas.read_csv(filename, header=None)
    for row in data[0]:
        kb.append(read_expr(row))

    if not check_integrity('xxx', 'yyy'):
        print("An error occurred while loading the knowledge base. The knowledge base contains contradictions")
        quit()


def get_all_patterns():
    file = open("chatbot.xml")
    soup = BeautifulSoup(file.read(), "lxml")
    file.close()

    categories = soup.findAll('pattern')

    for category in categories:
        ALL_CATEGORIES.append(category.text)
        if '*' not in category.text:
            CATEGORIES_WITHOUT_WILDCARDS.append(category.text)


# Employs different techniques to find the most similar question
def find_most_similar_question(question, language):
    response = find_most_similar_document(question, qapairs.questions + CATEGORIES_WITHOUT_WILDCARDS)

    # If the similarity is 0 then an appropriate match was not found
    if response['similarity'] == 0:
        if language != 'en':
            print(translate_text("I'm sorry, I'm not sure what you mean. Can you rephrase it?", language)[0])
        else:
            print("I'm sorry, I'm not sure what you mean. Can you rephrase it?")
    else:
        # Checks if the most similar response is a qa pair or a pattern without a wild card
        if response['index'] < len(qapairs.questions):
            # Prints most similar answer, adding a new line after each full stop to make it more readable
            output = qapairs.answers[response['index']].replace(". ", ".\n")
            if language != "en":
                output, _ = translate_text(output, language)

            print(output)
        else:
            # If the most similar result is a pattern without a wild card, input the pattern into the chat bot
            process_input(CATEGORIES_WITHOUT_WILDCARDS[response['index'] - len(qapairs.questions)], language)


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


def get_enemy_description(inp, parser, language):
    try:
        words = inp.split(' ')

        if words not in parser.ALL_ENEMIES and len(words) > 1:
            for word in words:
                if word in parser.ALL_ENEMIES:
                    inp = word

        output = parser.get_summary(inp)
        if language != "en":
            output, _ = translate_text(output, language)
        print(output)
    except (Exception):
        output = "Sorry I'm not sure what a '" + inp + "' is. Try asking something else"

        if language != 'en':
            output, _ = translate_text(output, language)
        print(output)


def get_enemy_weaknesses(inp, parser, language):
    try:
        words = inp.split(' ')

        if words not in parser.ALL_ENEMIES and len(words) > 1:
            for word in words:
                if word in parser.ALL_ENEMIES:
                    inp = word

        weaknesses = parser.get_enemy_weaknesses(inp)

        output = "A " + inp + " is weak to:"

        # print("A " + inp + " is weak to:")
        for weakness in weaknesses:
            # print(weakness)
            output += "\n" + weakness

        if language != 'en':
            output, _ = translate_text(output, language)
        print(output)
    except (Exception):
        output = "Sorry I'm not sure what a '" + inp + "' is. Try asking something else"

        if language != 'en':
            output, _ = translate_text(output, language)
        print(output)


def classify_image(filepath="", language="en"):
    if filepath == "":
        filepath = easygui.fileopenbox()

    prediction = TrainedModel().predict_local_image(path=filepath)

    output = "This image appears to contain " + prediction + "\n" + parser.get_summary(prediction)

    if language != 'en':
        output, _ = translate_text(output, language)

    print(output)

    # print("This image appears to contain " + prediction)
    # print(parser.get_summary(prediction))


def generate_image():
    TrainedGan().generate_and_display_image()


def evaluate_expression(expression_string, verbose=False):
    expression_string = expression_string.replace(" ", "")
    expr = read_expr(expression_string)

    if ResolutionProver().prove(expr, kb, verbose=verbose):
        return 'Correct'
    else:
        # Checks if the inverse of the expression is true to determine if their expression is false
        inv_expression = read_expr("-" + expression_string)

        if ResolutionProver().prove(inv_expression, kb, verbose=verbose):
            return "Incorrect"
        else:
            return "I don't know"


def answer_user_question(subject, object):
    expression_string = subject + '(' + object + ')'
    print(evaluate_expression(expression_string))


def expand_knowledge_base(subject, object):
    expression_string = subject + ' (' + object + ')'

    kb.append(read_expr(expression_string))

    if check_integrity('xxx', 'yyy'):
        print('OK, I will remember that', object, 'is', subject)
    else:
        print("This is contradictory")
        # Removes contradictory sentence
        kb.pop()


def guess_the_item():
    print('Enter the weight of the item:')
    while True:
        try:
            weight_value = float(input('> '))
        except ValueError:
            print('Please enter a number')
            continue
        break

    print('Enter the gold value of the item:')
    while True:
        try:
            gold_value = float(input('> '))
        except ValueError:
            print('Please enter a number')
            continue
        break

    print("I would guess that this item is " + fuzzy_logic(gold_value, weight_value))


# https://www.quora.com/How-do-I-remove-determiners-and-conjunctions-from-string-in-Python
def remove_connectives(words):
    stopWords = set(stopwords.words('english'))
    words = words.split(" ")
    filter_words = ""
    for word in words:
        if word not in stopWords:
            filter_words += word

    return filter_words


def process_input(user_input, language=''):
    responseAgent = "aiml"

    # Strips punctuation from input
    user_input = user_input.translate(str.maketrans('', '', string.punctuation))

    user_input = user_input.lower()

    # user_input = spell_check_sentence(user_input)
    if language == '':
        user_input, language = translate_text(user_input, to_lang='en')

    # print("Body: " + str(bod))

    if responseAgent == "aiml":
        answer = kernel.respond(user_input)

        if '#' in answer[0]:
            params = answer.split('$')
            phrase = params[-1]
            cmd = params[0]
            if cmd == "#0":
                if language != 'en':
                    phrase, _ = translate_text(phrase, language)
                print(phrase)
                quit()
            if cmd == "#1":
                get_enemy_description(params[1], parser, language)
            if cmd == "#2":
                get_enemy_weaknesses(params[1], parser, language)
            if cmd == "#3":
                # QA pairs
                find_most_similar_question(params[1], language)
            if cmd == "#4":
                classify_image(language=language)
            if cmd == "#5":
                print("Sure! I'll generate you an image of a bear now!")
                generate_image(language=language)
            if cmd == "#6":  # if input pattern is "I know that * is *"
                object, subject = params[1].split(' is ')
                expand_knowledge_base(remove_connectives(subject), remove_connectives(object), language)
            if cmd == "#7":  # if the input pattern is "check that * is *"
                object, subject = params[1].split(' is ')
                answer_user_question(remove_connectives(subject), remove_connectives(object), language)
            if cmd == '#8':  # Fuzzy logic
                guess_the_item(language)
        else:
            if language != 'en':
                answer, _ = translate_text(answer, language)
                print(answer)


if __name__ == "__main__":
    get_all_patterns()
    qapairs.load_qa_pairs()

    kernel = aiml.Kernel()

    kernel.setTextEncoding(None)

    kernel.bootstrap(learnFiles="chatbot.xml")

    parser = WitcherWikiParser()

    load_knowledge_base()

    FS = init_fuzzy_system()

    while True:
        process_input(input('> '))

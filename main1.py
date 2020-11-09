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

questions = []
answers = []

def handle_qa_pairs(question):
    response = tf_idf(question)

    if response['similarity'] == 0:
        print("I'm sorry, I'm not sure what you mean :( Can you rephrase it?")
    else:
        print(response)
        # Prints most similar answer, adding a new line after each full stop to make it more readable
        print(answers[response['index']].replace(". ", ".\n"))


def load_qa_pairs(dir='qapairs.csv'):
    file = open(dir, 'r')

    fileReader = reader(file)

    for row in fileReader:
        questions.append(row[0])
        answers.append(row[1])
    file.close()


def tf_idf(inp, allQuestions=questions):
    # if inp is str:
    sentences = [inp] + allQuestions
    # else:
    #     sentences = inp + allQuestions

    vectorizer = text.TfidfVectorizer()
    # Calculates tf.idf
    vectors = vectorizer.fit_transform(sentences)
    # Gets list of all words used in each sentence
    bagOfWords = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pandas.DataFrame(denselist, columns=bagOfWords)
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


load_qa_pairs()

kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

parser = WitcherWikiParser()

print(parser.get_summary('ekimmara'))

# for enemy in parser.ALL_BEASTS:
#     print(enemy + ":")
#     print(parser.get_summary(enemy))

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
                # TODO remove this code and change the command to something else
                name = params[1]
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

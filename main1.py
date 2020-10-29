import aiml
from Witcher_Wiki_Parser import WitcherWikiParser

name = ''

kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

parser = WitcherWikiParser()

for b in parser.ALL_BEASTS:
    print("A " + b + " is weak to:")

    weaknesses = parser.get_beast_weaknesses(b)

    print("A " + b + " is weak to:")
    for weakness in weaknesses:
        print(weakness)
    print("\n ----------- \n")

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

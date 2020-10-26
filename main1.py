import aiml
import wikia
import wikipedia
from Witcher_Wiki_Parser import Witcher_Wiki_Parser

name = ''

kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

parser = Witcher_Wiki_Parser()

print(parser.get_beast_weaknesses("Bruxa"))



# print("Welcome to the chat bot")
#
# wikipedia.API_URL = "http://{lang}{sub_wikia}.wikia.com/api/v1/{action}"
#
# p = wikipedia.page("Bruxa")
#
# print(p.title)
#
# # wikia.set_lang("en")
# # wikia.set_lang("nl")
# # wikia.API_URL = "https://witcher.fandom.com"
# # test = wikia.page("elderscrolls", "Dragons (Lore)")
#
# print("Summary:")
# # print(test.section_lists("Contents"))

# print(test)



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
        else:
            print(answer)

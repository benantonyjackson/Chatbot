import aiml

kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

print("Welcome to the chat bot")

while True:
    userInput = input("> ")

    responseAgent = "aiml"

    if responseAgent == "aiml":
        answer = kernal.respond(userInput)

        if '#' == answer[0]:
            cmd, phrase = answer.split("$")

            if cmd == "#0":
                print(phrase)
                break

        else:
            print(answer)


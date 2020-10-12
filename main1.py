import aiml

kernal = aiml.Kernel()

kernal.setTextEncoding(None)

kernal.bootstrap(learnFiles="chatbot.xml")

print("Welcome to the chat bot")

while True:
    userInput = input()

    responseAgent = "aiml"

    if responseAgent == "aiml":
        answer = kernal.respond(userInput)
        print(answer)
        break
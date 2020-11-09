import requests
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import wikia


class WitcherWikiParser:
    BASE_URL = 'https://witcher.fandom.com/'
    WIKI_URL = BASE_URL + "wiki"

    types = []
    ALL_BEASTS = []
    BEAST_MAP = {}

    def __init__(self):
        self.get_full_beitiary()

    def get_full_beitiary(self):
        allBeasts = []
        r = requests.get(self.BASE_URL + "The_Witcher_3_bestiary")

        soup = BeautifulSoup(r.content, "lxml")

        types = soup.findAll("span", {"class": "mw-headline"})

        for type in types:
            self.types.append(type.text)

        allBeastCategories = soup.findAll("div", {"class": "divTable"})

        for beastCategory in allBeastCategories:
            # print(beastCategory)
            beastCatSoup = BeautifulSoup(str(beastCategory), "lxml")

            beasts = beastCatSoup.findAll("a")

            for beast in beasts:
                allBeasts.append(beast.text.lower())
                self.BEAST_MAP[beast.text.lower()] = self.BASE_URL + beast['href']
                self.BEAST_MAP[beast['title'].lower()] = self.BASE_URL + beast['href']

        self.ALL_BEASTS = allBeasts

    def get_full_page(self, page_title):
        r = requests.get(self.BASE_URL + page_title)

        soup = BeautifulSoup(r.content, "lxml")

        mydivs = soup.findAll("div", {"data-source": "Susceptibility"})

        print(mydivs)

        for div in mydivs:
            print(div)

        return "Done!"

    def strip_script(self, data):
        pass

    def get_and_save_page(self, beast):
        # requests.get(self.WIKI_URL + "/" + page)
        print("Fetching page " + beast)
        page = beast
        if beast.lower() == "wargs" or beast.lower() == "warg":
            # TODO Fix titling issues with wolves
            beast = "Wolves"
        elif beast.lower() == "lubberkin" or beast.lower() == "lubberkins":
            # TODO Fix titling issues with wolves
            beast = "Botchling"
        elif beast.lower() == "kikimore" or beast.lower() == "kikimores":
            return
        elif beast.lower() == "fugas":
            # TODO See if you can find a combat guide for Fugas
            return
        elif beast.lower() == "godling":
            return
        elif beast.lower() == "dettlaff van der eretein":
            #T ODO See if you can find a combat guide for fighting Dettlaff
            return
            # return ["Sorry, I'm not quite sure how best to fight Dettlaff. Would you like to learn more about Detlaff?"]

        print("Requesting " + beast)
        r = requests.get(self.BEAST_MAP[beast.lower()])

        print("Writting" + page)
        file = open("page_backups" + "/" + page + ".html", "w")
        file.write(str(r.content))
        file.close()
        print("done")


    def get_beast_weaknesses(self, beast):

        if beast.lower() == "wargs" or beast.lower() == "warg":
            # TODO Fix titling issues with wolves
            beast = "Wolves"
        if beast.lower() == "lubberkin" or beast.lower() == "lubberkins":
            # TODO Fix titling issues with wolves
            beast = "Botchling"
        if beast.lower() == "kikimore" or beast.lower() == "kikimores":
            return ['There is more that one type of kikimore',
                    'Implement a follow up question to ask which type of kikimore']
        if beast.lower() == "fugas":
            # TODO See if you can find a combat guide for Fugas
            return ['Sorry, I do not know how best to fight Fugas. Would you like to know a bit more about Fugas?']
        if beast.lower() == "godling":
            return ['You do not need to fight Godlings at any point. Would you like to learn more about Goldings']
        if beast.lower() == "dettlaff van der eretein":
            # TODO See if you can find a combat guide for fighting Dettlaff
            return ["Sorry, I'm not quite sure how best to fight Dettlaff. Would you like to learn more about Detlaff?"]

        r = requests.get(self.BEAST_MAP[beast.lower()])

        soup = BeautifulSoup(r.content, "lxml")
        mydivs = soup.findAll("div", {"data-source": "Susceptibility"})

        #TODO This is a bit flimsy, try to find a better way!
        witcher3Weaknesses = mydivs[-1]

        weaknessSoup = BeautifulSoup(str(witcher3Weaknesses), "lxml")

        weaknesses = weaknessSoup.findAll("a")

        ret = []

        for weakness in weaknesses:
            ret.append(weakness.text)

        return ret

    def get_summary(self, enemy):
        enemy = enemy.lower()
        if enemy == "moreau's golem":
            return "Long years of solitary study tend to make mages somewhat eccentric. As the years pass, laypeople being to irritate them more and more: they are dense, unreliable, disobedient and determined not to understand the gravity of mages' work. They display emotion when they should show discipline and self-mastery. No wonder mages have long considered the best companions to be artificial constructs they themselves bring to life and design to follow their orders and meet their needs. Professor Moreau was no exception in this regard. His golem was his dutiful servant and companion, in good times and bad. Moreau's golem was also an excellent guardian: massive, unyielding and devilishly strong. All in all, he was a tough nut for a witcher to crack!"
        if enemy == "daphne's wraith":
            return "During his stay in Toussaint, Geralt became involved with a curiuos case of gynodendromorphy - that is to say, a woman who had been turned into a tree. When one cut into this tree's bark, it bled, and when the wind blew through its leaves, one could hear muffled sobs. Geralt investigated the matter and learned magic (or possibly a curse) was responsible for the transformation, and it surely had something to do with a certain sad episode form the woman's past. The love of Daphne's life, a knight errant, had gone to the witch of Lynx Crag and never returned, leaving her to wait for him forever, filled with sadness and longing."
        if enemy == "jenny o' the woods":
            return "Jenny o' the Woods was the name given to a powerful nightwraith that haunted the fields near the village of Midcopse in Velen."
        if enemy == "ekimmara":
            return "An ekimma or ekimmara is a type of lower vampire like fleders are. These tend to be more vicious and animalistic."
        if enemy == "leshen":
            return 'A leshy (Polish: Leszy, in English sometimes spelled Leshii, Leszi, or Lessun), also known as leshen or spriggan in the game series, is a forest monster that is described as something that "lives only to kill". When they kill something or someone, they do not leave much for any carrion eaters.'
        if enemy == "imp":
            return "Janne, or better known as Imp, was a doppler who lived in Novigrad around 1272."
        if enemy == "ice troll":
            return "Ice trolls are a breed of semi-intelligent troll species adapted for the cold regions of the Northern Realms and Skellige. Unlike their rock and normal brethren, they usually cannot speak the Common Speech. Their bestiary entry is obtained by reading The World Underground."
        if enemy == "foglet":
            return "Foglers, foglings, or foglets (Polish: Mglak) are magical beings said to have arrived during the Conjunction of the Spheres. They tend to be found mainly in swamps, forests, or in the mountains and lure their prey into traps using magic. They have glowing eyes and mouths, keen ears and long pointed fingers and feed on the bodies of their victims."
        if enemy == "slyzards":
            return "Dracolizard or Slyzard (Polish: Oszluzg), is a large, grey-colored, flying reptile, sometimes as big as a dragon, for which dracolizards are sometimes mistaken. It is said that Crinfrid Reavers made sure that all members of this species were killed in Redania. Geralt also stated once that nothing in the world is able to parry the blow given by a dracolizard's tail."
        if enemy == "berserkers":
            return wikia.summary("witcher", enemy)
        print(self.BEAST_MAP[enemy].split("/")[-1])
        return wikia.summary("witcher", self.BEAST_MAP[enemy].split("/")[-1])
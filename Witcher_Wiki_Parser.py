import requests
from html.parser import HTMLParser
from bs4 import BeautifulSoup


class WitcherWikiParser():
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
        # p = Parser()
        # p.feed(str(r.content))

        soup = BeautifulSoup(r.content, "lxml")

        mydivs = soup.findAll("div", {"data-source": "Susceptibility"})

        print(mydivs)

        for div in mydivs:
            print(div)

        return "Done!"

    def strip_script(self, data):
        pass

    def get_beast_weaknesses(self, beast):

        if beast.lower() == "wargs" or beast.lower() == "warg":
            #TODO Fix titling issues with wolves
            beast = "Wolves"
        if beast.lower() == "lubberkin" or beast.lower() == "lubberkins":
            #TODO Fix titling issues with wolves
            beast = "Botchling"
        if beast.lower() == "kikimore" or beast.lower() == "kikimores":
            return ['There is more that one type of kikimore',
                    'Implement a follow up question to ask which type of kikimore']
        if beast.lower() == "fugas":
            #TODO See if you can find a combat guide for Fugas
            return ['Sorry, I do not know how best to fight Fugas. Would you like to know a bit more about Fugas?']
        if beast.lower() == "godling":
            return ['You do not need to fight Godlings at any point. Would you like to learn more about Goldings']
        if beast.lower() == "dettlaff van der eretein":
            #TODO See if you can find a combat guide for fighting Dettlaff
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

class Parser(HTMLParser):
    text = ''

    def handle_starttag(self, tag, attrs):
        # print("Start tag:", tag)
        if tag == "body":
            for attr in attrs:
                print("     attr:", attr)

    def handle_endtag(self, tag):
        pass
        # print("Encountered an end tag :", tag)

    def handle_data(self, data):
        pass
        # print("Encountered some data  :", data)
import requests
from html.parser import HTMLParser
from html2json import collect
from bs4 import BeautifulSoup

class Witcher_Wiki_Parser():
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
                allBeasts.append(beast.text)
                self.BEAST_MAP[beast.text] = self.BASE_URL + beast['href']


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
        r = requests.get(self.BEAST_MAP[beast])

        print(self.BEAST_MAP[beast])

        soup = BeautifulSoup(r.content, "lxml")
        mydivs = soup.findAll("div", {"data-source": "Susceptibility"})

        # print(mydivs)

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
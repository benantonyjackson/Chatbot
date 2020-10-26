import requests
from html.parser import HTMLParser
from html2json import collect
from bs4 import BeautifulSoup

class Witcher_Wiki_Parser():
    BASE_URL = 'https://witcher.fandom.com/wiki/'
    def __init__(self):
        pass


    def get_full_page(self, page_title):
        r = requests.get(self.BASE_URL + page_title)
        p = Parser()
        p.feed(str(r.content))

        soup = BeautifulSoup(r.content, "lxml")

        mydivs = soup.findAll("div", {"data-source": "Susceptibility"})

        for div in mydivs:
            print(div.text)

        return "Done!"

    def strip_script(self, data):
        pass

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
import requests
from bs4 import BeautifulSoup
import wikia


class WitcherWikiParser:
    BASE_URL = 'https://witcher.fandom.com/'
    WIKI_URL = BASE_URL + "wiki"

    # Stores all of the different types of enemies
    TYPES = []

    # Stores all enemies
    ALL_ENEMIES = []
    # Dictionary that maps each enemies name to the url to its wiki page
    ENEMY_MAP = {}

    def __init__(self):
        self.get_full_bestiary()

    def get_full_bestiary(self):
        allEnemies = []

        # Requests the bestiary which contains a list of all enemies in the game
        r = requests.get(self.BASE_URL + "The_Witcher_3_bestiary")
        soup = BeautifulSoup(r.content, "lxml")
        
        # Uses the headlines to construct list of enemy types
        types = soup.findAll("span", {"class": "mw-headline"})
        for type in types:
            self.TYPES.append(type.text)

        allBeastCategories = soup.findAll("div", {"class": "divTable"})
        
        # loops through ea
        for enemyCategory in allBeastCategories:
            enemyCatagorySoup = BeautifulSoup(str(enemyCategory), "lxml")

            enemies = enemyCatagorySoup.findAll("a")

            for enemy in enemies:
                allEnemies.append(enemy.text.lower())
                self.ENEMY_MAP[enemy.text.lower()] = self.BASE_URL + enemy['href']
                self.ENEMY_MAP[enemy['title'].lower().replace(" (creature)", "")] = self.BASE_URL + enemy['href']

        allEnemies.append('wargs')
        allEnemies.append('warg')
        allEnemies.append('lubberkin')
        allEnemies.append('lubberkins')
        allEnemies.append('kikimore')
        allEnemies.append('kikimores')
        allEnemies.append('fugas')
        allEnemies.append('godling')
        allEnemies.append('dettlaff van der eretein')

        self.ENEMY_MAP['horse'] = self.BASE_URL + "Horse"
        self.ENEMY_MAP['horses'] = self.BASE_URL + "Horse"

        self.ALL_ENEMIES = allEnemies

    def get_enemy_weaknesses(self, enemy):
        if enemy.lower() == "wargs" or enemy.lower() == "warg":
            enemy = "Wolves"
        if enemy.lower() == "lubberkin" or enemy.lower() == "lubberkins":
            enemy = "Botchling"
        if enemy.lower() == "kikimore" or enemy.lower() == "kikimores":
            enemy = 'kikimore warrior'
        if enemy == 'fugas':
            return ['Sorry, I do not know how best to fight Fugas.']
        if enemy.lower() == "godling":
            return ['You do not need to fight Godlings at any point.']
        if enemy.lower() == "dettlaff van der eretein":
            return ["Sorry, I'm not quite sure how best to fight Dettlaff."]

        r = requests.get(self.ENEMY_MAP[enemy.lower()])

        soup = BeautifulSoup(r.content, "lxml")
        mydivs = soup.findAll("div", {"data-source": "Susceptibility"})

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
            return wikia.summary("witcher", "werebear")

        return wikia.summary("witcher", self.ENEMY_MAP[enemy].split("/")[-1])

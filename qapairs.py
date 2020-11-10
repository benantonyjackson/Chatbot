from csv import reader


class QApairs:

    def __init__(self):
        self.questions = []
        self.answers = []

    def load_qa_pairs(self, dir='qapairs.csv'):
        file = open(dir, 'r')

        fileReader = reader(file)

        for row in fileReader:
            self.questions.append(row[0])
            self.answers.append(row[1])
        file.close()
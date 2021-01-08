from os import listdir


bears = 'dataset/train/bears'

i = 0

print(listdir(bears))

bad_bears = []

for fileName in listdir(bears):
    try:
        f = open(bears + "/" + fileName, "rb")

        data = f.read(4).decode()

        i += 1
        print(i)
        print(data)

        if "RIFF" == data:
            bad_bears.append(f)
    except:
        pass

print(bad_bears)

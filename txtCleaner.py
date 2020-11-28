import re

textfilec = open("tweets.txt", "r")
textfileo = open("out.txt", "w")
text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', textfilec.read())


for line in text.split("\n"):
    tmp = line
    if tmp[:1] == "@":
        tmp = ""

    while tmp[:1] == ".":
        tmp = tmp[1:]
        # print(tmp)
        # print('.')
    #
    # if tmp.find("&amp;"):
    #     tmp = tmp[:tmp.find("&amp;")] + tmp[tmp.find("&amp;") + 4:]

    textfileo.write(tmp)

# rand = random.randint(0, 1000)
#
# print(text.find(" ", rand))





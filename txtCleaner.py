import re

textfile = open("tweet.csv", "r")

text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', textfile.read())

textfilec = open("tweets.txt", "w")


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

    textfilec.write(tmp)





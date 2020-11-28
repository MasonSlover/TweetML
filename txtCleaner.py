import re
import random

textfilec = open("tweets.txt", "r")
textfileo = open("cleanedTweets.txt", "w")

#remove all URLs using regular expression
text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', textfilec.read())

# Remove starting @s and .s
for line in text.split("\n"):
    tmp = line
    if tmp[:1] == "@":
        tmp = ""

    while tmp[:1] == ".":
        tmp = tmp[1:]

    #
    # if tmp.find("&amp;"):
    #     tmp = tmp[:tmp.find("&amp;")] + tmp[tmp.find("&amp;") + 4:]

    textfileo.write(tmp)


#Select random words to use as starting phrases for ML model
lines = open("cleanedTweets.txt").readlines()
line = lines[0]

words = line.split()
for i in range (0, 100):
    myword = random.choice(words)

    print(myword, end = '", "')







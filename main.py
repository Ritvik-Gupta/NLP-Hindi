import nltk

nltk.download("punkt")

import warnings

import matplotlib.pyplot as plt

# library imports
import pandas as pd

warnings.filterwarnings("ignore")

# printing list of files available to us
from os import walk as osWalk
from os.path import join as pathJoin

datasetPath = {}
for dirname, _, filenames in osWalk("./dataset"):
    for filename in filenames:
        datasetPath[filename[:-4]] = pathJoin(dirname, filename)
        print(datasetPath[filename[:-4]])


authorBible = pd.read_csv(datasetPath["bible with authors"]).drop("Unnamed: 0", axis=1)
authorBible.head()


with open(datasetPath["bible books"], mode="r", encoding="utf-8-sig") as bibleBooks:
    books = bibleBooks.read()
books = books.split("\n")
ntBooks = [i.strip('"') for i in books[39:66]]
print(ntBooks)


df = pd.DataFrame(
    100 * authorBible.groupby("Book Name").size() / len(authorBible),
    columns=["% occurrences"],
)
df["Testament"] = df.index.to_series().map(lambda x: 1 if x in ntBooks else 0)
df = df.sort_values("% occurrences", ascending=False)
df.head()


from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

hindiFont = FontProperties(fname=datasetPath["nirmala"])
redPatch = Patch(color="red", alpha=0.7, label="Old Testament")
bluePatch = Patch(color="blue", alpha=0.7, label="New Testament")

plt.grid()
plt.bar(
    df.index,
    df["% occurrences"],
    align="center",
    alpha=0.5,
    color=df["Testament"].apply(lambda x: ["red", "blue"][x]),
)

plt.xticks(df.index, color="b", fontproperties=hindiFont, rotation=90, fontsize=12)
plt.yticks(fontsize=15)
plt.ylabel("% occurrences", fontsize=20)
plt.title("Percentage Book wise portions", fontsize=20)
plt.legend(handles=[redPatch, bluePatch])
plt.gca().margins(x=0)
plt.gcf().canvas.draw()

tickLabels = plt.gca().get_xticklabels()
maxSize = max([tick.get_window_extent().width for tick in tickLabels])
inchMargin = 0.2

margin = inchMargin / plt.gcf().get_size_inches()[0]
plt.gcf().subplots_adjust(left=margin, right=1.0 - margin)
plt.gcf().set_size_inches(
    maxSize / plt.gcf().dpi * 100 + 2 * inchMargin, plt.gcf().get_size_inches()[1]
)


df = pd.DataFrame(
    100 * authorBible.groupby("Authors").size() / len(authorBible),
    columns=["% occurrences"],
)
df = df.sort_values("% occurrences", ascending=False)
df.head()

plt.bar(df.index, df["% occurrences"], align="center", alpha=0.5)
plt.xticks(df.index, color="b", rotation=90, fontsize=12)
plt.yticks(fontsize=15)
plt.ylabel("% occurrences", fontsize=15)
plt.title("Percentage portions of authors", fontsize=15)
plt.gca().margins(x=0)
plt.gcf().canvas.draw()

tickLabels = plt.gca().get_xticklabels()
maxSize = max([tick.get_window_extent().width for tick in tickLabels])
inchMargin = 0.2

margin = inchMargin / plt.gcf().get_size_inches()[0]
plt.gcf().subplots_adjust(left=margin, right=1.0 - margin)
plt.gcf().set_size_inches(
    maxSize / plt.gcf().dpi * 100 + 2 * inchMargin, plt.gcf().get_size_inches()[1]
)


set(authorBible[authorBible["Authors"] == "unknown"]["Book Name"])


df_O = pd.DataFrame(
    100
    * authorBible[authorBible["Testament Code"] == 0].groupby("Authors").size()
    / len(authorBible[authorBible["Testament Code"] == 0]),
    columns=["% occurrences"],
)
df_O = df_O.sort_values("% occurrences", ascending=False)
df_N = pd.DataFrame(
    100
    * authorBible[authorBible["Testament Code"] == 1].groupby("Authors").size()
    / len(authorBible[authorBible["Testament Code"] == 1]),
    columns=["% occurrences"],
)
df_N = df_N.sort_values("% occurrences", ascending=False)

f, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [3, 1]})
axes[0].bar(df_O.index, df_O["% occurrences"], align="center", alpha=0.5)
plt.sca(axes[0])
plt.xticks(df_O.index, color="b", rotation=90, fontsize=12)
plt.title("Authors of Old Testament")
plt.ylabel("% occurences")

axes[1].bar(df_N.index, df_N["% occurrences"], align="center", alpha=0.5, color="r")
plt.sca(axes[1])
plt.xticks(df_N.index, color="b", rotation=90, fontsize=12)
plt.title("Authors of New Testament")
plt.tight_layout()

"""It can be easily seen here that relatively recent New Testament Books have lesser unknown writers (4%) as compared to the Old Testament (28%). <br>
We get an interesting finding here that to the contrary of popular belief, it is not Paul who has written most of the new testament but it is Luke. Sure, Paul has written more books but volume-wise(no. of verses), It is Luke who gave maximum contribution. [Original manuscripts](https://apologika.blogspot.com/2014/05/who-wrote-most-of-new-testament.html) also support this fact.

## Starting Text Analysis
**Stopwords** are generally the most common words in a language and for our purpose of analysis, they have to be eliminated from the text. `Hindi_StopWords.txt` is a custom made list of 270 stopwords.
"""

with open(datasetPath["stopwords"], encoding="utf-8") as stopwords:
    stopword = stopwords.read().strip("\ufeff")
stopword = stopword.split(", ")
stopword = [i.strip("'") for i in stopword]
print(stopword)


# Commented out IPython magic to ensure Python compatibility.
from nltk import word_tokenize as tokenizeWord

with open(
    datasetPath["full text bible"], mode="r", encoding="utf-8-sig"
) as fullTextBible:
    text = fullTextBible.read()
# %matplotlib inline

stopwords = set(stopword)


class StemWords:
    suffixCollection = {
        2: [
            "कर",
            "ाओ",
            "िए",
            "ाई",
            "ाए",
            "नी",
            "ना",
            "ते",
            "ती",
            "ाँ",
            "ां",
            "ों",
            "ें",
        ],
        3: [
            "ाकर",
            "ाइए",
            "ाईं",
            "ाया",
            "ेगी",
            "ेगा",
            "ोगी",
            "ोगे",
            "ाने",
            "ाना",
            "ाते",
            "ाती",
            "ाता",
            "तीं",
            "ाओं",
            "ाएं",
            "ुओं",
            "ुएं",
            "ुआं",
        ],
        4: [
            "ाएगी",
            "ाएगा",
            "ाओगी",
            "ाओगे",
            "एंगी",
            "ेंगी",
            "एंगे",
            "ेंगे",
            "ूंगी",
            "ूंगा",
            "ातीं",
            "नाओं",
            "नाएं",
            "ताओं",
            "ताएं",
            "ियाँ",
            "ियों",
            "ियां",
        ],
        5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
    }

    @classmethod
    def generate(cls, word):
        for suffixLen, suffixes in cls.suffixCollection.items():
            if len(word) > suffixLen + 1:
                for suffix in suffixes:
                    if word.endswith(suffix):
                        return word[:-suffixLen]
        return word


"""### Most frequent words in Hindi Bible"""

from collections import Counter as collectionCounter

wordcount = {}
# To eliminate duplicates, we will split by punctuation, and use case demiliters.
for word in text.split():
    word = word.replace(".", "")
    word = word.replace(",", "")
    word = word.replace(":", "")
    word = word.replace(";", "")
    word = word.replace('"', "")
    word = word.replace("!", "")
    word = StemWords.generate(word)

    if word not in stopwords:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

# most common word
wordFreq = {}
for word, count in collectionCounter(wordcount).most_common(20):
    wordFreq[word] = count
print(wordFreq)

freqDf = pd.DataFrame(list(wordFreq.items()), index=range(20), columns=["word", "freq"])
fig, ax = plt.subplots(figsize=(25, 10))
ax.barh(freqDf["word"], freqDf["freq"], align="center")
ax.set_xlabel("Word frequencies", fontsize=20)
ax.set_title("Top 20 most frequent words in Hindi Bible", fontsize=20)
plt.yticks(
    range(len(wordFreq.keys())),
    list(wordFreq.keys()),
    fontproperties=hindiFont,
    fontsize=20,
)


from string import punctuation

from nltk.probability import FreqDist

tokens = tokenizeWord(text)
customStopwords = set(list(stopwords) + list(punctuation + "।" + "॥"))
filteredStopwords = [word for word in tokens if word not in customStopwords]

# removing numeric digits from list of words
filteredStopwords = [i for i in filteredStopwords if not i.isdigit()]
freqDist = FreqDist(filteredStopwords)

print("In HHBD Hindi Bible")
print(f"प्रेम appears for {freqDist['प्रेम']} times")
print(f"डर appears for {freqDist['डर']} times")

"""For you curious ones out there, here's a dictionary of words with their frequencies that might interest you."""

checkWords = ["यीशु", "मसीह", "उद्धारकर्ता", "उद्धार", "क्रूस"]
checkWordFreq = {}
for checkWord in checkWords:
    checkWordFreq[checkWord] = freqDist[checkWord]

print(checkWordFreq)

freqDist.pop("राजा", None)
freqDist

"""`freq` dictionary contains frequency of each word, we are taking frequency as a score of a particular word's importance. 
> *I have intentionally removed word 'राजा' so that our algorithm won't give significance to verses containing lineages of various Kings*
"""

sents = []
for i in text.split("॥"):
    sents.append(i.split("।"))
sents = [item for sublist in sents for item in sublist]

"""In the above code, We split our full bible text by *Puran Virams* & *Ardh Virams* i.e. '॥' & '।' and stored the resulted sentences in a flattened list `sents`. Next, we're ranking these sentences based on significance scores of their constituting words, in dictionary `ranking`."""

from collections import defaultdict

ranking = defaultdict(int)
for i, sent in enumerate(sents):
    for token in tokenizeWord(sent):
        if token in freqDist:
            ranking[i] += freqDist[token]

"""We'll use these ranking to arrive at significant sentences as per the ranking calculated above."""

from heapq import nlargest

sentsIdx = nlargest(1, ranking, key=ranking.get)
summary = [sents[j] for j in sorted(sentsIdx)]
summary

"""It gives Joshua (यहोशू) chapter 22, verses 21-27 as the most significant verses in the Hindi Bible.

### Hierarchical Clustering
Here, I have done hierarchical Clustering on the first 10 sentences of Bible i.e Genesis (उत्पत्ति) 1:1-10. Sentences (denoted by a title words 'जल', 'ऊपर', 'आकाश' etc.) are clustered based on their cosine distances and their hierarchy is found out by Ward's method which merges cluster based on their cosine distances. <sub>(Cosine distance = 1- cosine similarity)</sub> <br><br>Cosine distance  is opposite to cosine similarity which calculates similarity between two vectors based on the cosine ratio of the angle made between them two. See, if you can make sense of below dendrogram.
"""

from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosSimilarity

# define vectorizer parameters
tfidfVectorizer = TfidfVectorizer(
    max_df=0.8,
    max_features=200000,
    min_df=0.2,
    stop_words=stopwords,
    use_idf=True,
    tokenizer=tokenizeWord,
    ngram_range=(1, 3),
)
tfidfMatrix = tfidfVectorizer.fit_transform(sents[:10])

dist = 1 - cosSimilarity(tfidfMatrix)
linkageMatrix = ward(dist)
titles = [
    "परमेश्वर",
    "पृथ्वी",
    "उजियाला",
    "अन्धियारे",
    "दिन",
    "सांझ",
    "पहिला",
    "जल",
    "ऊपर",
    "आकाश",
]
fig, ax = plt.subplots(figsize=(6, 5))  # set size

ax = dendrogram(linkageMatrix, orientation="left", labels=titles)

plt.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom="off",  # ticks along the bottom edge are off
    top="off",  # ticks along the top edge are off
    labelbottom="off",
)
plt.grid()
plt.yticks(fontproperties=hindiFont, fontsize=15)
plt.tight_layout()  # show plot with tight layout
plt.show()

"""# Resource-based Sentiment Analysis
#### Hindi WordNet
Hindi WordNet (developed by IIT Bombay) is a similar resource like the WordNet in English, which captures lexical and semantic relations between Hindi words. Here we're only concerned with sentiments so I am using **SentiWordNet**, which contain sentiment polarity of Hindi words while clubbing their synonyms also.<br>
Reading text file `HSWN_WN.txt` containing Hindi SentiWordNet into a Pandas Dataframe **data**.
"""

data = pd.read_csv(
    datasetPath["sentiment word net"],
    delimiter=" ",
    names=["POS TAG", "HWN ID", "+ve score", "-ve score", "Related words"],
    header=None,
)
data.head()

wordsDict = {}
for i in data.index:
    words = data["Related words"][i].split(",")
    for word in words:
        wordsDict[word] = (
            data["POS TAG"][i],
            data["+ve score"][i],
            data["-ve score"][i],
        )

print(f"The size of the Hindi Sentiment Word Net : {len(wordsDict)} words")

"""Clearly, this small amount of words won't be enough.
### Using Machine Translation to improve our sentiment resource
`hindi word list.csv` is just a list of Hindi words and their Hindi Translations, translated using Google's translation API. We're using *TextBlob* to get sentiment polarity of these translated Hindi words. This way we have more Hindi words to check sentiment polarity from.
"""

from textblob import TextBlob

posData = pd.read_csv(datasetPath["word list"], header=None, names=["Hindi", "English"])
polarityList = []
for i in posData["English"].tolist():
    blob = TextBlob(i)
    polarityList.append(blob.sentiment.polarity)
posData["polarity"] = polarityList
posData.head()

sentimentResource = set(list(wordsDict.keys()) + list(posData["Hindi"]))
print(
    f"We have {len(set(filteredStopwords))} unique words without stopwords in Hindi Bible"
)
print(
    f"And we have total {len(sentimentResource)} unique words in our sentiment resources"
)

remaining = [i for i in set(filteredStopwords) if i not in sentimentResource]
print(
    f"The remaining words i.e. total unique words - (Sentiment Resource): {len(set(remaining))} words"
)


def sentiment(text):
    words = tokenizeWord(text)
    words = [i for i in words if i not in customStopwords]
    posPolarity = 0
    negPolarity = 0
    # adverbs, nouns, adjective, verb are only used
    allowedWords = ["a", "v", "r", "n"]
    for word in words:
        if word in wordsDict:
            # if word in dictionary, it picks up the positive and negative score of the word
            posTag, pos, neg = wordsDict[word]
            if posTag in allowedWords:
                if pos > neg:
                    posPolarity += pos
                elif neg > pos:
                    negPolarity += neg
        elif word in posData["Hindi"]:
            polarity = posData[posData["Hindi"] == word]["polarity"]
            if polarity >= 0:
                posPolarity += polarity
            elif polarity < 0:
                negPolarity += polarity

    # calculating the no. of positive and negative words in total in a review to give class labels
    if posPolarity > negPolarity:
        return 1, posPolarity
    else:
        return 0, -negPolarity


print("For statment: मैं इस उत्पाद से बहुत खुश हूँ")
print(
    f"Overall sentiment and it's polarity is {sentiment('मैं इस उत्पाद से बहुत खुश हूँ')}"
)


fullList = []
bookFlag = range(66)
for j in bookFlag:
    chapterTxt = []
    for i in authorBible.index:
        if authorBible["Book"][i] == bookFlag[j]:
            chapterTxt.append(authorBible["Text"][i])
    fullList.append("".join(chapterTxt))

print(f"Length of the resulting list: {len(fullList)}")

books = [i.strip('"') for i in books[0:66]]
print(books)


polarityList = []
for i in fullList:
    polarityList.append(sentiment(i)[1])

polarityDict = dict(zip(books, polarityList))
polarityDf = pd.DataFrame(
    {"Book": list(polarityDict.keys()), "Polarity": list(polarityDict.values())}
)

fig, ax = plt.subplots(figsize=(12, 23))
ax.barh(polarityDf["Book"], polarityDf["Polarity"], color="green")
ax.set_xlabel("Sentiment Scores", fontsize=15)
ax.set_ylabel("Book Name", fontsize=15)
ax.set_title("Cumulative Sentiment Score for each book", fontsize=20)
plt.yticks(list(polarityDict.keys()), fontproperties=hindiFont, fontsize=12)

# Natural Language Processing on Hindi Bible text + Sentiment Analysis and Classification

The [Ipynb Notebook](https://colab.research.google.com/drive/1KugDrrZM7zf1fqenUS6sMDn__rMbtQ8-#scrollTo=NjVPP6yiXudD) can be found here.


## Introduction

Analysis of Hindi Bible text of HHBD version from Bible Society of India (BSI) with NLP techniques. To perform sentiment analysis on all of the Bible books while also bringing to the surface some interesting findings based on facts such as which is the most significant verse based on the frequency of common words, who wrote most of the New Testament, etc. 

## Preprocessing of Data

The data was parsed into a **CSV** format. Most of the details like Book no., Chapter No. etc. were extracted from Verseid field in the file and some details like "Book names in Hindi" and "Authors' names" were added from external sources.

The original [datasets](https://www.kaggle.com/kapilverma/hindi-bible?select=HSWN_WN.txt) can be found here.

## Exploratory Visualization

First, classifying the book titles from **bible books.txt** to the New Testament and Old Testament for classification purpose in our incoming visualizations.

## WordCloud

Using **full text bible.txt** which is a compilation of whole Hindi text of the Bible to make the WordCloud. Can only be viewed in the [Colab Ipynb Notebook](https://colab.research.google.com/drive/1KugDrrZM7zf1fqenUS6sMDn__rMbtQ8-#scrollTo=NjVPP6yiXudD) due to limitations of my system. Not present in **main.py**

## Resource-based Sentiment Analysis

### Hindi Word Net

Hindi WordNet (developed by IIT Bombay) is a similar resource like the WordNet in English, which captures lexical and semantic relations between Hindi words. Using only **sentiment word net** as we are only concerned with the Sentiment Analysis of it. It contains sentiment polarity of Hindi words while clubbing their synonyms as well.
<br/>
Reading text file **sentiment word net.txt** containing Hindi Sentiment WordNet into a Pandas Dataframe data.

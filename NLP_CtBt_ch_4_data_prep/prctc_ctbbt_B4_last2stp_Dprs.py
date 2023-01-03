# Building a ChatBot with Deep-NLP


# Impoting the libraries
import numpy as np
import tensorflow as tf
import re
import time


# --------------------------     Part 1 : Data Preprocessing     ------------------------
# Importing the dataset
lines_in_cnvrstn = open("./cornell_movie_dialogs_corpus/movie_lines.txt", encoding='utf-8', errors='ignore').read().split('\n')
cnvrstn = open("./cornell_movie_dialogs_corpus/movie_conversations.txt", encoding='utf-8', errors='ignore').read().split('\n')


# Creating a dictionary that maps each linea and its ID
id_2_line = {};
for lyn in lines_in_cnvrstn:
    _line = lyn.split(" +++$+++ ")
    if len(_line) == 5:
        id_2_line[_line[0]] = _line[4] # creates the dicttionary
        # _line[0] is id "key in dictionary" and _line[4] is the line "as value of the key"


# Creating the list of all of the conversations
cnvrstn_ids = []
for cvstn in cnvrstn[:-1]:
    _cnvrstn =  cvstn.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    cnvrstn_ids.append(_cnvrstn.split(","))


# Getting seperately the questions and answers
raw_questn = []
raw_ans = []
for cvstn in cnvrstn_ids:
    for i in range(len(cvstn) - 1):
        raw_questn.append(id_2_line[cvstn[i]])  # using the "id_2_line" dictionary, by id-key
        raw_ans.append(id_2_line[cvstn[i+1]]) 
        # range(len(cvstn) - 1) is used because of cvstn[i+1]
        # notice we are using both "cnvrstn_ids" and "id_2_line"


# Doing the first cleaning of the texts
def clean_text(text):
    text = text.lower();
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text) 
    text = re.sub(r"she's", "she is", text) 
    text = re.sub(r"that's", "that is", text) 
    text = re.sub(r"what's", "what is", text) 
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text) 
    text = re.sub(r"\'ve", " have", text) 
    text = re.sub(r"\'re", " are", text) 
    text = re.sub(r"\'d", " would", text) 
    text = re.sub(r"won't", "will not", text) 
    text = re.sub(r"can't", "cannot", text) 
    text = re.sub(r"[-()#/@;:<>{}+=~|.?,]", "", text)
    return text


# Cleaning the questions 
clean_questn = []
for qes in raw_questn:
    clean_questn.append(clean_text(qes))

# Cleaning the answers
clean_ans = []
for aNs in raw_ans:
    clean_ans.append(clean_text(aNs))


# create a dictionary that maps each word to its number of occurrences.
word2count = {}
for questn in clean_questn:
    for word in questn.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for ans in clean_ans:
    for word in ans.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


# Creating two dictionaries that map the questions words and the answers word to a unique integers
threshold = 20
word_number = 0
questnWrd2Int = {}
# 'word' is "key" and 'count' is "value"
for word, count in word2count.items():
    if count >= threshold:
        questnWrd2Int[word] = word_number
        word_number += 1

word_number = 0
ansWrd2Int = {}
for word, count in word2count.items():
    if count >= threshold:
        ansWrd2Int[word] = word_number
        word_number += 1


# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for tkn in tokens:
    # adding 'token' as "key" and 'unique int' as "value"
    questnWrd2Int[tkn] = len(questnWrd2Int) + 1

for tkn in tokens:
    # adding 'token' as "key" and 'unique int' as "value"
    ansWrd2Int[tkn] = len(ansWrd2Int) + 1


# Creating inverse dictionary of the ansWrd2Int, notice ':' is used to crete dictionary
ans_Int_2_Wrd = {w_i: w for w, w_i in ansWrd2Int.items()}

# Adding EOS tokens to the end of every answers
for i in range(len(clean_ans)):
    clean_ans[i] += " <EOS>" # notice a space is added to seperate <EOS>

# python prctc_ctbbt.py
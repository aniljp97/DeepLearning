"""
Part 7:
There are comments marking each part of the problem. All results printed to the output.
"""

import nltk


f = open("nlp_input.txt", 'r')
text = f.read()

# Tokenized text to words
wtokens = nltk.word_tokenize(text)

# Applying lemmatization to each word
lem = nltk.stem.WordNetLemmatizer()
l = []
for w in wtokens:
    l.append(lem.lemmatize(w))
print("\tLemmatization:\n", l)
print()

# Finding all trigrams for the words
t = []
i = 0
while i < len(wtokens) - 2:
    t.append((wtokens[i], wtokens[i+1], wtokens[i+2]))
    i += 1
print("\tTrigrams:\n", t)
print()

# Extract the top 10 of the most repeated trigrams based on their count
t_dict = {}
for e in t:
    if(e in t_dict):
        t_dict[e] += 1
    else:
        t_dict[e] = 1
t_top_10 = dict(sorted(t_dict.items(), key=lambda x: x[1], reverse=True)[:10])
print("\tTop 10 of the most repeated trigrams based on their count:\n", t_top_10)
print()

# Find all the sentences with the most repeated tri-grams
s_tri = []
stokens = nltk.sent_tokenize(text)
for s in stokens:
    swtokens = nltk.word_tokenize(s)
    i = 0
    while i < len(swtokens) - 2:
        if (swtokens[i], swtokens[i + 1], swtokens[i + 2]) in t_top_10.keys():
            s_tri.append(s)
        i += 1
print("\tSentences with the most repeated tri-grams:\n", s_tri)
print()

# Extract those sentences and concatenate
all_s_tri = ""
for s in s_tri:
    all_s_tri += s

# Print the concatenated result
print("\tConcatenated sentences with most repeated tri-grams:\n", all_s_tri)
print()


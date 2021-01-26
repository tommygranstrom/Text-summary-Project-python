import numpy as np
import pandas as pd 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Loading wordembeddings")
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

print("Wordembeddings loaded")

def remove_stopwords(sen,stop_words):    
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def createVectorRep(textCentenses,stopwords):
    #Create vocabulary
    my_vocabulary = []
    from nltk.tokenize import word_tokenize
    for i in textCentenses:
        my_vocabulary.append(word_tokenize(i))
    my_vocabulary = [y for x in my_vocabulary for y in x] # flatten list
    my_vocabulary = list(set(my_vocabulary))

    #Create your vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words=stopwords,vocabulary=my_vocabulary)
    #Create tfidf vector rep
    smatrix = vectorizer.transform(textCentenses) #Sparse matrix

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer(norm="l2")
    tfidf_transformer.fit(smatrix)
    tf_idf_vectortt = tfidf_transformer.transform(smatrix)
    tf_idf_vector = tf_idf_vectortt.todense()
    return tf_idf_vector 


def makeTextSummary(text,textLanguage,sl):
    #Pre processing of raw text
    sentences = []
    from nltk.tokenize import sent_tokenize
    sentences.append(sent_tokenize(text)) #Sentences tokenize
    sentences = [y for x in sentences for y in x] # flatten list
    clean_sentences = [s.lower() for s in sentences] #Lower case ü
    clean_sentences = pd.Series(clean_sentences).str.replace("ü","u") #If german 
    clean_sentences = pd.Series(clean_sentences).str.replace("å", "a") # if Swedish
    clean_sentences = pd.Series(clean_sentences).str.replace("ä", "a")
    clean_sentences = pd.Series(clean_sentences).str.replace("ö", "o")
    clean_sentences = pd.Series(clean_sentences).str.replace("?", " ")
    clean_sentences = pd.Series(clean_sentences).str.replace("[^a-zA-Z]", " ") # Remove special charachters
    from nltk.corpus import stopwords
    stop_words = stopwords.words(textLanguage) #Load stop words for corresponding language
    
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    clean_sentences = [remove_stopwords(r.split(),stop_words) for r in clean_sentences]
    sentence_vectors2 = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors2.append(v)
    sim_mat2 = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat2[i][j] = cosine_similarity(sentence_vectors2[i].reshape(1,100), sentence_vectors2[j].reshape(1,100))[0,0]
    nx_graph2 = nx.from_numpy_array(sim_mat2)
    scores2 = nx.pagerank(nx_graph2)
    ranked_sentences2 = sorted(((scores2[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    #make nice output
    print("\n---")
    print("Text to summarize:")
    print(text)
    summary = ""
    for i in range(sl-1):
        summary = summary + " " + str(ranked_sentences2[i][1])
    print("\nSummary: ")
    print(summary)
    print("---\n")
    return summary

#Load article dataset
texts = []
summaries = []
with open("sumTexts.csv") as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        a = lines[i].split(";")
        texts.append(a[0])
        summaries.append(a[1])

for i in range(len(texts)):
    makeTextSummary(texts[i],'english',3)


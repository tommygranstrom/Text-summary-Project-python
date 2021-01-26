import numpy as np
import pandas as pd 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

    sentence_vectors1 = createVectorRep(clean_sentences,stop_words)#TFIDF
    # similarity matrix
    sim_mat1 = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat1[i][j] = cosine_similarity(sentence_vectors1[i], sentence_vectors1[j])[0,0]
    nx_graph1 = nx.from_numpy_array(sim_mat1)
    scores1 = nx.pagerank(nx_graph1)
    ranked_sentences1 = sorted(((scores1[i],s) for i,s in enumerate(sentences)), reverse=True)

    #make nice output
    print("\n---")
    print("Text to summarize:")
    print(text)
    summary = ""
    for i in range(sl-1):
        summary = summary + " " + str(ranked_sentences1[i][1])
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


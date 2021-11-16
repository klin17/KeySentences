

# from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import spacy
import os
import csv

nlp = spacy.load("en_core_web_sm")

# def read_article(file_name):
#     file = open(file_name, "r")
#     filedata = file.readlines()
#     article = filedata[0].split(". ")
#     sentences = []

#     for sentence in article:
#         print(sentence)
#         sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
#     sentences.pop() 
    
#     return sentences

# def sentence_similarity(sent1, sent2, stopwords=None):
#     if stopwords is None:
#         stopwords = []
 
#     sent1 = [w.lower() for w in sent1]
#     sent2 = [w.lower() for w in sent2]
 
#     all_words = list(set(sent1 + sent2))
 
#     vector1 = [0] * len(all_words)
#     vector2 = [0] * len(all_words)
 
#     # build the vector for the first sentence
#     for w in sent1:
#         if w in stopwords:
#             continue
#         vector1[all_words.index(w)] += 1
 
#     # build the vector for the second sentence
#     for w in sent2:
#         if w in stopwords:
#             continue
#         vector2[all_words.index(w)] += 1
 
#     return 1 - cosine_distance(vector1, vector2)
 
# def build_similarity_matrix(sentences, stop_words):
#     # Create an empty similarity matrix
#     similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
#     for idx1 in range(len(sentences)):
#         for idx2 in range(len(sentences)):
#             if idx1 == idx2: #ignore if both are same sentences
#                 continue 
#             similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

#     return similarity_matrix


# def generate_summary(file_name, top_n=5):
#     stop_words = stopwords.words('english')
#     summarize_text = []

#     # Step 1 - Read text anc split it
#     sentences =  read_article(file_name)

#     # Step 2 - Generate Similary Martix across sentences
#     sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

#     # Step 3 - Rank sentences in similarity martix
#     sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
#     scores = nx.pagerank(sentence_similarity_graph)

#     # Step 4 - Sort the rank and pick top sentences
#     ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
#     print("Indexes of top ranked_sentence order are ", ranked_sentence)    

#     for i in range(top_n):
#       summarize_text.append(" ".join(ranked_sentence[i][1]))

#     # Step 5 - Offcourse, output the summarize texr
#     print("Summarize Text: \n", ". ".join(summarize_text))

def baseline(path, top_n=3):
    summarize_text = []

    # Step 1 - Read text and split it
    sentences = []
    vectors = []
    with open(path, "r", encoding="utf-8") as file:
        lines = [l.strip() for l in file.readlines() if len(l.strip()) > 0]
        for doc in nlp.pipe(lines):
            for sent in doc.sents:
                sentences.append(sent.text)
                vectors.append(sent.vector)
        
    # Step 2 - Generate Similary Martix across sentences
    def find_similarity(s1idx, s2idx):
        return 1 - cosine_distance(vectors[s1idx], vectors[s2idx])

    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = find_similarity(idx1, idx2)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    # print("loop starting {}".format(top_n))
    for i in range(top_n):
        # print("appending {}".format(i))
        summarize_text.append(ranked_sentence[i][1])

    # # Step 5 - Offcourse, output the summarize texr
    # print("Summarize Text: \n", " ".join(summarize_text))
    return summarize_text

def get_key_sentences(labeled_path):
    key_sentences = []
    with open(labeled_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            sentence, label = line["sentence"], line["label"]
            if label == "True":
                key_sentences.append(sentence)
    return key_sentences

# def compare(labeled):


if __name__ == "__main__":
    print("finished imports")

    raw_name = "000c835555db62e319854d9f8912061cdca1893e"
    story_file = os.path.join("stories", raw_name + ".txt")
    labeled_file = os.path.join("labeled_data2_0.65", raw_name + "_labeled.csv")

    key_sentences = get_key_sentences(labeled_file)
    print("key sentences:")
    # print(key_sentences)
    for sentence in key_sentences:
        print(repr(sentence))
    top_n = len(key_sentences)

    print("finished extracting key sentences")

    predicted = baseline(story_file, top_n=top_n)

    # print("predicted:")
    # print(predicted)

    print("results")
    for sentence in predicted:
        print(repr(sentence))
        print(sentence in key_sentences)
    

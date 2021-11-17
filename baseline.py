

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
    found_highlight = False
    with open(path, "r", encoding="utf-8") as file:
        lines = [l.strip() for l in file.readlines() if len(l.strip()) > 0]
        for doc in nlp.pipe(lines):
            for sent in doc.sents:
                if sent.text == "@highlight":
                    # print("found highlight")
                    found_highlight = True
                if not found_highlight:
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

    # print(sentences)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    # print("loop starting {}".format(top_n))
    if top_n >= len(ranked_sentence):
        print("WARN: n capped from {} to {}".format(top_n, len(ranked_sentence)))
        top_n = len(ranked_sentence)
    for i in range(top_n):
        # print("appending {}".format(i))
        summarize_text.append(ranked_sentence[i][1])

    # # Step 5 - Offcourse, output the summarize texr
    # print("Summarize Text: \n", " ".join(summarize_text))
    return summarize_text, len(sentences)

def get_key_sentences(labeled_path):
    key_sentences = []
    with open(labeled_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            sentence, label = line["sentence"], line["label"]
            if label == "True":
                key_sentences.append(sentence)
    return key_sentences


if __name__ == "__main__":
    print("finished imports")

    data_dir = "labeled_data2_0.7"
    files = os.listdir(data_dir)
    num_files = len(files)
    count = 0
    scores = []
    for file in files[:500]:
        if count % 10 == 0:
            print(count)
        basename = os.path.basename(file)
        pre, ext = os.path.splitext(basename)
        raw_name = pre.split("_")[0]

        story_file = os.path.join("stories", raw_name + ".txt")
        labeled_file = os.path.join("labeled_data2_0.65", raw_name + "_labeled.csv")

        key_sentences = get_key_sentences(labeled_file)
        # print("key sentences:")
        # print(key_sentences)
        # for sentence in key_sentences:
        #     print(repr(sentence))
        top_n = k = len(key_sentences)

        # print("finished extracting key sentences")

        predicted, s = baseline(story_file, top_n=top_n)

        # print("predicted:")
        # print(predicted)

        # print("results")
        tp = 0
        for sentence in predicted:
            # print(repr(sentence))
            # print(sentence in key_sentences)
            if sentence in key_sentences:
                tp += 1
        scores.append(tp/(k))
        count += 1
    print(np.mean(scores))


    

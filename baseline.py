

# from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import spacy
import os
import csv
import time

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

def find_similarity(s1idx, s2idx, vs):
    return 1 - cosine_distance(vs[s1idx], vs[s2idx])

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
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = find_similarity(idx1, idx2, vectors)

    # print(sentences)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    try:
        # scores = nx.pagerank_numpy(sentence_similarity_graph)
        scores = nx.pagerank(sentence_similarity_graph, max_iter=1000)
        # print("success: {}".format(len(sentences)))
    except:
        # print("failed: {}".format(len(sentences)))
        # print(sentences)
        return False, len(sentences)

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

    data_dir = "labeled_data2_0.8"
    story_dir = "stories"

    files = os.listdir(data_dir)
    num_files = len(files)
    print("Total files: {}".format(num_files))
    count = 0
    score_sum = 0
    scores = []
    sum_tp = sum_fp = sum_tn = sum_fn = 0
    sum_k = 0
    num_fails = 0
    time0 = time.time()
    prev_time = time.time()
    for file in files:
        if count % 556 == 0:
            this_time = time.time()
            print("{}/{} = {}%, in {} seconds".format(count, num_files, 100*count/num_files, this_time - prev_time))
            prev_time = this_time
        basename = os.path.basename(file)
        pre, ext = os.path.splitext(basename)
        raw_name = pre.split("_")[0]

        story_file = os.path.join(story_dir, raw_name + ".txt")
        labeled_file = os.path.join(data_dir, raw_name + "_labeled.csv")

        key_sentences = get_key_sentences(labeled_file)
        # print("key sentences:")
        # print(key_sentences)
        # for sentence in key_sentences:
        #     print(repr(sentence))
        top_n = k = len(key_sentences)

        # print("finished extracting key sentences")

        predicted, s = baseline(story_file, top_n=top_n)

        if not predicted:
            num_fails += 1
            continue

        # print("predicted:")
        # print(predicted)

        # print("results")
        tp = fp = 0
        for sentence in predicted:
            # print(repr(sentence))
            # print(sentence in key_sentences)
            if sentence in key_sentences:
                tp += 1
            else:
                fp += 1
        fn = len(set(key_sentences) - set(predicted))
        tn = s - (fn + tp + fp)

        score_sum += tp/k
        
        sum_tp += tp
        sum_fp += fp
        sum_tn += tn
        sum_fn += fn
        sum_k += k
        scores.append(tp/(k))
        count += 1
    print("finished {} files".format(count))
    print("total time: {} seconds".format(prev_time - time0))
    print("Failures: {}".format(num_fails))
    print("average: {}".format(np.mean(scores)))
    print(sum_tp, sum_tn, sum_fp, sum_fn, sum_k)
    print(sum_tp + sum_tn + sum_fp + sum_fn)
    print(score_sum / count)
    print("accuracy: sum_tp/sum_k = {}".format(sum_tp/sum_k))
    precision = sum_tp/(sum_tp + sum_fp)
    print("precision: tp/(tp+fp) = {}".format(precision))
    recall = sum_tp/(sum_tp + sum_fn)
    print("recall: tp/(tp+fn) = {}".format(recall))
    print("f1: 2*precision*recall/(precision + recall) = {}".format(2*precision*recall/(precision + recall)))



    

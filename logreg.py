
import spacy
import os
import csv
import numpy as np
import math
import time
#import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler

nlp = spacy.load("en_core_web_sm")

#doc will be in csv form, gets labels and raw sentences
def get_doc_raw_data(lab_doc_path):
    sentences = []
    labels = []
    with open(lab_doc_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            sentences.append(line["sentence"])
            labels.append(line["label"])
    return sentences, labels

def get_cosine_sim(a, b):
    norma = norm(a)
    return dot(a, b)/(norma*norma)

def get_list_for_cont_n(sent_embeds, n):
    to_ret = []
    for i in range(len(sent_embeds)):
        low_bound = max(0, i - n)
        up_bound = min(len(sent_embeds) - 1, i + n)
        cont_window = sent_embeds[low_bound:(up_bound+1)]
        cont_window_vect = np.mean(cont_window, axis = 0).tolist()
        to_ret.append(get_cosine_sim(cont_window_vect, sent_embeds[i]))
    return to_ret
        
def get_doc_LR_inputs(lab_doc_path):
    raw_sents, raw_labels = get_doc_raw_data(lab_doc_path)
    #vector_sents_list = [nlp(sent).vector.reshape(1,-1) for sent in raw_sents]
    nlp_sents_list = list(nlp.pipe(raw_sents))
    embed_sents_list = [nlp_sent.vector.tolist() for nlp_sent in nlp_sents_list]
    num_n_ents_list = [len(nlp_sent.ents) for nlp_sent in nlp_sents_list]
    sent_lengths_list = [len(nlp_sent) for nlp_sent in nlp_sents_list]
    
    sent_lengths_bychar_list = [len(raw_sent) for raw_sent in raw_sents]
    
    sent_char_tok_ratio_list = []
    #print(type)
    # for i in range(len(raw_sents)):
    #     sent_char_tok_ratio_list.append(sent_lengths_bychar_list[i]/sent_lengths_list[i])
    for charlen, toklen in zip(sent_lengths_bychar_list, sent_lengths_list):
        sent_char_tok_ratio_list.append(charlen / toklen)
    
    doc_vector = np.mean(embed_sents_list, axis = 0).tolist()
    doc_sim_list = [get_cosine_sim(doc_vector, sent_vector) for sent_vector in embed_sents_list]
    
    # n = 1, 2, 3, 4, 5
    cont_winds_lists = [get_list_for_cont_n(embed_sents_list, n) for n in range(1, 6)]
    
    feats = []
    feats.append(num_n_ents_list)
    feats.append(sent_lengths_list)
    feats.append(sent_lengths_bychar_list)
    feats.append(sent_char_tok_ratio_list)
    feats.append(doc_sim_list)
    
    feats.extend(cont_winds_lists)
    # embed_feats = np.array(embed_sents_list).T.tolist()
    # feats.extend(embed_feats)
            
    vector_sents_matrix = np.array(feats).T.tolist()
    
    return raw_labels, vector_sents_matrix

#testing on single doc
def LR_on_doc(lab_doc_path):
    labels, sents_matrix = get_doc_LR_inputs(lab_doc_path)
    X_train, X_test, y_train, y_test = train_test_split(sents_matrix, labels, random_state=42)
    LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', fit_intercept=False)
    LR.fit(X_train, y_train)
    pred_labels = LR.predict(X_test)
    actual_labels = y_test
    overall_test_results = []
    for i in range(len(pred_labels)):
        overall_test_results.append({'Number': i, 'pred': pred_labels[i], 'act': actual_labels[i]})
    for result in overall_test_results:
        print(result)
        print()
        
def LR_on_first_n(dir_name, n, train_ratio):
    raw_file_names = os.listdir(dir_name)[0:n]
    LR_x_inputs = []
    LR_labels = []
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    time0 = time.perf_counter()
    t_prev = time.perf_counter()
    train_cutoff = range(math.floor(train_ratio * n))
    print(f"{train_cutoff=}")
    for c, raw_name in enumerate(raw_file_names):
        file_path = os.path.join("labeled_data2_0.8", raw_name)
        doc_labels, doc_matrix = get_doc_LR_inputs(file_path)
        LR_x_inputs.append(doc_matrix)
        LR_labels.append(doc_labels)
        if c in train_cutoff:
            X_train.append(doc_matrix)
            y_train.append(doc_labels)
        else:
            X_test.append(doc_matrix)
            y_test.append(doc_labels)
        # c += 1
        if c % 453 == 0:
            t_now = time.perf_counter()
            print("finished {}/{}={:.5f}%, in {:.5f} s".format(c, n, 100*c/n, t_now - t_prev))
            t_prev = t_now
    time1 = time.perf_counter()
    print("Created features in {:.5f} s".format(time1 - time0))
    #X_train, X_test, y_train, y_test = train_test_split(LR_x_inputs, LR_labels, random_state=42)
    
    
    X_train_list = []
    for doc_matrix in X_train:
        for sent_vector in doc_matrix:
            X_train_list.append(sent_vector)
    X_train_matrix = X_train_list
    #X_train_matrix = np.concatenate(X_train_list)
    y_train_list = []
    for doc in y_train:
        for label in doc:
            y_train_list.append(label)
    LR = LogisticRegression(multi_class='multinomial', fit_intercept=False, solver='lbfgs', penalty='l2')
    
    scaler = StandardScaler()
    X_train_matrix = scaler.fit_transform(X_train_matrix)
    #X_test = scaler.transform(X_test)
    LR.fit(X_train_matrix, y_train_list)
    
    num_sentences = 0
    # num_act_false = 0
    num_act_true = 0
    # num_pred_false = 0
    num_pred_true = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0

    # pred_labels = [LR.predict(scaler.transform(xtest)) for xtest in X_test]
    # act_labels = y_test
    pred_labels = []
    act_labels = []
    for i in range(len(X_test)):
        X_test[i] = scaler.transform(X_test[i])
        pred_labels.append(LR.predict(X_test[i]))
        act_labels.append(y_test[i])

    for predictions, actuals in zip(pred_labels, act_labels):
        for pred, actual in zip(predictions, actuals):
            if(pred == 'True' and actual == 'True'):
                tp += 1
            elif(pred == 'True' and actual == 'False'):
                fp += 1
            elif(pred == 'False' and actual == 'True'):
                fn += 1
            elif(pred == 'False' and actual == 'False'):
                tn += 1

            if(pred == 'True'):
                num_pred_true += 1
            # elif(pred == 'False'):
            #     num_pred_false += 1
            # else:
            #     print(f"{pred=}")

            if(actual == 'True'):
                num_act_true += 1
            # elif(actual == 'False'):
            #     num_act_false += 1
            # else:
            #     print(f"{actual=}")

            num_sentences += 1

    timefinish = time.perf_counter()
    print("Total time: {:.5f}".format(timefinish - time0))
    print(f"{num_sentences=}")
    print(f"{num_act_true=}")
    print(f"{num_pred_true=}")
    # print(f"{num_act_false=}")
    # print(f"{num_pred_false=}")
    # print(f"actuals: {num_act_false + num_act_true}")
    # print(f"preds: {num_pred_false + num_pred_true}")
    print(f"{fp=}, {fn=}, {tp=}, {tn=}")
    print(f"Accuracy = {(tp + tn) / (tp + tn + fn + fp)}")
    precision = tp / (tp + fp)
    print(f"Precision = {precision}")
    recall = tp / (tp + fn)
    print(f"Recall = {recall}")
    print(f"F1 = {2 * precision * recall / (precision + recall)}")
    return pred_labels, act_labels, LR.coef_
#def get_train_set()

if __name__ == "__main__":
    dir_name = "labeled_data2_0.8"
    train_split = 0.75
    # n = 22651   # all the files = 22651
    n = 22651
    print(f"Running on {n} files")
    pred_labels, act_labels, coefs = LR_on_first_n(dir_name, n, train_split)


    # import os
    # import math
    # n = 22651
    # split_index = math.floor(.75 * n)
    # dir_name = "labeled_data2_0.8"
    # raw_name = os.listdir(dir_name)[100]
    # #story_file = os.path.join("stories", raw_name + ".txt")
    # labeled_file = os.path.join("labeled_data2_0.8", raw_name)
    # #from Log_Reg_Rel_Functions import *
    # #LR_on_doc(labeled_file)
    # train_set = []

    # for i in range(split_index):
    #     train_set.append(i)

    # results = LR_on_first_n(dir_name, n, train_set)
    # pred_labels = results[0]
    # act_labels = results[1]
    # coefs = results[2]

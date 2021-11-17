
import os
import spacy
import numpy as np
# import re
import csv
import time

# returns two lists: story (all the non highlight lines which are not '@highlight') 
# and the highlights (all the lines that follow @highlight)
# note: story contains the lines in the original file, not necessarily separated into sentences
def separate_highlights(filereader):
    highlights = []
    story = []
    was_highlight = False
    for line in filereader:
        stripped = line.strip()
        if len(stripped) > 0:
            # assumes that highlights always occur after the story
            if was_highlight:
                highlights.append(stripped)
            if len(highlights) == 0 and stripped != "@highlight":
                story.append(stripped)
            was_highlight = stripped == "@highlight"
    return story, highlights

# uses spacy pipeline to sentize and vectorize sentences in story and highlights
def sent_and_vectorize(story, highlights, nlp):
    highlight_vectors = [d.vector for d in nlp.pipe(highlights)]
    # story_vectors = [d.vector for d in nlp.pipe(story)]
    story_vectors = []
    story_sents = []
    for doc in nlp.pipe(story):
        for sent in doc.sents:
            story_sents.append(sent.text)
            story_vectors.append(sent.vector)
    return story_sents, story_vectors, highlight_vectors

# generates a list of csv writer rows for each sentence along with 
# labels donoting whether sentence is key based on cosine similarity above some threshold
def label_sents(story_sents, story_vectors, highlight_vectors, threshold):
    similarities = []
    stacked = np.stack(highlight_vectors)
    scaling = np.sqrt(np.sum(np.square(stacked), axis=1))

    labeled_data = []
    sim_count = 0
    for i, vec in enumerate(story_vectors):
        normalized_vec = vec / np.sqrt(np.sum(vec * vec))
        dots = stacked @ normalized_vec
        similarities = dots / scaling
        sim_idx = np.argmax(similarities)
        similarity = similarities[sim_idx]
        # similar_highlight = highlights[sim_idx]
        if similarity > threshold:
            # print(story[i])
            # print("is similar to: ", similar_highlight)
            sim_count += 1
            labeled_data.append({"sentence": story_sents[i], "label": "True"})
        else:
            labeled_data.append({"sentence": story_sents[i], "label": "False"})
    
    if sim_count < 1:
        # print("WARN: no key sentences found with threshold {}".format(threshold))
        # print("skipping file")
        return False

    return labeled_data

# converts the .txt file with @highlight tags into 
# a labeled csv file with one sentence per line, labeled with True if sentence is key, else False
def process_file(path, out_path, threshold):
    with open(path, encoding="utf-8") as f:
        story, highlights = separate_highlights(f)
        story_sents, story_vectors, highlight_vectors = sent_and_vectorize(story, highlights, nlp)
        
        # for s in story_sents:
        #     print(s)

        labeled_data = label_sents(story_sents, story_vectors, highlight_vectors, threshold)

        if not labeled_data:
            # print("skipping file: {}".format(path))
            return False

        # write the result to file
        with open(out_path, "w", encoding="utf-8") as outf:
            fieldnames = ['sentence', 'label']
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(labeled_data)
        return True

if __name__ == "__main__":
    time0 = time.time()
    # load spaCy
    nlp = spacy.load("en_core_web_sm")

    for threshold in [0.7]:

        data_dir = "stories"
        out_dir = "labeled_data2_" + str(threshold)

        # note: each txt file in data_dir is a single story
        # note: each story has multiple highlights labeled by @highlight
        num_fails = 0
        files = os.listdir(data_dir)
        num_files = len(files)
        print("There are {} files".format(num_files))
        count = 0
        prev_time = time.time()
        for file in files:
            count += 1
            # create the file path
            path = os.path.join(data_dir, file)
            pre, ext = os.path.splitext(file)
            out_path = os.path.join(out_dir, pre + "_labeled.csv")
            # process the file
            # print(path)
            success = process_file(path, out_path, threshold)
            if not success:
                num_fails += 1
            if count % 2314 == 0:
                this_time = time.time()
                print("Finished {} files: {}/{} = {}%, in {} seconds".format(count, count, num_files, 100*count/num_files, this_time - prev_time))
                prev_time = this_time
        
        print("Produced {} files with key sentences".format(num_files - num_fails))
        print("Failed on {} files".format(num_fails))
        print(f"{100*(num_files - num_fails)/num_files}% failure rate")
        print(f"{100*num_fails/num_files}% failure rate")
        print("total time: {} seconds".format(prev_time - time0))
    # print("Failed to find key sentences for {} files out of {} total".format(num_fails, num_files))
    

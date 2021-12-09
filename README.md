# KeySentences
Project to extract key sentences from a text.


## Requirements
- spaCy, including the `en_core_web_sm` pipeline
- numpy
- nltk
- networkx


## Data Set Acquisition and Transformation
Download using the cnn "stories" linked here: https://cs.nyu.edu/~kcho/DMQA/ which should download as "cnn_stories.tgz"
1. Unzip the .tgz
1. Convert each .story file into a .txt file
1. For each file:
    - Separate the story from the labeled highlights
    - Use spacy on each line of the story, pulling out the sentences (doc.sents) and the vector of each sentence (sent.vector)
    - Use spacy on each highlight to convert into vectors (doc.vector)
    - Generate cosine similarities between each sentence and highlights
    - Use max similarity as similarity score for each sentence
    - Denote each sentence with similarity score above some threshold as "key sentences" ie label=True, else denote as "non key sentence" ie label=False
    - Create a text file with (label, sentence) pairs

Step 2 can be done using change_extensions.py
Useage: 

`python change_extensions.py dirToAlter fromExt toExt`

eg:

```python change_extensions.py cnn_stories .story .txt```

Step 3 can be done using data_cleaning.py (which takes around 100 minutes per run)
Useage:

`python data_cleaning.py`

[//]: # (these are comments)

[//]: # (Results for data_cleaning with different thresholds:)
[//]: # (Threshold | Files Produced | Files Failed | Percent Retention |)
[//]: # (---|---|---|---)
[//]: # (0.65 | 60961 | 9894 | 86.04%)
[//]: # (0.7 | 48271 | 22584 | 68.13%)
[//]: # (0.8 | 17670 | 53185 | 24.94%)
[//]: # (0.9 | 3513 | 67340 | 4.96%)

Revised Results for data_cleaning with different thresholds:
Threshold | Files Produced | Files Failed | Percent Failure | Time (s)
---|---|---|---|---
0.65 | 79459 | 13120 | 14.171680402683114% | 8938
0.7 | 62674 | 29905 | 32.30214195443891% | 7738
0.75 | 41380 | 51199 | 55.30303848604975% | 11460
0.8 | 22651 | 69928 | 75.53332829259335% | 10209\*
0.9 | 4523 | 88056 | 95.11444280020307% | 9211

\* estimated time (output of 31309 includes time when program was paused)

Took around 150 min per run

## Baseline

The baseline is a model that uses pagerank on sentence vectors and selects the top n ranked sentences
where n is the number of key sentences in the article.

### Steps:
1. Read the input file and split the text into sentences and vectors for each sentence
2. Generate a similarity matrix holding the cosine similarities between each sentence using the vectors
3. Use PageRank on this matrix to find the most important sentences
4. Output the top n sentences from PageRank

### Analysis:
#### Raw output:
pagerank_numpy():
```
2822 684393 26455 26446 29277
740116
accuracy: sum_tp/sum_k = 0.09638965741025378
precision: tp/(tp+fp) = 0.09638965741025378
recall: tp/(tp+fn) = 0.0964192975263086
f1: 2*precision*recall/(precision + recall) = 0.09640447519002478
```

pagerank():
```
Failures: 705
average: 0.07432243896315996
2761 660491 25593 25584 28354
714429
0.07432243896315983
accuracy: sum_tp/sum_k = 0.09737603160047965
precision: tp/(tp+fp) = 0.09737603160047965
recall: tp/(tp+fn) = 0.09740695007937908
f1: 2*precision*recall/(precision + recall) = 0.09739148838603855
```

The numbers in the first line are tp, tn, fp, fn, k, respectively, with k being the total number of key sentences

Since we use the number of key sentences in the article as n, the number of false positives and false negatives is exactly the same. Thus the precision and recall tell us the same information: how many of the top n ranked sentences from pagerank are actually key sentences on average. Accuracy, true positives / total number of actual positives, is not useful since there are not many key sentences in general.

The baseline took around 33 minutes to run on 22651 documents.

## Logistic Regression
The actual model uses Logistic Regression to do binary classification on sentences of an article.

### Features:
- number of named entities
- number of tokens in the sentence
- number of chars in the sentence
- character to token ratio
- similarity to document
- similarity to a context (with window sizes 1 through 5)
- sentence embedding

### Analysis:
#### Raw output:
With all features:
```
Total time: 1728.25267
num_sentences=185567
num_act_true=7277
num_pred_true=87504
num_act_false=178290
num_pred_false=98063
actuals: 185567
preds: 185567
fp=81759, fn=1532, tp=5745, tn=96531
Accuracy = 0.5511540306196684
Precision = 0.06565414152495885
Recall = 0.7894736842105263
F1 = 0.12122682816176236
```

Without embeddings:
```
Created features in 1758.85846 s
Total time: 1764.96069
num_sentences=185567
num_act_true=7277
num_pred_true=81537
num_act_false=178290
num_pred_false=104030
actuals: 185567
preds: 185567
fp=76443, fn=2183, tp=5094, tn=101847
Accuracy = 0.5762931986829555
Precision = 0.062474704735273556
Recall = 0.7000137419266181
F1 = 0.11471164456054224
```

Only embeddings:
```
185567
7277
19
17
7275
2
178273
```
In order: total number of sentences, number of highlights, number of sentences predicted as highlights, fp, fn, tp, tn.

Takes around 30 minutes to run on 22651 documents.
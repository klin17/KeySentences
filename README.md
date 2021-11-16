# KeySentences
Project to extract key sentences from a text.


## Requirements
- spaCy, including the `en_core_web_sm` pipeline
- numpy


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

Results for data_cleaning with different thresholds:
Threshold | Files Produced | Files Failed | Percent Retention |
---|---|---|---
0.65 | 60961 | 9894 | 86.04%
0.7 | 48271 | 22584 | 68.13%
0.8 | 17670 | 53185 | 24.94%
0.9 | 3513 | 67340 | 4.96%

Revised Results for data_cleaning with different thresholds:
Threshold | Files Produced | Files Failed | Percent Failure | Time (s)
---|---|---|---|---
0.65 | 79459 | 13120 | 14.171680402683114% | 8938

Took around 150 min per run

## Initial Testing
Will be done in `testing.py`
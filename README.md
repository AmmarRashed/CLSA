## Ammar Rashed.

## Abstract

The project is about using multiple labelled textual corpora from different languages to leverage
the representation power of resources-rich “source” languages (e.g. English) into enhancing the models of
“target” languages that are relatively less available (e.g. Bengali, Swahili, ...etc.). This project focuses on
sentiment classification of movie reviews collected from IMBD for English as the source language, and
Beyazperde for Turkish as the target language. The aim of the project is not to learn a new word
embeddings for the target language, but to rather use the existing ones (e.g. Fasttext and Google’s
word2vec) to generalize the sentiment knowledge from the source language to the target language. One
caveat is that, the motivation of this project is to mitigate the effect of data scarcity in many languages
(e.g. Swahili, Bengali ...etc.) and not to obtain an optimal representation. That being said, given enough
data, existing algorithms (e.g. Logistic Regression, RandomForest ...etc. using only doc2vec or BOW for
sentence representation) will suffice in learning reasonably powerful sentiment classification models.
Turkish, though does not have as abundant textual resources as English for example, has arguably big-
enough corpus to train powerful word embeddings that are per se capable of achieving satisfactory
results in sentiment classification tasks. However, my aim in this project is trying new approaches and
architectures that have not been tried before, improving them to the degree that they perform at least as
good as monolingual classifiers. In this project, I show that using languages that are rich with labelled
corpora can make up for the scarcity of their relatively poor peer languages in labelled corpora, and, thus,
eliminating the need for tedious tasks of creating and collecting large labelled corpora.

## Performance Metrics:

Scores given to the movie reviews range from 1 to 10. For ratings with floating numbers and
ratings that range from 1 to 5, I scale the ratings floor them so that all ratings range from 1 to 10. For the
classification accuracy, I take the mean of the absolute difference between predicted scores and actual
scores of testing movie reviews. Code 1. To insure the robustness of validation, I run 10-fold test for 10
trials. I use different random state for each trial. However, I noticed that all trials gave exactly similar
results. Therefore, I will report only the scores of each of the 10-folds for just one trial.


## Data:

Movie Reviews
- English: <a href="https://www.imdb.com/">IMDB</a> 500 x reviews.
- Turkish: <a href="http://www.beyazperde.com/"> BeyazPerde</a> 500 x reviews. Stemmed (okullarından > okul) and cleaned (Ö, Ü > O, U)

*The approaches proposed in this project are not affected by the bias in the ratio of Score.

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/data_stats.png?raw=true">

## Language Models

- <a href="https://code.google.com/archive/p/word2vec/"> Google's word2vec (English) </a>
- <a href="https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md"> Facebook's Fasttext (English and Turkish)</a>


<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/data_reps.png?raw=true">

## Approaches

### Selective Waves

- Overview

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/matrices.png?raw=true">

- Merging step

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/high_level_merging2.png?raw=true">
<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/high_level_merging.png?raw=true">

- Training

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/training_reg_class.png?raw=true">

- Prediction

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/pred_reg_class.png?raw=true">



### Quadro-bridge

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/quadro_bridge.png?raw=true">


## Results 

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/perf.png?raw=true">


## Discussion

<img src="https://github.com/AmmarRashed/CLSA/blob/quadrobridge/misc/pros_cons.png?raw=true">

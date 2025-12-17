# Bigram language models & perplexity

This project implements a bigram language model using English Wikipedia corpus. The goal of the lab is to estimate bigram probabilities, commpute surprisal, and evaluate a model using perplexity on a held-out test set.

## Description

The lab explores how statistical language models are built from data and how their performance can be evaluated.
The program:
* Trains a bigram language model from text
* Computes conditional probabilities of word pairs
* Calculates surprisal for bigrams
* Applies LaPlace smoothing for unseen events
* Computes perplexity on a separate test corpus
* Evaluates how training data size affects model performance

The program includes functions to:
* Read and tokenize raw text
* Extract unigram frequencies
* Extract bigram frequencies
* Compute bigram probabilities
* Calculate surprisal values
* Handle unseen bigrams using LaPlace smooothing
* Compute total surprisal over a test set
* Calculate perplexity

### Dependencies

* A wiki.train and a wiki.test file are used
* Libraries used: nltk, collections, math, sys, time

### Executing program

* The program is executed from the command line and expects 2 arguments:
```
python3 lab2.py TRAIN_FILE TEST_FILE
```

## Authors

Charlotte, Natural Language Processing lab

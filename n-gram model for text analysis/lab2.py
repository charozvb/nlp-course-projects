from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
from math import log
import sys, time

def get_lines(path):
    with open(path, "r") as f:
        lines = f.read()
    return lines

def get_unigrams(lst_sentences):
    d_unigram = defaultdict(int)
    for sentence in lst_sentences:
        d_unigram['<s>'] += 1
        d_unigram['<e>'] += 1
        for words in word_tokenize(sentence):
            d_unigram[words] += 1
    return d_unigram

def get_bigrams(lst_sentences):
    d_bigram = defaultdict(int)
    for sentence in lst_sentences:
        sentence = word_tokenize(sentence)
        d_bigram[('<s>', sentence[0].lower())] += 1
        d_bigram[(sentence[-1], '<e>')] += 1
        for words in range(1, len(sentence)-1):
            bigram = sentence[words].lower(), sentence[words+1].lower()
            d_bigram[bigram] += 1
    return d_bigram

def get_surprisal(p):
    surprisal = - (log(p, 2))
    return surprisal

def get_bigram_surprisal(d_unigram, d_bigram):
    frequency = []
    d_surprisal = defaultdict(int)
    for unigram in d_unigram:
        for bigram in d_bigram:
            if bigram[0] == unigram:
                frequency = d_unigram.get(unigram)
                conditionalprob = ((d_bigram[bigram] + 1) / (frequency + len(d_unigram)))
                surprisal = get_surprisal(conditionalprob)
                d_surprisal[bigram] = surprisal
    return d_surprisal

def get_test_surp_unseen(unigram_freq, bigram):
    unseen_surprisal = - log(1/(unigram_freq[bigram[0]] + (len(unigram_freq))), 2)
    return unseen_surprisal

def get_perplexity(unigram_freq, bigram_surprisal, test_sentences):
    total_surp = 0.0
    word_count = 0
    for sentence in test_sentences:
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.append("<e>")
        for i in range(len(sentence)-1):
            bigram = sentence[i], sentence[i+1]
            word_count += 1
            if bigram in bigram_surprisal:
                total_surp += bigram_surprisal[bigram]
            else:
                total_surp += get_test_surp_unseen(unigram_freq, bigram)
    perplexity = 2**(total_surp/word_count)
    return perplexity

def main():
    train = get_lines(sys.argv[1])
    train_sentences = sent_tokenize(train)


    #print unigram:
    unigram_freq = get_unigrams(train_sentences)
    print(unigram_freq.get('around'))
    print(unigram_freq.get('select'))
    print(unigram_freq.get('<s>'))
    print(unigram_freq.get('<e>'))

    #print bigram:
    bigram_freq = get_bigrams(train_sentences)
    print(bigram_freq.get(('agreed','to')))
    print(bigram_freq.get(('into','aspects')))
    print(bigram_freq.get(('<s>','while')))
    print(bigram_freq.get(('.','<e>')))

    #print surprisal:
    print(get_surprisal(1)) 
    print(get_surprisal(0.5))
    print(get_surprisal(0.3))
    print(get_surprisal(0.1))

    bi_surp = get_bigram_surprisal(unigram_freq, bigram_freq)

    #print get_bigram_surprisal:
    print(bi_surp.get(('this','is')))
    print(bi_surp.get(('this','issue')))
    print(bi_surp.get(('and','why')))
    print(bi_surp.get(('which','can')))

    #perplexity:
    perplexity = get_perplexity(bi_surp, unigram_freq, train)
    print(perplexity)

if __name__=="__main__":
    main()



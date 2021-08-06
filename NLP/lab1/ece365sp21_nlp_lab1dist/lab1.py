from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import udhr

def get_freqs(corpus, puncts):
    freqs = {}
    ### BEGIN SOLUTION
    for c in puncts:
        corpus = corpus.replace(c,' ')
    for d in '1234567890':
        corpus = corpus.replace(d,' ')
    corpus = corpus.split()
    corpus = [w.lower() for w in corpus]
    for w in corpus:
        if w not in freqs:
            freqs[w] = 1
        else:
            freqs[w] += 1
    ### END SOLUTION
    return freqs

def get_top_10(freqs):
    top_10 = []
    ### BEGIN SOLUTION
    freqs = sorted(freqs.items(),key=lambda x:x[1],reverse=True)
    top_10 = [i[0] for i in freqs[:10]]
    ### END SOLUTION
    return top_10

def get_bottom_10(freqs):
    bottom_10 = []
    ### BEGIN SOLUTION
    freqs = sorted(freqs.items(),key=lambda x:x[1])
    bottom_10 = [i[0] for i in freqs[:10]]
    ### END SOLUTION
    return bottom_10

def get_percentage_singletons(freqs):
    ### BEGIN SOLUTION
    count = 0
    for i in freqs.values():
        if i == 1:
            count += 1
    perc = count/len(freqs)*100
    ### END SOLUTION
    return perc

def get_freqs_stemming(corpus, puncts):
    ### BEGIN SOLUTION
    porter = PorterStemmer()
    freqs = {}
    for c in puncts:
        corpus = corpus.replace(c,' ')
    for d in '1234567890':
        corpus = corpus.replace(d,' ')
    corpus = corpus.split()
    corpus = [porter.stem(w.lower()) for w in corpus]
    for w in corpus:
        if w not in freqs:
            freqs[w] = 1
        else:
            freqs[w] += 1
    ### END SOLUTION
    return freqs

def get_freqs_lemmatized(corpus, puncts):
    ### BEGIN SOLUTION
    wordnet_lemmatizer = WordNetLemmatizer()
    freqs = {}
    for c in puncts:
        corpus = corpus.replace(c,' ')
    for d in '1234567890':
        corpus = corpus.replace(d,' ')
    corpus = corpus.split()
    corpus = [wordnet_lemmatizer.lemmatize(w.lower(), pos="v") for w in corpus]
    for w in corpus:
        if w not in freqs:
            freqs[w] = 1
        else:
            freqs[w] += 1
    ### END SOLUTION
    return freqs

def size_of_raw_corpus(freqs):
    ### BEGIN SOLUTION
    ### END SOLUTION
    return len(freqs)

def size_of_stemmed_raw_corpus(freqs_stemming):
    ### BEGIN SOLUTION
    ### END SOLUTION
    return len(freqs_stemming)

def size_of_lemmatized_raw_corpus(freqs_lemmatized):
    ### BEGIN SOLUTION
    ### END SOLUTION
    return len(freqs_lemmatized)

def percentage_of_unseen_vocab(a, b, length_i):
    ### BEGIN SOLUTION
    ### END SOLUTION
    return len(set(a)-set(b))/length_i

def frac_80_perc(freqs):
    ### BEGIN SOLUTION
    freq = 0
    words = 0
    total = sum(freqs.values())
    freqs = sorted(freqs.items(),key=lambda x:x[1],reverse=True)
    for i in freqs:
        freq += i[1]
        words += 1
        if freq/total >= 0.8:
            break
    frac = words/len(freqs)
    ### END SOLUTION
    return frac

def plot_zipf(freqs):
    ### BEGIN SOLUTION
    word_rank = list(range(1,len(freqs)+1))
    freq = []
    freqs = sorted(freqs.items(),key=lambda x:x[1],reverse=True)
    for i in freqs:
        freq.append(i[1])
    plt.plot(word_rank,freq)
    plt.xlabel("rank of words")
    plt.ylabel("frequency")
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.

def get_TTRs(languages):
    TTRs = {}
    for lang in languages:
        words = udhr.words(lang)
        ### BEGIN SOLUTION
        n_types = []
        for i in range(100,1400,100):
            n_types.append(len(set([w.lower() for w in words[:i]])))
        TTRs[lang] = n_types
        ### END SOLUTION
    return TTRs

def plot_TTRs(TTRs):
    ### BEGIN SOLUTION
    for key in TTRs:
        plt.plot(range(100,1400,100),TTRs[key],label=key) 
    plt.xlabel('amount of tokens ')
    plt.ylabel('count of types')
    plt.title('TTR Plot')
    plt.legend()
    plt.xticks(range(100,1400,100))
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.

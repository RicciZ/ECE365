# ECE365
## Data Science and Engineering

### Fundamentals of ML
#### lab1
- numpy
- matrix

#### lab2
- Bayes Classifier
- LDA (Linear Discriminant Analysis)
    - sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
- K-NN (K Nearest Neighbors)
    - sklearn.neighbors.KNeighborsClassifier()

#### lab3
- Bernoulli Naive Bayes Classifier
    - sklearn.naive_bayes.BernoulliNB()
- Multinomial Naive Bayes Classifier
    - sklearn.naive_bayes.MultinomialNB()
- Support Vector Machine
    - sklearn.svm.LinearSVC()
    - sklearn.svm.SVC()
- Logistic Regression
    - sklearn.linear_model.LogisticRegression()
- K-NN
    - sklearn.neighbors.KNeighborsClassifier()
- Confusion Matrix
    - TP, FP, FN, TN
    - TPR = TP/T, FPR = FP/F
- Cross Validation
    - sklearn.model_selection.cross_val_score()
        - for one parameter in the classifier
    - sklearn.model_selection.GridSearchCV()
        - for multiple parameters in the classifier

#### lab4
- K-Means
    - sklearn.cluster.KMeans()
- Compress Image with K-Means
- Generate Prototypes for K-Means
- Residual Sum of Squares
    - Benchmark RSS: Always predict the response as zero (or mean response if not centered).
- Ordinary Least Squares
    - sklearn.linear_model.LinearRegression()
- Ridge
    - sklearn.linear_model.Ridge()
    - give small weight for irrelevant features
- Lasso
    - sklearn.linear_model.Lasso()
    - give zero weight for irrelevant features

#### lab5
- eigendecomposition
    - np.linalg.eigh()
- Singular Value Decomposition (SVD)
    - np.linalg.svd()
- Principal Component Analysis (PCA)
    - sklearn.decomposition.PCA()

### Genomics
#### lab1
- DNA reads split/check purity/count

#### lab2
- Smith Waterman Algorithm (dp for alignment)
- accelerate matching with the property of dictionary (hash) of python

#### lab3
- Single-Nucleotide Polymorphisms (SNP)
- logistic regression
    - statsmodels.api.Logit()

#### lab4
- Expectation Maximization Algorithm (EM)
- Single-Cell RNA-seq
- t-Distributed Stochastic Neighbor Embedding
    - sklearn.manifold.TSNE()

### Natural Language Processing (NLP)
#### lab1
- corpus
- stemming
- lemmatization
- Type-to-Token Ratio (TTR)

#### lab2
- bag of words
- prune
- linear classification
- Naive Bayes
- logistic regression
- precision, recall, F1

#### lab3
- tokenization
- padding
- n-gram probability
- language model

#### lab4
- word vectors
- co-occurrence matrix
- word2vec word embedding
- cosine similarity
- polysemous words
- synonyms and antonyms


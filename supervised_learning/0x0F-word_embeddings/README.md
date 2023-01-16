Resources
Read or watch:

An Introduction to Word Embeddings
Introduction to Word Embeddings
Natural Language Processing|Bag Of Words Intuition
Natural Language Processing|TF-IDF Intuition| Text Prerocessing
Word Embedding - Natural Language Processing| Deep Learning
Word2Vec Tutorial - The Skip-Gram Model
Word2Vec Tutorial Part 2 - Negative Sampling
GloVe Explained
FastText: Under the Hood
ELMo Explained
Definitions to skim

Natural Language Processing
References:

Efficient Estimation of Word Representations in Vector Space (Skip-gram, 2013)
Distributed Representations of Words and Phrases and their Compositionality (Word2Vec, 2013)
GloVe: Global Vectors for Word Representation (website)
GloVe: Global Vectors for Word Representation (2014)
fastText (website)
Bag of Tricks for Efficient Text Classification (fastText, 2016)
Enriching Word Vectors with Subword Information (fastText, 2017)
Probabilistic FastText for Multi-Sense Word Embeddings (2018)
ELMo (website)
Deep contextualized word representations (ELMo, 2018)
sklearn.feature_extraction.text.CountVectorizer
sklearn.feature_extraction.text.TfidfVectorizer
genism.models.word2vec
genism.models.fasttext
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is natural language processing?
What is a word embedding?
What is bag of words?
What is TF-IDF?
What is CBOW?
What is a skip-gram?
What is an n-gram?
What is negative sampling?
What is word2vec, GloVe, fastText, ELMo?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
All of your files must be executable
A README.md file, at the root of the folder of the project, is mandatory
Your code should follow the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Download Gensim 3.8.x
pip install --user gensim==3.8
Download Keras 2.2.5
pip install --user keras==2.2.5
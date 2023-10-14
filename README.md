This repository implements the Word2Vec vector generation for a given corpus using Skip-Gram as defined in the paper: https://arxiv.org/pdf/1310.4546.pdf

The Research paper defined following 4 optimizations on top of default skip-gram implementation with softmax objective. A total of 4 have been implemeted, baring hiererchial softmax.

The Research paper mentions following variations (including original):
  1. Tokenize the corpus into unigrams, then calculate average probability using softmax function in outer layer.(2. Skip-Gram Model)
  2. Tokenize the corpus into unigrams, then calculate average probability using hiererchial function in outer layer. (2.1. Hierchial Softmax)
  3. Tokenize the corpus into unigrams, remove the most frequent words defined by objective. Follow it up with either softmax or hierchial softmax in outer layer. (2.3. Subsampling of Frequent Words)
  4. Tokenize the corpus into unigrams, remove the frequent words, learn bigrams and update vocab. Follow it up with either softmax or hierchial softmax in outer layer. (4. Learning Phrases)
  5. Tokenize the corpus into unigrams, remove the frequent words, learn bigrams and update vocab. Follow it up with either softmax or hierchial softmax in outer layer. But backpropogate only to K negative samples. (2.2. Negative Subsampling)

Of the above 5, the 2nd implementation i.e. Hierchial softmax is not implemented. The drawbacks for this are that the Word2Vec model takes a lot of time to train of large corpus. (Parameters > 2.5mil on RTX3060ti). 
The following is mapping between a python file, and the method from research paper.

  - Method 1 - word2vec_base.py (CPU) & word2vec_base_th.py (GPU)
  - Method 2: NA
  - Method 3: word2vec_freq_word_sampling.py (CPU) & word2vec_freq_word_sampling_th.py(GPU)
  - Method 4: word2vec_freqwords_phrases.py (CPU) & word2vec_freqwords_phrases_th.py (GPU)
  - Method 5: word2vec_negative_sampling_np.py (CPU) & word2vec_negative_sampling_np_th.py (GPU)

Each of class inside above files have been standardized. The following are models defined inside python files (names are self explainatory)

```
gpu_models = [
        Word2VecLearnPhraseSampleFreqWordGPU, Word2VecFreqWordSamplingGPU, Word2VecGPU, Word2VecNegSamplingGPU
]

cpu_models = [
    Word2VecCPU, Word2VecFreqWordSamplingCPU, Word2VecLearnPhraseSampleFreqWordCPU, Word2VecNegSamplingCPU
]
```

Let's say for reference, we want to train a specifc model, either `Word2VecLearnPhraseSampleFreqWordCPU` or `Word2VecFreqWordSamplingGPU`. We can train as follows:

```
model = Word2VecFreqWordSamplingGPU(corpus, embedding_dim=300) # Other params are defaults
model.train(num_epochs=10)
predictions = model.predict("any word in vocab", 2)
```

A sample file `main.py` is pushed which can walk you through sample training and prediction using any of created models.

Limitations:
Currently while testing, A model with 496800 parameters, 39851 word-pairs, takes 11min per epoch. Scaling is issue as expected with softmax. 

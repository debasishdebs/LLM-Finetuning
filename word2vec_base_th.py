# Skip-Gram with softmax activation
# Not implemented using specific formula and not implemented hirerchial softmax and Sampling negtive samples to reduce training and
# Learning Phrases using unigram-bigram score and no sampling of frequent words

import re
from collections import defaultdict
import torch as th
import datetime as dt
from word2vec import Word2VecBase


class Word2VecGPU(Word2VecBase):
    def __init__(self, corpus, embedding_dim, context_window=2, learning_rate=0.025):
        # Initialize the Word2Vec model with hyperparameters
        super().__init__()
        self.corpus = corpus
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.num_workers = 8
        self.build_vocabulary()

    def __str__(self):
        return "Word2Vec-Softmax-GPU"

    def build_vocabulary(self):
        # Build the vocabulary from the preprocessed corpus
        words = self.preprocess(self.corpus)
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1

        # Filter words based on subsampling threshold
        # self.vocabulary = [word for word, count in word_counts.items() if count / len(words) > self.subsampling_threshold]
        self.vocabulary = [word for word, count in word_counts.items()]

        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)
        self.initialize_embeddings()
        print(f"Total parameters in model: {self.vocab_size * self.embedding_dim} and num tokens: {self.vocab_size}")

    def train_pair(self, context_word, target_word):
        # context_word, target_word = args[0], args[1]
        # Calculate loss and update vectors for a context-target word pair
        context_vector = self.word_vectors[self.word_to_index[context_word]]
        target_index = self.word_to_index[target_word]
        probabilities = self.forward_pass(self.word_vectors[target_index])
        self.backward_pass(self.word_to_index[context_word], target_index, probabilities)

        C = 0
        loss = 0
        for m in range(self.vocab_size):
            # if(self.y_train[j][m]):
            if (self.word_vectors[target_index][m]):
                # print(self.output_matrix[m])
                loss += -1 * self.output_matrix[m]
                C += 1
        loss += C * th.log(th.sum(th.exp(self.output_matrix)))
        return loss

    def forward_pass(self, target_vector):
        # Forward pass calculates the loss for a context-target word pair

        # Calculate the dot products between input_vector and all hidden_layer. This would be input to hidden layer. Defined by y = w1.dot(i)
        self.hidden_matrix = th.matmul(self.U_weights.T, target_vector)
        # Calculate the dot products between hidden_layer and output_layer. This would be then applied to softmax to get probabilities. Defined by o = w2.dot(y)
        self.output_matrix = th.matmul(self.V_weights.T, self.hidden_matrix)
        # Softmax formula: P(target_index|context_vector) = exp(dot_product) / sum(exp(all_dot_products))
        probabilities = self.softmax(self.output_matrix) # self.softmax(self.output_matrix)

        return probabilities

    def backward_pass(self, context_index, target_index, probabilities):
        # Backward pass updates vectors using gradient descent
        e = probabilities.reshape(-1, 1) - self.word_vectors[context_index].reshape(self.vocab_size, 1)

        dLdV = th.matmul(self.hidden_matrix.reshape(-1, 1), e.T)
        X = self.word_vectors[target_index].reshape(self.vocab_size, 1)
        dLdU = th.matmul(X, th.matmul(self.V_weights, e).T)

        self.update_vectors(self.learning_rate, dLdU, dLdV)

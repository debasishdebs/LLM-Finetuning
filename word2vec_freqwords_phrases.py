# Skip-Gram with sampling on frequent words
# Learning phrases


import numpy as np
import re
from collections import defaultdict
import math
import datetime as dt


class Word2VecLearnPhraseSampleFreqWordCPU:
    def __init__(self, corpus, embedding_dim, context_window=2, subsampling_threshold=1e-5, learning_rate=0.025):
        self.corpus = corpus
        self.embedding_dim = embedding_dim
        self.context_window = context_window

        self.subsampling_threshold = subsampling_threshold
        self.discard_prob_threshold = 0.98

        self.phrase_threshold = 0.25
        self.phrase_discounting_coeff = subsampling_threshold

        self.learning_rate = learning_rate
        self.build_vocabulary()

    def __str__(self):
        return "Word2Vec-Softmax-FreqWordSampling-LearnPhrases-CPU"

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def discard_probability(self, word_frequency):
        """
        Calculate the probability of discarding a word based on its frequency.

        Args:
            word_frequency (int): The frequency of the word in the corpus.
            threshold (float): The chosen threshold (default is 1e-5).

        Returns:
            float: The probability of discarding the word.
        """
        probability = 1.0 - math.sqrt(self.subsampling_threshold / word_frequency)
        return probability

    def get_word_frequency(self, word):
        word_count = self.phrase_freq[word]
        total_words = sum(self.word_counts.values())
        return word_count / total_words

    def build_vocabulary(self):
        words = self.preprocess(self.corpus)
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        self.word_counts = word_counts

        # First we create phrases using bigram-unigram scoring.
        unigram_scores, bigram_scores = self.generate_unigram_bigram_scores(words)
        filtered_bigrams = [b[0] for b in bigram_scores if b[1] > self.phrase_threshold]
        vocabulary = filtered_bigrams + [u[0] for u in unigram_scores]

        # Now we create a word frequency dict for combined vocab including unigrams and bigrams
        self.phrase_freq = {phrase: self.word_freq[phrase] if len(phrase.split(" ")) == 1 else self.bigram_freq[phrase]
                            for phrase in vocabulary}

        # self.vocabulary = [word for word, count in word_counts.items() if count / len(words) > self.subsampling_threshold]
        self.vocabulary = [word for word in self.phrase_freq if
                           self.discard_probability(self.get_word_frequency(word)) <= self.discard_prob_threshold]
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        self.vocabulary_freq = {
            phrase: self.word_freq[phrase] if len(phrase.split(" ")) == 1 else self.bigram_freq[phrase]
            for phrase in self.vocabulary}
        self.vocab_size = len(self.vocabulary)
        self.initialize_embeddings()
        print(f"Total parameters in model: {self.vocab_size * self.embedding_dim}")

    def generate_unigram_bigram_scores(self, words):
        word_freq = defaultdict(int)
        bigram_freq = defaultdict(int)

        # Count word frequencies and bigram frequencies in the corpus
        for i in range(1, len(words)):
            word = words[i]
            prev_word = words[i - 1]
            if i == 1:
                word_freq[words[i - 1]] += 1
            word_freq[word] += 1
            bigram = f"{prev_word} {word}"
            bigram_freq[bigram] += 1

        self.word_freq = word_freq
        self.bigram_freq = bigram_freq

        # Score unigrams and bigrams to learn phrases
        scored_words = []
        for word in word_freq:
            score = word_freq[word]
            scored_words.append((word, score))

        scored_bigrams = []
        for bigram in bigram_freq:
            prev_word, word = bigram.split()
            scored_bigrams.append(
                (bigram, self.calculate_word_pair_score(bigram_freq[bigram], word_freq[prev_word], word_freq[word])))

        return scored_words, scored_bigrams

    def calculate_word_pair_score(self, count_wi_wj, count_wi, count_wj):
        """
        Calculate the score of a word pair (wi, wj).

        Args:
            count_wi_wj (int): Co-occurrence count of words wi and wj.
            count_wi (int): Count of word wi.
            count_wj (int): Count of word wj.
            delta (float): A constant (typically small, e.g., 1e-5) to prevent division by zero.

        Returns:
            float: The calculated score for the word pair (wi, wj).
        """
        score = (count_wi_wj - self.phrase_discounting_coeff) / (count_wi * count_wj)
        return score

    def initialize_embeddings(self):
        # Initialize word and context vectors randomly
        self.word_vectors = np.eye(self.vocab_size)
        self.U_weights = np.random.randn(self.vocab_size, self.embedding_dim)
        self.V_weights = np.random.randn(self.embedding_dim, self.vocab_size)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            t_epoch = dt.datetime.now()
            idx = 1
            for context_word, target_word in self.generate_training_data():
                total_loss += self.train_pair(context_word, target_word)
                if idx % 10000 == 0:
                    print(f"Done with {idx} word-target pair")
                idx += 1
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.vocabulary)}.Tot Pairs: {idx}. Time taken: {dt.datetime.now() - t_epoch}")
            new_learning_rate = self.learning_rate * 1 / ((1 + self.learning_rate * epoch))
            print(f"Changing alpha from {self.learning_rate} to {new_learning_rate}")
            self.learning_rate = new_learning_rate

    def generate_training_data(self):
        # Generate training data by iterating through the corpus
        words = self.preprocess(self.corpus)
        words = [word for word in words if word in self.vocabulary]
        for i, target_word in enumerate(words):
            if target_word not in self.word_to_index:
                continue  # Skip words not in vocabulary

            # Context window defines the range of words to consider as context
            start = max(0, i - self.context_window)
            end = min(len(words), i + self.context_window + 1)
            for j in range(start, end):
                if i == j:
                    continue  # Skip the target word itself
                context_word = words[j]
                yield context_word, target_word

    def train_pair(self, context_word, target_word):
        # Calculate loss and update vectors for a context-target word pair
        context_vector = self.word_vectors[self.word_to_index[context_word]]
        target_index = self.word_to_index[target_word]
        probabilities = self.forward_pass(self.word_vectors[target_index])
        self.backward_pass(context_vector, target_index, probabilities, gradient=1.0)

        C = 0
        loss = 0
        for m in range(self.vocab_size):
            # if(self.y_train[j][m]):
            if (self.word_vectors[target_index][m]):
                # print(self.output_matrix[m])
                loss += -1 * self.output_matrix[m]
                C += 1
        loss += C * np.log(np.sum(np.exp(self.output_matrix)))
        return loss

    def forward_pass(self, target_vector):
        # Forward pass calculates the loss for a context-target word pair

        # Calculate the dot products between input_vector and all hidden_layer. This would be input to hidden layer. Defined by y = w1.dot(i)
        self.hidden_matrix = np.dot(self.U_weights.T, target_vector)
        # Calculate the dot products between hidden_layer and output_layer. This would be then applied to softmax to get probabilities. Defined by o = w2.dot(y)
        self.output_matrix = np.dot(self.V_weights.T, self.hidden_matrix)
        # Softmax formula: P(target_index|context_vector) = exp(dot_product) / sum(exp(all_dot_products))
        probabilities = self.softmax(self.output_matrix)

        return probabilities

    def backward_pass(self, context_vector, target_index, probabilities, gradient):
        # Backward pass updates vectors using gradient descent
        e = probabilities.reshape(-1, 1) - self.word_vectors[target_index].reshape(self.vocab_size, 1)

        dLdV = np.dot(self.hidden_matrix.reshape(-1, 1), e.T)
        X = context_vector.reshape(self.vocab_size, 1)
        dLdU = np.dot(X, np.dot(self.V_weights, e).T)

        self.update_vectors(self.learning_rate, dLdU, dLdV)

    def softmax(self, x):
        # Softmax function to calculate probabilities
        e_x = np.exp(x - np.max(x))  # Exponential of dot products
        return e_x / e_x.sum()  # Probabilities for all words

    def update_vectors(self, learning_rate, dLdU, dLdV):
        # Update vectors using gradient descent
        self.U_weights = self.U_weights - learning_rate * dLdU
        self.V_weights = self.V_weights - learning_rate * dLdV

    def predict(self, word, number_of_predictions):
        if word in self.vocabulary:
            index = self.word_to_index[word]
            X = [0 for i in range(self.vocab_size)]
            X[index] = 1
            prediction = self.forward_pass(X)
            output = {}
            for i in range(self.vocab_size):
                output[prediction[i]] = i

            top_context_words = []
            for k in sorted(output, reverse=True):
                top_context_words.append(self.vocabulary[output[k]])
                if len(top_context_words) >= number_of_predictions:
                    break

            return top_context_words
        else:
            print("Word not found in dictionary")

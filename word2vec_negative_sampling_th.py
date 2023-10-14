import numpy as np
import re
from collections import defaultdict, OrderedDict
import math
import datetime as dt
import torch as th
from word2vec import Word2VecBase

# Skip-Gram with sampling on frequent words
# Learning phrases
# Noise distribution


class Word2VecNegSamplingGPU(Word2VecBase):
    def __init__(self, corpus, embedding_dim, context_window=2, subsampling_threshold=1e-5, negative_samples=5, learning_rate=0.025):
        super().__init__()
        self.corpus = corpus
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.noise_distribution = th.zeros(int(1e8), dtype=th.int16)

        self.num_negative_samples = negative_samples

        self.subsampling_threshold = subsampling_threshold
        self.discard_prob_threshold = 0.95

        self.phrase_threshold = 0.2
        self.phrase_discounting_coeff = subsampling_threshold

        self.learning_rate = learning_rate
        self.build_vocabulary()

    def __str__(self):
        return "Word2Vec-Softmax-FreqWordSampling-LearnPhrases-NegSampling-GPU"

    def create_noise_distribution(self):
        distribution = self.get_word_probability(0.75)
        distribution = list(distribution.values())
        previous_j = 0
        for i, value in enumerate(distribution):
            j = int(self.noise_distribution.shape[0] * value)
            self.noise_distribution[previous_j:previous_j + j] = i
            previous_j = previous_j + j

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

    def get_word_probability(self, pow=1.0):
        tot_words = sum([np.power(v, pow) for v in self.word_counts.values()])
        word_prob = OrderedDict((word, float(self.phrase_freq[word]) / float(tot_words))
                                for word in self.phrase_freq.keys())
        return word_prob

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

        # self.vocabulary = [word for word, count in word_counts.items() if count /
        # len(words) > self.subsampling_threshold]
        self.vocabulary = [word for word in self.phrase_freq if
                           self.discard_probability(self.get_word_frequency(word)) <= self.discard_prob_threshold]
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        self.vocabulary_freq = {
            phrase: self.word_freq[phrase] if len(phrase.split(" ")) == 1 else self.bigram_freq[phrase]
            for phrase in self.vocabulary}
        self.vocab_size = len(self.vocabulary)

        # Now that we have word frequency, let's create noise distribution table for negative sampling
        self.create_noise_distribution()

        self.initialize_embeddings()
        print(f"Total parameters in model: {self.vocab_size * self.embedding_dim} and num tokens: {self.vocab_size}")

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

    def sample_negative_word(self, target_word):
        # Sample a negative word not equal to the target_word
        negative_word = target_word
        while negative_word == target_word:
            index = th.randint(low=0, high=self.noise_distribution.shape[0], size=(1, ))
            negative_word_index = self.noise_distribution[index]
            negative_word = self.vocabulary[negative_word_index[0]]
        return negative_word

    def train_pair(self, context_word, target_word):
        # Calculate loss and update vectors for a context-target word pair
        target_index = self.word_to_index[target_word]
        probs = self.forward_pass(self.word_vectors[target_index])
        self.backward_pass(self.word_to_index[context_word], target_index)

        C = 0
        loss = 0
        for m in range(self.vocab_size):
            # if(self.y_train[j][m]):
            if (self.word_vectors[target_index][m]):
                loss += -1 * self.output_matrix[m].item()
                C += 1
        loss += C * th.log(th.sum(th.exp(self.output_matrix)))
        return loss

    def forward_pass(self, target_vector):
        # context_vector = self.word_vectors[self.word_to_index[context_word]]
        # target_vector = self.word_vectors[self.word_to_index[target_word]]
        #
        # # print(target_vector.view(1, -1).shape, self.U_weights.shape)
        # U_i = th.matmul(target_vector.view(1, -1), self.U_weights)
        # V_c = th.matmul(context_vector, self.V_weights.T)
        # self.output_matrix = th.matmul(self.V_weights.T, U_i.T)
        # pos_prob = self.sigmoid(th.multiply(U_i.T, V_c))
        #
        # # Generate a list of negative samples

        mask_th = th.Tensor([[1] * self.vocab_size]).to(self.device)
        idx_mask = (th.matmul(mask_th.float(), (self.word_vectors == target_vector).T.float()) == self.vocab_size)[0]
        target_index = (idx_mask == True).nonzero(as_tuple=False).item()

        target_word = self.vocabulary[target_index]
        target_vector = self.word_vectors[target_index]
        self.negative_samples = [self.sample_negative_word(target_word) for _ in range(self.num_negative_samples)]
        # # Calculate negative probabilities for sampled words using the sigmoid function
        # neg_probs = []
        # for neg_sample in self.negative_samples:
        #     neg_idx = self.word_to_index[neg_sample]
        #     neg_vec = self.word_vectors[neg_idx]
        #
        #     U_i = th.matmul(target_vector.view(1, -1), self.U_weights)
        #     V_i = th.matmul(neg_vec, self.V_weights.T)
        #
        #     neg_prob = self.sigmoid(-1 * th.mul(U_i.T, V_i))
        #     neg_probs.append(neg_prob)
        #
        # return pos_prob, th.stack(neg_probs)
        # Calculate the dot products between input_vector and all hidden_layer. This would be input to hidden layer. Defined by y = w1.dot(i)
        self.hidden_matrix = th.matmul(self.U_weights.T, target_vector)
        # Calculate the dot products between hidden_layer and output_layer. This would be then applied to softmax to get probabilities. Defined by o = w2.dot(y)
        self.output_matrix = th.matmul(self.V_weights.T, self.hidden_matrix)
        # Softmax formula: P(target_index|context_vector) = exp(dot_product) / sum(exp(all_dot_products))
        probabilities = self.softmax(self.output_matrix)

        return probabilities

    @staticmethod
    def negative_log_likelihood(probabilities, context_vector):
        likelihood = sum(th.mul(probabilities, context_vector))
        return -1 * th.log(likelihood)

    def backward_pass(self, context_idx, target_idx):
        pos_sample_vector = self.word_vectors[context_idx]
        neg_idxs = [self.word_to_index[w] for w in self.negative_samples]
        neg_sample_vectors = self.word_vectors[neg_idxs]

        probabilities = []
        for idx in neg_idxs:
            # print(self.word_vectors[idx].type(th.int).dtype, self.word_vectors[target_idx].type(th.int).dtype)
            inner_term = th.matmul(self.V_weights[self.word_vectors[idx].type(th.int)],
                                self.U_weights[self.word_vectors[target_idx].type(th.int)])
            inner_term = self.sigmoid(inner_term)
            neg_loss = th.mul(inner_term, self.U_weights[self.word_vectors[target_idx].type(th.int)])
            probabilities.append(neg_loss)

        positive_term = self.sigmoid(th.matmul(self.V_weights[self.word_vectors[context_idx].type(th.int)],
                                            self.U_weights[self.word_vectors[target_idx].type(th.int)]))
        pos_prob = positive_term - 1
        pos_loss = th.mul(pos_prob, self.U_weights[self.word_vectors[target_idx].type(th.int)])

        probabilities.append(pos_loss)
        dedw_neg = sum(probabilities)
        dedw_pos = th.matmul(pos_prob, self.U_weights[self.word_vectors[target_idx].type(th.int)].T)
        self.V_weights[self.word_vectors[context_idx].type(th.int)] = self.V_weights[self.word_vectors[context_idx].type(th.int)] - self.learning_rate * dedw_pos
        self.U_weights[self.word_vectors[target_idx].type(th.int)] = self.U_weights[self.word_vectors[target_idx].type(th.int)] - self.learning_rate * dedw_neg

    def sigmoid(self, x):
        # return 1 / (1 + th.exp(-x))
        return th.sigmoid(x)

import torch as th
import re
import datetime as dt
import tqdm


class Word2VecBase(object):
    def __init__(self, **kwargs):
        self.device = "cuda" if th.cuda.is_available() else "cpu"

    def initialize_embeddings(self):
        # Initialize word and context vectors randomly
        self.word_vectors = th.eye(self.vocab_size, device=self.device)
        self.U_weights = th.randn(self.vocab_size, self.embedding_dim, device=self.device)
        self.V_weights = th.randn(self.embedding_dim, self.vocab_size, device=self.device)

    def generate_training_data(self):
        # Generate training data by iterating through the corpus
        training_data = []
        for i, target_word in enumerate(self.vocabulary):
            # Context window defines the range of words to consider as context
            start = max(0, i - self.context_window)
            end = min(len(self.vocabulary), i + self.context_window + 1)
            for j in range(start, end):
                if i == j:
                    continue  # Skip the target word itself
                context_word = self.vocabulary[j]
                training_data.append([context_word, target_word])
        return training_data

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def softmax(self, x):
        # Softmax function to calculate probabilities
        e_x = th.exp(x - th.max(x))  # Exponential of dot products
        return e_x / e_x.sum()  # Probabilities for all words

    def update_vectors(self, learning_rate, dLdU, dLdV):
        # Update vectors using gradient descent
        self.U_weights = self.U_weights - learning_rate * dLdU
        self.V_weights = self.V_weights - learning_rate * dLdV

    def train(self, num_epochs):
        t_start = dt.datetime.now()

        training_data = self.generate_training_data()
        print(f"Total word-target pairs for training {len(training_data)}")

        for epoch in range(num_epochs):
            total_loss = 0
            t_epoch = dt.datetime.now()
            idx = 1
            for X_train in (pbar:= tqdm.tqdm(training_data)):
                pbar.set_postfix_str(X_train[1])

                context_word, target_word = X_train[0], X_train[1]

                total_loss += self.train_pair(context_word, target_word)

                if idx % 10000 == 0:
                    print(f"Done with {idx} word-target pair")
                idx += 1
            tot_time = dt.datetime.now() - t_epoch
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.vocabulary)}.Tot Pairs: {idx}. Time per pair in ms(4 matmuls): {(tot_time / idx).total_seconds() * 1000} Time taken: {tot_time}")
            new_learning_rate = self.learning_rate * 1 / ((1 + self.learning_rate * epoch))
            print(f"Changing alpha from {self.learning_rate} to {new_learning_rate}")
            self.learning_rate = new_learning_rate
        print(f"Total training time: {dt.datetime.now() - t_start}")

    def predict(self, word, number_of_predictions):
        if word in self.vocabulary:
            index = self.word_to_index[word]
            X = [0 for i in range(self.vocab_size)]
            X[index] = 1
            prediction = self.forward_pass(th.tensor(X, device=self.device).float())
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

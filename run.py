#!/usr/bin/env python
import numpy as np
import pickle
import time

class sparse(dict):
    def __missing__(self, key):
        return 0

class Glove():
    def __init__(self, input_file, vec_size=200, window_size=5, epochs=1, learning_rate=0.05):
        with open(input_file, 'r') as train_file:
            self.corpus = train_file.read().split()
        self.vocab = {}
        index = 0
        print("Read corpus")
        for word in self.corpus:
            if word not in self.vocab:
                self.vocab[word] = [1, index]
                index += 1
            else:
                self.vocab[word][0] += 1
        print("Created vocab tracker")
        self.id_corpus = [self.vocab[word][1] for word in self.corpus]
        del self.corpus
        print("Created id corpus, deleted corpus")
        #self.matrix = pickle.load(open('matrix_3_1493506660.p', 'rb'))
        self.matrix = sparse()
        self.init_matrix(window_size)
        self.matrix = list(self.matrix.items())
        del self.id_corpus
        print("Created matrix, converted to list, deleted id corpus")
        #pickle.dump(self.matrix, open('matrix_%d_%d.p' % (window_size, int(time.time())), 'wb'))
        self.vector_table_context = np.random.normal(0, 0.15, (len(self.vocab), vec_size))
        self.vector_table_center = np.random.normal(0, 0.15, (len(self.vocab), vec_size))
        self.biases_context = np.random.normal(0, 0.15, len(self.vocab))
        self.biases_center = np.random.normal(0, 0.15, len(self.vocab))
        self.train(vec_size, epochs, learning_rate)

    def train(self, vec_size, epochs, learning_rate):
        matrix_list = self.matrix
        adagrad_vec_center = np.ones(((len(self.vocab)), vec_size), dtype=np.float64)
        adagrad_vec_context = np.ones(((len(self.vocab)), vec_size), dtype=np.float64)
        adagrad_bias_context = np.ones((len(self.vocab)), dtype=np.float64)
        adagrad_bias_center = np.ones((len(self.vocab)), dtype=np.float64)
        for epoch in range(epochs):
            np.random.shuffle(matrix_list)
            cost = 0
            track = 0
            for (center_id, context_id), cooccur in matrix_list:
                common_grad = np.dot(self.vector_table_center[center_id], self.vector_table_context[context_id]) + self.biases_center[center_id] + self.biases_context[context_id] - np.log(cooccur)
                fx = (cooccur / 100) ** 0.75 if cooccur < 100 else 1
                cost += fx * (common_grad ** 2)
                grad_center = fx * self.vector_table_context[context_id] * common_grad
                grad_context = fx * self.vector_table_center[center_id] * common_grad
                grad_biases = fx * common_grad
                self.vector_table_center[center_id] -= learning_rate * grad_center / np.sqrt(adagrad_vec_center[center_id])
                self.vector_table_context[context_id] -= learning_rate * grad_context / np.sqrt(adagrad_vec_context[context_id])
                self.biases_center[center_id] -= learning_rate * grad_biases / np.sqrt(adagrad_bias_center[center_id])
                self.biases_context[context_id] -= learning_rate * grad_biases / np.sqrt(adagrad_bias_context[context_id])

                adagrad_vec_center[center_id] += grad_center ** 2
                adagrad_vec_context[context_id] += grad_context ** 2
                adagrad_bias_center[center_id] += grad_biases ** 2
                adagrad_bias_context[context_id] += grad_biases ** 2

                if track % 1000000 == 0:
                    print("common_grad: %f" % common_grad)
                    print("fx: %f" % fx)
                    print("cooccur: %f" % cooccur)
                    print("Cost: %f, Iter: %d / %d" % (cost, track, len(matrix_list)))
                track += 1
            print("Epoch: %d, Cost: %f" % (epoch, cost))
            print()

        
    def init_matrix(self, window_size):
        for i, id in enumerate(self.id_corpus):
            for seek in range(-window_size, window_size + 1):
                if seek == 0 or i + seek < 0 or i + seek >= len(self.id_corpus):
                    continue
                self.matrix[id, self.id_corpus[i+seek]] += 1 / np.sqrt(abs(seek))

    def get_id(self, word):
        return self.vocab[word][1]

    def get(self, words):
        return dict([(word, self.vector_table_center[self.get_id(word)] + self.vector_table_context[self.get_id(word)]) for word in words if word in self.vocab])

def run(input_file, vocab_file, output_file):
    vec_size=300
    window_size=8
    epochs=15
    learning_rate=0.05
    np.random.seed(1)

    glove = Glove(input_file, vec_size=vec_size, window_size=window_size, epochs=epochs, learning_rate=learning_rate)
    with open(vocab_file, 'r') as test_file:
        vocab = [word.strip() for word in test_file.readlines()]
    vecs = glove.get(vocab)
    with open(output_file, 'w') as output_file:
        for word, vec in vecs.items():
            print(word + ' ' + ' '.join(str("%.3f" % x) for x in vec), file=output_file)

if __name__ == "__main__":
    run('train.txt', 'test.txt', 'vectors.txt')

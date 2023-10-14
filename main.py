from word2vec_base import Word2VecCPU
from word2vec_base_th import Word2VecGPU
from word2vec_freq_word_sampling import Word2VecFreqWordSamplingCPU
from word2vec_freq_word_sampling_th import Word2VecFreqWordSamplingGPU
from word2vec_freqwords_phrases import Word2VecLearnPhraseSampleFreqWordCPU
from word2vec_freqwords_phrases_th import Word2VecLearnPhraseSampleFreqWordGPU
from word2vec_negative_sampling_np import Word2VecNegSamplingCPU
from word2vec_negative_sampling_th import Word2VecNegSamplingGPU
import os
import torch as th


if __name__ == '__main__':
    gpu_models = [
        Word2VecLearnPhraseSampleFreqWordGPU, Word2VecFreqWordSamplingGPU, Word2VecGPU, Word2VecNegSamplingGPU
    ]

    cpu_models = [
        Word2VecCPU, Word2VecFreqWordSamplingCPU, Word2VecLearnPhraseSampleFreqWordCPU, Word2VecNegSamplingCPU
    ]

    th.multiprocessing.set_start_method('spawn')

    dataset1_f = os.path.join(os.getcwd() + "/interview_ds.txt")
    dataset2_f = os.path.join(os.getcwd() + "/interview_ds_2.txt")

    with open(dataset1_f, "r") as f:
        corpus1 = f.read()
    with open(dataset2_f, "r") as f:
        corpus2 = f.read()

    corpus = corpus1 + corpus2

    model = Word2VecFreqWordSamplingGPU(corpus[:int(1e5)], embedding_dim=300)
    print(f"Running model {model}")
    model.train(10)
    predictions = model.predict("name", 3)
    print(predictions)

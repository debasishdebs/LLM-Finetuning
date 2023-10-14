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
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True

    dataset1_f = os.path.join(os.getcwd() + "/interview_ds.txt")
    dataset2_f = os.path.join(os.getcwd() + "/interview_ds_2.txt")

    with open(dataset1_f, "r") as f:
        corpus1 = f.read()
    with open(dataset2_f, "r") as f:
        corpus2 = f.read()

    corpus = corpus1 + corpus2
    # gpu_models = [Word2VecNegSamplingGPU]

    for word2vec_model in gpu_models[:-1]:
        model = word2vec_model(corpus1[:int(1e6)], embedding_dim=100)
        print(f"Running model {model}")
        model.train(3)
        predictions = model.predict("puppy", 3)
        print(predictions, " are predictiosn for word ", "puppy")
        print(50*"-")

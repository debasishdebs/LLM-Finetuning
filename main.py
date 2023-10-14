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

    corpus = """
        Artificial intelligence (AI) is a multidisciplinary field that intersects computer science, mathematics, psychology, and many other domains. The quest to create machines that can mimic human intelligence has led to remarkable advances in AI, and one of the core components of AI research is natural language processing (NLP). NLP involves teaching machines to understand, interpret, and generate human language.
        The concept of AI and machine learning has been around for decades, but recent years have seen an explosion in the development and deployment of AI technologies. This surge is driven by the growth of big data, increased computing power, and innovative algorithms. Machine learning, a subset of AI, focuses on building systems that can learn from data and make predictions or decisions based on that data.
        The field of NLP plays a vital role in AI and machine learning, as it deals with human language, one of the most complex and nuanced data forms. Language is dynamic, and understanding it requires not just recognizing words but grasping their meanings in context, nuances, idioms, and cultural references. NLP applications encompass a wide range of tasks, from simple language translation to complex natural language understanding.
        One of the fundamental challenges in NLP is representing language in a way that machines can process and analyze it effectively. This is where word embeddings come into play. Word embeddings are vector representations of words, and they have revolutionized NLP by providing an efficient means of capturing semantic relationships between words. Among the various word embedding techniques, Word2Vec stands out as one of the most influential and widely adopted.
        Word2Vec is a neural network-based model developed by researchers at Google, including Tomas Mikolov. It is designed to learn distributed representations of words in a continuous vector space. These word vectors have remarkable properties, making them incredibly useful for various NLP applications. The primary idea behind Word2Vec is that words with similar meanings should be close together in the vector space. By training on large text corpora, the model can learn these word embeddings.
        Training a Word2Vec model involves preprocessing a substantial text corpus, which can be a collection of books, articles, or any text data. The corpus should be diverse and extensive to ensure the model captures a wide range of vocabulary and linguistic patterns. The text data is preprocessed to remove punctuation, special characters, and stop words, which are common but low-information words like "the," "and," and "in."
        Next, hyperparameters are set to configure the Word2Vec model. These include the embedding dimension, context window size, subsampling threshold, negative sampling count, and learning rate. The embedding dimension determines the size of the word vectors. A larger dimension can capture more complex relationships but requires more data.
        The context window size specifies how many words to consider on each side of the target word when training the model. A larger window may capture more contextual information, while a smaller window may focus on syntactic relationships.
        Subsampling is an important technique to filter out high-frequency words that do not contribute much information, as they often occur in various contexts. By removing these frequent words, the model can focus on learning more meaningful representations.
        Negative sampling is another crucial component of Word2Vec training. Instead of updating the model parameters for all words in the vocabulary, negative sampling selects a small set of non-context words for each target word. This speeds up training and focuses on the most relevant terms.
        The learning rate determines the step size in the parameter updates. Properly tuning the learning rate is essential for training stability and convergence.
        With these parameters set, the Word2Vec model is ready for training. The training process involves multiple epochs, where the model iterates through the entire corpus, predicting context words for each target word. The parameters are updated to maximize the likelihood of these predictions. This iterative process gradually refines the word vectors.
        Word2Vec is known for its simplicity and effectiveness. The word vectors it produces are widely used for various NLP tasks, including sentiment analysis, document classification, information retrieval, and recommendation systems. These word embeddings allow machines to understand and process text data more efficiently.
        The impact of Word2Vec and similar word embedding techniques has extended to fields beyond NLP. They find applications in fields like image processing, recommendation systems, and more. These representations enable computers to understand and operate on a broader range of data types.
        In conclusion, Word2Vec and word embeddings have transformed the landscape of natural language processing and machine learning. They provide a bridge between the complexity of human language and the computational capabilities of machines. As AI continues to advance, these techniques remain a vital part of the toolkit for understanding and generating human language.
    """

    th.multiprocessing.set_start_method('spawn')

    dataset1_f = os.path.join(os.getcwd() + "/interview_ds.txt")
    dataset2_f = os.path.join(os.getcwd() + "/interview_ds_2.txt")

    with open(dataset1_f, "r") as f:
        corpus1 = f.read()
    with open(dataset2_f, "r") as f:
        corpus2 = f.read()

    # corpus = corpus1 + corpus2

    model = Word2VecFreqWordSamplingGPU(corpus1[:int(1e5)], embedding_dim=300)
    print(f"Running model {model}")
    model.train(10)

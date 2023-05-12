import gensim
import os

abspath = os.path.dirname(os.path.abspath(__file__))
reloaded_word_vectors = gensim.models.KeyedVectors.load(os.path.join(abspath, "../vectors/vectors.kv"))

w1 = "dirty"
print("Most similar to {0}".format(w1), reloaded_word_vectors.most_similar(positive=w1))

# look up top 6 words similar to 'polite'
w1 = ["polite"]
print(
    "Most similar to {0}".format(w1),
    reloaded_word_vectors.most_similar(
        positive=w1,
        topn=6))

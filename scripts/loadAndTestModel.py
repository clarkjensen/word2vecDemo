import gensim
import os
import sys

# switch these and the tests when switching which model to test (hotel reviews or HR search queries)
# dataset_file = "../reviews_data.txt.gz"
dataset_file = "../search-results-by-url.csv"

# Split the file location into directory path and file name with extension
dir_path, file_name_ext = os.path.split(dataset_file)

# Split the file name and extension
file_name, file_ext = os.path.splitext(file_name_ext)

abspath = os.path.dirname(os.path.abspath(__file__))

reloaded_word_vectors = gensim.models.KeyedVectors.load(os.path.join(abspath, "../vectors/" + file_name + ".kv"))

w1 = "q.anyPlace=Newcastle%20upon%20tyne"
print("Most similar to {0}".format(w1), reloaded_word_vectors.most_similar(positive=w1))
print("Least similar to {0}".format(w1), list(reversed(reloaded_word_vectors.most_similar("q.anyPlace=Newcastle%20upon%20tyne", topn=sys.maxsize)[-10:])))


# w1 = "dirty"
# print("Most similar to {0}".format(w1), reloaded_word_vectors.most_similar(positive=w1))

# # look up top 6 words similar to 'polite'
# w1 = ["polite"]
# print(
#     "Most similar to {0}".format(w1),
#     reloaded_word_vectors.most_similar(
#         positive=w1,
#         topn=6))

# # look up top 6 words similar to 'france'
# w1 = ["france"]
# print(
#     "Most similar to {0}".format(w1),
#     reloaded_word_vectors.most_similar(
#         positive=w1,
#         topn=6))

# # look up top 6 words similar to 'shocked'
# w1 = ["shocked"]
# print(
#     "Most similar to {0}".format(w1),
#     reloaded_word_vectors.most_similar(
#         positive=w1,
#         topn=6))

# # look up top 6 words similar to 'shocked'
# w1 = ["beautiful"]
# print(
#     "Most similar to {0}".format(w1),
#     reloaded_word_vectors.most_similar(
#         positive=w1,
#         topn=6))

# # get everything related to stuff on the bed
# w1 = ["bed", 'sheet', 'pillow']
# w2 = ['couch']
# print(
#     "Most similar to {0}".format(w1),
#     reloaded_word_vectors.most_similar(
#         positive=w1,
#         negative=w2,
#         topn=10))

# # similarity between two different words
# print("Similarity between 'dirty' and 'smelly'",
#       reloaded_word_vectors.similarity(w1="dirty", w2="smelly"))

# # similarity between two identical words
# print("Similarity between 'dirty' and 'dirty'",
#       reloaded_word_vectors.similarity(w1="dirty", w2="dirty"))

# # similarity between two unrelated words
# print("Similarity between 'dirty' and 'clean'",
#       reloaded_word_vectors.similarity(w1="dirty", w2="clean"))


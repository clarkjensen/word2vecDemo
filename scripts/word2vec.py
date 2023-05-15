import os
import logging
import pandas as pd
import gzip
import gensim

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

# switch between these to build a model either on hotel reviews, or on HR search queries
dataset_file = "../reviews_data.txt.gz"
# dataset_file = "../search-results-by-url.csv"

def show_file_contents(input_file):
    df = pd.read_csv(input_file, encoding='utf-8', header=0, dtype=str)
    print(df.head(20))


def read_input(input_file):
    """This method reads the input file which can be in csv, tsv, txt, or gzip format."""

    file_extension = os.path.splitext(input_file)[1]
    logging.info("reading file {0}...this may take a while".format(input_file))

    if file_extension == '.csv':
        df = pd.read_csv(input_file, encoding='utf-8', header=0, dtype=str)
        df.dropna(inplace=True)  # remove rows with missing data
        # for review in df.iloc[:, 0]:
        for i, review in enumerate(df.iloc[:, 0]):
            if "?" in review:
                if (i % 1000 == 0):
                    logging.info("read {0} lines".format(i))

                # this is specific to the HR Search queries -- not applicable to all csv files
                query_params = review.split("?")[1]
                query_params = query_params.replace("&", " ")

                # this is a temporary solution -- we should use the simple_preprocess, but it doesn't
                # work with things like symbols or numbers (like those in HR Search queries).  It 
                # improves the training speed if we use it.
                query_params = query_params.split(" ")
                yield query_params
                # yield gensim.utils.simple_preprocess(query_params)

    elif file_extension == '.tsv':
        df = pd.read_csv(input_file, delimiter='\t', encoding='utf-8', header=0, dtype=str)
        df.dropna(inplace=True)  # remove rows with missing data
        for i, review in enumerate(df.iloc[:, 0]):
            if "?" in review:
                if (i % 1000 == 0):
                    logging.info("read {0} lines".format(i))

                # this is specific to the HR Search queries -- not applicable to all csv files
                query_params = review.split("?")[1]
                query_params = query_params.replace("&", " ")

                # this is a temporary solution -- we should use the simple_preprocess, but it doesn't
                # work with things like symbols or numbers (like those in HR Search queries).  It 
                # improves the training speed if we use it.
                query_params = query_params.split(" ")
                yield query_params
                # yield gensim.utils.simple_preprocess(query_params)

    elif file_extension == '.txt':
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):

                if (i % 10000 == 0):
                    logging.info("read {0} reviews".format(i))
                # do some pre-processing and return list of words for each review
                # text
                yield gensim.utils.simple_preprocess(line)

    elif file_extension == '.gz':
        with gzip.open(input_file, 'rb') as f:
            for i, line in enumerate(f):

                if (i % 10000 == 0):
                    logging.info("read {0} reviews".format(i))
                # do some pre-processing and return list of words for each review
                # text
                yield gensim.utils.simple_preprocess(line)

    else:
        logging.error("unsupported file type: {0}".format(file_extension))




if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, dataset_file)

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        vector_size=150,
        window=10,
        min_count=2,
        workers=10)

    model.train(documents, total_examples=len(documents), epochs=1)

    # Split the file location into directory path and file name with extension
    dir_path, file_name_ext = os.path.split(dataset_file)

    # Split the file name and extension
    file_name, file_ext = os.path.splitext(file_name_ext)

    # save only the word vectors, and save it to a file that is similarly named to the source file
    model.wv.save(os.path.join(abspath, "../vectors/" + file_name + ".kv"))
# Running the Word2Vec Tutorial
1. From the command line, first, clone this repo.
```
git clone https://github.com/clarkjensen/word2vecDemo.git
```
2. Run `python3 scripts/word2vec.py` to create the model (should take about 3 minutes with one epoch of training).  It's going to make a model of hotel reviews by default.  Change the input file on line 35.  You can also make the model more accurate by increasing the epochs (line 50).  10 epochs is a good upper-limit.

3. Review and test the model by running `python3 scripts/loadAndTestModel.py`.  You can add more tests to that file.

Contextualized Word Representations from Distant Supervision with and for NER
================================================================
NER with Wikipedia Distant Supervision Contextualized Embeddings
This repository contains the source code for the NER system presented in the following research publication ([link](http://todo))

    Abbas Ghaddar and Philippe Langlais 
    Contextualized Word Representations from Distant Supervision with and for NER
    TODO

This code is based on the original [bert](https://github.com/google-research/bert) implementation
## Requirements

* python 3.6
* tensorflow>=1.13
* pyhocon (for parsing the configurations)
* fasttext==0.8.3

## Prepare the Data
1. Follow instruction in /data in order to obtain the data, and change the path of `data_dir` in the `experiments.config` file. 

2. Change the `raw_path` variables for [conll](http://www.cnts.ua.ac.be/conll2003/ner/) and [ontonotes](http://conll.cemantix.org/2012/data.html) datasets in `experiments.config` file to `path/to/conll-2003` and `path/to/conll-2012/v4/data` respectively. For conll dataset please rename eng.train eng.testa eng.testb files to conll.train.txt conll.dev.txt conll.test.txt respectively. Also, change `DATA_DIR` in `train_ner.sh` and `cache_emb.sh`.

3. Run: 
 
```
$ python preprocess.py {conll|ontonotes}
$ cd data
$ sh cache_emb.sh {conll|ontonotes}
```

## Training
Once the data preprocessing is completed, you can train and test a model with:
```
$ cd data
$ sh train_ner.sh {conll|ontonotes}
```

## Citation

Please cite the following paper when using our code: 

```
@InProceedings{TODO,
  title={Contextualized Word Representations from Distant Supervision with and for NER}},
  author={Ghaddar, Abbas	and Langlais, Phillippe},
  TODO
}

```

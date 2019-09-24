Contextualized Word Representations from Distant Supervisionwith and for NER
================================================================
NER with Wikipedia Distant Supervision Contextualized Embeddings
This repository contains the source code for the NER system presented in the following research publication ([link](http://todo))

    Abbas Ghaddar and Philippe Langlais 
    Contextualized Word Representations from Distant Supervision with and for NER
    TODO
    
## Requirements

* python 3.6
* tensorflow>=1.13
* pyhocon (for parsing the configurations)
* fasttext == 0.8.3

## Prepare the Data
1. Download the data from [here](https://drive.google.com/open?id=TODO) and unzip the files in data directory.

2. Change the `raw_path` variables for [conll](http://www.cnts.ua.ac.be/conll2003/ner/) and [ontonotes](http://conll.cemantix.org/2012/data.html) datasets in `experiments.config` file to `path/to/conll-2003` and `path/to/conll-2012/v4/data` respectively. For conll dataset please rename eng.train eng.testa eng.testb files to conll.train.txt conll.dev.txt conll.test.txt respectively. 

3. Run: 
 
```
$ python preprocess.py {conll|ontonotes}
$ sh data/cache_emb.sh {conll|ontonotes}
```

## Training
Once the data preprocessing is completed, you can train and test a model with:
```
$ sh data/train_ner.sh {conll|ontonotes}
```

## Citation

Please cite the following paper when using our code: 

```
@InProceedings{ghaddar2018coling,
  title={Contextualized Word Representations from Distant Supervision with and for NER}},
  author={Ghaddar, Abbas	and Langlais, Phillippe},
  TODO
}

```

# HedModTmplGen

Code for ACL 2019 long paper [Ensuring Readability and Data-fidelity using Head-modifier Templates in Deep Type Description Generation](http://arxiv.org) based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## 1 Dependencies

- Python 3.5+

- PyTorch 1.0 

```bash
pip install -r requirements.txt
```

## 2 Template acquicision

### 1) StanfordCoreNLP

Download stanford corenlp from [here](https://github.com/Lynten/stanford-corenlp), and place them in the `corenlp/` folder.

The start CoreNLP by running the following commands within the folder,

```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
```

### 2) Run a demo

Demo can be found in `raw/annotate_desc.py`

```bash
python annotate_desc.py
```

## 3 Quick start

### 1) Dataset preparation

#### Quick version

Download the processed dataset from [here](link), and place them in the `data/dataNK/` folders.

#### More troublesome version

Download the raw infobox data from [here](link), place them in the `raw/infobox/dataNK/`, and run the following to convert raw infobox to sentences.

```bash
python build_dataset.py --config condig/demo-prep.yml
```

It may take a while. :).

### 2) Preprocess

Preprocess dataset into *.pt. 

```bash
python preprocess.py --config config/demo-prep.yml
```

### 3) Train

```bash
python train.py --config config/demo-train.yml
```

### 4) Test

```bash
python test.py --config config/demo-test.yml
```

### 5) Evaluate

#### a) BLEU, ROUGE, METEOR, CIDEr

Simply add `-report_bleu` in test commands, or run

```
python evaluation.py $result_file$ $golden_file$
```

#### b) ModCopy, HedAcc

Start Stanford CoreNLP, then in the folder `eval/`, run

```bash
python modcopy.py -src somewhere/src-test.txt -res result_file
python hedacc.py -src somewhere/src-test.txt -tgt tgt-test.txt -res result_file
```

## 4 Citation

If you find our code or paper useful to your research, please kindly cite our paper.

```tex
@inproceedings{chen2019template,
	title={Ensuring Readability and Data-fidelity using Head-modifier Templates in Deep Type Description Generation},
	author={Jiangjie Chen, Ao Wang, Haiyun Jiang, Suo Feng, Chenguang Li and Yanghua Xiao},
	booktitle={ACL},
	year={2019},
}
```



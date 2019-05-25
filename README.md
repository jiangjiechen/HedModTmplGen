# ISA-NLG

## Environment

- StanfordCoreNLP

```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
```

## Data Format

- `.infobox`: {qid \t {pid, DESC/TMPL_DESC, NAME} \t o{;o}* \n}+. 
  - Typically raw `.infobox` file is composed of IDs. Sampled infobox are composed of strings.
- `.dict`:  {id \t str \n}*

## Preprocess:

- `python rdf2table.py`: extract infobox from rdf, multilingual supported
- `python deunicode.py`: run this twice to deunicode
- `python prop_dealer.py`: deal with properties dicts or sth.
- `python wikidata_sample.py`: evenly sample dataset from raw infobox
- `python dict_dealer.py pid.dict/qid.dict xx.infobox` : segment .dict, pid and qid alike
- `python assemble.py`: assemble ids and dicts & annotate descriptions into templates
- `python deunicode_desc.py`: deunicode descriptions and names. Twice. Jesus.
- `python build_vocab.py`: build vocabulary for infobox into word\tcount.

- `python build_dataset.py -config.xx_prep.yml`: convert raw infobox to sents.
- `python preprocess.py --config config/xx_prep.yml`: opennmt style preprocess, separate w/ previous custom dataloading methods

## Train:

- `python train.py --config config/xx.yml`: literally.

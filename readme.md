This is the code for [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition](https://arxiv.org/abs/2208.04534)

We found previous nested NER related work used different sentence tokenizations, resulting in different number of 
sentences and entities, which would make the comparison between different papers unfair. To solve this 
issue, we propose using the pre-processing scripts under `preprocess` to get the ACE2004, ACE2005 and Genia datasets.
Please refer the [readme](preprocess/readme.md) for more details.

To run the genia dataset, using
```shell
python train.py -n 5 --lr 7e-6 --cnn_dim 200 --biaffine_size 400 --n_head 4 -b 8 -d genia  --logit_drop 0 --cnn_depth 3 
```

for ACE2004, using
```shell
python train.py -n 50 --lr 2e-5 --cnn_dim 120 --biaffine_size 200 --n_head 5 -b 48 -d ace2004 --logit_drop 0.1 --cnn_depth 2
```

for ACE2005, using
```shell
python train.py -n 50 --lr 2e-5 --cnn_dim 120 --biaffine_size 200 --n_head 5 -b 48 -d ace2005 --logit_drop 0 --cnn_depth 2 
```

Here, we set `n_heads`, `cnn_dim` and `biaffine_size` for small number of parameters, based on our experiment, reduce `n_head` and
enlarge `cnn_dim` and `biaffine_size` should get slightly better performance.

### Customized data
If you want to use your own data, please organize your data line like the following way, the data folder should 
have the following files
```text
customized_data/
    - train.jsonelines
    - dev.jsonlines
    - test.jsonlines
```
in each file, each line should be a json object, like the following
```text
{"tokens": ["Our", "data", "suggest", "that", "lipoxygenase", "metabolites", "activate", "ROI", "formation", "which", "then", "induce", "IL-2", "expression", "via", "NF-kappa", "B", "activation", "."], "entity_mentions": [{"entity_type": "protein", "start": 12, "end": 13, "text": "IL-2"}, {"entity_type": "protein", "start": 15, "end": 17, "text": "NF-kappa B"}, {"entity_type": "protein", "start": 4, "end": 5, "text": "lipoxygenase"}, {"entity_type": "protein", "start": 4, "end": 6, "text": "lipoxygenase metabolites"}]}
```
the entity `start` and `end` is inclusive and exclusive, respectively.

* [update in 20220818]  
We add pre-processing code to extract Genia entities from raw data. We split train/dev/test
based on documents to facilitate document-level NER study.

## Nest NER data pre-processing
The Nested NER task aims to extract entities from text, and entities may overlap with each other.
Previously, a lot of work used the three datasets ACE2004, ACE2005 and Genia to verify their model. 
Although for ACE2004 and ACE2005, almost all of previous work follow the document split as suggested in [Joint Mention Extraction and Classification with Mention Hypergraphs](https://aclanthology.org/D15-1102.pdf),
we find that the statistics between each paper are different, the main difference lies in the number 
of sentences. This divergence are mainly caused by using different sentence tokenizer. To facilitate 
future research in this direction, we suggest to use the following pre-processing procedures. And for Genia, since 
it is publicly available, we just add the data in this repo, if you use Genia, please do not forget to cite the 
origianl paper:[GENIA corpus—a semantically annotated corpus for bio-textmining.](https://academic.oup.com/bioinformatics/article/19/suppl_1/i180/227927?login=true)


In order to make this pre-processing easier to follow, we did not use widely standford-corenlp to 
split sentences, but use the pythonic nltk package to do it. 

The code is adapted from the `process_ace.py` from [oneie](http://blender.cs.illinois.edu/software/oneie/) code.

### requirements
```text
python==3.8.13
beautifulsoup4==4.11.1
bs4==4.11.1
lxml==4.9.1
nltk==3.7
```

The ACE2004 corpus can be downloaded from https://catalog.ldc.upenn.edu/LDC2005T09  
The ACE2005 corpus can be downloaded from https://catalog.ldc.upenn.edu/LDC2006T06  

Previous paper generally follow the document split from [Joint Mention Extraction and Classification with Mention Hypergraphs](https://aclanthology.org/D15-1102.pdf).
The document split are presented in
```text
- splits
    - ace2004
        - dev.txt
        - test.txt
        - train.txt
    - ace2005
        - dev.txt
        - test.txt
        - train.txt
```
In this repo, we follow this split. 

After download the ACE2004 and ACE2005 raw corpus, please unzip the corpus and 
place them into the data folder, the folder should look like
```text
- data
    - ace05  # This is the ACE2005
        - data
            - Arabic
            - Chinese
            - English
        - docs
        - dtd
        - index.html
    - ace_multilang_tr  # This is the ACE2004
        - data
            - Arabic
            - Chinese
            - English
        - docs
        - dtd
        - index.html
```



### For ACE2004
Simply run the following code
```bash
python process_ace2004.py
```
You should get similar output as follows (If you run this code for the first time, nltk may 
ask you to download a model, please follow the instruction to download it)
```text
Converting the dataset to JSON format
#SGM files: 451
100%|██████████| 451/451 [00:05<00:00, 86.37it/s] 
Converting the dataset to OneIE format
skip num: 0
Splitting the dataset into train/dev/test sets
```
After this, you can find the following files in outputs/ace2005 folder
```text
- outputs
    - ace20054
        - dev.jsonlines
        - test.jsonlines
        - train.jsonlines
        - english.jsonlines
        - english.oneie.jsonlines
```
### For ACE2005
Simplely run the following code
```bash
python process_ace2005.py
```
You should get similar output as follows (If you run this code for the first time, nltk may 
ask you to download a model, please follow the instruction to download it)
```text
Converting the dataset to JSON format
#SGM files: 599
100%|██████████| 599/599 [00:11<00:00, 53.32it/s]
Converting the dataset to OneIE format
skip num: 0
Splitting the dataset into train/dev/test sets
```
After this, you can find the following files in outputs/ace2005 folder
```text
- outputs
    - ace2005
        - dev.jsonlines
        - test.jsonlines
        - train.jsonlines
        - english.jsonlines
        - english.oneie.jsonlines
```
each line is a json object, it should look like the following (the start is inclusive and end is exclusive)
```text
{
    "doc_id": "CNN_IP_20030405.1600.00-3",
    "sent_id": "CNN_IP_20030405.1600.00-3-0",
    "tokens":
    [
        "JULIET",
        "BREMNER",
        ",",
        "ITV",
        "NEWS",
        "(",
        "voice",
        "-",
        "over",
        ")"
    ],
    "sentence": " JULIET BREMNER, ITV NEWS (voice-over)",
    "entity_mentions":
    [
        {
            "id": "CNN_IP_20030405.1600.00-3-E32-70",
            "text": "JULIET BREMNER",
            "entity_type": "PER",
            "mention_type": "NAM",
            "entity_subtype": "Individual",
            "start": 0,
            "end": 2
        },
        {
            "id": "CNN_IP_20030405.1600.00-3-E40-69",
            "text": "ITV NEWS",
            "entity_type": "ORG",
            "mention_type": "NAM",
            "entity_subtype": "Media",
            "start": 3,
            "end": 5
        }
    ]
}

```

### Genia
For Genia, we generally follow the train/dev/test split as  [BARTNER](https://aclanthology.org/2021.acl-long.451/) and [W2NER](https://arxiv.org/pdf/2112.10070.pdf).
But we find that in each split, there exist some duplicated text, such as `(ABSTRACT TRUNCATED AT 250 WORDS)`, we keep one 
sentence for each duplicated sentences. Besides, there are some annotation conflict in Genia, we fix them manually, the 
result files are included in the `outputs/genia` folder.


### Statistics
You can use the following command to get the statistics for each dataset
```shell
python statistics.py -f outputs/genia
```


The Statistics for ACE2004, ACE2005 and Genia are as follows  

|          |                   |        | ACE2004 |       |        | ACE2005 |       |        | Genia |       |
|:--------:|:-----------------:|:------:|:-------:|:-----:|:------:|:-------:|:-----:|--------|-------|-------|
|          |                   | Train  |   Dev   | Test  | Train  |   Dev   | Test  | Train  | Dev   | Test  |
|          |    Total Sent.    |  6297  |   742   |  824  |  7178  |   960   | 1051  | 14957  | 1667  | 1850  |
| Sentence | Avg. Sent. Length | 23.36  |  24.26  | 24.03 | 20.87  |  20.57  | 18.65 | 25.35  | 26.04 | 26.03 |
|          | Max Sent. Length  |  120   |   98    |  113  |  139   |   99    |  88   | 149    | 149   | 106   |
|          |    Total Ent.     | 22231  |  2514   | 3036  | 25300  |  3321   | 3099  | 45133  | 5365  | 5506  |
|  Entity  | Avg. Ent. Length  |  2.63  |  2.67   | 2.68  |  2.42  |  2.26   | 2.40  | 1.96   | 1.97  | 2.09  |
|          |   # Nested Ent.   | 10176  |  1092   | 1422  | 10005  |  1214   | 1186  | 7995   | 1067  | 1199  |
|  Tokens  |     # Tokens.     | 147128 |  17998  | 19798 | 149843 |  19745  | 19603 | 379231 | 43409 | 48159 |





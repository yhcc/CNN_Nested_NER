from fastNLP.io import Pipe
from transformers import AutoTokenizer
import numpy as np
import sparse
from tqdm import tqdm
import json
from collections import Counter
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle, iob2


class UnifyPipe(Pipe):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 'roberta' in model_name:
            self.add_prefix_space = True
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        elif 'deberta' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif 'bert' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        else:
            raise RuntimeError(f"Unsupported {model_name}")


class SpanNerPipe(UnifyPipe):
    def __init__(self, model_name, max_len=400):
        super(SpanNerPipe, self).__init__(model_name)
        self.matrix_segs = {}  # 用来记录 matrix 最后一维的分别代表啥意思，dict的顺序就是label的顺序，所有value
        self.max_len = max_len

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        labels = set()
        for ins in data_bundle.get_dataset('train'):
            raw_ents = ins['raw_ents']
            for s, e, t in raw_ents:
                labels.add(t)
        labels = list(sorted(labels))
        label2idx = {l:i for i,l in enumerate(labels)}
        def get_new_ins(bpes, spans, indexes):
            bpes.append(self.sep)
            cur_word_idx = indexes[-1]
            indexes.append(0)
            # int8范围-128~127
            matrix = np.zeros((cur_word_idx, cur_word_idx, len(label2idx)), dtype=np.int8)
            ent_target = []
            for _ner in spans:
                s, e, t = _ner
                matrix[s, e, t] = 1
                matrix[e, s, t] = 1
                ent_target.append((s, e, t))
            matrix = sparse.COO.from_numpy(matrix)
            assert len(bpes)<=512, len(bpes)
            new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                               word_len=cur_word_idx, matrix=matrix, ent_target=ent_target)
            return new_ins

        def process(ins):
            raw_words = ins['raw_words']  # List[str]
            raw_ents = ins['raw_ents']  # List[(s, e, t)]
            old_ent_str = Counter()
            has_ent_mask = np.zeros(len(raw_words), dtype=bool)
            for s, e, t in raw_ents:
                old_ent_str[''.join(raw_words[s:e+1])] += 1
                has_ent_mask[s:e+1] = 1
            punct_indexes = []
            for idx, word in enumerate(raw_words):
                # is_upper = True
                # if idx<len(raw_words):
                #     is_upper = raw_words[idx][0].isupper()
                if self.split_name in ('train', 'dev'):
                    if word[-1] == '.' and has_ent_mask[idx] == 0:  # truncate too long sentence.
                        punct_indexes.append(idx+1)

            if len(punct_indexes) == 0 or punct_indexes[-1] != len(raw_words):
                punct_indexes.append(len(raw_words))

            raw_sents = []
            raw_entss = []
            last_end_idx = 0
            for p_i in punct_indexes:
                raw_sents.append(raw_words[last_end_idx:p_i])
                cur_ents = [(s-last_end_idx, e-last_end_idx, t) for s, e, t in raw_ents if last_end_idx<=s<=e<p_i]
                raw_entss.append(cur_ents)
                last_end_idx = p_i

            bpes = [self.cls]
            indexes = [0]
            spans = []
            ins_lst = []
            new_ent_str = Counter()
            for _raw_words, _raw_ents in zip(raw_sents, raw_entss):
                _indexes = []
                _bpes = []
                for s, e, t in _raw_ents:
                    new_ent_str[''.join(_raw_words[s:e+1])] += 1

                for idx, word in enumerate(_raw_words, start=0):
                    if word in word2bpes:
                        __bpes = word2bpes[word]
                    else:
                        __bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                       add_special_tokens=False)
                        word2bpes[word] = __bpes
                    _indexes.extend([idx]*len(__bpes))
                    _bpes.extend(__bpes)
                next_word_idx = indexes[-1]+1
                if len(bpes) + len(_bpes) <= self.max_len:
                    bpes = bpes + _bpes
                    indexes += [i + next_word_idx for i in _indexes]
                    spans += [(s+next_word_idx-1, e+next_word_idx-1, label2idx.get(t), ) for s, e, t in _raw_ents]
                else:
                    new_ins = get_new_ins(bpes, spans, indexes)
                    ins_lst.append(new_ins)
                    indexes = [0] + [i + 1 for i in _indexes]
                    spans = [(s, e, label2idx.get(t), ) for s, e, t in _raw_ents]
                    bpes = [self.cls] + _bpes
            if bpes:
                ins_lst.append(get_new_ins(bpes, spans, indexes))

            assert len(new_ent_str - old_ent_str) == 0 and len(old_ent_str-new_ent_str)==0
            return ins_lst

        for name in data_bundle.get_dataset_names():
            self.split_name = name
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins in tqdm(ds, total=len(ds), desc=name, leave=False):
                # in case there exist some overlong sentences, but no sentence will be overlong if follow the provided pre-processing
                ins_lst = process(ins)
                for ins in ins_lst:
                    new_ds.append(ins)
            data_bundle.set_dataset(new_ds, name)

        setattr(data_bundle, 'label2idx', label2idx)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        self.matrix_segs['ent'] = len(label2idx)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        dl = SpanLoader().load(paths)
        return self.process(dl)


class SpanLoader(Loader):
    def _load(self, path):
        ds = DataSet()
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                entities = data['entity_mentions']
                tokens = data['tokens']
                raw_ents = []
                for ent in entities:
                    raw_ents.append((ent['start'], ent['end']-1, ent['entity_type']))
                _raw_ents = list(set(raw_ents))
                if len(_raw_ents) != len(raw_ents):
                    print("Detect duplicate entities...")
                ds.append(Instance(raw_words=tokens, raw_ents=raw_ents))
        return ds

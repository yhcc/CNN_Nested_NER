"""
This script extracts IE annotations from ACE2005 (LDC2006T06).

Usage:
python process_ace.py \

"""
import collections
import copy
import os
import re
import json
import glob
import sys

import tqdm
from collections import Counter
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from dataclasses import dataclass
from argparse import ArgumentParser
from nltk import (sent_tokenize as sent_tokenize_,
                  wordpunct_tokenize as wordpunct_tokenize_)

TAG_PATTERN = re.compile('<[^<>]+>', re.MULTILINE)

# DOCS_TO_REVISE_SENT = {
#     }


def mask_escape(text: str) -> str:
    """Replaces escaped characters with rare sequences.

    Args:
        text (str): text to mask.

    Returns:
        str: masked string.
    """
    return text.replace('&amp;', 'ҪҪҪҪҪ').replace('&lt;', 'ҚҚҚҚ').replace('&gt;', 'ҺҺҺҺ')


def unmask_escape(text: str) -> str:
    """Replaces masking sequences with the original escaped characters.

    Args:
        text (str): masked string.

    Returns:
        str: unmasked string.
    """
    return text.replace('ҪҪҪҪҪ', '&amp;').replace('ҚҚҚҚ', '&lt;').replace('ҺҺҺҺ', '&gt;')


def wordpunct_tokenize(text: str, language: str = 'english') -> List[str]:
    """Performs word tokenization. For English, it uses NLTK's
    wordpunct_tokenize function. For Chinese, it simply splits the sentence into
    characters.

    Args:
        text (str): text to split into words.
        language (str): available options: english, chinese.

    Returns:
        List[str]: a list of words.
    """
    if language == 'chinese':
        return [c for c in text if c.strip()]
    return wordpunct_tokenize_(text)


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace('\n', ' ')

    def char_offsets_to_token_offsets(self, tokens: List[Tuple[int, int, str]]):
        """Converts self.start and self.end from character offsets to token
        offsets.

        Args:
            tokens (List[int, int, str]): a list of token tuples. Each item in
                the list is a triple (start_offset, end_offset, text).
        """
        start_ = end_ = -1
        for i, (s, e, _) in enumerate(tokens):
            if s == self.start:
                start_ = i
            if e == self.end:
                end_ = i + 1
        if start_ == -1 or end_ == -1 or start_ > end_:
            raise ValueError('Failed to update offsets for {}-{}:{} in {}'.format(
                self.start, self.end, self.text, tokens))
        self.start, self.end = start_, end_

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            dict: a dict of instance variables.
        """
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end
        }

    def remove_space(self):
        """Removes heading and trailing spaces in the span text."""
        # heading spaces
        text = self.text.lstrip(' ')
        self.start += len(self.text) - len(text)
        # trailing spaces
        text = text.rstrip(' ')
        self.text = text
        self.end = self.start + len(text)

    def copy(self):
        """Makes a copy of itself.

        Returns:
            Span: a copy of itself."""
        return Span(self.start, self.end, self.text)


@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    entity_subtype: str
    mention_type: str
    value: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            'text': self.text,
            'entity_id': self.entity_id,
            'mention_id': self.mention_id,
            'start': self.start,
            'end': self.end,
            'entity_type': self.entity_type,
            'entity_subtype': self.entity_subtype,
            'mention_type': self.mention_type
        }
        if self.value:
            entity_dict['value'] = self.value
        return entity_dict


@dataclass
class Sentence(Span):
    sent_id: str
    tokens: List[str]
    entities: List[Entity]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'sent_id': self.sent_id,
            'tokens': [t for t in self.tokens],
            'entities': [entity.to_dict() for entity in self.entities],
            'start': self.start,
            'end': self.end,
            'text': self.text,
        }


@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'doc_id': self.doc_id,
            'sentences': [sent.to_dict() for sent in self.sentences]
        }


# def revise_sentences(sentences: List[Tuple[str, int, int]],
#                      doc_id: str) -> List[Tuple[int, int, str]]:
#     """Automatic sentence tokenization may have errors for a few documents.
#
#     Args:
#         sentences (List[Tuple[str, int, int]]): a list of sentence tuples.
#         doc_id (str): document ID.
#
#     Returns:
#         List[Tuple[str, int, int]]: a list of revised sentence tuples.
#     """
#     sentences_ = []
#
#     offset_list = DOCS_TO_REVISE_SENT[doc_id]
#     first_part_offsets = {offset for offset, _ in offset_list}
#     second_part_offsets = {offset for _, offset in offset_list}
#
#
#     for sentence_idx, (text, start, end) in enumerate(sentences):
#         if start in first_part_offsets:
#             next_text, next_start, next_end = sentences[sentence_idx + 1]
#             space = ' ' * (next_start - end)
#             sentences_.append((text + space + next_text, start, next_end))
#         elif start in second_part_offsets:
#             continue
#         else:
#             sentences_.append((text, start, end))
#
#     return sentences_


def read_sgm_file(path: str, entities,
                  language: str = 'english') -> List[Tuple[str, int, int]]:
    """Reads a SGM text file.

    Args:
        path (str): path to the input file.
        language (str): document language. Valid values: "english" or "chinese".

    Returns:
        List[Tuple[str, int, int]]: a list of sentences. Each item in the list
            is a tuple of three elements, sentence text, start offset, and end
            offset.
    """
    data = open(path, 'r', encoding='utf-8').read()
    # data = re.sub('&AMP;', '&', data)

    # Get the offset of <HEADLINE>
    min_offset = max(0, TAG_PATTERN.sub('', data.replace('<HEADLINE>', '⁂')).find('⁂'))  # only cares content after the <HEADLINE> tag
    clean_text = TAG_PATTERN.sub('', data)[min_offset:]  # since no annotation before min_offset, delete it
    for entity in entities:
        assert len(entity.text) <= entity.end - entity.start
        if 'NYT20001122.2009.0333-E29' == entity.entity_id and entity.mention_id in ('29-34',):
            entity.end = entity.end + 4
            entity.text = entity.text.replace('&', '&AMP;')
        if entity.entity_id in {'NYT20001122.2009.0333-E25', 'NYT20001128.1940.0266-E2',
            'APW20001225.2035.0477-E1', 'APW20001225.2035.0477-E2', 'APW20001225.2035.0477-E53',
            'APW20001225.2035.0477-E2', 'APW20001205.2150.0791-E4'}  and entity.mention_id in ('25-30', '2-6', '1-6', '2-8', '53-105', '2-106', '4-13'):
            entity.end = entity.end + 4
            entity.text = entity.text.replace('&', '&AMP;')
        if 'NYT20001127.2214.0372-E20' == entity.entity_id and entity.mention_id == '20-28':
            entity.end = entity.end + 4
            entity.text = entity.text.replace('AT&T', 'AT&AMP;T')
        assert entity.start>min_offset
        entity.start -= min_offset  # shift for min_offset
        entity.end -= min_offset

    # make sure all entity separate from its neighbor character
    entity_pos = [collections.defaultdict(list) for _ in range(len(clean_text))]
    for entity in entities:
        entity_pos[entity.start]['start'].append(entity)
        entity_pos[entity.end]['end'].append(entity)
    shift_for_more_space = 0
    for i in range(len(entity_pos)):
        if len(entity_pos[i])>0:
            starts = entity_pos[i].get('start', [])
            ends = entity_pos[i].get('end', [])
            assert not(len(starts)!=0 and len(ends)!=0)  # cannot overlap with each other
            if starts:
                for ent in starts:
                    ent.start += shift_for_more_space
                start = starts[0].start
                if start>0 and clean_text[start-1]!=' ':
                    clean_text = clean_text[:start] + ' ' + clean_text[start:]
                    shift_for_more_space += 1
                    for ent in starts:
                        ent.start += 1
            if ends:
                for ent in ends:
                    ent.end += shift_for_more_space
                end = ends[0].end
                if end<len(clean_text) and clean_text[end] != ' ':
                    clean_text = clean_text[:end] + ' ' + clean_text[end:]
                    shift_for_more_space += 1

    # delete extrac space
    clean_text = re.sub('\n', ' ', clean_text)
    shifts = []
    shift = 0
    last_space = False
    for i in range(len(clean_text)):
        if clean_text[i] == ' ' and last_space:
            shift += 1
        shifts.append(shift)
        last_space = clean_text[i] == ' '
    clean_text = re.sub(' +', ' ', clean_text)

    # check all entity can be found in the same way as in the raw text
    within_entity = [0]*len(clean_text)
    for entity in entities:
        entity.start = entity.start - shifts[entity.start]
        entity.end = entity.end - shifts[entity.end]
        assert re.sub(' ', '', entity.text) == re.sub('[\n ]', '', clean_text[entity.start:entity.end]), (entity.text, clean_text[entity.start:entity.end])
        for i in range(entity.start, entity.end):
            within_entity[i] += 1

    sentences = sent_tokenize_(clean_text)
    if 'NYT20001218.2144.0401.sgm' in path:  # this sentence need special process
        sent1 = sentences.pop(0)
        sent2 = sentences.pop(0)
        sentences.insert(0, sent1+sent2)

    # To make sure no entity is splitted into two sentences
    offset = 0
    idx = 0
    new_sentences = []
    while idx<len(sentences):
        sent = sentences[idx]
        start = offset
        if idx<len(sentences)-1:
            _sent = sent
            while within_entity[offset+len(sent)-1]>0 and within_entity[offset+len(sent)]>0:
                offset += len(sent) + 1
                idx += 1
                sent = sentences[idx]
                _sent = _sent + ' ' + sent
            offset += len(sent) + 1
        else:
            _sent = sent
            offset += len(sent) + 1

        new_sentences.append((_sent, start, offset))
        idx += 1
    sentences = new_sentences

    assert sum(map(lambda x:len(x[0]), sentences)) == len(clean_text) - len(sentences)  # consider the space between sentences
    # check no entity is truncated into sentences
    sentence = '\n'.join([x[0] for x in sentences])
    # make sure the entity should still the same as expected
    for entity in entities:
        assert re.sub(' ', '', entity.text) == re.sub(' ', '', sentence[entity.start:entity.end]), (entity.text, sentence[entity.start:entity.end])

    return sentences


def read_apf_file(path: str,
                  time_and_val: bool = False
                 ) -> Tuple[str, str, List[Entity]]:
    """Reads an APF file.

    Args:
        path (str): path to the input file.
        time_and_val (bool): extract times and values or not.

    Returns:
        doc_id (str): document ID.
        source (str): document source.
        entity_list (List[Entity]): a list of Entity instances
    """
    data = open(path, 'r', encoding='utf-8').read()
    soup = BeautifulSoup(data, 'lxml-xml')

    # metadata
    root = soup.find('source_file')
    source = root['SOURCE']
    doc = root.find('document')
    doc_id = doc['DOCID']

    entity_list = []

    # entities: nam, nom, pro
    for entity in doc.find_all('entity'):
        entity_id = entity['ID']
        entity_type = entity['TYPE']
        entity_subtype = entity['CLASS']
        for entity_mention in entity.find_all('entity_mention'):
            mention_id = entity_mention['ID']
            mention_type = entity_mention['TYPE']
            # TODO if you want to add head, you can use entity_mention.find('head')
            head = entity_mention.find('extent').find('charseq')
            start, end, text = int(head['START']), int(head['END']), head.text
            entity_list.append(Entity(start, end, text,
                                      entity_id, mention_id, entity_type,
                                      entity_subtype, mention_type))

    if time_and_val:
        # entities: value
        for entity in doc.find_all('value'):
            enitty_id = entity['ID']
            entity_type = entity['TYPE']
            entity_subtype = entity.get('SUBTYPE', None)
            for entity_mention in entity.find_all('value_mention'):
                mention_id = entity_mention['ID']
                mention_type = 'VALUE'
                extent = entity_mention.find('extent').find('charseq')
                start, end, text = int(extent['START']), int(
                    extent['END']), extent.text
                entity_list.append(Entity(start, end, text,
                                          entity_id, mention_id, entity_type,
                                          entity_subtype, mention_type))

        # entities: timex
        for entity in doc.find_all('timex2'):
            entity_id = entity['ID']
            enitty_type = entity_subtype = 'TIME'
            value = entity.get('VAL', None)
            for entity_mention in entity.find_all('timex2_mention'):
                mention_id = entity_mention['ID']
                mention_type = 'TIME'
                extent = entity_mention.find('extent').find('charseq')
                start, end, text = int(extent['START']), int(
                    extent['END']), extent.text
                entity_list.append(Entity(start, end, text,
                                          entity_id, mention_id, entity_type,
                                          entity_subtype, mention_type,
                                          value=value))

    # remove heading/tailing spaces
    for entity in entity_list:
        entity.remove_space()

    return doc_id, source, entity_list


def process_entities(entities: List[Entity],
                     sentences: List[Tuple[str, int, int]]
                    ) -> List[List[Entity]]:
    """Cleans entities and splits them into lists

    Args:
        entities (List[Entity]): a list of Entity instances.
        sentences (List[Tuple[str, int, int]]): a list of sentences.

    Returns:
        List[List[Entity]]: a list of sentence entity lists.
    """
    sentence_entities = [[] for _ in range(len(sentences))]

    # assign each entity to the sentence where it appears
    not_assigned = []
    for entity in entities:
        start, end = entity.start, entity.end
        assigned = False
        for i, (_, s, e) in enumerate(sentences):
            if start >= s and end <= e:
                sentence_entities[i].append(entity)
                assigned = True
                break
        if not assigned:
            not_assigned.append(entity)
    assert len(not_assigned) == 0

    sentence_entities_cleaned = [[] for _ in range(len(sentences))]
    for i, entities in enumerate(sentence_entities):
        if not entities:
            continue
        # prefer longer entities
        # entities.sort(key=lambda x: (x.end - x.start), reverse=True)
        # chars = [0] * max([x.end for x in entities])
        already_seen_span = set()
        for entity in entities:
            if (entity.start, entity.end, entity.entity_type) in already_seen_span:
                continue
            already_seen_span.add((entity.start, entity.end, entity.entity_type))
            sentence_entities_cleaned[i].append(entity)
        sentence_entities_cleaned[i].sort(key=lambda x: x.start)

    return sentence_entities_cleaned


def tokenize(sentence: Tuple[str, int, int],
             entities: List[Entity],
             language: str = 'english'
            ) -> List[Tuple[int, int, str]]:
    """Tokenizes a sentence.
    Each sentence is first split into chunks that are entity/event spans or words
    between two spans. After that, word tokenization is performed on each chunk.

    Args:
        sentence (Tuple[str, int, int]): Sentence tuple (text, start, end)
        entities (List[Entity]): A list of Entity instances.
        events (List[Event]): A list of Event instances.

    Returns:
        List[Tuple[int, int, str]]: a list of token tuples. Each tuple consists
        of three elements, start offset, end offset, and token text.
    """
    text, start, end = sentence
    text = mask_escape(text)

    # split the sentence into chunks
    splits = {0, len(text)}
    for entity in entities:
        splits.add(entity.start - start)
        splits.add(entity.end - start)
    splits = sorted(list(splits))
    chunks = [(splits[i], splits[i + 1], text[splits[i]:splits[i + 1]])
              for i in range(len(splits) - 1)]

    # tokenize each chunk
    chunks = [(s, e, t, wordpunct_tokenize(t, language=language))
              for s, e, t in chunks]

    # merge chunks and add word offsets
    tokens = []
    for chunk_start, chunk_end, chunk_text, chunk_tokens in chunks:
        last = 0
        chunk_tokens_ = []
        for token in chunk_tokens:
            token_start = chunk_text[last:].find(token)
            if token_start == -1:
                raise ValueError(
                    'Cannot find token {} in {}'.format(token, text))
            token_end = token_start + len(token)
            chunk_tokens_.append((token_start + start + last + chunk_start,
                                  token_end + start + last + chunk_start,
                                  unmask_escape(token)))
            last += token_end
        tokens.extend(chunk_tokens_)
    return tokens


def convert(sgm_file: str,
            apf_file: str,
            time_and_val: bool = False,
            language: str = 'english') -> Document:
    """Converts a document.

    Args:
        sgm_file (str): path to a SGM file.
        apf_file (str): path to a APF file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese. Defaults to 'english'.

    Returns:
        Document: a Document instance.
    """
    doc_id, source, entities = read_apf_file(apf_file, time_and_val=time_and_val)
    sentences = read_sgm_file(sgm_file, entities, language=language)

    # Reivse sentences
    # sentences = revise_sentences(sentences, doc_id)

    # Process entities
    sentence_entities = process_entities(entities, sentences)
    # Tokenization
    sentence_tokens = [tokenize(s, ent, language=language) for s, ent
                       in zip(sentences, sentence_entities)]

    # Convert span character offsets to token indices
    sentence_objs = []
    for i, (toks, ents, sent) in enumerate(zip(
            sentence_tokens, sentence_entities, sentences)):
        sent_id = '{}-{}'.format(doc_id, i)
        # _toks = [t for _, _, t in toks]
        _toks = [token[-1] for token in toks]
        for entity in ents:
            entity.char_offsets_to_token_offsets(toks)
            # entity.text = entity.text
            assert re.sub(' ', '', entity.text) == ''.join(_toks[entity.start:entity.end]), \
                    (re.sub(' ', '', entity.text), ''.join(_toks[entity.start:entity.end]), entity.mention_id)

        sentence_objs.append(Sentence(start=sent[1],
                                      end=sent[2],
                                      text=sent[0],
                                      sent_id=sent_id,
                                      tokens=_toks,
                                      entities=ents))
    return Document(doc_id, sentence_objs)


def convert_batch(input_path: str,
                  output_path: str,
                  time_and_val: bool = False,
                  language: str = 'english'):
    """Converts a batch of documents.

    Args:
        input_path (str): path to the input directory. Usually, it is the path
            to the LDC2006T06/data/English or LDC2006T06/data/Chinese folder.
        output_path (str): path to the output JSON file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese. Defaults to 'english'.
    """
    if language == 'english':
        sgm_files = glob.glob(os.path.join(input_path, '**', '*.sgm'))
    elif language == 'chinese':
        sgm_files = glob.glob(os.path.join(input_path, '**', 'adj', '*.sgm'))
    else:
        raise ValueError('Unknown language: {}'.format(language))
    print('Converting the dataset to JSON format')
    print('#SGM files: {}'.format(len(sgm_files)))
    progress = tqdm.tqdm(total=len(sgm_files))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as w:
        for sgm_file in sgm_files:
            progress.update(1)
            apf_file = sgm_file.replace('.sgm', '.apf.xml')
            doc = convert(sgm_file, apf_file, time_and_val=time_and_val,
                          language=language)
            w.write(json.dumps(doc.to_dict()) + '\n')
    progress.close()


def convert_to_oneie(input_path: str,
                     output_path: str):
    """Converts files to OneIE format.

    Args:
        input_path (str): path to the input file.
        output_path (str): path to the output file.
    """
    print('Converting the dataset to OneIE format')
    skip_num = 0
    with open(input_path, 'r', encoding='utf-8') as r, \
            open(output_path, 'w', encoding='utf-8') as w:
        for line in r:
            doc = json.loads(line)
            for sentence in doc['sentences']:
                tokens = sentence['tokens']
                skip = False
                for token in tokens:
                    if len(token.strip()) == 0:
                        skip = True
                if skip:
                    skip_num += 1
                    continue

                # entities
                entities = []
                for entity in sentence['entities']:
                    entities.append({
                        'id': entity['mention_id'],
                        'text': entity['text'],
                        'entity_type': entity['entity_type'],
                        'mention_type': entity['mention_type'],
                        'entity_subtype': '',
                        'start': entity['start'],
                        'end': entity['end']
                    })
                    assert ''.join(tokens[entity['start']:entity['end']]) == re.sub(' ', '', entity['text']), \
                        (''.join(tokens[entity['start']:entity['end']]), re.sub(' ', '', entity['text']))

                sent_obj = {
                    'doc_id': doc['doc_id'],
                    'sent_id': sentence['sent_id'],
                    'tokens': tokens,
                    'sentence': sentence['text'],
                    'entity_mentions': entities
                }
                w.write(json.dumps(sent_obj) + '\n')
    print('skip num: {}'.format(skip_num))


def split_data(input_file: str,
               output_dir: str,
               split_path: str):
    """Splits the input file into train/dev/test sets.

    Args:
        input_file (str): path to the input file.
        output_dir (str): path to the output directory.
        split_path (str): path to the split directory that contains three files,
            train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
            files is a document ID.
    """
    print('Splitting the dataset into train/dev/test sets')
    train_docs, dev_docs, test_docs = Counter(), Counter(), Counter()
    # load doc ids
    with open(os.path.join(split_path, 'train.txt')) as r:
        train_docs.update([l.split('/')[-1] for l in r.read().strip('\n').split('\n')])
    with open(os.path.join(split_path, 'dev.txt')) as r:
        dev_docs.update([l.split('/')[-1] for l in r.read().strip('\n').split('\n')])
    with open(os.path.join(split_path, 'test.txt')) as r:
        test_docs.update([l.split('/')[-1] for l in r.read().strip('\n').split('\n')])

    # split the dataset
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(os.path.join(output_dir, 'train.jsonlines'), 'w') as w_train, \
        open(os.path.join(output_dir, 'dev.jsonlines'), 'w') as w_dev, \
        open(os.path.join(output_dir, 'test.jsonlines'), 'w') as w_test:
        for line in r:
            inst = json.loads(line)
            doc_id = inst['doc_id']
            if doc_id in train_docs:
                train_docs[doc_id] += 1
                w_train.write(line)
            elif doc_id in dev_docs:
                dev_docs[doc_id] += 1
                w_dev.write(line)
            elif doc_id in test_docs:
                test_docs[doc_id] += 1
                w_test.write(line)

    for name, counter in zip(['train', 'dev', 'test'], [train_docs, dev_docs, test_docs]):
        for key, value in counter.items():
            if value == 0:
                sys.stderr.write(f'Fail to find document:{key} for split:{name}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default='data/ace_multilang_tr/data', help='Path to the input folder')
    parser.add_argument('-o', '--output', default='outputs/ace2004/', help='Path to the output folder')
    parser.add_argument('-s', '--split', default=None,
                        help='Path to the split folder')
    parser.add_argument('--time_and_val', action='store_true',
                        help='Extracts times and values')

    args = parser.parse_args()
    args.lang = 'english'
    input_dir = os.path.join(args.input, args.lang.title())

    # Create a tokenizer based on the model name
    # Convert to doc-level JSON format
    json_path = os.path.join(args.output, '{}.jsonlines'.format(args.lang))
    convert_batch(input_dir, json_path, time_and_val=args.time_and_val,
                  language=args.lang)

    # Convert to OneIE format
    oneie_path = os.path.join(args.output, '{}.oneie.jsonlines'.format(args.lang))
    convert_to_oneie(json_path, oneie_path)

    # Split the data
    split_data(oneie_path, args.output, 'splits/ace2004')
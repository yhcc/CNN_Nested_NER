# -*- coding: utf-8 -*-
"""
23 Jan 2017
To convert GENIA XML annotation file into line-by-line format easier to be processed
The output file will be in the following format:
    For tokenized files (.tok):
        The first line is the tokenized text, separated by spaces
        The second line is the POS tags (post-processed, oversplit tokens have * as POS)
        The third line is the list of annotations, in token offset
        The fourth line is blank
    For non-tokenized files (.span):
        The first line is the original text
        The second line is the list of annotations, in character offset
        The third line is blank
For both types of output files, the annotation is in the following format separated by space:
    First token is the list of indices in the format <start>,<end>(+<start>,<end>)*
    Second token is the type of the entity (e.g., G#protein_molecule)
"""

# Import statements
from __future__ import print_function

import collections
import json

from bs4 import BeautifulSoup as BS
import sys
import os
import re
import math

class Token(object):
    def __init__(self, text, orig_text, start, end, after, before, postag, orig_postag):
        '''Token object to faithfully represent a token

        To be represented faithfully, a token needs to hold:
        - text: The text it is covering, might be normalized
        - orig_text: The original text it is covering, found in the original text
        - start: The start index in the sentence it appears in
        - end: The end index in the sentence it appears in
        - after: The string that appear after this token, but before the next token
        - before: The string that appear before this token, but after the previous token
        - postag: The POS tag of this token, might be adjusted due to oversplitting
        - orig_postag: The POS tag of the original token this token comes from
        Inspired by CoreLabel in Stanford CoreNLP
        '''
        self.text = text
        self.orig_text = orig_text
        self.start = start
        self.end = end
        self.after = after
        self.before = before
        self.postag = postag
        self.orig_postag = orig_postag

class Span(object):
    def __init__(self, start, end):
        '''Span object represents any span with start and end indices'''
        self.start = start
        self.end = end

    def get_text(self, text):
        return text[self.start:self.end]

    def contains(self, span2):
        return self.start <= span2.start and self.end >= span2.end

    def overlaps(self, span2):
        if ((self.start >= span2.start and self.start < span2.end) or
                (span2.start >= self.start and span2.start < self.end)):
            return True
        return False

    def equals(self, span2):
        return self.start == span2.start and self.end == span2.end

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '{},{}'.format(self.start, self.end)

class Annotation(object):
    def __init__(self, spans, label, text=None, parent=None, dis=False):
        '''Annotation object defines an annotation by a list of spans and its label.

        Optionally, this object can hold the containing text, so that the text of this annotation can be recovered by calling get_text method.

        If this annotation is discontiguous (more than one spans), the parent specifies the annotation that contains all the discontiguous entities in the same coordinated expression
        '''
        self.spans = spans
        self.label = label
        self.text = text
        self.parent = parent
        self.dis = dis

    def get_text(self):
        return ' ... '.join(span.get_text(self.text) for span in self.spans)

    def overlaps(self, ann2):
        for span in self.spans:
            for span2 in ann2.spans:
                if span.overlaps(span2):
                    return True
        return False

    def contains(self, ann2):
        for span2 in ann2.spans:
            this_span_is_contained = False
            for span in self.spans:
                if span.contains(span2):
                    this_span_is_contained = True
                    break
            if not this_span_is_contained:
                return False
        return True

    def equals(self, ann2):
        if ann2 is None:
            return False
        for span, span2 in zip(self.spans, ann2.spans):
            if not span.equals(span2):
                return False
        if self.label != ann2.label:
            return False
        return True

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '{} {}'.format('+'.join(str(span) for span in self.spans), self.label)

class Sentence(object):
    def __init__(self, sentence_xml):
        self.text = sentence_xml.get_text()
        self.tokens = self.get_tokens(sentence_xml)
        self.span_annotations, self.token_annotations = self.get_annotations(sentence_xml)

    @staticmethod
    def get_tokens(sentence_xml):
        '''Returns the list of tokens from a sentence

        '''
        tokens = []
        idx = 0
        before = ''
        for token in sentence_xml.find_all('w'):
            token_txt = token.get_text()
            postag = token['c']
            if len(tokens)>=1:
                tokens[-1].after = ' '
            tokens.append(Token(token_txt, token_txt, idx, idx+len(token_txt), '', before, postag, postag))
            idx += len(token_txt) + 1  # add space
            before = ' '
        _text = re.sub(' ', '', ''.join([tok.text for tok in tokens]))
        text = re.sub(' ', '', sentence_xml.get_text())
        assert text.startswith(_text)
        if text!=_text:
            left_text = text[len(_text):]
            tokens.append(Token(left_text, left_text, idx, idx+len(left_text), '', ' ', '*', '*'))
        return tokens

    def get_annotations(self, sentence_xml):
        '''Extracts annotations from a sentence annotation'''
        text = self.text
        tokens = self.tokens
        span_annotations = []
        _text = ' '.join([tok.text for tok in tokens])
        Sentence.process_annotation(sentence_xml, _text, span_annotations, 0)
        token_annotations = []
        # Converts the character-based annotations into token-based annotations
        for annotation in span_annotations:
            span_tokens = []
            for span in annotation.spans:
                span_tokens.append(Sentence.span_to_token(span, tokens))
            token_annotation = Annotation(span_tokens, annotation.label, annotation.text, annotation.parent,
                                          annotation.dis)
            token_annotations.append(token_annotation)
        return span_annotations, token_annotations

    @staticmethod
    def span_to_token(span, tokens):
        '''Returns the list of tokens that covers the given list of character spans'''
        start = -1
        end = -1
        for idx, token in enumerate(tokens):
            if span.start < token.end and start == -1:
                start = idx
            if token.end <= span.end:
                end = idx + 1
        return Span(start, end)

    @staticmethod
    def normalize_lex(lex):
        return lex.replace('-_','-').replace('_-','-').replace('__','_').replace('*_','*').replace('\\*','*').strip('_')

    @staticmethod
    def process_annotation(parent_annotation, text, span_annotations, idx, allow_nest=False):
        '''The method that processes the children of a BeautifulSoup tag
        '''
        # in order to avoid match unexpected span, first get the start_idx for each con
        annotation_starts = []
        tokens = []
        for child in parent_annotation.childGenerator():
            if child.get_text().strip() == '':
                continue
            child = BS(str(child), 'lxml')
            # if '<cons' in str(child):
            if bool(child.find('cons')):
                annotation_starts.append(idx+len(' '.join(tokens)))
            _tokens = [tok.text for tok in Sentence.get_tokens(child)]
            tokens += _tokens
        for _idx, annotation in enumerate(parent_annotation.find_all('cons', recursive=False)):
            ann_txt = r' '.join([tok.text for tok in Sentence.get_tokens(annotation)])
            found_match = None
            for i in range(annotation_starts[_idx], len(text)-len(ann_txt)+1):
                if text[i:i+len(ann_txt)] == ann_txt:
                    found_match = (i, i+len(ann_txt))
                    break

            if found_match is None:
                raise('1:Cannot find {} in {} at {} ({})'.format(annotation.get_text(), text[idx:], idx, text))
            else:
                ann_idx = found_match[0]
            # sanity check
            if not allow_nest and _idx+1<len(annotation_starts):
                assert found_match[-1]<=annotation_starts[_idx+1]
            Sentence.process_annotation(annotation, text, span_annotations, ann_idx, allow_nest=True)
            try:
                ann_lex = annotation['lex']
            except:
                ann_lex = r' '.join([tok.text for tok in Sentence.get_tokens(annotation)])
            ann_lex = Sentence.normalize_lex(ann_lex)
            try:
                ann_sem = annotation['sem']
            except:
                # No sem means this is part of discontiguous entity, should have been handled by the discontiguous entity handler below when it processes the parent
                idx = ann_idx + len(ann_txt)
                continue
            if not ann_sem.startswith('('):
                # This is a contiguous entity
                # Just add it into the list of annotations
                span_annotations.append(Annotation([Span(ann_idx, ann_idx+len(ann_txt))], ann_sem, text))
            # Find all possible constituents of the discontiguous entities
            sub_cons = annotation.find_all('cons', recursive=True)
            if (len(sub_cons) > 1 and (ann_sem.startswith('(') or ann_lex.startswith('*') or ann_lex.endswith('*'))):
                # This contains a discontiguous entity
                # We need to find the spans of each discontiguous entity
                combined_ann = Annotation([Span(ann_idx, ann_idx+len(ann_txt))], ann_sem, text)
                sub_anns = Sentence.parse_lex(ann_lex, ann_sem)
                sub_cons_ann = []
                # Find the character span of each constituent
                for sub_con in sub_cons:
                    sub_con_txt = r' '.join([tok.text for tok in Sentence.get_tokens(sub_con)])
                    sub_con_idx = text.find(sub_con_txt, idx)
                    if sub_con['lex'].startswith('(') or '*' not in sub_con['lex']:
                        # This is contiguous entity, should have been handled by case 1 above
                        continue
                    if sub_con_idx == -1:
                        # This means a constituent cannot be found in its parent constituent, a bug in this script
                        print(sub_cons_ann)
                        raise Exception('2:Cannot find {} in {} at {} ({})'.format(sub_con_txt, text[idx:], idx, text))
                    sub_cons_ann.append((Sentence.normalize_lex(sub_con['lex']), Span(sub_con_idx, sub_con_idx+len(sub_con_txt))))
                    idx = sub_con_idx + len(sub_con_txt)
                # Map each entity to its character span(s)
                for sub_lex, sub_sem in sub_anns:
                    spans = Sentence.find_spans(sub_lex, text, sub_cons_ann)
                    span_annotations.append(Annotation(spans, sub_sem, text, combined_ann, dis=True))
            idx = ann_idx + len(ann_txt)

    @staticmethod
    def parse_lex(lex, sem):
        result = []
        for sub_lex, sub_sem in zip(Sentence.split_lex(lex), Sentence.split_lex(sem)):
            if '#' in sub_sem:
                if sub_lex == 'amino-terminal_(729-766)_region':
                    # Special case, since the text is:
                    # "Deletions of a relatively short amino- (729-766) or carboxy- terminal (940-984) region"
                    #
                    sub_lex = 'amino-(729-766)_terminal_region'
                result.append((Sentence.normalize_lex(sub_lex), sub_sem))
        return result

    @staticmethod
    def split_lex(lex, idx=None):
        '''Parses a lex attribute (might be nested) into a list of basic lex form (i.e., no nested lex)
        '''
        if idx is None:
            idx = [0]
        result = []
        if idx[0] == len(lex):
            return result
        if lex[idx[0]] == '(' and re.match('^\\((AND|OR|BUT_NOT|AS_WELL_AS|VERSUS|TO|NOT_ONLY_BUT_ALSO|NEITHER_NOR|THAN) .+', lex):
            idx[0] = lex.find(' ', idx[0])
            while idx[0] < len(lex) and lex[idx[0]] == ' ':
                idx[0] += 1
                result.extend(Sentence.split_lex(lex, idx))
        else:
            open_bracket_count = 0
            end_of_lex = -1
            for pos, char in enumerate(lex[idx[0]:]):
                if char == '(':
                    open_bracket_count += 1
                elif char == ')':
                    if open_bracket_count > 0:
                        open_bracket_count -= 1
                    else:
                        end_of_lex = pos+idx[0]
                        break
                elif char == ' ':
                    end_of_lex = pos+idx[0]
                    break
            if end_of_lex == -1:
                end_of_lex = len(lex)
            result.append(lex[idx[0]:end_of_lex])
            idx[0] = end_of_lex
        return result

    @staticmethod
    def find_spans(lex, text, cons):
        '''Given an entity from the lex attribute of cons, and the list of possible constituents (with their character spans), return the list of character spans that forms the entity

        The search of the components of the discontiguous entities depends on the lex of the original discontiguous entity (e.g., in (AND class_I_interferon class_II_interferon)) by trying to match the string with the lex of the constituents found inside that tags (e.g., "class*", "*I*", "*II*", "*interferon" for the above case).

        Various checks have been in place to ensure that the entire string is found, while allowing some minor differences.
        '''
        spans = []
        lex_idx = 0
        prev_lex_idx = -1
        lex = lex.lower().strip('*')
        for con_lex, con_span in cons:
            con_lex = con_lex.strip('*').lower()
            con_lex_idx = lex.find(con_lex, lex_idx)
            if con_lex_idx - lex_idx >= 2:
                # Ensure that we don't skip over too many characters
                con_lex_idx = -1
            if con_lex_idx - lex_idx == 1:
                # Skipping one character might be permissible, given that it's not an important character
                if lex[lex_idx] not in ' -_/*':
                    con_lex_idx = -1
            if con_lex_idx == -1:
                # We didn't find this constituent in the parent string
                # Normally we would just skip this and continue checking the next constituent,
                # However, in some cases, a constituent is a prefix of the next constituent.
                # In that case, we might need to back-off the previous match, and try the longer constituent.
                # For example, when trying to match "class_II_interferon", we might match "*I*" to the first "I" of "class_II_interferon", which is incorrect, as it should be matched with "*II*" which comes after "*I*". So the following is the backing-off mechanism to try to match that.
                # Since this theoretically not 100% accurate, each back-off action is logged, and we need to check whether the back-off was correct.
                # For GENIA dataset, there are 53 cases of backing-off, and all of them have been verified correct
                con_lex_idx = lex.find(con_lex, prev_lex_idx)
                if con_lex_idx != -1 and con_lex_idx < lex_idx and len(con_lex) > spans[-1].end-spans[-1].start:
                    print('Found {} from backing off from {} for {}, please check ({}) {}'.format(con_lex, spans[-1].get_text(text), lex, text, cons))
                    del(spans[len(spans)-1])
                    spans.append(Span(con_span.start, con_span.end))
                    prev_lex_idx = lex_idx
                    lex_idx = con_lex_idx + len(con_lex)
                    if con_lex.endswith('-'):
                        lex_idx -= 1
                else:
                    continue
            else:
                spans.append(Span(con_span.start, con_span.end))
                prev_lex_idx = lex_idx
                lex_idx = con_lex_idx + len(con_lex)
                if con_lex.endswith('-'):
                    lex_idx -= 1
        diff = abs(lex_idx-len(lex.rstrip('*')))
        if diff >= 1:
            # To check whether the entity is completed
            print('Cons: {}'.format(cons))
            if diff == 1:
                print('WARNING: differ by one: "{}", found: "{}"'.format(lex, lex[:lex_idx]))
            else:
                print('\n===\nCannot find complete mention of "{}" in "{}", found only "{}"\n===\n'.format(lex, text, lex[:lex_idx]))
        for idx in range(len(spans)-1, 0, -1):
            if spans[idx].start == spans[idx-1].end or text[spans[idx-1].end:spans[idx].start] == ' ':
                spans[idx-1].end = spans[idx].end
                del(spans[idx])
        return spans

def split_train_dev_test(sentences, train_pct=0.8, dev_pct=0.1, test_pct=0.1):
    count = len(sentences)
    train_count = int(train_pct*count)
    dev_count = int(math.ceil(dev_pct*count))
    test_count = int(math.ceil(test_pct*count))
    train_count -= train_count + dev_count + test_count - count
    start_test_idx = train_count+dev_count
    return sentences[:train_count], sentences[train_count:start_test_idx], sentences[start_test_idx:]

def filter_annotations(anns, remove_disc=True, remove_over=False, use_five_types=True):
    result = []
    for ann in anns:
        ann = Annotation(ann.spans[:], ann.label, ann.text, ann.parent, ann.dis)
        if use_five_types:
            if not re.match('G#(DNA|RNA|cell_line|cell_type|protein).*', ann.label):
                continue
            if ann.label not in ['G#cell_line', 'G#cell_type']:
                ann.label = ann.label[:ann.label.find('_')]
        if remove_disc and ann.dis:
            continue
        if remove_disc and len(ann.spans) > 1:
            ann.spans = [Span(ann.spans[0].start, ann.spans[-1].end)]
        if remove_over or (remove_disc and ann.parent is not None):
            need_to_be_removed = False
            for idx in reversed(range(len(result))):
                ann2 = result[idx]
                if not remove_over and remove_disc and ann.parent is not None and ann.parent != ann2.parent:
                    continue
                if ann2.overlaps(ann):
                    if ann2.contains(ann):
                        need_to_be_removed = True
                    elif ann.contains(ann2):
                        del(result[idx])
                    else:
                        # Neither is contained within the other, not nested! Remove one arbitrarily, easier to remove the latter
                        need_to_be_removed = True
            if need_to_be_removed:
                continue
        result.append(ann)
    return result

def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(folder, 'data', 'GENIAcorpus3.02.merged.fixed.xml')
    output_dir = os.path.join(folder, 'outputs', 'genia')
    os.makedirs(output_dir, exist_ok=True)
    with open(xml_path, 'r', encoding='utf8') as infile:
        soup = BS(infile.read(), 'lxml')

    articles = soup.find_all('article')
    docs = {}
    for article in articles:
        doc = {}
        doc_id = article.find_all('bibliomisc')[0].get_text()
        sents = []
        sentences = article.find_all('sentence')
        for idx, sentence in enumerate(sentences):
            sent = Sentence(sentence)
            sents.append(sent)
        doc['sents'] = sents
        docs[doc_id] = doc

    doc2split = {}
    genia_split_path = os.path.join(folder, 'splits', 'genia')
    for name in ['train', 'dev', 'test']:
        with open(os.path.join(genia_split_path, f'{name}.txt'), 'r') as f:
            for line in f:
                doc2split[line.strip()] = name
    result_data = {
        'train': [],
        'test': [],
        'dev': []
    }
    for doc_id, doc in docs.items():
        split = doc2split[doc_id]
        for idx, sent in enumerate(doc['sents']):
            token_anns = filter_annotations(sent.token_annotations)
            entity_mentions = []
            tokens = [token.text for token in sent.tokens]
            add_ents = set()
            for ann in token_anns:
                s, e = ann.spans[0].start, ann.spans[0].end
                t = ann.label
                if (s, e, t) in add_ents:
                    continue
                add_ents.add((s, e, t))
                assert s<e and e<=len(tokens)
                entity_mentions.append({
                    'start': s,
                    'end': e,
                    'entity_type': t.split('#')[-1],
                    'text': ' '.join(tokens[s:e])
                })
            result_data[split].append(
                {
                    'tokens': tokens,
                    'doc_id': doc_id,
                    'sent_id': doc_id + '-' + str(idx),
                    'entity_mentions': entity_mentions
                }
            )
    for key, value in result_data.items():
        with open(os.path.join(output_dir, f'{key}.jsonlines'), 'w') as f:
            for v in value:
                f.write(json.dumps(v) + '\n')

if __name__ == '__main__':
    main()


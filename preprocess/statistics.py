
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', default='outputs/genia', help='Where to read train.jsonlines, dev.jsonlines, test.jsonlines')

args = parser.parse_args()

folder = args.folder

def calculate(path):
    max_sent_len = 0
    total_lengths = 0
    total_ent_length = 0
    max_ent_length = 0
    num_ents = 0
    overlapped_ent_num = 0
    document_ids = set()
    num_sent = 0
    with open(path, 'r') as f:
        for line in f:
            num_sent += 1
            data = json.loads(line.strip())
            document_ids.add(data['doc_id'])
            max_sent_len = max(max_sent_len, len(data['tokens']))
            total_lengths += len(data['tokens'])
            flags = [0]*len(data['tokens'])
            for ent in data['entity_mentions']:
                num_ents += 1
                start, end = ent['start'], ent['end']
                total_ent_length += ent['end'] - ent['start']
                max_ent_length = max(max_ent_length, ent['end'] - ent['start'])
                for i in range(start, end):
                    flags[i] += 1
            for ent in data['entity_mentions']:
                start, end = ent['start'], ent['end']
                if any([flags[i]>1 for i in range(start, end)]):
                    overlapped_ent_num += 1

    print(f"For {path}")
    print("total sentence ", num_sent)
    print("average sentence length ", total_lengths/num_sent)
    print("max sentence length ", max_sent_len)

    print('num_entities ', num_ents)
    print('average entity length  ', total_ent_length/num_ents)
    print('max entity length  ', max_ent_length)
    print("Number of nested entity ", overlapped_ent_num)

    print("Document ", len(document_ids))
    print("Number of tokens ", total_lengths)
    print()


for name in ['train', 'dev', 'test']:
    path = f'{folder}/{name}.jsonlines'
    calculate(path)




import torch
from fastNLP import Metric
import numpy as np
from .metrics_utils import _compute_f_rec_pre, decode


class NERMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, allow_nested=True):
        super(NERMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres

    def update(self, ent_target, scores, word_len):
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class
        ent_scores = (ent_scores + ent_scores.transpose(1, 2))/2
        span_pred = ent_scores.max(dim=-1)[0]

        span_ents = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        for ents, span_ent, ent_pred in zip(ent_target, span_ents, ent_scores.cpu().numpy()):
            pred_ent = set()
            for s, e, l in span_ent:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                if score[ent_type]>=self.ent_thres:
                    pred_ent.add((s, e, ent_type))
            ents = set(map(tuple, ents))
            self.tp += len(ents.intersection(pred_ent))
            self.pre += len(pred_ent)
            self.rec += len(ents)

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        return res




def _compute_f_rec_pre(tp, rec, pre):
    pre = tp/(pre+1e-6)
    rec = tp/(rec+1e-6)
    f = 2*pre*rec/(pre+rec+1e-6)
    return round(f*100, 2), round(rec*100, 2), round(pre*100, 2)


def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area.
    """
    for start in range(seq_len):
        for end in range(start, seq_len):
            yield (start, end)


def decode(scores, length, allow_nested=False, thres=0.5):
    batch_chunks = []
    for idx, (curr_scores, curr_len) in enumerate(zip(scores, length.cpu().tolist())):
        curr_non_mask = scores.new_ones(curr_len, curr_len, dtype=bool).triu()
        tmp_scores = curr_scores[:curr_len, :curr_len][curr_non_mask].cpu().numpy()  # -1 x 2

        confidences, label_ids = tmp_scores, tmp_scores>=thres
        labels = [i for i in label_ids]
        chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != 0]
        confidences = [conf for label, conf in zip(labels, confidences) if label != 0]

        assert len(confidences) == len(chunks)
        chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
        chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)
        if len(chunks):
            batch_chunks.append(set([(s, e, l) for l,s,e in chunks]))
        else:
            batch_chunks.append(set())
    return batch_chunks


def is_overlapped(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def is_nested(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def is_clashed(chunk1: tuple, chunk2: tuple, allow_nested: bool=True):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)


def filter_clashed_by_priority(chunks, allow_nested: bool=True):
    filtered_chunks = []
    for ck in chunks:
        if all(not is_clashed(ck, ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)

    return filtered_chunks

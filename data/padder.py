import torch
from fastNLP import Padder


class Torch3DMatrixPadder(Padder):
    def __init__(self, num_class, pad_val=-11, batch_size=32, max_len=512):
        super(Torch3DMatrixPadder, self).__init__(pad_val=pad_val, dtype=int)
        self.buffer = torch.full((batch_size, max_len, max_len, num_class), fill_value=self.pad_val,
                                 dtype=torch.float32).clone()

    def __call__(self, field):
        max_len = max([len(f) for f in field])
        buffer = self.buffer[:len(field), :max_len, :max_len].clone()
        buffer.fill_(self.pad_val)
        for i, f in enumerate(field):
            buffer[i, :len(f), :len(f)] = torch.from_numpy(f)

        return buffer

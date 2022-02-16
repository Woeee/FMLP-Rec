# @Time   : 2022/2/14
# @Author : Hui Yu
# @Email  : ishyu@outlook.com

import torch
import random
from torch.utils.data import Dataset

class FMLPRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.max_len = args.max_seq_length

        if data_type=='train':
            for seq in user_seq:
                input_ids = seq[-(self.max_len + 2):-2]  # keeping same as train set
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
        elif data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])
        else:
            self.user_seq = user_seq

        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length


    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.args.item_size)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len
        # Associated Attribute Prediction
        # Masked Attribute Prediction

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                #torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                #torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
            )

        return cur_tensors

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item
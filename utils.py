# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2019/11/07 22:11:33
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from pytorch_pretrained_bert import BertTokenizer

logger = logging.getLogger(__name__)

bert_model = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')

# Adjusted
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS', 
#         'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')

VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-ns', 'I-ns')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2

# Added
divider = '==='

class NerDataset(Dataset):
    def __init__(self, f_path):
        with open(f_path, 'r', encoding='utf-8') as fr:
            # Adjusted
            # entries = fr.read().strip().split('\n\n')
            entries = fr.read().strip().split('\n\n')

        sents, tags_li = [], [] # list of lists
        for entry in entries:
            lines = entry.splitlines()
            i = lines.index(divider)

            # Adjusted
            # words = [line.split()[0] for line in entry.splitlines()]
            # tags = ([line.split()[-1] for line in entry.splitlines()])

            words_user = [line.split()[0] for line in lines[:i]]
            words_agent = [line.split()[0] for line in lines[i+1:]]
            tags_user = ([line.split()[-1] for line in lines[:i]])
            tags_agent = ([line.split()[-1] for line in lines[i+1:]])

            # Adjusted
            
            # if len(words) > MAX_LEN:
            #     # 先对句号分段
            #     word, tag = [], []
            #     for char, t in zip(words, tags):
                    
            #         if char != '。':
            #             if char != '\ue236':   # 测试集中有这个字符
            #                 word.append(char)
            #                 tag.append(t)
            #         else:
            #             sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
            #             tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
            #             word, tag = [], []            
            #     # 最后的末尾
            #     if len(word):
            #         sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
            #         tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
            #         word, tag = [], []

            # else:
            #     sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
            #     tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])


            if len(words_user) + len(words_agent) > MAX_LEN:
                print("length exceeds {}, where user length is {} and agent length is {}".format(MAX_LEN, len(words_user), len(words_agent)))
                local_max_len = min(len(words_user), words_agent)
                sents.append(["[CLS]"] + words_user[:local_max_len] + ["[SEP]"] + words_agent[:local_max_len] + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags_user[:local_max_len] + ['[SEP]'] + tags_agent[:local_max_len] + ["[SEP]"])
            else:
                sents.append(["[CLS]"] + words_user + ["[SEP]"] + words_agent + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags_user + ['[SEP]'] + tags_agent + ["[SEP]"])


        self.sents, self.tags_li = sents, tags_li
                

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []

        # Adjusted

        # for w, t in zip(words, tags):
        #     tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
        #     xx = tokenizer.convert_tokens_to_ids(tokens)
        #     # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

        #     # 中文没有英文wordpiece后分成几块的情况
        #     is_head = [1] + [0]*(len(tokens) - 1)
        #     t = [t] + ['<PAD>'] * (len(tokens) - 1)
        #     yy = [tag2idx[each] for each in t]  # (T,)

        #     x.extend(xx)
        #     is_heads.extend(is_head)
        #     y.extend(yy)


        # This is for tokenzied set
        # for w, t in zip(words, tags):
        #     # Avoid tokenizing again sine ##ding will be split into # 
        #     # tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
        #     tokens = [w]
        #     xx = tokenizer.convert_tokens_to_ids(tokens)
        #     # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

        #     is_head = [1] + [0]*(len(tokens) - 1)
        #     t = [t] + ['<PAD>'] * (len(tokens) - 1)
        #     yy = [tag2idx[each] for each in t]  # (T,)

        #     x.extend(xx)
        #     is_heads.extend(is_head)
        #     y.extend(yy)

        # This is for untokeinzed set


        assert len(words)==len(tags), f"len(words)={len(words)}, len(tags)={len(tags)}, words={words}, tags={tags}"
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

            is_head = [1] + [0]*(len(tokens) - 1)
            t = [t] + ['<PAD>'] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens
    

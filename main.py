# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils import data
from model import Net
from crf import Bert_BiLSTM_CRF
from utils import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag


os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def train(model, iterator, optimizer, criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        # Adjusted
        # words, x, is_heads, tags, y, seqlens = batch
        words, x, is_heads, tags, y, seqlens, mask = batch

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        _y = y # for monitoring
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(x, y, mask=mask) # logits: (N, T, VOCAB), y: (N, T)

        # logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        # y = y.view(-1)  # (N*T,)
        # writer.add_scalar('data/loss', loss.item(), )

        # loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i==0:
            print("=====sanity check======")
            #print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            # print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")


        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f, device):
    model.eval()

    path = "{}_prediction".format(f)

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # Adjusted
            # words, x, is_heads, tags, y, seqlens = batch
            words, x, is_heads, tags, y, seqlens, mask = batch

            x = x.to(device)
            mask = mask.to(device)
            # y = y.to(device)

            _, y_hat = model(x, mask=mask)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open(path, 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())

            # This includes the [SEP] in between the 2 sentences
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                if t != "[SEP]":
                    fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open(path, 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open(path, 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    # final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    # with open(final, 'w', encoding='utf-8') as fout:
    #     result = open(path, "r", encoding='utf-8').read()
    #     fout.write(f"{result}\n")

    #     fout.write(f"precision={precision}\n")
    #     fout.write(f"recall={recall}\n")
    #     fout.write(f"f1={f1}\n")

    # Added
    #os.remove("temp")

    print("precision=%.5f"%precision)
    print("recall=%.5f"%recall)
    print("f1=%.5f"%f1)

    # Added
    # start_idx = 0
    # end_idx = 0
    # test_pred_lines = []
    # test_true_lines = []
    # seq_lines = pd.DataFrame(test_outputs["tokens"])
    # for i,seq in enumerate(seq_lines["tokens"]):
    #     start_idx = end_idx
    #     end_idx = start_idx + len(seq)
    #     adju_pred_line = parse_line(test_y_ns[start_idx:end_idx])
    #     test_true_line = test_outputs["true_labels"][start_idx:end_idx]
    #     test_pred_lines.append(adju_pred_line)
    #     test_true_lines.append(test_true_line)
    # rose_metric(test_true_lines,test_pred_lines)

    # # Metrics —— Token
    # test_pred_tokens = parse_token(test_y_ns)
    # test_true_tokens = parse_token(test_outputs["true_labels"])
    # token_metric(test_true_tokens,test_pred_tokens)

    I_num_proposed = len(y_pred[y_pred==5])
    I_num_correct = (np.logical_and(y_true==y_pred, y_true==5)).astype(np.int).sum()
    I_num_gold = len(y_true[y_true==5])

    B_num_proposed = len(y_pred[y_pred==4])
    B_num_correct = (np.logical_and(y_true==y_pred, y_true==4)).astype(np.int).sum()
    B_num_gold = len(y_true[y_true==4])

    O_num_proposed = len(y_pred[y_pred==3])
    O_num_correct = (np.logical_and(y_true==y_pred, y_true==3)).astype(np.int).sum()
    O_num_gold = len(y_true[y_true==3])

    try:
        I_precision = I_num_correct / I_num_proposed
    except ZeroDivisionError:
        I_precision = 1.0

    try:
        I_recall = I_num_correct / I_num_gold
    except ZeroDivisionError:
        I_recall = 1.0

    try:
        I_f1 = 2*I_precision*I_recall / (I_precision + I_recall)
    except ZeroDivisionError:
        if I_precision*I_recall==0:
            I_f1=1.0
        else:
            I_f1=0

    try:
        B_precision = B_num_correct / B_num_proposed
    except ZeroDivisionError:
        B_precision = 1.0

    try:
        B_recall = B_num_correct / B_num_gold
    except ZeroDivisionError:
        B_recall = 1.0

    try:
        B_f1 = 2*B_precision*B_recall / (B_precision + B_recall)
    except ZeroDivisionError:
        if B_precision*B_recall==0:
            B_f1=1.0
        else:
            B_f1=0

    try:
        O_precision = O_num_correct / O_num_proposed
    except ZeroDivisionError:
        O_precision = 1.0

    try:
        O_recall = O_num_correct / O_num_gold
    except ZeroDivisionError:
        O_recall = 1.0

    try:
        O_f1 = 2*O_precision*O_recall / (O_precision + O_recall)
    except ZeroDivisionError:
        if O_precision*O_recall==0:
            O_f1=1.0
        else:
            O_f1=0
    
    macro_precision = (I_precision + B_precision + O_precision) / 3
    macro_recall = (I_recall + B_recall + O_recall) / 3
    macro_f1 = (I_f1 + B_f1 + O_f1) / 3

    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w', encoding='utf-8') as fout:
        result = open(path, "r", encoding='utf-8').read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")
        fout.write(f"marco_precision={macro_precision}\n")
        fout.write(f"marco_recall={macro_recall}\n")
        fout.write(f"macro_f1={macro_f1}\n")

    # Added
    #os.remove("temp")

    print("marco_precision=%.5f"%macro_precision)
    print("marco_recall=%.5f"%macro_recall)
    print("marco_f1=%.5f"%macro_f1)





    return precision, recall, f1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/02")

    # Adjusted
    # parser.add_argument("--trainset", type=str, default="processed/processed_training_bio.txt")
    # parser.add_argument("--validset", type=str, default="processed/processed_dev_bio.txt")

    parser.add_argument("--trainset", type=str, default="raw/processed_training_bio.txt")
    parser.add_argument("--validset", type=str, default="raw/processed_dev_bio.txt")
    parser.add_argument("--evaluateset", type=str, default="raw/processed_test.txt")

    parser.add_argument("--evaluate_epoch", type=int, default=0)
    parser.add_argument("--gradient", type=int, default=1)

    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Bert_BiLSTM_CRF(tag2idx, gradient=hp.gradient).cuda()
    print('Initial model Done')
    # model = nn.DataParallel(model)

    train_dataset = NerDataset(hp.trainset)

    # Adjusted
    if hp.evaluate_epoch > 0:
        eval_dataset = NerDataset(hp.evaluateset)
    else:
        eval_dataset = NerDataset(hp.validset)
    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if hp.evaluate_epoch > 0:
        path = "./{}/{}.pt".format(hp.logdir, hp.evaluate_epoch)
        model.load_state_dict(torch.load(path))

        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, "evaluate")
        precision, recall, f1 = eval(model, eval_iter, fname, device)

    else:
        print('Start Train...,')
        for epoch in range(1, hp.n_epochs+1):  # 每个epoch对dev集进行测试

            train(model, train_iter, optimizer, criterion, device)

            print(f"=========eval at epoch={epoch}=========")
            if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
            fname = os.path.join(hp.logdir, str(epoch))
            precision, recall, f1 = eval(model, eval_iter, fname, device)

            torch.save(model.state_dict(), f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")


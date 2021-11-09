#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import sys

import json

post = '-ns'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a conversation graph to a set of connected components (i.e. threads).')
    parser.add_argument('--raw_list', help='List of raw text documents containing the raw log content as <filename>:...')
    parser.add_argument('--result', help='File containing the cluster content as <filename>:...')

    args = parser.parse_args()

    train_raw = open('./{}/train_dials.json'.format(args.raw_list))
    dev_raw = open('./{}/dev_dials.json'.format(args.raw_list))
    test_raw = open('./{}/test_dials.json'.format(args.raw_list))

    train_result = open('./{}/processed_training_bio.txt'.format(args.result), "w")
    dev_result = open('./{}/processed_dev_bio.txt'.format(args.result), "w")
    test_result = open('./{}/processed_test.txt'.format(args.result), "w")

    train_json = json.load(train_raw)

    for dialog in train_json:
        for turn in dialog["dialogue"]:
            assert len(turn["user"]["transcript"]) == len(turn["user"]["label"])
            for ut, ul, in zip(turn["user"]["transcript"], turn["user"]["label"]):
                if ul == "B" or ul == "I":
                    ul = ul + post
                if ut.strip() != "" and ul.strip() != "":
                    train_result.write("{} {}\n".format(ut, ul))

            train_result.write("===\n".format(ut, ul))

            assert len(turn["agent"]["transcript"]) == len(turn["agent"]["label"])
            for at, al in zip(turn["agent"]["transcript"], turn["agent"]["label"]):
                if al == "B" or al == "I":
                    al = al + post
                if at.strip() != "" and al.strip() != "":
                    train_result.write("{} {}\n".format(at, al))
            train_result.write("\n".format(at, al))
    
    dev_json = json.load(dev_raw)

    for dialog in dev_json:
        for turn in dialog["dialogue"]:
            assert len(turn["user"]["transcript"]) == len(turn["user"]["label"])
            for ut, ul, in zip(turn["user"]["transcript"], turn["user"]["label"]):
                if ul == "B" or ul == "I":
                    ul = ul + post
                if ut.strip() != "" and ul.strip() != "":
                    dev_result.write("{} {}\n".format(ut, ul))

            dev_result.write("===\n".format(ut, ul))

            assert len(turn["agent"]["transcript"]) == len(turn["agent"]["label"])
            for at, al in zip(turn["agent"]["transcript"], turn["agent"]["label"]):
                if al == "B" or al == "I":
                    al = al + post
                if at.strip() != "" and al.strip() != "":
                    dev_result.write("{} {}\n".format(at, al))
            dev_result.write("\n".format(at, al))

    test_json = json.load(test_raw)

    for dialog in test_json:
        for turn in dialog["dialogue"]:
            assert len(turn["user"]["transcript"]) == len(turn["user"]["label"])
            for ut, ul, in zip(turn["user"]["transcript"], turn["user"]["label"]):
                if ul == "B" or ul == "I":
                    ul = ul + post
                if ut.strip() != "" and ul.strip() != "":
                    test_result.write("{} {}\n".format(ut, ul))

            test_result.write("===\n".format(ut, ul))

            assert len(turn["agent"]["transcript"]) == len(turn["agent"]["label"])
            for at, al in zip(turn["agent"]["transcript"], turn["agent"]["label"]):
                if al == "B" or al == "O":
                    al = al + post
                if at.strip() != "" and al.strip() != "":
                    test_result.write("{} {}\n".format(at, al))
            train_result.write("\n".format(at, al))


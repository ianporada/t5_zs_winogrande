import argparse
import csv
from dataclasses import dataclass
import os
import json
import logging

from tqdm.auto import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed
)


logger = logging.getLogger(__name__)

args_dict = dict(
    data_dir="data",
    output_dir="output",
    eval_type="dev",
    model_name_or_path='t5-11b',
    max_seq_length=80,
    max_decode_len=10,
    eval_batch_size=32,
    seed=0,
    num_workers=4,
)
args = argparse.Namespace(**args_dict)

set_seed(args.seed)

tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
model.parallelize()
model.eval()

@dataclass(frozen=True)
class InputExample:
    example_id: str
    sentence: str
    opt1: str
    opt2: str


def get_examples(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = [json.loads(jline) for jline in f]
    examples = [
        InputExample(
            example_id=line['qID'],
            sentence=line['sentence'],
            opt1=line['option1'],
            opt2=line['option2'],
        )
        for line in lines
    ]
    return examples


class WinograndeDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.inputs = []
        self.opt1_targets = []
        self.opt2_targets = []
        self._build()
    # get item
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        opt1_target_ids = self.opt1_targets[index]["input_ids"].squeeze()
        opt2_target_ids = self.opt2_targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        opt1_target_mask = self.opt1_targets[index]["attention_mask"].squeeze()
        opt2_target_mask = self.opt2_targets[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask,
                "opt1_target_ids": opt1_target_ids, "opt1_target_mask": opt1_target_mask,
                "opt2_target_ids": opt2_target_ids, "opt2_target_mask": opt2_target_mask,
                }
    # get len
    def __len__(self):
        return len(self.inputs)
    # build
    def _build(self):
        fname = os.path.join(self.args.data_dir, "%s.jsonl" % self.args.eval_type)
        examples = get_examples(fname)
        for example in examples:
            self._create_features(example)
    # get output str
    def get_output_str(self, opt):
        return "<extra_id_0> %s" % opt
    # create features
    def _create_features(self, example):
        input_ = example.sentence.replace('_', '<extra_id_0>')
        opt1_target = self.get_output_str(example.opt1)
        opt2_target = self.get_output_str(example.opt2)
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.args.max_seq_length, pad_to_max_length=True, return_tensors="pt",
        )
        # tokenize targets
        opt1_tokenized_targets = self.tokenizer.batch_encode_plus(
            [opt1_target], max_length=self.args.max_decode_len, pad_to_max_length=True, return_tensors="pt",
        )
        opt2_tokenized_targets = self.tokenizer.batch_encode_plus(
            [opt2_target], max_length=self.args.max_decode_len, pad_to_max_length=True, return_tensors="pt",
        )
        self.inputs.append(tokenized_inputs)
        self.opt1_targets.append(opt1_tokenized_targets)
        self.opt2_targets.append(opt2_tokenized_targets)


dataset = WinograndeDataset(tokenizer, args)
loader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

est = []
for batch in tqdm(loader):
    opt1_labels = batch["opt1_target_ids"]
    opt1_labels[opt1_labels[:, :] == tokenizer.pad_token_id] = -100
    opt2_labels = batch["opt2_target_ids"]
    opt2_labels[opt2_labels[:, :] == tokenizer.pad_token_id] = -100
    with torch.torch.no_grad():
        opt1_outputs = model(
            input_ids=batch["source_ids"].cuda(),
            attention_mask=batch["source_mask"].cuda(),
            labels=opt1_labels.cuda(),
            decoder_attention_mask=batch['opt1_target_mask'].cuda()
        )
        opt1_logits = opt1_outputs['logits']
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        opt1_loss = loss_fct(opt1_logits.permute(0, 2, 1), opt1_labels.cuda())
        opt2_outputs = model(
            input_ids=batch["source_ids"].cuda(),
            attention_mask=batch["source_mask"].cuda(),
            labels=opt2_labels.cuda(),
            decoder_attention_mask=batch['opt1_target_mask'].cuda()
        )
        opt2_logits = opt2_outputs['logits']
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        opt2_loss = loss_fct(opt2_logits.permute(0, 2, 1), opt2_labels.cuda())
    est.extend((torch.mean(opt1_loss, 1) < torch.mean(opt2_loss, 1)).flatten().tolist())

output_est = [1 if x else 2 for x in est]
output_est = [[x]*5 for x in output_est]
with open('%s_est.lst' % args.eval_type, 'w') as f:
    csv.writer(f).writerows(output_est)

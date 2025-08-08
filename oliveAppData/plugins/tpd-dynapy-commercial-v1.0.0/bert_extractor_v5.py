# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##############################################################################
#
# Code modified to extract pre-computed feature vectors from BERT.
# Main additions include the handling of the following:
#
# - the output format is pickle (*.pkl) instead of json
# - the output includes the sentence id 
#
# info: nikoletta.basiou@sri.com
############################################################################# 

"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import h5py
import os
import sys
import numpy as np

from transformers import XLMRobertaTokenizer, XLMRobertaModel, BertTokenizer, BertModel
import torch

class XLMRobertaExtractor(object):

    def __init__(self, model_dir, max_seq_length, layers="-1", do_lower_case=True):
        self.max_seq_length = max_seq_length
        self.layer_indexes = [int(x) for x in layers.split(",")]
#        try:
        self.model = XLMRobertaModel.from_pretrained(model_dir, config=os.path.join(model_dir, 'config.json'), output_hidden_states=True)
#        except OSError:
#            f = os.path.join(model_dir, 'pytorch_model.bin')
#            self.model = XLMRobertaModel.from_pretrained(f, config=os.path.join(model_dir, 'config.json'), output_hidden_states=True)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(os.path.join(model_dir, 'sentencepiece.bpe.model'), do_lower_case=do_lower_case)
        self.ENABLE_CHUNKING_AND_MEAN = True
        self.pooling_layer = "-1.-2.-3.-4"
        self.pooling_strategy = "pool_mean"
        if self.pooling_strategy == 'pool_sep':
            self.emb_token_index = -1
        elif self.pooling_strategy == 'pool_cls':
            self.emb_token_index = 0
        else:
            self.emb_token_index = -1

    def extract(self, in_list):

        examples = read_examples_with_sentids(in_list, 'train')
        layer_indexes = [int(x) for x in self.pooling_layer.split(".")]
        device = self.model.device

        if self.ENABLE_CHUNKING_AND_MEAN:
            features = convert_examples_to_features(
                examples=examples, seq_length=12000, tokenizer=self.tokenizer)
        else:
            features = convert_examples_to_features(
                examples=examples, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        sent_emb_array = [] # array to store the BERT sentence embedding as stored in [CLS]
        logids = []

        for feat in features:
            all_layers = []
            if len(feat.tokens) > self.max_seq_length:
                clean_tokens = feat.tokens
                shift = int(self.max_seq_length / 2)
                embeds = []
                ind = 0
                chunks_no = 0
                this_all_layers = []
                while ind + shift <= len(feat.tokens):
                    this_all_layers = []
                    this_tokens = clean_tokens[ind:ind+self.max_seq_length-2]
                    data = self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt')
                    data['input_ids'] = data['input_ids'].to(device)
                    data['attention_mask'] = data['attention_mask'].to(device)
                    this_outputs = self.model(**data)
                    #this_outputs = self.model(**self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt'))
                    this_hidden_states = this_outputs[2]

                    for l in layer_indexes:
                        this_all_layers.append(this_hidden_states[l])

                    this_sent_embed = apply_pooling(self.pooling_strategy, this_all_layers, self.emb_token_index)
                    embeds.append(this_sent_embed.detach().cpu().numpy())

                    ind += shift
                    chunks_no += 1

                if len(feat.tokens) - 2 - ind > 0:
                    this_all_layers = []
                    this_tokens = clean_tokens[ind:]
                    data = self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt')
                    data['input_ids'] = data['input_ids'].to(device)
                    data['attention_mask'] = data['attention_mask'].to(device)
                    this_outputs = self.model(**data)
#                    this_outputs = self.model(**self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt'))
                    this_hidden_states = this_outputs[2]

                    for l in layer_indexes:
                        this_all_layers.append(this_hidden_states[l])

                    this_sent_embed = apply_pooling(self.pooling_strategy, this_all_layers, self.emb_token_index)

                    embeds.append(this_sent_embed.detach().cpu().numpy())

                    chunks_no += 1

                embed = np.vstack(embeds).mean(0)
            else:
                data = self.tokenizer.encode_plus(feat.tokens, add_special_tokens=True, return_tensors='pt')
                data['input_ids'] = data['input_ids'].to(device)
                data['attention_mask'] = data['attention_mask'].to(device)
                outputs = self.model(**data)
#                outputs = self.model(**self.tokenizer.encode_plus(feat.tokens, add_special_tokens=True, return_tensors='pt'))

                hidden_states = outputs[2]

                for l in layer_indexes:
                    all_layers.append(hidden_states[l])

                sent_embed = apply_pooling(self.pooling_strategy, all_layers, self.emb_token_index)
                embed = sent_embed.detach().cpu().numpy()


            sent_emb_array.append(embed)
            logids.append(feat.sent_id)

        return logids, np.vstack(sent_emb_array)



class BertExtractor(object):

    def __init__(self, model_dir, max_seq_length, layers="-1", do_lower_case=True):
        self.max_seq_length = max_seq_length
        self.layer_indexes = [int(x) for x in layers.split(",")]
        f = os.path.join(model_dir, 'pytorch_model.bin')
        self.model = BertModel.from_pretrained(f, config=os.path.join(model_dir, 'config.json'), output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, 'vocab.txt'), do_lower_case=do_lower_case)
        self.ENABLE_CHUNKING_AND_MEAN = True
        self.pooling_layer = "-1.-2.-3.-4"
        self.pooling_strategy = "pool_mean"
        if self.pooling_strategy == 'pool_sep':
            self.emb_token_index = -1
        elif self.pooling_strategy == 'pool_cls':
            self.emb_token_index = 0
        else:
            self.emb_token_index = -1

    def extract(self, in_list):

        examples = read_examples_with_sentids(in_list, 'train')
        layer_indexes = [int(x) for x in self.pooling_layer.split(".")]
        device = self.model.device

        if self.ENABLE_CHUNKING_AND_MEAN:
            features = convert_examples_to_features(
                examples=examples, seq_length=12000, tokenizer=self.tokenizer)
        else:
            features = convert_examples_to_features(
                examples=examples, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        sent_emb_array = [] # array to store the BERT sentence embedding as stored in [CLS]
        logids = []

        for feat in features:
            all_layers = []
            # feat.tokens = [x.decode('utf-8') for x in feat.tokens]
            if len(feat.tokens) > self.max_seq_length:
                clean_tokens = feat.tokens
                shift = int(self.max_seq_length / 2)
                embeds = []
                ind = 0
                chunks_no = 0
                this_all_layers = []
                while ind + shift <= len(feat.tokens):
                    this_all_layers = []
                    this_tokens = clean_tokens[ind:ind+self.max_seq_length-2]
                    data = self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt')
                    data['input_ids'] = data['input_ids'].to(device)
                    data['attention_mask'] = data['attention_mask'].to(device)
                    this_outputs = self.model(**data)
                    #this_outputs = self.model(**self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt'))
                    this_hidden_states = this_outputs[2]

                    for l in layer_indexes:
                        this_all_layers.append(this_hidden_states[l])

                    this_sent_embed = apply_pooling(self.pooling_strategy, this_all_layers, self.emb_token_index)
                    embeds.append(this_sent_embed.detach().cpu().numpy())

                    ind += shift
                    chunks_no += 1

                if len(feat.tokens) - 2 - ind > 0:
                    this_all_layers = []
                    this_tokens = clean_tokens[ind:]
                    data = self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt')
                    data['input_ids'] = data['input_ids'].to(device)
                    data['attention_mask'] = data['attention_mask'].to(device)
                    this_outputs = self.model(**data)
                    #this_outputs = self.model(**self.tokenizer.encode_plus(this_tokens, add_special_tokens=True, return_tensors='pt'))
                    this_hidden_states = this_outputs[2]

                    for l in layer_indexes:
                        this_all_layers.append(this_hidden_states[l])

                    this_sent_embed = apply_pooling(self.pooling_strategy, this_all_layers, self.emb_token_index)

                    embeds.append(this_sent_embed.detach().cpu().numpy())

                    chunks_no += 1

                embed = np.vstack(embeds).mean(0)
            else:
                data = self.tokenizer.encode_plus(feat.tokens, add_special_tokens=True, return_tensors='pt')
                data['input_ids'] = data['input_ids'].to(device)
                data['attention_mask'] = data['attention_mask'].to(device)
                outputs = self.model(**data)
#                outputs = self.model(**self.tokenizer.encode_plus(feat.tokens, add_special_tokens=True, return_tensors='pt'))

                hidden_states = outputs[2]

                for l in layer_indexes:
                    all_layers.append(hidden_states[l])

                sent_embed = apply_pooling(self.pooling_strategy, all_layers, self.emb_token_index)
                embed = sent_embed.detach().cpu().numpy()


            sent_emb_array.append(embed)
            logids.append(feat.sent_id)

        return logids, np.vstack(sent_emb_array)




class InputExample(object):

  def __init__(self, unique_id, text_a, text_b, sent_id):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b
    self.sent_id = sent_id

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, sent_id):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
    self.sent_id = sent_id


def apply_pooling(pooling_strategy, all_layers, emb_token_index=-1):
    token_vecs = torch.cat(tuple(all_layers), dim=1)[0]

    if pooling_strategy == 'pool_mean':
        # mean pooling: get the mean of token embeddings
        pooled_embed = torch.mean(token_vecs, dim=0)
    elif pooling_strategy == 'pool_max':
        # max pooling: get the max of token embeddings
        pooled_embed, sent_embed_ind = torch.max(token_vecs, dim=0)
    elif pooling_strategy == 'pool_mean_max':
        # mean_max concat: concatenate the mean and max of token embeddings
        mean_sent_embed = torch.mean(token_vecs, dim=0)
        max_sent_embed, sent_embed_ind = torch.max(token_vecs, dim=0)
        pooled_embed = torch.cat((mean_sent_embed, max_sent_embed))
    elif pooling_strategy == 'pool_cls' or FLAGS.pooling_strategy == 'pool_sep':
        # cls/sep embedding: get the embedding of 'CLS' or 'SEP' token
        pooled_embed = token_vecs[emb_token_index]
    elif pooling_strategy == 'pool_concat_layers_mean':
        # concatenate the token embeddings from multiple layers and take the mean of token embeddings
        token_vecs=torch.cat(tuple(all_layers), dim=2)[0]
        pooled_embed = torch.mean(token_vecs, dim=0)

    return pooled_embed

def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []

    # The two lines below are commented because RoBERTa type of embeddings have diffent start/end tokens
    #tokens.append("[CLS]")
    #input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    # The two lines below are commented because RoBERTa type of embeddings have diffent start/end tokens
    #tokens.append("[SEP]")
    #input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      # The two lines below are commented because RoBERTa type of embeddings have diffent start/end tokens
      #tokens.append("[SEP]")
      #input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            sent_id=example.sent_id))
  return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      #print("v1 line:", line)
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)

      #print("v1 text_a:", text_a)
      #print("v1 text_b:", text_b)

      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
    #print("v1 examples:", examples)
  return examples


def read_examples_with_sentids(input_list, set_type='train'):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  for input_line in input_list:
      sent_id, raw_text = input_line.split(' ', 1)
      line = raw_text
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)

      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b, sent_id=sent_id))
      unique_id += 1

  return examples

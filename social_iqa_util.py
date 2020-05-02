# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import json
import glob
import json
import logging
import os
from typing import List

import tqdm
import pickle

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SocialIQaProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        #middle = os.path.join(data_dir, "train/middle")
        #high = self._read_txt(high)
        #middle = self._read_txt(middle)
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        #middle = os.path.join(data_dir, "train/middle")
        #high = self._read_txt(high)
        #middle = self._read_txt(middle)
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_tst.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        #middle = os.path.join(data_dir, "train/middle")
        #high = self._read_txt(high)
        #middle = self._read_txt(middle)
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"]
            #race_id = "%s-%s" % (set_type, data_raw["race_id"])
            #article = data_raw["article"]
            #for i in range(len(data_raw["answers"])):
                #truth = str(ord(data_raw["answers"][i]) - ord("A"))
                #question = data_raw["questions"][i]
                #options = data_raw["options"][i]

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question,
                    contexts=[context,context,context],
                    endings=[endings[0], endings[1], endings[2]],#, options[3]
                    label=label,
                )
            )
        return examples


class SocialIQaOBProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        qapath = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")
        with open(qapath, 'r', encoding='utf-8') as f:
            data = f.readlines()

        kbpath = os.path.join(data_dir, "socialIQa_v1.4_trn_knowledge_reranked10.pkl")
        kb_data = pickle.load( open(kbpath, "rb" ) )
        return self._create_examples(data, kb_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        qapath = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(qapath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        kbpath = os.path.join(data_dir, "socialIQa_v1.4_dev_knowledge_reranked10.pkl")
        kb_data = pickle.load( open(kbpath, "rb" ) )
        return self._create_examples(data, kb_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_tst.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        kbpath = os.path.join(data_dir, "socialIQa_v1.4_tst_knowledge_reranked10.pkl")
        kb_data = pickle.load( open(kbpath, "rb" ) )
        return self._create_examples(data, kb_data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    # answer = context + answer
    # context = concat kb
    def _create_examples(self, lines, kb_data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            
            context_a_list = kb_data[idx]['answerA']
            context_b_list = kb_data[idx]['answerB']
            context_c_list = kb_data[idx]['answerC']

            context_a = ""
            for l in context_a_list[:1]:
                context_a += l.replace("\n",". ")
            context_a = context_a[:-1]

            context_b = ""
            for l in context_b_list[:1]:
                context_b += l.replace("\n",". ")
            context_b = context_b[:-1]

            context_c = ""
            for l in context_c_list[:1]:
                context_c += l.replace("\n",". ")
            context_c = context_c[:-1]
            
            
            question = item["context"] + item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"]
            #race_id = "%s-%s" % (set_type, data_raw["race_id"])
            #article = data_raw["article"]
            #for i in range(len(data_raw["answers"])):
                #truth = str(ord(data_raw["answers"][i]) - ord("A"))
                #question = data_raw["questions"][i]
                #options = data_raw["options"][i]

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question,
                    contexts=[context_a,context_b,context_c],
                    endings=[endings[0], endings[1], endings[2]],#, options[3]
                    label=label,
                )
            )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending
            if len(text_a) == 0:
                logger.info("context of example %d have length 0"  % (ex_index))
                text_a = " "

            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))

    return features


processors = {
    "socialiqa": SocialIQaProcessor,
    "socialiqa_ob": SocialIQaOBProcessor,
    }

#MULTIPLE_CHOICE_TASKS_NUM_LABELS = {    "socialiqa", 3   }

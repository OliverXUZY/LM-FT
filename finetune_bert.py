#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from src.trainArgs import task_to_keys, task_num_labels, DataTrainingArguments, ModelArguments
import src

import torch
import torch.nn as nn


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


# hold out 'mrpc' for testing for now
task_names = ['cola', 'sst2', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'snli', 
            'trec', 'mpqa', 'cr', 'sst5', 'mr', 'subj']

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}") # not print training args for now

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    for meta_epoch in range(10):
        tasks_subset = random.sample(task_names,data_args.num_task)
        tasks = {}
        for task_name in tasks_subset:
            tasks[task_name]=src.load_taskdata(task_name, model_args, data_args, training_args, logger)
            if task_name == "snli":
                tasks[task_name]['train'] = tasks[task_name]['train'].filter(lambda example: example["label"] != -1)
                tasks[task_name]['validation'] = tasks[task_name]['validation'].filter(lambda example: example["label"] != -1)
                tasks[task_name]['test'] = tasks[task_name]['test'].filter(lambda example: example["label"] != -1)
            
            if task_name == "cr":
                tasks[task_name]['train'] = tasks[task_name]['train'].filter(lambda example: example["sentence"] != None)
                tasks[task_name]['validation'] = tasks[task_name]['validation'].filter(lambda example: example["sentence"] != None)
                
            if not training_args.do_predict:
                if tasks[task_name].get('test'):
                    del tasks[task_name]['test']
            if task_name == 'mnli':
                mnli = tasks[task_name]
                mnli['validation'] = concatenate_datasets([mnli['validation_matched'], mnli['validation_mismatched']])
                mnli['test'] = concatenate_datasets([mnli['test_matched'], mnli['test_mismatched']])
                del mnli['validation_mismatched'], mnli['validation_matched']
                del mnli['test_mismatched'],mnli['test_matched']
                # tasks[task_name] = mnli # not needed, since del is in-place

            if task_name == 'wnli':
                ## too little sample size 
                train_sample = 200
                val_sample = 50
            else:
                train_sample = data_args.num_train_sample_per_task
                val_sample = data_args.num_val_sample_per_task
            # sample subset per task(categories)
            train_set = tasks[task_name]['train']
            val_set = tasks[task_name]['validation']
            
            tasks[task_name]['train'] = train_set.select(random.sample(range(len(train_set)), train_sample))
            tasks[task_name]['validation'] = val_set.select(random.sample(range(len(val_set)), val_sample))

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            # num_labels=5, # not important, we will append our own head
            # finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

        ## load classifier, each dataset has its own classifier
        head_path = os.path.join(training_args.output_dir,"head")
        if not os.path.exists(head_path):
            os.makedirs(head_path)
        classifiers = {}
        for task_name in tasks_subset:
            classifiers[task_name] = nn.Linear(768, task_num_labels[task_name], bias=True).to(DEVICE)
            fname = os.path.join(head_path,f"{task_name}_head.pth")
            if os.path.isfile(fname):
                classifiers[task_name].load_state_dict(torch.load(fname))
            

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        max_seq_length = data_args.max_seq_length

        
        for task in tasks:
            # print(task)
            # Preprocessing the raw_datasets  !!! per task
            sentence1_key, sentence2_key = task_to_keys[task]
            # print(sentence1_key, sentence2_key)
            def preprocess_function(examples):
                # Tokenize the texts
                args = (
                    (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
                )
                result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

                return result
            with training_args.main_process_first(desc="dataset map pre-processing"):
                tasks[task] = tasks[task].map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
        

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
        # we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None
        
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        
        for task in tasks:
            trainer.model.classifier = classifiers[task]
            trainer.model.num_labels = task_num_labels[task]
            trainer.train_dataset = tasks[task]["train"]
            trainer.eval_dataset = tasks[task]["validation"]
            # trainer.tokenizer = tokenizer

            # Training
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples =  len(tasks[task]["train"])
            metrics["train_samples"] = min(max_train_samples, len(tasks[task]["train"]))

            trainer.save_model("./test/save")  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            
            torch.save(classifiers[task].state_dict(), os.path.join(training_args.output_dir,"head", f"{task}_head.pth"))



if __name__ == "__main__":
    main()
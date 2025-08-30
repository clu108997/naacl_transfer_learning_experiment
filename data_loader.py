# Copyright (c) 2019-present, Thomas Wolf.
# All rights reserved. This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.
import logging
import math
import os
from argparse import ArgumentParser
from pprint import pformat

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ignite.contrib.handlers import CosineAnnealingScheduler, create_lr_scheduler_with_warmup
from ignite.engine import Engine, Events
from ignite.metrics import Loss, MetricsLambda

from ignite.handlers import ExpStateScheduler

from pytorch_pretrained_bert import BertTokenizer


from pretraining_model import TransformerWithLMHead
from utils import get_and_tokenize_dataset, average_distributed_scalar, add_logging_and_checkpoint_saving, WEIGHTS_NAME

logger = logging.getLogger(__file__)

def get_data_loaders(args, tokenizer):
    """ Prepare the dataloaders for training and evaluation """
    datasets = get_and_tokenize_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Convert to Tensor and reshape in blocks of the transformer's input length")
    for split_name in ['train', 'valid']:
        tensor = torch.tensor(datasets[split_name], dtype=torch.long)
        num_sequences = (tensor.size(0) // args.num_max_positions) * args.num_max_positions
        datasets[split_name] = tensor.narrow(0, 0, num_sequences).view(-1, args.num_max_positions)

    logger.info("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(datasets['train']) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(datasets['valid']) if args.distributed else None
    train_loader = DataLoader(datasets['train'], sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(datasets['valid'], sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Seq length): {}".format(datasets['train'].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(datasets['valid'].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler, datasets['train_num_words'], datasets['valid_num_words']



def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='wikitext-2', help="One of ('wikitext-103', 'wikitext-2') or a dict of splits paths.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")

    parser.add_argument("--embed_dim", type=int, default=410, help="Embeddings dim")
    parser.add_argument("--hidden_dim", type=int, default=2100, help="Hidden dimension")
    parser.add_argument("--num_max_positions", type=int, default=256, help="Max input length")
    parser.add_argument("--num_heads", type=int, default=10, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=16, help="NUmber of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Normal initialization standard deviation")
    parser.add_argument("--sinusoidal_embeddings", action="store_true", help="Use sinusoidal embeddings")

    parser.add_argument("--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--n_warmup", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--eval_every", type=int, default=-1, help="Evaluate every X steps (-1 => end of epoch)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradient")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="Learning Rate Scheduler Type")
    parser.add_argument("--gamma", type=float, default=0.9, help="gamma argument for exponential scheduler")
    parser.add_argument("--end_lr", type=float, default=0.0, help="end value for scheduler lr")
    

    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log on main process only, logger.warning => log on all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))  # This is a logger.info: only printed on the first process

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, model and optimizer")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)  # Let's use a pre-defined tokenizer
    args.num_embeddings = len(tokenizer.vocab)  # We need this to create the model at next line (number of embeddings to use)
    model = TransformerWithLMHead(args)
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info("Model has %s parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Prepare model for distributed training if needed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler, train_num_words, valid_num_words = get_data_loaders(args, tokenizer)


if __name__ == "__main__":
    train()
    # get_data_loaders()

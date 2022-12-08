import random
import json

import numpy as np
import torch
from transformers import AdamW, BertTokenizer, RobertaTokenizer, DebertaTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup

from common.constants import *
from common.evaluators.bert_hierarchical_evaluator import BertHierarchicalEvaluator
from common.trainers.bert_hierarchical_trainer import BertHierarchicalTrainer
from common.trainers.bert_hierarchical_curriculum_trainer import BertHierarchicalCurriculumTrainer
from datasets.bert_processors.congressional_hearing_processor import CongressionalHearingProcessor
from datasets.bert_processors.congressional_hearing_explanations_processor import CongressionalHearingExplanationsProcessor
from datasets.bert_processors.congressional_hearing_explanations_stratified_processor import CongressionalHearingExplanationsStratifiedProcessor
from datasets.bert_processors.congressional_hearing_easy_processor import CongressionalHearingEasyProcessor
from datasets.bert_processors.congressional_hearing_med_processor import CongressionalHearingMedProcessor
from datasets.bert_processors.congressional_hearing_hard_processor import CongressionalHearingHardProcessor
from models.bert_hier.args import get_args
from models.bert_hier.model import BertHierarchical, RobertaHierarchical, XLNetHierarchical, DebertaHierarchical


def evaluate_split(model, processor, tokenizer, args, save_file, split='dev'):
    evaluator = BertHierarchicalEvaluator(model, processor, tokenizer, args, split)
    scores_fine, scores_coarse = evaluator.get_scores(silent=True)
    print_save_scores(scores_coarse, 'COARSE', save_file+'_coarse', args, split)
    print_save_scores(scores_fine, 'FINE', save_file+'_fine', args, split)


def print_save_scores(scores, score_type, save_file, args, split):
    if args.is_regression:
        rmse, kendall, pearson, spearman, pearson_spearman, avg_loss = scores[0][:6]
        print('\n' + score_type + ': ' + LOG_HEADER_REG)
        print(LOG_TEMPLATE_REG.format(split.upper(), rmse, kendall, pearson, spearman, pearson_spearman, avg_loss))
    else:
        precision, recall, f1, accuracy, avg_loss = scores[0][:5]
        print('\n' + score_type + ': ' + LOG_HEADER_CLASS)
        print(LOG_TEMPLATE_CLASS.format(split.upper(), accuracy, precision, recall, f1, avg_loss))

    scores_dict = dict(zip(scores[1], scores[0]))
    with open(save_file, 'w') as f:
        f.write(json.dumps(scores_dict))


def create_optimizer_scheduler(model, args, num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                                num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)
    return optimizer, scheduler


def run_main(args, curr_fold):
    print('Args: ', args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)
    print('FP16:', args.fp16)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    metrics_dev_json = args.metrics_json + '_dev'
    metrics_test_json = args.metrics_json + '_test'

    dataset_map = {
        'CongressionalHearing': CongressionalHearingProcessor,
        'CongressionalHearingExplanations': CongressionalHearingExplanationsProcessor,
        'CongressionalHearingExplanationsStratified': CongressionalHearingExplanationsStratifiedProcessor,
        'CongressionalHearingEasy': CongressionalHearingEasyProcessor,
        'CongressionalHearingMed': CongressionalHearingMedProcessor,
        'CongressionalHearingHard': CongressionalHearingHardProcessor
    }

    tokenizer_map = {
        'bert': BertTokenizer,
        'roberta': RobertaTokenizer,
        'xlnet': XLNetTokenizer,
        'deberta': DebertaTokenizer
    }

    model_map = {
        'bert': BertHierarchical,
        'roberta': RobertaHierarchical,
        'xlnet': XLNetHierarchical,
        'deberta': DebertaHierarchical
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    if args.task == TASK_REGRESSION:
        args.num_labels = 1
        args.is_multilabel = False
        args.is_regression = True
    else:
        args.num_labels = dataset_map[args.dataset].NUM_CLASSES
        args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL
        args.is_regression = False
    args.parent_to_child_index_map = {0: (0, 1), 1: (2, 3), 2: (4, 5)}
    args.is_hierarchical = False

    processor = dataset_map[args.dataset](args)

    if not args.trained_model: ## PRIYA
        save_path = os.path.join(args.save_path, processor.NAME)
        os.makedirs(save_path, exist_ok=True)

    pretrained_vocab_path = args.model

    if args.model_family == 'deberta':
        tokenizer = tokenizer_map[args.model_family].from_pretrained(pretrained_vocab_path, add_prefix_space=True)
    else:
        tokenizer = tokenizer_map[args.model_family].from_pretrained(pretrained_vocab_path)

    # tokenizer = tokenizer_map[args.model_family].from_pretrained(pretrained_vocab_path)

    train_examples = None
    num_train_optimization_steps = None
    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

    # model_path = './model_checkpoints/bert/CongressionalHearingEasyFolds/fold0/2022-12-05_02-05-58.pt'
    # model_json = torch.load(model_path)
    
    # print(model_json)
    # import pdb; pdb.set_trace()
    # ["model"]
    model = model_map[args.model_family](model_name=args.model,
                                         num_fine_labels=args.num_labels, num_coarse_labels=args.num_coarse_labels,
                                         use_second_input=args.use_second_input)
    # model = torch.load()
    # model_by_fold = ['./model_checkpoints/bert/CongressionalHearingEasyFolds/fold0/2022-12-06_01-44-08.pt', 
    # './model_checkpoints/bert/CongressionalHearingEasyFolds/fold1/2022-12-06_02-12-09.pt',
    # './model_checkpoints/bert/CongressionalHearingEasyFolds/fold2/2022-12-06_02-40-13.pt',
    # './model_checkpoints/bert/CongressionalHearingEasyFolds/fold3/2022-12-06_03-08-18.pt']
    
    # model_by_fold = ['./model_checkpoints/bert/CongressionalHearingMedFolds/fold0/2022-12-06_04-56-16.pt',
    # './model_checkpoints/bert/CongressionalHearingMedFolds/fold1/2022-12-06_06-20-53.pt',
    # './model_checkpoints/bert/CongressionalHearingMedFolds/fold2/2022-12-06_07-16-07.pt',
    # './model_checkpoints/bert/CongressionalHearingMedFolds/fold3/2022-12-06_08-11-19.pt']
    # # if curr_fold is not None:
    # print('Fold: ', curr_fold, 'Model: ', model_by_fold[curr_fold])
    # model = torch.load(model_by_fold[curr_fold])
    model.to(device)

    # Prepare optimizer
    optimizer, scheduler = create_optimizer_scheduler(model, args, num_train_optimization_steps)

    if args.use_curriculum:
        trainer = BertHierarchicalCurriculumTrainer(model, optimizer, processor,
                                      scheduler, tokenizer, args)
    else:
        trainer = BertHierarchicalTrainer(model, optimizer, processor,
                                      scheduler, tokenizer, args)

    trainer.train()
    
    model = torch.load(trainer.snapshot_path)
    # else:
    #   model = torch.load(trainer.snapshot_path)
    # model = torch.load('./model_checkpoints/bert/CongressionalHearingEasyFolds/fold3/2022-12-05_04-46-18.pt')
    # model = torch.load('./model_checkpoints/bert/CongressionalHearingMedFolds/fold3/2022-12-05_08-22-59.pt')
    # model = torch.load('./model_checkpoints/bert/CongressionalHearingMedFolds/fold1/2022-12-05_06-32-33.pt')
    # # model = torch.load('./model_checkpoints/bert/CongressionalHearingEasyFolds/fold0/2022-12-05_02-05-58.pt')
    # model = torch.load('./model_checkpoints/bert/CongressionalHearingMedFolds/fold0/2022-12-05_02-55-08.pt')
    
    
    if trainer.training_converged:
        if args.evaluate_dev:
            evaluate_split(model, processor, tokenizer, args, metrics_dev_json, split='dev')
        if args.evaluate_test:
            evaluate_split(model, processor, tokenizer, args, metrics_test_json, split='test')

    return trainer.training_converged


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    run_main(args)

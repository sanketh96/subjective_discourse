import torch
import numpy as np
from datasets.bert_processors.abstract_processor import convert_examples_to_features
from datasets.bert_processors.congressional_hearing_explanations_processor import CongressionalHearingExplanationsProcessor
from models.bert_hier.model import RobertaHierarchical
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import os
import models.args
from tqdm import tqdm
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_args():
    command = '--dataset CongressionalHearingExplanations --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 4 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_student_expert_7_concat_test.json --first-input-column 16 --use-second-input --second-input-column 2 --use-third-input --third-input-column 38 --use_expert_model --expert_model_path_fold_0 ./model_checkpoints/bert/CongressionalHearingFoldsExplanations/fold0/2022-12-08_00-00-16.pt --expert_model_path_fold_1 ./model_checkpoints/bert/CongressionalHearingFoldsExplanations/fold1/2022-12-08_01-20-56.pt --expert_model_path_fold_2 ./model_checkpoints/bert/CongressionalHearingFoldsExplanations/fold2/2022-12-08_02-43-36.pt --expert_model_path_fold_3 ./model_checkpoints/bert/CongressionalHearingFoldsExplanations/fold3/2022-12-08_04-05-10.pt'
    parser = models.args.get_args()
    parser.add_argument('--model', default=None, type=str, required=True)
    parser.add_argument('--model-family', type=str, default='bert', choices=['bert', 'xlnet', 'roberta', 'albert', 'deberta'])
    parser.add_argument('--dataset', type=str, default='SST-2', choices=['SST-2', 'AGNews', 'Reuters',
                                                                         'CongressionalHearing', 'CongressionalHearingExplanations',
                                                                         'CongressionalHearingBinary', 'AAPD', 'IMDB',
                                                                         'Yelp2014'])
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'bert'))
    parser.add_argument('--cache-dir', default='cache', type=str)
    parser.add_argument('--trained-model', default=None, type=str)
    parser.add_argument('--fp16', action='store_true', help='use 16-bit floating point precision')

    parser.add_argument('--max-seq-length',
                        default=128,
                        type=int,
                        help='The maximum total input sequence length after WordPiece tokenization. \n'
                             'Sequences longer than this will be truncated, and sequences shorter \n'
                             'than this will be padded.')

    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for')

    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass')

    parser.add_argument('--loss-scale',
                        type=float,
                        default=0,
                        help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n'
                             '0 (default value): dynamic loss scaling.\n'
                             'Positive power of 2: static loss scaling value.\n')

    parser.add_argument('--pos-weights',
                        type=str,
                        default=None,
                        help='Comma-separated weights for positive examples in each class to use during the loss')
    parser.add_argument('--pos-weights-coarse',
                        type=str,
                        default=None,
                        help='Comma-separated weights for positive examples in each coarse class to use during the loss')
    parser.add_argument('--loss', type=str, default='cross-entropy',
                        choices=['cross-entropy', 'mse'],
                        help='Loss to use during training for multi-label classification.')
    parser.add_argument('--num-coarse-labels',
                        type=int,
                        default=3,
                        help='Number of coarse-grained labels.')
    parser.add_argument('--id-column', type=int, default=0)
    parser.add_argument('--label-column', type=int, default=1)
    parser.add_argument('--first-input-column', type=int, default=2)
    parser.add_argument('--use-second-input', action='store_true')
    parser.add_argument('--second-input-column', type=int, default=3)
    parser.add_argument('--use-third-input', action='store_true')
    parser.add_argument('--third-input-column', type=int, default=12)
    parser.add_argument('--use-fourth-input', action='store_true')
    parser.add_argument('--fourth-input-column', type=int, default=12)
    parser.add_argument('--num_train_restarts', type=int, default=3)
    parser.add_argument('--use_expert_model', action='store_true')
#     parser.add_argument('--expert_model_path', type=str, default=None)
    parser.add_argument('--expert_model_path_fold_0', type=str, default=None)
    parser.add_argument('--expert_model_path_fold_1', type=str, default=None)
    parser.add_argument('--expert_model_path_fold_2', type=str, default=None)
    parser.add_argument('--expert_model_path_fold_3', type=str, default=None)
    parser.add_argument('--finetune_last_layers_only', action='store_true')
    parser.add_argument('--num_last_layers', type=int, default=None)
    args = parser.parse_args(command.split())
    return args

def reduce_dimensions(data, dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(data)
    return pca.transform(data)

if __name__ == '__main__':
    args = get_args()
    args.fold_num = 0

    pretrained_vocab_path = args.model
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_vocab_path)
    model = torch.load('./model_checkpoints/student_expert_7/CongressionalHearingFoldsExplanations/fold0/2022-12-08_19-49-10.pt')

    processor = CongressionalHearingExplanationsProcessor(args)
    eval_examples = processor.get_test_examples(args.data_dir, is_expert=False)
    eval_features = convert_examples_to_features(eval_examples, args.max_seq_length,
                                                        tokenizer, use_guid=True)

    unpadded_input_ids = [f.input_ids for f in eval_features]
    unpadded_input_mask = [f.input_mask for f in eval_features]
    unpadded_segment_ids = [f.segment_ids for f in eval_features]

    padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
    padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
    padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
    label_ids_fine = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    doc_ids = torch.tensor([f.guid for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids_fine, doc_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    cls_tokens = []
    labels = [] # first class only
    args.device='cuda'
    model.to(args.device).eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, doc_ids = batch
            logits_coarse, logits_fine, output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)  # batch-size, num_classes
            cls_token = output[:, 0, :]
            cls_tokens.append(cls_token.cpu())
            labels.append(label_ids[:,0].cpu()) # first class only

    rep_array = torch.cat(cls_tokens).numpy()
    print(rep_array.shape)
    labels_arr = torch.cat(labels).numpy()
    print(labels_arr.shape)

    x = reduce_dimensions(rep_array, 2)
    fig = plt.figure()
    plt.scatter(x[:,0], x[:,1], c=labels_arr)
    plt.title('PCA of the CLS representation for class 1')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    # plt.legend()
    fig.tight_layout()
    plt.savefig('pca_cls_1.png')
    plt.show()
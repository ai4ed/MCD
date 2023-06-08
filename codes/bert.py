from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=2.5e-05)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--max_len', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=421)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--num_hidden_layers', type=int, default=4)
parser.add_argument('--num_attention_heads', type=int, default=1)
parser.add_argument('--num_area_attention_heads', type=int, default=1)
parser.add_argument('--output_dense_num', type=int, default=1)
parser.add_argument('--sparse_emb_dim', type=int, default=4)
parser.add_argument('--refit',default='dev_rmse')
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--save_dir',default='model/wd_bert')
parser.add_argument('--emb_name', type=str, default="edu_roberta_cls")
parser.add_argument('--feature_type_file', type=str, default="feature_filter")
parser.add_argument('--wdl_mode', type=str, default="WideDeepEF")
parser.add_argument('--area_attention', type=int, default=1)
parser.add_argument('--area_key_mode', type=str, default="mean")
parser.add_argument('--area_value_mode', type=str, default="sum")
parser.add_argument('--max_area_width', type=int, default=4)
# args = parser.parse_args([])
args = parser.parse_args()
print("args is ",args)

import warnings
import os
warnings.filterwarnings("ignore")

import sys
import copy
import torch
torch.set_num_threads(2) 
from torch import nn
import torch.nn.functional as F
sys.path.append('utils/hbm')
from run_hbm import BertConfig
BertLayerNorm = torch.nn.LayerNorm
from utils.competition_utils import set_seed
from utils.metrics_utils import *

seed = args.seed
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.model_utils import load_data_loader


data = load_data_loader(args)

# ## Train

def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


bert_config = BertConfig(seq_length=args.max_len,
                         hidden_size=args.deep_feature_dim,
                         num_labels=args.num_labels,
                         num_hidden_layers=args.num_hidden_layers,
                         num_attention_heads=args.num_attention_heads)
bert_config.pooled = False

from models.wdl_bert_layers import DeepBert
    
args.device = device
args.bert_config = bert_config
args.dnn_hidden_units = [1024,512,256,128,64,32][-args.output_dense_num:]
config = copy.deepcopy(args)

model = DeepBert(config)
model.apply(init_weights)

from models.model_utils import train_model
dfhistory,final_report = train_model(data,model,args,device)
print(final_report)
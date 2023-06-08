from argparse import ArgumentParser
parser = ArgumentParser()
# model train
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--max_len', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--save_dir', default='model/wd_bert')
parser.add_argument('--max_timesteps', type=int, default=400)
parser.add_argument('--refit', default='dev_rmse')
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--use_bias', type=int, default=1)

# wide
parser.add_argument('--feature_type_file', type=str, default="feature_filter")
parser.add_argument('--sparse_emb_dim', type=int, default=1)
parser.add_argument('--wide_dnn_num', type=int, default=2)

# deep
## lstm
parser.add_argument('--emb_name', type=str, default="edu_roberta_cls")
parser.add_argument('--lstm_output_dim', type=int, default=256)
parser.add_argument('--lstm_layer_num', type=int, default=1)
parser.add_argument('--use_bidirectional', type=int, default=1)
parser.add_argument('--deep_use_bias', type=int, default=1)
parser.add_argument('--deep_dnn_num', type=int, default=2)
parser.add_argument('--deep_dnn_dropput', type=float, default=0.1)

## attention
parser.add_argument('--use_attention', type=int, default=1)
parser.add_argument('--num_area_attention_heads', type=int, default=1)
parser.add_argument('--area_attention', type=int, default=1)
parser.add_argument('--area_key_mode', type=str, default="mean")
parser.add_argument('--area_value_mode', type=str, default="sum")
parser.add_argument('--max_area_width', type=int, default=4)
# deep pooling mode
parser.add_argument('--deep_output_pooling_mode', type=str, default="avg",help="avg/gate_avg/wgate_avg/gate_channel_avg/gate_channel_sum/gate_softmax/se_att/ca_att/g_mlp")
parser.add_argument('--se_att_r', type=int, default=1,help="0,1,2,3")
parser.add_argument('--merge_order',type=str,default="aa_gate",help="the order between self-attention and gate. aa_gate/gate_aa/concat/concat_1")
#multi hop
parser.add_argument('--hop', type=int, default=0,help="0,1,2,3")
parser.add_argument('--hop_w_share', type=int, default=0,help="the weight in hop is share or not")
parser.add_argument('--hop_mode', type=str, default="add",help="add,or concat")
# loss type
parser.add_argument("--loss_type", type=str, default="ce")
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=2)

# adversarial training
parser.add_argument("--adt_type", type=str, default="None")
parser.add_argument("--adt_alpha", type=float, default=0.2)
parser.add_argument("--adt_epsilon", type=float, default=1)
parser.add_argument("--adt_k", type=int, default=3)


args = parser.parse_args()
args.hop_w_share = args.hop_w_share==1


import warnings
warnings.filterwarnings("ignore")

import copy
import os
import torch
torch.set_num_threads(2) 
from utils.metrics_utils import *
from utils.competition_utils import set_seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config to bool
args.use_attention = args.use_attention == 1
args.use_bidirectional = args.use_bidirectional == 1
args.use_bias = args.use_bias==1
# args.wide_use_bias = args.wide_use_bias==1
args.deep_use_bias = args.deep_use_bias==1
args.save_model = args.save_model==1
args.area_attention = args.area_attention==1
args.device = device

print("args is ", args)
set_seed(args.seed)  # 设置种子

raw_dense_list = [1024, 512, 256, 128, 64,32]

from models.model_utils import load_data_loader


data = load_data_loader(args)


from models.lstm_layers import Deep

class DeepModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.deep_model = Deep(config)
    def forward(self, x_wide, x_deep):
        deep_logit = self.deep_model(x_deep)
        return deep_logit, deep_logit, deep_logit

config = copy.deepcopy(args)
config.deep_dnn_hidden_units = raw_dense_list[-args.deep_dnn_num:]
model = DeepModel(config)

from models.model_utils import train_model
dfhistory,final_report = train_model(data,model,args,device)
print(final_report)

config_path = final_report['save_path'].replace(".model",'_config.pkl')
import pickle
with open(config_path,'wb') as f:
    pickle.dump(config,f)

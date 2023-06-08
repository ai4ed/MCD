
import torch
import torch.nn as nn
import torch.nn.functional  as F
from deepctr_torch.layers import DNN

from .area_attention import AreaAttention,MultiHeadAreaAttention


def merge_emb(a,b,merge_mode):
    if merge_mode=="add":
        return a+b
    elif merge_mode=="concat":
        return torch.cat([a,b],axis=-1)
    
from .attention_layers import SEAtt,CaAtt,gMLP

class DeepFlatten(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if "deep_output_pooling_mode" not in self.config:
            self.config.deep_output_pooling_mode = "avg"
        self.bilstm = nn.LSTM(input_size=config.deep_feature_dim,
                              num_layers=config.lstm_layer_num,
                              hidden_size=config.max_timesteps,
                              bidirectional=config.use_bidirectional,
                              proj_size=config.lstm_output_dim,
                              batch_first=True)

        input_dim = config.lstm_output_dim * \
            2 if config.use_bidirectional else config.lstm_output_dim
        if config.use_attention:
            if config.area_attention:
                area_attn_core = AreaAttention(
                    key_query_size=input_dim,
                    area_key_mode=config.area_key_mode,
                    area_value_mode=config.area_value_mode,
                    max_area_width=config.max_area_width,
                    memory_width=config.max_len,
                    dropout_rate=0.2,
                )
                self.area_attn = MultiHeadAreaAttention(
                    area_attention=area_attn_core,
                    num_heads=config.num_area_attention_heads,
                    key_query_size=input_dim,
                    key_query_size_hidden=input_dim,
                    value_size=input_dim,
                    value_size_hidden=input_dim
                )
            else:
                self.self_attn = nn.MultiheadAttention(
                input_dim, num_heads=1, batch_first=True)
        
            if self.config.hop!=0:
                hop_mode_v = 1 if config.hop_mode=="add" else 2
                if self.config.hop_w_share:
                    self.w_hop = nn.Linear(input_dim*hop_mode_v,input_dim)
                else:
                    if self.config.hop==1:
                        self.w_hop1 = nn.Linear(input_dim*hop_mode_v,input_dim)
                    elif self.config.hop==2:
                        self.w_hop1 = nn.Linear(input_dim*hop_mode_v,input_dim)
                        self.w_hop2 = nn.Linear(input_dim*hop_mode_v,input_dim)
                    elif self.config.hop==3:
                        self.w_hop1 = nn.Linear(input_dim*hop_mode_v,input_dim)
                        self.w_hop2 = nn.Linear(input_dim*hop_mode_v,input_dim)
                        self.w_hop3 = nn.Linear(input_dim*hop_mode_v,input_dim)
                    else:
                        raise ValueError(f"The hop value should in [0,1,2,3], not {self.config.hop}")
        self.output_dim = input_dim
        if self.config.deep_output_pooling_mode in ["gate_channel_avg","gate_softmax_avg","gate_channel_sum"]:
            self.gate_linear = nn.Linear(input_dim,1)
        if self.config.deep_output_pooling_mode=='wgate_avg':
            self.kernel = nn.Parameter(torch.Tensor(config.max_len,input_dim))
            nn.init.xavier_uniform_(self.kernel)
        if self.config.deep_output_pooling_mode=='se_att':
            r = config.se_att_r
            self.se_att = SEAtt(config.max_len,r)
        if self.config.deep_output_pooling_mode =="g_mlp":
            r = config.se_att_r
            self.g_mlp = gMLP(config.max_len,input_dim,r)
        if self.config.deep_output_pooling_mode=='ca_att':
            r = config.se_att_r
            self.ca_att = CaAtt(config.max_len,input_dim,r)
        if self.config.merge_order.startswith("concat"):
            self.merge_pooling = nn.AdaptiveAvgPool1d(input_dim//2)
            if self.config.merge_order =='concat_14':
                self.concat_14_aa_linear = nn.Sequential(nn.Linear(input_dim,input_dim),
                                                         nn.Sigmoid())
                self.concat_14_aa_gate = nn.Sequential(nn.Linear(input_dim,input_dim),
                                                       nn.Sigmoid())
                self.concat_14_merge = nn.Sequential(nn.Linear(input_dim*2,input_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(input_dim,input_dim),
                                                     nn.ReLU()
                                                     )
                
        
        
    def forward(self, x):
        x_out, (h_n, c_n) = self.bilstm(x)
#         print("x_out_aa.shape is ",x_out_aa.shape)
        if self.config.use_attention:
            if self.config.area_attention:
                x_out_aa = self.area_attn(x_out, x_out, x_out)#(batch_size, num_queries, value_size)
                if self.config.hop!=0:
                    q = x_out_aa
                    if self.config.hop_w_share:#w是共享的
                        for _ in range(self.config.hop):
                            q = self.w_hop(merge_emb(x_out_aa,q,self.config.hop_mode))
                            x_out_aa = self.area_attn(q, q, q)
                    else:
                        if self.config.hop==1:
                            q = self.w_hop1(merge_emb(x_out_aa,q,self.config.hop_mode))
                            x_out_aa = self.area_attn(q, q, q)
                        elif self.config.hop==2:
                            q = self.w_hop1(merge_emb(x_out_aa,q,self.config.hop_mode))
                            x_out_aa = self.area_attn(q, q, q)
                            q = self.w_hop2(merge_emb(x_out_aa,q,self.config.hop_mode))
                            x_out_aa = self.area_attn(q, q, q)
                        elif self.config.hop==3:
                            q = self.w_hop1(merge_emb(x_out_aa,q,self.config.hop_mode))
                            x_out_aa = self.area_attn(q, q, q)
                            q = self.w_hop2(merge_emb(x_out_aa,q,self.config.hop_mode))
                            x_out_aa = self.area_attn(q, q, q)
                            q = self.w_hop3(merge_emb(x_out_aa,q,self.config.hop_mode))
                            x_out_aa = self.area_attn(q, q, q)
            else:
                x_out_aa = self.self_attn(x_out, x_out, x_out, need_weights=False)[0]
            
        if self.config.use_attention:
            if self.config.merge_order=="aa_gate":
                x_gate_input = x_out_aa
            elif self.config.merge_order.startswith("concat"):
                x_gate_input = x_out
        else:
            x_gate_input = x_out

        # print("attention x_out is ",x_out.shape)
        if self.config.deep_output_pooling_mode=='avg':
            x_out_gate = torch.mean(x_gate_input, axis=1)
        elif self.config.deep_output_pooling_mode=='gate_avg':#sigmoid(x)*x,逐点
            x_gate_input = x_gate_input.permute(0,2,1)#换完后是batch_size,emb_dim,max_len
            weights = torch.sigmoid(x_gate_input)
            x_out_gate = torch.mean(weights*x_gate_input,axis=-1)
        elif self.config.deep_output_pooling_mode=='wgate_avg':#sigmoid(wx)*x,逐点
            weights = torch.sigmoid(self.kernel*x_gate_input)#逐点
            x_out_gate = torch.mean(weights*x_gate_input,axis=1)#对max_len
        elif self.config.deep_output_pooling_mode=='gate_channel_avg':#sigmoid(x)*x,输入的每个位置一个权重
            weights = torch.sigmoid(self.gate_linear(x_gate_input))
            x_out_gate = torch.mean(weights*x_gate_input,axis=1)
        elif self.config.deep_output_pooling_mode=='gate_channel_sum':#sigmoid(x)*x
            weights = torch.sigmoid(self.gate_linear(x_gate_input))
            x_out_gate = torch.sum(weights*x_gate_input,axis=1)
        elif self.config.deep_output_pooling_mode=='gate_softmax_avg':#所有位置一起算softmax
            weights = F.softmax(self.gate_linear(x_gate_input),dim=1)
            x_out_gate = torch.mean(weights*x_gate_input,axis=1)
        elif self.config.deep_output_pooling_mode=='se_att':
            x_refit = self.se_att(x_gate_input)
            # print(f"x_refit shape is {x_refit.shape}")
            x_out_gate = torch.mean(x_refit,axis=1)#max_len
        elif self.config.deep_output_pooling_mode=='ca_att':
            x_refit = self.ca_att(x_gate_input)
            x_out_gate = torch.mean(x_refit,axis=1)#max_len
        elif self.config.deep_output_pooling_mode =="g_mlp":
            x_refit = self.g_mlp(x_gate_input)
            x_out_gate = torch.mean(x_refit,axis=1)#max_len
        # get final result
        if self.config.merge_order=="aa_gate":
            x_out = x_out_gate
        elif self.config.merge_order.startswith("concat"):
            concat_mode = self.config.merge_order
            x_out_aa = torch.mean(x_out_aa,axis=1)#avg pooling
            a = x_out_aa
            b = x_out_gate
            
            if concat_mode=="concat":#concat
                x_out = torch.cat([self.merge_pooling(a),
                                   self.merge_pooling(b)],axis=-1)
            elif concat_mode=="concat_1":#sum
                x_out = a + b
            elif concat_mode=="concat_2":#inter
                x_out = a * b
            elif concat_mode=="concat_3":#sigmoid(a)*a concat b
                x_out = torch.cat([self.merge_pooling(torch.sigmoid(a)*a),
                                   self.merge_pooling(b)],axis=-1)
            elif concat_mode=="concat_4":#sigmoid(a)*a sum b
                x_out = torch.sigmoid(a)*a + b
            elif concat_mode=="concat_5":#sigmoid(a)*a inter b
                x_out = torch.sigmoid(a)*a * b
            elif concat_mode=="concat_6":#a concat sigmoid(b)*b
                x_out = torch.cat([self.merge_pooling(a),
                                   self.merge_pooling(torch.sigmoid(b)*b)],axis=-1)
            elif concat_mode=="concat_7":#a sum sigmoid(b)*b
                x_out = a + torch.sigmoid(b)*b
            elif concat_mode=="concat_8":#a inter sigmoid(b)*b
                x_out = a * torch.sigmoid(b)*b
            elif concat_mode=="concat_9":#sigmoid(a)*a concat sigmoid(b)*b
                x_out = torch.cat([self.merge_pooling(torch.sigmoid(a)*a),
                                   self.merge_pooling(torch.sigmoid(b)*b)],axis=-1)
            elif concat_mode=="concat_10":#sigmoid(a)*a sum sigmoid(b)*b
                x_out = torch.sigmoid(a)*a + torch.sigmoid(b)*b
            elif concat_mode=="concat_11":#sigmoid(a)*a inter sigmoid(b)*b
                x_out = torch.sigmoid(a)*a * torch.sigmoid(b)*b
            elif concat_mode=="concat_12":#sigmoid(a) inter b
                x_out = torch.sigmoid(a) * b
            elif concat_mode=="concat_13":#a inter sigmoid(b)
                x_out = a * torch.sigmoid(b)
            elif concat_mode=="concat_14":#a inter sigmoid(b)
                a = self.concat_14_aa_linear(a)*a
                b = self.concat_14_aa_gate(b)*b
                x_out = self.concat_14_merge(torch.cat([a,b],axis=-1))
                
            
        return x_out
    
 
 
 
 
class Deep(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepFlatten(config)
        dnn_hidden_units = config.deep_dnn_hidden_units if "deep_dnn_hidden_units" in config else  (256, 128)
        dnn_dropout = config.deep_dnn_dropput if "deep_dnn_dropput" in config else 0
        self.dnn = DNN(self.model.output_dim, dnn_hidden_units, dropout_rate=dnn_dropout)
        self.out = nn.Linear(dnn_hidden_units[-1], config.num_labels,bias=config.deep_use_bias)

    def forward(self, x,return_details=False):
        x_out_flatten = self.model(x)
        # print(f"x_out_flatten shape is {x_out_flatten.shape}")
        x_out = self.dnn(x_out_flatten)
        # print(f"x_out shape is {x_out.shape}")
        out = self.out(x_out)
        # print(f"out shape is {out.shape}")
        if return_details:
            details = {"x_out_flatten":x_out_flatten,"x_out":x_out}
            return out,details
        return out
 
 
class DeepFlattenOur(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if "deep_output_pooling_mode" not in self.config:
            self.config.deep_output_pooling_mode = "avg"
        self.bilstm = nn.LSTM(input_size=config.deep_feature_dim,
                              num_layers=config.lstm_layer_num,
                              hidden_size=config.max_timesteps,
                              bidirectional=config.use_bidirectional,
                              proj_size=config.lstm_output_dim,
                              batch_first=True)
        input_dim = config.lstm_output_dim * \
            2 if config.use_bidirectional else config.lstm_output_dim
      
        area_attn_core = AreaAttention(
            key_query_size=input_dim,
            area_key_mode=config.area_key_mode,
            area_value_mode=config.area_value_mode,
            max_area_width=config.max_area_width,
            memory_width=config.max_len,
            dropout_rate=0.2,
        )
        self.area_attn = MultiHeadAreaAttention(
            area_attention=area_attn_core,
            num_heads=config.num_area_attention_heads,
            key_query_size=input_dim,
            key_query_size_hidden=input_dim,
            value_size=input_dim,
            value_size_hidden=input_dim
        )
        self.output_dim = input_dim
        r = config.se_att_r
        self.se_att = SEAtt(config.max_len,r)

    def forward(self, x,return_details=False):
        x_out, (h_n, c_n) = self.bilstm(x)
        x_out_aa = self.area_attn(x_out, x_out, x_out,return_details=return_details)#(batch_size, num_queries, value_size)
        if return_details:
            x_out_aa,details = x_out_aa
        x_refit = self.se_att(x_out_aa)
        x_out = torch.mean(x_refit,axis=1)#max_len
        
        if return_details:
            return x_out,details
        return x_out
    

class DeepOur(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepFlattenOur(config)
        dnn_hidden_units = config.deep_dnn_hidden_units if "deep_dnn_hidden_units" in config else  (256, 128)
        dnn_dropout = config.deep_dnn_dropput if "deep_dnn_dropput" in config else 0
        self.dnn = DNN(self.model.output_dim, dnn_hidden_units, dropout_rate=dnn_dropout)
        self.out = nn.Linear(dnn_hidden_units[-1], config.num_labels,bias=config.deep_use_bias)

    def forward(self, x,return_details=False):
        x_out_flatten = self.model(x,return_details=return_details)
        if return_details:
            x_out_flatten,details = x_out_flatten
        print(f"x_out_flatten shape is {x_out_flatten.shape}")
        x_out = self.dnn(x_out_flatten)
        # print(f"x_out shape is {x_out.shape}")
        out = self.out(x_out)
        # print(f"out shape is {out.shape}")
        if return_details:
            details.update({"x_out_flatten":x_out_flatten,"x_out":x_out})
            return out,details
        return out
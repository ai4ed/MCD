import torch
import torch.nn as nn
import torch.nn.functional  as F

class SEAtt(nn.Module):
    def __init__(self, input_dim, r=32):
        """
        Squeeze-and-Excitation attention(SE block) references from Coordinate Attention for Efficient Mobile Network Design
        Args:
            input_dim (_type_): _description_
            r (int, optional): _description_. Defaults to 32.
        """
        super().__init__()
        self.s_layer = nn.Linear(input_dim,input_dim//r)
        self.e_layer = nn.Linear(input_dim//r,input_dim)
        
    def forward(self,x):
        """_summary_

        Args:
            x (_type_): (batch_size,max_len,emb_dim)
        Returns:
            _type_: (batch_size,max_len,emb_dim)
        """
        x = x.permute(0,2,1)#结果是batch_size,emb_dim,max_len
        weights = torch.sigmoid(self.e_layer(F.relu(self.s_layer(x))))
        # print(x.shape,weights.shape)
        refit_x = (x*weights).permute(0,2,1)
        return refit_x
    
class gMLP(nn.Module):
    def __init__(self, max_len,input_dim, r=32):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(input_dim, input_dim//r),nn.GELU())
        self.s_layer = nn.Linear(max_len,max_len)
        self.norm = nn.LayerNorm(input_dim//(r*2))
        self.proj_out = nn.Linear(input_dim//(r*2),input_dim)
        
    def forward(self,x):
        """
        Args:
            x (_type_): (batch_size,max_len,emb_dim)
        Returns:
            _type_: (batch_size,max_len,emb_dim)
        """
        x = self.proj_in(x)
        res, gate = x.chunk(2, dim = -1)#res (batch_size,max_len,emb_dim/(r*2))
        #sgu模块
        gate = self.norm(gate)#batch_size,max_len,dim
        gate = self.s_layer(gate.permute(0,2,1)).permute(0,2,1)
        output = self.proj_out(gate*res)
        return output

class SELayer(nn.Module):
    def __init__(self, input_dim, r=32):
        super().__init__()
        self.s_layer = nn.Linear(input_dim, input_dim//r)
        self.e_layer = nn.Linear(input_dim//r, input_dim)

    def forward(self, x):
        g = torch.sigmoid(self.e_layer(F.relu(self.s_layer(x))))
        return g


class CaAtt(nn.Module):
    """_summary_
    Coordinate Attention for Efficient Mobile Network Design
    Args:
        nn (_type_): _description_
    """
    def __init__(self, max_len, input_dim, r=1):
        super().__init__()
        self.pool_sentence = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_dim = nn.AdaptiveAvgPool2d((1, None))

        # merge
        self.f1 = nn.Linear(max_len+input_dim, max_len+input_dim)

        self.se_sentence = SELayer(max_len, r=r)
        self.se_dim = SELayer(input_dim, r=r)

    def forward(self, x):
        """
        Args:
            x (_type_): (batch_size,max_len,emb_dim)
        Returns:
            _type_: (batch_size,max_len,emb_dim)
        """
        identity = x
        n, h, w = x.size()
        # pooling
        # [1, max_len, 1],对dim的结果pooling，得到长度和max_len一致
        x_s_raw = self.pool_sentence(x)
        x_d_raw = self.pool_dim(x).permute(
            0, 2, 1)  # [1, 1, emb_dim]->[1, emb_dim, 1]

        # merge
        y = torch.cat([x_s_raw, x_d_raw], dim=1)
        y_merge = F.relu(self.f1(y.squeeze(-1)))

        # split
        x_s, x_d = torch.split(y_merge, [h, w], dim=1)
        # se
        x_s = self.se_sentence(x_s).unsqueeze(-1)
        x_d = self.se_dim(x_d).unsqueeze(-1).permute(0, 2, 1)
        # output of our coordinate attention
        x_s = x_s.expand(-1, h, w)
        x_d = x_d.expand(-1, h, w)
        refit_x = identity*x_s*x_d
        return refit_x
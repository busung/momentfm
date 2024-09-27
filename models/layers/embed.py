import math
import warnings

import torch
import torch.nn as nn

import torch.nn.init as init

from momentfm.utils.masking import Masking

import torch.nn.functional as F

import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="MOMENT"):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if (
            self.model_name == "MOMENT"
            or self.model_name == "TimesNet"
            or self.model_name == "GPT4TS"
        ):
            return self.pe[:, : x.size(2)]
        else:
            return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.transpose(1, 2)
        # batch_size x seq_len x d_model
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self, c_in, d_model, model_name, embed_type="fixed", freq="h", dropout=0.1
    ):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(
            d_model=d_model, model_name=model_name
        )
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 512,
        patch_len: int = 8,
        stride: int = 8,
        dropout: int = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
        num_domain: int = 0,
        concat_other_layer: bool = False,
        using_weight: bool = False,
        domain_standardization: bool = False,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding
        self.num_domain = num_domain
        self.concat_other_layer = concat_other_layer
        self.using_weight = using_weight

        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        self.domain_standardization = domain_standardization
        
        initial_tensor = torch.randn(1,5)

        if num_domain:
            #init.xavier_uniform_(initial_tensor.unsqueeze(0))  # (1, num_cols)로 변환 후 초기화

            # 동일한 행을 반복하여 2D 텐서 생성
            initial_tensor = initial_tensor.repeat(num_domain, 1)  # (num_rows, num_cols) 형태로 복제

        # nn.Parameter로 설정
        self.domain = nn.Parameter(initial_tensor,requires_grad = False)
        
        if self.using_weight:
            # temp_weight = torch.ones((self.num_domain,self.num_domain))
            # for i in range(self.num_domain):
            #     temp_weight[i,i] = 0
            # self.domain_weight = nn.Parameter(temp_weight)
            
            #self.domain = nn.Parameter(init.xavier_uniform_(initial_tensor))
            
            #self.domain = nn.Parameter(torch.abs(torch.randn_like(initial_tensor)))
            
            self.lin_l = nn.Linear(5,5,bias=False)
            self.lin_r = nn.Linear(5,5,bias=False)
            
            # self.lin_l = nn.Linear(5,5)
            # self.lin_r = nn.Linear(5,5)
            
            self.att_l = nn.Parameter(torch.ones(5))
            self.att_r = nn.Parameter(torch.ones(5))
            
            self.concat_linear = nn.Linear(10,5)
            
            self.attention = torch.zeros(self.num_domain,self.num_domain)
            self.attention = nn.Parameter(self.attention,requires_grad=False)
            
        
        if (self.num_domain>0) & self.concat_other_layer:
            self.value_embedding = nn.Linear(patch_len, d_model, bias = value_embedding_bias)
            self.concat_embedding = nn.Linear(d_model + 5, d_model, bias = value_embedding_bias)
        elif (self.num_domain>0) & ~self.concat_other_layer:
            self.value_embedding = nn.Linear(patch_len+5, d_model, bias = value_embedding_bias)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
    def backward_hook(self):
        def hook(grad):
            grad[self.fixed_row] = 0
            return grad
        self.embedding.weight.register_hook(hook)

    def custom_softmax(self,x,except_num):
        exp_x = torch.exp(x)  # e^x
        sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)  # 각 행(row)마다 합계 계산
        sum_exp_x = sum_exp_x - exp_x[except_num]
        return exp_x / sum_exp_x  # e^x / sum(e^x)
    
    def custom_log_softmax(self,x,except_num):
        exp_x = torch.exp(x)  # e^x
        sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)  # 각 행(row)마다 합계 계산
        sum_exp_x = sum_exp_x - exp_x[except_num]
        log_sum_exp_x = torch.log(sum_exp_x)
        return x - log_sum_exp_x  # e^x / sum(e^x)


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, domain = None) -> torch.Tensor:
        
        mask = Masking.convert_seq_to_patch_view(
            mask, patch_len=self.patch_len
        ).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = (
            mask.repeat_interleave(self.d_model, dim=-1)
            .unsqueeze(1)
            .repeat(1, n_channels, 1, 1)
        )
        # mask : [batch_size x n_channels x n_patches x d_model]
        # x : [batch_size * n_chnnels * n_patches * patch_len]
        #self.domain grad_requires 찍어보기
        if (self.num_domain>0) & self.concat_other_layer:
            batch_size = x.size(0)
            n_channels = x.size(1)
            number_of_patches = x.size(2)
            if self.using_weight:
                # weight = self.domain_weight[domain,:].float()
                # weight = weight/torch.sum(weight).item()
                
                # # selected_row = self.domain_weight[domain,:].float().clone()

                # # 특정 요소의 기울기 고정을 위한 Mask 생성
                # # mask = torch.ones_like(selected_row)
                # # mask[domain] = 0
                # # non_fixed_part = selected_row[mask.bool()]
                # # softmax_part = F.softmax(non_fixed_part, dim=0)
                
                # domain_par = self.domain[domain,:].float()
                # aggregated_domain = torch.matmul(weight,self.domain)
                # domain_par = domain_par + aggregated_domain
                
                # #domain_par = torch.matmul(weight,self.domain)
                # expanded_domain_vec = domain_par.expand(batch_size, n_channels, number_of_patches, 5)
                ######################################
                temp_mask = torch.ones(self.num_domain)
                temp_mask[domain] = 0
                
                domain_tensor = self.domain.clone()
                
                if self.domain_standardization:
                    avg = self.domain.mean(dim=1,keepdim=True)
                    std = self.domain.mean(dim=1,keepdim=True)
                    
                    domain_tensor = (self.domain-avg)/std
                                           
                selected_domain = self.lin_l(domain_tensor[domain,:].float())
                #(5) * (5) => scalar (1)
                scalar_domain = torch.matmul(selected_domain,self.att_l)
                
                #((n-1)*5)
                other_domain = self.lin_r(domain_tensor[temp_mask.bool()].float())
                
                #(n-1*5) * (5) => (n-1)
                scalar_other_domain = torch.matmul(other_domain,self.att_r)
                alpha = F.leaky_relu(scalar_other_domain + scalar_domain,negative_slope=0.01)
                
                # (n)
                attention = F.softmax(alpha,dim=0)
                temp_attention = attention.cpu()
                #temp_attention = torch.exp(temp_attention)

                if not np.isclose((torch.sum(temp_attention)).detach().numpy(),1.0):
                    print("not 1")
                    print((torch.sum(temp_attention)).detach().numpy())
                
                # (n-1,1) * (n-1)
                updated_other_domain = attention.unsqueeze(-1) * other_domain
                updated_other_domain = torch.sum(updated_other_domain,dim=0)
                
                concated_domain = torch.cat((selected_domain,updated_other_domain),dim=-1)
                
                updated_selected_domain = self.concat_linear(concated_domain)
                
                with torch.no_grad():
                    self.attention[domain,:][temp_mask.bool()] = attention
                    domain_tensor[domain,:] = updated_selected_domain
                    self.domain = domain_tensor
                
                expanded_domain_vec = updated_selected_domain.expand(batch_size, n_channels, number_of_patches, 5)
                #####################################################
                # temp_mask = torch.ones(self.num_domain)
                # temp_mask[domain] = 0
                # selected_domain = self.lin_l(self.domain[domain,:].float())
                # #(5) * (5) => scalar (1)
                # scalar_domain = torch.matmul(selected_domain,self.att_l)
                
                # #((n-1)*5)
                # other_domain = self.lin_r(self.domain.float())
                
                # #(n-1*5) * (5) => (n-1)
                # scalar_other_domain = torch.matmul(other_domain,self.att_r)
                # alpha = F.leaky_relu(scalar_other_domain + scalar_domain,negative_slope=0.01)
                
                # # (n)
                # attention = F.softmax(alpha,dim=0)
                # temp_attention = attention.cpu()
                # #temp_attention = torch.exp(temp_attention)

                # if not np.isclose((torch.sum(temp_attention)).detach().numpy(),1.0):
                #     print("not 1")
                #     print((torch.sum(temp_attention)).detach().numpy())
                
                # # (n-1,1) * (n-1)
                # updated_other_domain = attention.unsqueeze(-1) * other_domain
                # updated_other_domain = torch.sum(updated_other_domain[temp_mask.bool()],dim=0)
                
                # self.attention = attention
                
                # concated_domain = torch.cat((selected_domain,updated_other_domain),dim=-1)
                
                # updated_selected_domain = self.concat_linear(concated_domain)
                
                # expanded_domain_vec = updated_selected_domain.expand(batch_size, n_channels, number_of_patches, 5)
                
            else:
                domain_par = self.domain[domain,:].float()
                #expanded_domain_vec = domain_par.unsqueeze(0).unsqueeze(0)
                expanded_domain_vec = domain_par.expand(batch_size, n_channels, number_of_patches, 5)

            x = self.value_embedding(x)
            x = torch.cat((x, expanded_domain_vec), dim=-1)
            x = self.concat_embedding(x)
            
            x = mask * x + (1 - mask) * self.mask_embedding

            if self.add_positional_embedding:
                x = x + self.position_embedding(x)
            
            return self.dropout(x)

        elif (self.num_domain>0) & ~self.concat_other_layer:
            
            batch_size = x.size(0)
            n_channels = x.size(1)
            number_of_patches = x.size(2)
            
            if self.using_weight:
                # weight = self.domain_weight[domain,:].float()
                # weight = weight/torch.sum(weight).item()
                
                # # selected_row = self.domain_weight[domain,:].float().clone()

                # # 특정 요소의 기울기 고정을 위한 Mask 생성
                # # mask = torch.ones_like(selected_row)
                # # mask[domain] = 0
                # # non_fixed_part = selected_row[mask.bool()]
                # # softmax_part = F.softmax(non_fixed_part, dim=0)
                
                # domain_par = self.domain[domain,:].float()
                # aggregated_domain = torch.matmul(weight,self.domain)

                # domain_par = domain_par + aggregated_domain
                
                # #domain_par = torch.matmul(weight,self.domain)
                # expanded_domain_vec = domain_par.expand(batch_size, n_channels, number_of_patches, 5)
                
                ######################################
                temp_mask = torch.ones(self.num_domain)
                temp_mask[domain] = 0
                
                domain_tensor = self.domain.clone()
                
                if self.domain_standardization:
                    avg = self.domain.mean(dim=1,keepdim=True)
                    std = self.domain.mean(dim=1,keepdim=True)
                    
                    domain_tensor = (self.domain-avg)/std
                    
                                    
                selected_domain = self.lin_l(domain_tensor[domain,:].float())
                #(5) * (5) => scalar (1)
                scalar_domain = torch.matmul(selected_domain,self.att_l)
                
                #((n-1)*5)
                other_domain = self.lin_r(domain_tensor[temp_mask.bool()].float())
                
                #(n-1*5) * (5) => (n-1)
                scalar_other_domain = torch.matmul(other_domain,self.att_r)
                alpha = F.leaky_relu(scalar_other_domain + scalar_domain,negative_slope=0.01)
                
                # (n)
                attention = F.softmax(alpha,dim=0)
                temp_attention = attention.cpu()
                #temp_attention = torch.exp(temp_attention)

                if not np.isclose((torch.sum(temp_attention)).detach().numpy(),1.0):
                    print("not 1")
                    print((torch.sum(temp_attention)).detach().numpy())
                
                # (n-1,1) * (n-1)
                updated_other_domain = attention.unsqueeze(-1) * other_domain
                updated_other_domain = torch.sum(updated_other_domain,dim=0)
                
                concated_domain = torch.cat((selected_domain,updated_other_domain),dim=-1)
                
                updated_selected_domain = self.concat_linear(concated_domain)
                
                with torch.no_grad():
                    self.attention[domain,:][temp_mask.bool()] = attention                
                    domain_tensor[domain,:] = updated_selected_domain
                    self.domain = domain_tensor
                
                expanded_domain_vec = updated_selected_domain.expand(batch_size, n_channels, number_of_patches, 5)
                #####################################################
                # temp_mask = torch.ones(self.num_domain)
                # temp_mask[domain] = 0
                # selected_domain = self.lin_l(self.domain[domain,:].float())
                # #(5) * (5) => scalar (1)
                # scalar_domain = torch.matmul(selected_domain,self.att_l)
                
                # #((n-1)*5)
                # other_domain = self.lin_r(self.domain.float())
                
                # #(n-1*5) * (5) => (n-1)
                # scalar_other_domain = torch.matmul(other_domain,self.att_r)
                # alpha = F.leaky_relu(scalar_other_domain + scalar_domain,negative_slope=0.01)
                
                # # (n)
                # attention = F.softmax(alpha,dim=0)
                # temp_attention = attention.cpu()
                # #temp_attention = torch.exp(temp_attention)

                # if not np.isclose((torch.sum(temp_attention)).detach().numpy(),1.0):
                #     print("not 1")
                #     print((torch.sum(temp_attention)).detach().numpy())
                
                # # (n-1,1) * (n-1)
                # updated_other_domain = attention.unsqueeze(-1) * other_domain
                # updated_other_domain = torch.sum(updated_other_domain[temp_mask.bool()],dim=0)
                
                # self.attention = attention
                
                # concated_domain = torch.cat((selected_domain,updated_other_domain),dim=-1)
                
                # updated_selected_domain = self.concat_linear(concated_domain)
                
                # expanded_domain_vec = updated_selected_domain.expand(batch_size, n_channels, number_of_patches, 5)

            else:
                domain_par = self.domain[domain,:].float()
                #expanded_domain_vec = domain_par.unsqueeze(0).unsqueeze(0)
                expanded_domain_vec = domain_par.expand(batch_size, n_channels, number_of_patches, 5)

            x = torch.cat((x, expanded_domain_vec), dim=-1)
            
            # Input encoding
            x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
            
            if self.add_positional_embedding:
                x = x + self.position_embedding(x)

            return self.dropout(x)
            # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding

        if self.add_positional_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            warnings.warn(
                "Stride and patch length are not equal. "
                "This may lead to unexpected behavior."
            )

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x

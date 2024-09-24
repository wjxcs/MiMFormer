import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange,reduce
from torch import nn
import torch.nn.init as init
from args_parse import args
from testt import FeaturesMemory

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weights)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weights)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

"""
文本编码器
"""


class cls_Attention(nn.Module):
    def __init__(self, input_dim):
        super(cls_Attention, self).__init__()

        # 可学习的线性变换层，用于将张量b用作Q，张量a用作K和V
        self.conv_q = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.conv_k = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.conv_v = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.scale = input_dim ** -0.5  # 1/sqrt(dim)

    def forward(self, a, b):
        batch_size, bands, h, w = a.size()

        # 将张量b调整为与张量a相同的形状
        b_reshaped = b.repeat(1, 1, h, w)

        # 使用线性变换计算 Q、K、V
        q = self.conv_q(b_reshaped)
        k = rearrange(self.conv_k(a),'b d h w -> b d w h')
        v = self.conv_v(a)
        # print("q shape is {}".format(q.shape)) 
        # print("k shape is {}".format(k.shape))
        # 计算点积
        dots = torch.matmul(q, k) * self.scale
        # 进行 softmax 归一化
        attention_weights = torch.nn.functional.softmax(dots, dim=1)

        # 使用权重对 Value 进行加权求和
        output = torch.matmul(attention_weights, v)
        # print("output shape is {}".format(output.shape))

        return output

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channel, out_channel,ker_size,pad):
        super(depthwise_separable_conv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channel, in_channel, kernel_size=ker_size,padding=pad, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class Gen_FeatureEXM(nn.Module):
    def __init__(self,in_ch):
        super(Gen_FeatureEXM,self).__init__()
        self.out_ch = in_ch*3
        self.dwConv1  = depthwise_separable_conv(in_channel=in_ch,
                             out_channel=self.out_ch,
                             ker_size=3,
                             pad=1)
        self.dwConv2 = depthwise_separable_conv(in_channel=in_ch,
                            out_channel=self.out_ch,
                            ker_size=5,
                            pad=2)
        self.dwConv3 = depthwise_separable_conv(in_channel=in_ch,
                            out_channel=self.out_ch,
                            ker_size=7,
                            pad=3)

        self.conv1 = nn.Conv2d(in_channels=in_ch,
                               out_channels=in_ch,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_ch,
                               out_channels=in_ch,
                               kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(self.out_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        # self.bn4 = nn.BatchNorm2d(in_ch)
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.Gelu1 = nn.GELU()
        self.Gelu2 = nn.GELU()
    def forward(self,x):
        output = self.conv1(x)
        output = self.Gelu1(self.bn1(output))
        output1 = self.dwConv1(output)
        output2 = self.dwConv2(output)
        output3 = self.dwConv3(output)
        output = output3.mul(output1.mul(output2))
        # print(output.shape)
        output = self.bn2(output)
        output = self.conv2(output)
        output = self.Gelu2(self.bn3(output))
        return output

class FeatureFusionGate(nn.Module):
    def __init__(self, feature_dim, kernel_size=1):
        super(FeatureFusionGate, self).__init__()
        # 创建一个门控卷积层，用于计算专属特征的门控分数
        self.gate_conv = nn.Conv2d(feature_dim, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, exclusive_features, common_features):
        # 计算专属特征的门控分数
        gate_scores_exclusive = torch.sigmoid(self.gate_conv(exclusive_features))
        # print(gate_scores_exclusive.shape)
        # 计算通用特征的门控分数，它是1减去专属特征的门控分数
        gate_scores_common = 1 - gate_scores_exclusive

        # 使用门控分数调制专属特征和通用特征
        modulated_exclusive_features = exclusive_features * gate_scores_exclusive
        modulated_common_features = common_features * gate_scores_common

        # 融合两种调制后的特征
        fused_features = modulated_exclusive_features + modulated_common_features

        return fused_features





class concept_Tokenizer(nn.Module):
    def __init__(self,in_dim,out_dim,feats_channels,transform_channels,out_channels,tran_in_channels,num_feats_per_cls):
        super().__init__()
        self.conv = nn.Conv2d(in_dim,out_dim,kernel_size=1)
        self.FeaturesMemory = FeaturesMemory(args.cls, feats_channels=feats_channels, transform_channels=transform_channels,
        out_channels=out_channels, tran_in_channels=tran_in_channels,use_context_within_image=True, num_feats_per_cls=num_feats_per_cls)

    def forward(self,feat):
        output = self.conv(feat)
        print("output1 shape is {}".format(output.shape))
        output = rearrange(output,"b c h w -> b (h w) c")

        feat_1 = rearrange(feat,"b c h w -> b c (h w)")
        output = torch.einsum("bcn,bnd -> bcd",feat_1,output)
        output = F.softmax(output,dim=-1)
        output = reduce(output,"b (c s) d->b c d",'mean',s=8)

        # output = torch.einsum("bcn,bcd->bnd",output,feat_1)
        print("output1 shape is {}".format(output.shape))
        output = rearrange(output, "b c (h w) -> b c h w",h=16,w=16)

        memory,output = self.FeaturesMemory(output)
        output = rearrange(output,"b (c n) h w -> b c (n h w)",c=4,n=4)
        print("outputsd shape is {}".format(output.shape))
        return output



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x
class MyNet(nn.Module):
    def __init__(self, num_classes=args.cls, num_tokens=4, dim=1024, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(MyNet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=args.pca, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_spe = nn.Sequential(
            nn.Conv2d(in_channels=args.pca, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.clsAtt = cls_Attention(32)
        self.Gen_FeatureEXM = Gen_FeatureEXM(32)
        self.FeatureFusionGate = FeatureFusionGate(32)
        # Tokenization
        self.conceptT = concept_Tokenizer(in_dim=32,out_dim=256,feats_channels=1024,transform_channels=1024,
                                          out_channels=16,tran_in_channels=4,num_feats_per_cls=4)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, spe,mask=None):
        # print(x.shape)
        x = self.conv(x)
        
        output_1 = self.clsAtt(x,spe)
        output_2 = self.Gen_FeatureEXM(x)
        output = self.FeatureFusionGate(output_1,output_2)
        print("output shape {}".format(output.shape))
        concept_tokenes = self.conceptT(output)
        # update

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # print("cls_tokens shape {}".format(cls_tokens.shape))
        # print("concept_tokenes shape {}".format(concept_tokenes.shape))
        # print("pos_embedding shape {}".format(self.pos_embedding.shape))

        x = torch.cat((cls_tokens, concept_tokenes), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x




if __name__ == '__main__':
    model = MyNet()
    model.eval()
    # print(model)
    input = torch.randn(64, 30, 13, 13)
    spe = torch.randn(64, 32, 1, 1)
    y = model(input,spe)
    print(y.size())


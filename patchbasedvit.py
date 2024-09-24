import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange,repeat
from torch import nn
import torch.nn.init as init
from args_parse import args
from Memory import FeaturesMemory

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
        self.Gelu1 = nn.ReLU()
        self.Gelu2 = nn.ReLU()
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
        # print()
        # print("gate_scores_exclusive shape is {}".format(gate_scores_exclusive.shape))
        # 计算通用特征的门控分数，它是1减去专属特征的门控分数
        gate_scores_common = 1 - gate_scores_exclusive

        # 使用门控分数调制专属特征和通用特征
        modulated_exclusive_features = exclusive_features * gate_scores_exclusive
        modulated_common_features = common_features * gate_scores_common

        # 融合两种调制后的特征
        fused_features = modulated_exclusive_features + modulated_common_features

        return fused_features

class MultiScaleSemanticConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleSemanticConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        return conv1_out + conv3_out + conv5_out

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, img_size=12, patch_size=4, in_chans=32, embed_dim=256):
        super().__init__()
        # (H, W)
        img_size = pair(img_size)
        # (P, P)
        patch_size = pair(patch_size)
        # N = (H // P) * (W // P)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 可训练的线性投影 - 获取输入嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # (B, C, H, W) -> (B, D, (H//P), (W//P)) -> (B, D, N) -> (B, N, D)
        #   D=embed_dim=768, N=num_patches=(H//P)*(W//P)
        #   torch.flatten(input, start_dim=0, end_dim=-1)  # 形参：展平的起始维度和结束维度
        # 可见 Patch Embedding 操作 1 行代码 3 步到位
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

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
class MyNet_patch(nn.Module):
    def __init__(self, num_classes=args.cls, mode="test",num_tokens=9, dim=256, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(MyNet_patch, self).__init__()
        self.mode = mode
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
        # self.conceptT = concept_Tokenizer(in_dim=32,out_dim=4,feats_channels=1024,transform_channels=1024,
        #                                   out_channels=16,tran_in_channels=4,num_feats_per_cls=4,mode=self.mode)
        self.conceptT = PatchEmbed(13,4,32,256)
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, target=None,mask=None):
        # print(x.shape)
        batch_size, channels, height, width = x.shape

        spe = x[:, :, ((height - 1) // 2), ((width - 1) // 2)].view(batch_size, channels, 1, 1)
        # print(spe.shape)
        # x = self.conv(x)

        # spe = self.conv_spe(spe)
        output_1 = self.clsAtt(x,spe)
        # print("outputss clsAtt shape {}".format(output_1.shape))

        output_2 = self.Gen_FeatureEXM(x)
        # print("outputss Gen_FeatureEXM shape {}".format(output_2.shape))

        output = self.FeatureFusionGate(output_1,output_2)
        # print("outputss shape {}".format(output.shape))

        concept_tokenes = self.conceptT(output_2)
        # update
        # print("concept_tokenes shape {}".format(output.shape))
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # xs = self.to_cls_token(x[:, 0])
        x = torch.cat((cls_tokens, concept_tokenes), dim=1)
        # print("concept_tokenes shape {}".format(x.shape))
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        xs = self.to_cls_token(x[:, 0])
        x = self.nn1(xs)
        if self.mode == "train":
            # self.MemoryC.update_memory(semantic_tokens, category_index)

            return x,output_1, output_2
            # return x,output_1,output_2
        else:
            return x,xs


if __name__ == '__main__':
    model = MyNet_patch(mode="train")
    model.eval()
    # print(model)
    input = torch.randn(64, 32, 13, 13)
    spe = torch.randn(64, 32, 1, 1)
    target = torch.randn(64)
    y = model(input, target)
    print(y.size())


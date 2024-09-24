import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange,reduce
from torch import nn
import torch.nn.init as init
from args_parse import args
from Memory import FeaturesMemory

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


if __name__ == '__main__':
    model = PatchEmbed(13,4,30,256)
    model.eval()
    # print(model)
    input = torch.randn(64, 30, 13, 13)

    y = model(input)
    print(y.size())


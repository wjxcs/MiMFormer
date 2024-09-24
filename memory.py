import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import BertModel, BertTokenizer

from einops import rearrange,repeat
import  numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from Jiax.Semantic_de_cls.cls_semantic_de.selfattention import SelfAttentionBlock
from args_parse import args
'''features memory'''
"""
num_classes: 类别数量
feats_channels： 输入特征的通道数量
transform_channels：这是自注意力模块中变换后的通道数。虽然这个参数在 
out_channels： 这是最终输出特征的通道数。这个参数在 bottleneck 序列中被使用，同样不影响记忆张量的形状。
"""
class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels,tran_in_channels,
                 use_context_within_image=True, num_feats_per_cls=4, use_hard_aggregate=False, memory_data=None,**kwargs):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        # set attributes
        self.is_initialized = False
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.use_context_within_image = use_context_within_image
        self.use_hard_aggregate = use_hard_aggregate
        # self.conv = nn.Conv2d(num_classes,feats_channels,1)
        # init memory
        if memory_data is not None:
            self.memory = nn.Parameter(memory_data)
        else:
            # self.memory = 【num_classes, num_feats_per_cls, feats_channels】
            self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)
        # define self_attention module
        if self.num_feats_per_cls > 1:
            self.self_attentions = nn.ModuleList()
            for _ in range(self.num_feats_per_cls):
                self_attention = SelfAttentionBlock(
                    key_in_channels=tran_in_channels,
                    query_in_channels=tran_in_channels,
                    transform_channels=transform_channels,
                    out_channels=feats_channels,
                    share_key_query=False,
                    query_downsample=None,
                    key_downsample=None,
                    key_query_num_convs=2,
                    value_out_num_convs=1,
                    key_query_norm=True,
                    value_out_norm=True,
                    matmul_norm=True,
                    with_out_project=True,
                )
                self.self_attentions.append(self_attention)
            self.fuse_memory_conv = nn.Sequential(
                nn.Conv2d(feats_channels * self.num_feats_per_cls, feats_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(feats_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # 每个类别的特征数量为1
            self.self_attention = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=tran_in_channels,

                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
            )
        # whether need to fuse the contextual information within the input image
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels + tran_in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if use_context_within_image:
            self.self_attention_ms = SelfAttentionBlock(
                key_in_channels=tran_in_channels,
                query_in_channels=tran_in_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
            )
            self.bottleneck_ms = nn.Sequential(
                nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    '''forward'''
    def forward(self, feats, preds=None):
        batch_size, num_channels, h, w = feats.size()
        selected_memory_list = []
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:, idx, :]  # memory [cls,1,1024]
            selected_feat = rearrange(feats,"b dim h w ->  (dim h w) b")
            selected_memory = torch.einsum("cj,jb->bc", memory, selected_feat)

            selected_memory = F.softmax(selected_memory,dim=-1)
            selected_memory = torch.einsum("bc,cd -> bd",selected_memory,memory)

            selected_memory_list.append(selected_memory)

        # false
        if self.num_feats_per_cls > 1:
            relation_selected_memory_list = []
            for idx, selected_memory in enumerate(selected_memory_list):
                # --(B*H*W, C) --> (B, H, W, C)
                selected_memory = rearrange(selected_memory, "b (d h w) ->  b d h w",d=num_channels,h=h,w=w)
                relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))

            selected_memory = torch.cat(relation_selected_memory_list, dim=1)
            selected_memory = self.fuse_memory_conv(selected_memory)
        else:
            # print(self.num_feats_per_cls)
            assert len(selected_memory_list) == 1
            selected_memory = selected_memory_list[0]
            new_selected_memory = selected_memory
            for b in range(batch_size - 1):
                new_selected_memory = torch.cat((new_selected_memory, selected_memory), 1)
            # --feed into the self attention module
            selected_memory = new_selected_memory.permute(1, 2, 0).contiguous().unsqueeze(3)
            selected_memory = self.self_attention(feats, selected_memory)
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        return self.memory.data, memory_output

    def init_memory(self):

        # 初始化分词器和模型
        tokenizer = BertTokenizer.from_pretrained(r'/wenjiaxiang/Jiax/Semantic_de_cls/cls_semantic_de/bert')
        model = BertModel.from_pretrained(r'/wenjiaxiang/Jiax/Semantic_de_cls/cls_semantic_de/bert')
        # print(model)
        # 定义FeaturesMemory实例所需的参数
        num_classes = len(args.target_names)  # 类别数量

        # 对每个类别名称进行编码并获取BERT输出
        encoded_texts = tokenizer(args.target_names, padding=True, truncation=True, return_tensors='pt')

        # 如果您需要访问其他与分词相关的张量，也可以类似地进行
        token_type_ids = encoded_texts['token_type_ids']
        # print("token_type_ids.shape is {}".format(token_type_ids.shape))
        with torch.no_grad():
            outputs = model(**encoded_texts)

            last_hidden_states = outputs.last_hidden_state
        # print(encoded_texts.shape)

        cls_features = last_hidden_states[:, 0, :]  # (num_classes, feats_channels)

        # 调整cls_features的形状以匹配memory_model.memory的形状
        cls_features = rearrange(cls_features, 'c n -> c 1 n')  # 假设n是features的维度

        # 现在循环遍历每个类别和每个特征索引，并将cls_features的值赋给memory_model.memory
        for clsid in range(num_classes):
            for idx in range(self.num_feats_per_cls):
                # 使用rearrange调整形状以匹配memory_model.memory[clsid][idx].data的形状
                memoru_init = rearrange(cls_features[clsid], "c n -> (c n)")
                memoru_init = repeat(memoru_init, "n -> d n", d=self.num_feats_per_cls)

                # 确保memoru_init的形状与memory_model.memory[clsid][idx].data的形状一致
                self.memory[clsid].data.copy_(memoru_init)

        # 打印memory_model.memory.data的形状，验证赋值是否正确
        print(f"Shape of memory_model.memory.data: {self.memory.data.shape}")

    """
        patch_labels = [batch_size] 
        clsids = [batch_size] 
        FeaturesMemory = [num_classes, num_feats_per_cls, feats_channels]

    """

    def update(self, features, patch_labels, ignore_index=255, strategy='cosine_similarity', learning_rate=0.004,
               **kwargs):
        assert strategy in ['mean', 'cosine_similarity']
        # batch_size, num_channels, h, w = features.size()
        momentum = args.base_momentum
        base_lr = args.base_lr
        if args.adjust_by_learning_rate:
            momentum = momentum / base_lr * learning_rate

            args.base_momentum = momentum
        # use features to update memory
        patch_labels = patch_labels.long()

        unique_clsids = patch_labels.unique()

        patch_labels = repeat(patch_labels,"b -> b (repeat)",repeat=1024)
        # print("patch_labels shape is {}".format(patch_labels.shape))
        features = rearrange(features,"b c h w -> b (c h w)")
        # features = features.view(batch_size, num_channels)
        for clsid in unique_clsids:
            if clsid == ignore_index:continue
            # 检查是否需要初始化记忆特征
            feats_cls = features[patch_labels == clsid]
            # print("feats_cls {}".format(feats_cls.shape))
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)
                self.memory[clsid].data.copy_(feats_cls)

            else:
                assert strategy in ['cosine_similarity']
                # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
                feats_cls = rearrange(feats_cls,"(b c) -> b c",c=1024)

                relation = torch.matmul(
                    F.normalize(feats_cls, p=2, dim=1),
                    F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0),
                )
                argmax = relation.argmax(dim=1)
                # ----for saving memory during training
                for idx in range(self.num_feats_per_cls):
                    mask = (argmax == idx)
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    self.memory[clsid].data[idx].copy_(
                        self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)

        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)




    # syn the memory
# 导入必要的库
import torch
if __name__ == "__main__":
    # 定义模拟数据的参数
    num_classes = 16  # 类别数量
    feats_channels = 4  # 特征的通道数
    tran_in_channels = 4
    transform_channels = 1024  # 自注意力模块变换后的通道数
    out_channels = 4  # 最终输出特征的通道数
    num_feats_per_cls = 4  # 每个类别的特征表示数量
    batch_size = 64  # 批次大小
    height, width = 16, 16  # 特征图的尺寸

    # 创建模拟的特征数据
    features = torch.randn(batch_size, tran_in_channels, height, width)

    # 创建模拟的 patch labels，这里我们简单地使用一个随机生成的标签数组
    patch_labels = torch.randint(num_classes, (batch_size * 1,), dtype=torch.long)

    # 将 patch labels 重塑回原来的批次大小
    patch_labels = patch_labels.view(batch_size)

    dim = 1024
    # 初始化 FeaturesMemory 类的一个实例
    memory_model = FeaturesMemory(num_classes, dim, transform_channels, out_channels, tran_in_channels)
    memory_model.init_memory()
    # 定义更新策略和动量
    strategy = 'cosine_similarity'  # 或者 'cosine_similarity'
    base_momentum = 0.9
    adjust_by_learning_rate = False
    base_lr = 0.01  # 基础学习率，如果 adjust_by_learning_rate 为 True，则需要这个参数
    # print(features.shape)
    # 调用 update 方法进行测试
    memory_model.update(
        features,
        patch_labels,
        strategy=strategy,
        base_momentum=base_momentum,
        adjust_by_learning_rate=adjust_by_learning_rate,
        base_lr=base_lr
    )

    # 打印记忆张量的形状，确认是否符合预期
    print("memory_model.memory.data.shape is {}".format(memory_model.memory.data.shape))

    # 如果需要，可以继续测试 forward 方法
    print("features shape is {}".format(features.shape))
    memory_data, memory_output = memory_model.forward(features)

    # 打印输出的形状
    print("Memory data shape:", memory_data.shape)
    print("Memory output shape:", memory_output.shape)
import torch
import torch.nn.functional as F
import torch.nn as nn

class MemoryModule(nn.Module):
    def __init__(self, cls_num, token_dim, learning_rate=0.001, decay_rate=0.99):
        super(MemoryModule, self).__init__()
        self.cls_num = cls_num
        self.token_dim = token_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        self.memory_tensor = nn.Parameter(torch.randn(cls_num, token_dim))
        self.register_buffer('memory_freshness', torch.ones(cls_num))

    def forward(self, semantic_tokens):
        batch_size, num_tokens, token_dim = semantic_tokens.size()

        # 计算相似度
        similarity = torch.matmul(semantic_tokens, self.memory_tensor.T)  # [b, num_tokens, cls_num]
        # print(similarity.shape)
        # 计算注意力权重
        attention = F.softmax(similarity, dim=-1)  # [b, num_tokens, cls_num]

        # 更新token
        updated_tokens = torch.matmul(attention, self.memory_tensor)  # [b, num_tokens, token_dim]

        return updated_tokens, attention

    def update_memory(self, semantic_tokens, labels):
        # batch_size, num_tokens, token_dim = semantic_tokens.size()

        # 衰减记忆新鲜度
        self.memory_freshness *= self.decay_rate

        # 计算每个类的平均语义token
        semantic_tokens_mean = semantic_tokens.mean(dim=1)  # [b, token_dim]

        # 创建独热编码矩阵
        labels_one_hot = F.one_hot(labels, num_classes=self.cls_num).float()  # [b, cls_num]

        # 计算每个类的语义token和标签计数
        weighted_semantic_tokens = torch.matmul(labels_one_hot.T, semantic_tokens_mean)  # [cls_num, token_dim]
        label_counts = labels_one_hot.sum(dim=0)  # [cls_num]

        # 避免除以0的情况
        label_counts = label_counts.unsqueeze(1).clamp(min=1)  # [cls_num, 1]

        # 归一化加权语义token
        weighted_semantic_tokens /= label_counts

        # 更新记忆向量
        with torch.no_grad():
            self.memory_tensor.data = (1 - self.learning_rate) * self.memory_tensor.data + self.learning_rate * weighted_semantic_tokens

            # 重置更新类的记忆新鲜度
            self.memory_freshness = torch.where(label_counts.squeeze(1) > 0, torch.ones_like(self.memory_freshness), self.memory_freshness)

# 测试代码
if __name__ == "__main__":
    cls_num = 10
    token_dim = 32
    batch_size = 32
    num_tokens = 4

    memory_module = MemoryModule(cls_num, token_dim)
    semantic_tokens = torch.randn(batch_size, num_tokens, token_dim)
    labels = torch.randint(0, cls_num, (batch_size,))

    updated_tokens, attention = memory_module(semantic_tokens)
    memory_module.update_memory(semantic_tokens, labels)

    print("Updated Tokens:", updated_tokens.shape)
    print("Attention:", attention.shape)
    print("Memory Tensor:", memory_module.memory_tensor.shape)
    print("Memory Freshness:", memory_module.memory_freshness.shape)

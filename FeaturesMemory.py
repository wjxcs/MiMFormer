import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    def __init__(self, memory_size, token_dim):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.token_dim = token_dim
        self.memory = nn.Parameter(torch.zeros(memory_size, token_dim))
        nn.init.xavier_normal_(self.memory)  # 初始化策略：xavier_normal
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avrgpool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, semantic_tokens):
        # semantic_tokens: [b, 4, 32]
        b, n_tokens, token_dim = semantic_tokens.size()

        # 计算注意力权重
        ap = self.avrgpool(semantic_tokens)
        mp = self.maxpool(semantic_tokens)
        output = ap + mp

        # 数值稳定性处理，避免除零错误
        output = output + 1e-6

        re_semantic_tokens = (semantic_tokens * output).sum(1)

        attention_scores = torch.matmul(re_semantic_tokens, self.memory.t())  # [b, memory_size]

        # 数值稳定性处理，避免过大值
        attention_scores = torch.clamp(attention_scores, min=-1e6, max=1e6)
        category_indices = attention_scores.argmax(dim=1)  # [b]

        updated_semantic_tokens = []
        for i in range(b):
            category_index = category_indices[i]
            tokens = semantic_tokens[i]
            weighted_tokens = 0.4 * tokens * self.memory[category_index] + 0.6 * tokens
            updated_semantic_tokens.append(weighted_tokens.unsqueeze(0))

        updated_semantic_tokens = torch.cat(updated_semantic_tokens, dim=0)

        # 确保输入 Batch Normalization 的数据没有 NaN 或无穷大
        if torch.isnan(updated_semantic_tokens).any() or torch.isinf(updated_semantic_tokens).any():
            updated_semantic_tokens = torch.where(torch.isnan(updated_semantic_tokens),
                                                  torch.full_like(updated_semantic_tokens, 0), updated_semantic_tokens)
            updated_semantic_tokens = torch.where(torch.isinf(updated_semantic_tokens),
                                                  torch.full_like(updated_semantic_tokens, 0), updated_semantic_tokens)

        updated_semantic_tokens = self.bn(updated_semantic_tokens)

        return updated_semantic_tokens, category_indices, attention_scores

    def update_memory(self, semantic_tokens, category_indices):
        b, n_tokens, token_dim = semantic_tokens.size()
        for i in range(b):
            category_index = category_indices[i]
            new_memory_content = semantic_tokens[i].mean(dim=0)

            # 数值稳定性处理，避免 NaN 和无穷大
            new_memory_content = torch.where(torch.isnan(new_memory_content), torch.full_like(new_memory_content, 0),
                                             new_memory_content)
            new_memory_content = torch.where(torch.isinf(new_memory_content), torch.full_like(new_memory_content, 0),
                                             new_memory_content)

            if self.memory.data[category_index].eq(0).all():
                self.memory.data[category_index] = new_memory_content
            else:
                # 更新策略：滑动平均更新策略
                self.memory.data[category_index] = 0.6 * self.memory[category_index] + 0.4 * new_memory_content


# 示例使用
if __name__ == "__main__":
    b, n_tokens, token_dim = 8, 4, 32
    semantic_tokens = torch.randn(b, n_tokens, token_dim)
    memory_size = 10

    memory_module = MemoryModule(memory_size, token_dim)
    output_tokens, index, att_score = memory_module(semantic_tokens)
    print(output_tokens.shape)  # 期望输出: [8, 4, 32]

    # 更新记忆
    memory_module.update_memory(semantic_tokens, index)

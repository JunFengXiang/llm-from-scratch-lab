import torch

# 示例输入：4 个 token ID。
# 这些整数必须落在 [0, vocab_size) 范围内。
input_ids = torch.tensor([2, 3, 5, 1])

# 词表中共有 6 个 token。
vocab_size = 6

# 每个 token 用 3 维向量表示。
output_dim = 3

# 固定随机种子，使每次运行时初始化的 embedding 权重一致。
torch.manual_seed(123)

# Embedding 可以理解为一个可训练的查表矩阵。
# 权重矩阵形状为 (vocab_size, output_dim)，即 (6, 3)。
embedding_layer = torch.nn.Embedding(
    vocab_size,
    output_dim
)

# 查看整个 embedding 权重矩阵。
print(embedding_layer.weight)

# 输入形状为 (1,)，输出形状为 (1, 3)。
# 数字 3 被用作行索引，取出权重矩阵的第 3 行。
print(embedding_layer(torch.tensor([3])))

# 批量查表：输入中有 4 个 token ID，
# 所以输出形状为 (4, 3)。
print(embedding_layer(input_ids))

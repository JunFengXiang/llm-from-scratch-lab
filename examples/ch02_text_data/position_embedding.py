import torch
from sliding_window import create_dataloader_v1

# 本示例对应教材第 2.8 节。
# 当前代码先把 token ID 转换为 token embedding；
# 位置 embedding 会在后续代码中与 token embedding 相加。
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    # raw_text 是原始字符串，后面会由 GPT-2 分词器转换为 token ID。
    raw_text = file.read()

# GPT-2 词表的大小。
vocab_size = 50257

# 每个 token 用多少维向量表示。
# Embedding 层内部的权重矩阵形状是 (vocab_size, output_dim)。
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 每个训练样本包含 4 个 token。
max_length = 4

# 一次从 DataLoader 中取出 8 个样本。
batch_size = 8

# 相邻滑动窗口向前移动 4 个 token。
# 当 stride == max_length 时，相邻窗口之间没有重叠。
stride = 4

# 创建输入-目标 token 对组成的 DataLoader。
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

# iter(dataloader) 返回一个迭代器。
# next(data_iter) 会取出其中的下一个 batch。
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# inputs 和 targets 的形状通常都是 (batch_size, max_length)，即 (8, 4)。
print("inputs:")
print(inputs)
print("inputs.shape:", inputs.shape)

print("targets:")
print(targets)
print("targets.shape:", targets.shape)


# Embedding 层根据每个 token ID 查表。
# 输入形状为 (8, 4)，输出形状为 (8, 4, 256)：
# 每个 token ID 都被替换成一个 256 维向量。
token_embeddings = token_embedding_layer(inputs)

print("token_embeddings.shape:", token_embeddings.shape)

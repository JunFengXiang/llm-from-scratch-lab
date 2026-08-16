import tiktoken
import torch

# 读取教材中的原始文本。
# 这个文件通常从项目根目录运行，因此使用 data/ch02/... 路径。
with open("data/ch02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 使用 GPT-2 分词器把文本转换为 token ID 序列。
tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
print("total tokens:", len(enc_text))

# 从第 50 个 token 开始取一个示例片段，方便观察滑动窗口。
enc_sample = enc_text[50:]

# 上下文长度：每个训练样本包含多少个 token。
context_size = 4

# 输入序列 x 和目标序列 y 长度相同，但 y 比 x 向右移动一个 token。
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print("x:", x)
print("y:", y)

# 逐步展示自回归语言模型的训练目标：
# 给定前面的 context，模型要预测紧接着的 desired token。
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)


# Dataset 负责保存和访问单个样本；
# DataLoader 负责把多个样本组织成 batch。
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 每个元素都是一个长度为 max_length 的输入/目标张量。
        self.input_ids = []
        self.target_ids = []

        # 对完整文本进行分词，得到一维 token ID 序列。
        token_ids = tokenizer.encode(txt)

        # 从左到右创建滑动窗口。
        # stride 决定相邻窗口向前移动多少个 token：
        # stride=1 时重叠最多；stride=max_length 时窗口之间不重叠。
        for i in range(0, len(token_ids) - max_length, stride):
            # 输入窗口长度为 max_length。
            input_chunk = token_ids[i:i + max_length]

            # 目标窗口整体向右移动一个 token。
            # 例如 input=[10, 20, 30, 40]，
            # target=[20, 30, 40, 50]。
            target_chunk = token_ids[i + 1:i + max_length + 1]

            # 将 Python 列表转换为 PyTorch 张量，形状为 (max_length,)。
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 返回数据集中的样本数量。
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 根据索引返回一个 (input, target) 样本。
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    # 分词器负责把原始字符串转换为 token ID。
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建保存所有滑动窗口样本的 Dataset。
    dataset = GPTDatasetV1(
        txt,
        tokenizer,
        max_length,
        stride
    )

    # DataLoader 将多个形状为 (max_length,) 的样本拼成 batch。
    # 输出的 inputs 和 targets 形状通常为
    # (batch_size, max_length)。
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# 先用 batch_size=1、stride=1 观察连续重叠的窗口。
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False
)

# iter() 创建迭代器；连续两次 next() 读取前两个 batch。
data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)

print("first batch:", first_batch)
print("second batch:", second_batch)

# batch_size=1 时，形状为 (1, 4)；
# 第一个维度是 batch，第二个维度是 token 数。
print("input shape:", first_batch[0].shape)
print("target shape:", first_batch[1].shape)


# 再用 batch_size=8、stride=4 模拟教材第 2.8 节的设置。
# 因为 stride == max_length，所以窗口之间没有重叠。
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)

# 直接读取第一个 batch。
# inputs 和 targets 的形状通常都是 (8, 4)。
inputs, targets = next(iter(dataloader))

print(inputs.shape)
print(targets.shape)

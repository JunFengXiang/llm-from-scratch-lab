import torch
from sliding_window import create_dataloader_v1

with open("the-verdict.txt","r",encoding = "utf-8")as file:
    raw_text = file.read()
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

max_length = 4
batch_size = 8
stride = 4

dataloader = create_dataloader_v1(
    raw_text,
    batch_size = 8,
    max_length = max_length,
    stride = max_length,
    shuffle = False
)

#读取一个batch
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("inputs:")
print(inputs)
print("inputs.shape:",inputs.shape)

print("targets:")
print(targets)
print("targets.shape:",targets.shape)


#将tokenid 转化为词向量
token_embeddings = token_embedding_layer(inputs)

print("token_embeddings.shape:",token_embeddings.shape)
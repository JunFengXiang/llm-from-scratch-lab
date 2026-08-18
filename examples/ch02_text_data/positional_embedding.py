import torch
from sliding_window import create_dataloader_v1

with open("data/ch02/the-verdict.txt","r",encoding = "utf-8") as f:
    raw_text = f.read()

max_length = 4
vocab_size = 50257
output_dim = 256
dataloader = create_dataloader_v1(
    raw_text,
    batch_size = 8,
    max_length = max_length,
    stride = max_length,
    shuffle = False
)
inputs, targets = next(iter(dataloader))

print("Token IDs:\n",inputs)
print("\nInputs shape:",inputs.shape)
print("\nTargets shape::",targets.shape)

#词元嵌入

token_embedding_layer = torch.nn.Embedding(
    vocab_size,
    output_dim
)

token_embeddings = token_embedding_layer(inputs)

print("\nToken embedding shape:", token_embeddings.shape)

context_length = max_length

pos_embedding_layer = torch.nn.Embedding(
    context_length,
    output_dim
)

position_ids = torch.arange(context_length)
pos_embeddings = pos_embedding_layer(position_ids)
print("\nPosition IDs:",position_ids)
print("Position embeddings shape:",pos_embeddings.shape)


input_embeddings = token_embeddings + pos_embeddings

print("\nInput embeddings shape:",input_embeddings.shape)
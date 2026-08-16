import torch


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

max_length = 4
batch_size = 8
stride = 4


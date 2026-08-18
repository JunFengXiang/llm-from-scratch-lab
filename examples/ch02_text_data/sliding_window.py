import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)#对全部文本进行分词

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i+ max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids) #返回数据集的总行数
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]#返回数据集的指定行

def create_dataloader_v1(
                        txt,
                        batch_size=4, 
                        max_length=256,
                        stride=128,
                        shuffle = True,
                        drop_last = True,
                        num_workers = 0
                        ):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(
        txt,
        tokenizer,
        max_length,
        stride
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader

if __name__ == "__main__":

    with open("data/ch02/the-verdict.txt","r", encoding = "utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")

    enc_text = tokenizer.encode(raw_text)
    print("total tokens:",len(enc_text))
    enc_sample = enc_text[50:]

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print("x:",x)
    print("y:",y)
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context,"---->",desired)


    dataloader = create_dataloader_v1(
        raw_text,
        batch_size = 1,
        max_length = 4,
        stride = 1,
        shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    second_batch = next(data_iter)

    print("first batch:",first_batch)
    print("second batch:",second_batch)
    print("input shape:",first_batch[0].shape)
    print("target shape:",first_batch[1].shape)

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size = 8,
        max_length = 4,
        stride = 4,
        shuffle = False
    )

    inputs,targets = next(iter(dataloader))

    print(inputs.shape)
    print(targets.shape)
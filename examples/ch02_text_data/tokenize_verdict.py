
#完整版示例
from simple_tokenizer import SimpleTokenizerV1
import re
with open("data/ch02/the-verdict.txt","r",encoding = "utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
#set删除固定次元 按从小到大的顺序排列
vocab_size = len(all_words)
print(vocab_size)
print(all_words[:10])

vocab = {
    token: integer
    for integer,token in enumerate(all_words)
}

for i,item in enumerate(vocab.items()):
    print(item)
    if i>= 9:
        break


int_to_str = {
    integer: token
    for token, integer in vocab.items()
}

token_id = vocab["the"]
print(token_id)
print(int_to_str[token_id])

tokenizer = SimpleTokenizerV1(vocab)

text =  '''"It's the last he painted, you know."Mrs. Gisburn said with pardonable pride.'''
ids = tokenizer.encode(text)
print(ids)
#the_id = tokenizer.str_to_int["the"]

#print(the_id)
#print(tokenizer.int_to_str[the_id])

#print(tokenizer.encode("Hello, do you like tea?"))


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend([
    "<|endoftext|>",
    "<|unk|>",
])
vocab = {
    token: integer
    for integer,token in enumerate(all_tokens)
}

print(len(vocab.items()))#items是字典的属性 取出所有"词元,id组合"

for i,item in enumerate(list(vocab.items())[-5:]):
    print(item)
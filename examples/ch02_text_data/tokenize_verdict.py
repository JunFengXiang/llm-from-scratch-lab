# 完整版示例：从原始文本构建词表，再完成编码和解码。
from simple_tokenizer import SimpleTokenizerV2
from simple_tokenizer import SimpleTokenizerV1
import re

# 读取教材中的《The Verdict》文本。
with open("data/ch02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 按单词、标点、破折号和空白字符切分原始文本。
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# 去掉每个片段两端的空白，并过滤空字符串。
preprocessed = [
    item.strip()
    for item in preprocessed
    if item.strip()
]

print(len(preprocessed))
print(preprocessed[:30])

# set() 去除重复 token，sorted() 再按字典序排列。
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
print(all_words[:10])

# enumerate() 为每个 token 分配一个连续整数 ID。
vocab = {
    token: integer
    for integer, token in enumerate(all_words)
}

# 查看词表开头的若干个 (token, ID) 组合。
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 9:
        break


# 创建反向词表：整数 ID -> token 字符串。
int_to_str = {
    integer: token
    for token, integer in vocab.items()
}

# 正向查询和反向查询应该互相对应。
token_id = vocab["the"]
print(token_id)
print(int_to_str[token_id])

# V1 要求所有 token 都已经存在于词表。
tokenizer = SimpleTokenizerV1(vocab)

text = '''"It's the last he painted, you know."Mrs. Gisburn said with pardonable pride.'''
ids = tokenizer.encode(text)
print(ids)

# 为词表增加两个特殊 token。
# <|endoftext|> 可表示文本边界；
# <|unk|> 用于表示词表中没有出现过的 token。
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend([
    "<|endoftext|>",
    "<|unk|>",
])

# 根据新增后的 token 列表重新建立词表。
vocab = {
    token: integer
    for integer, token in enumerate(all_tokens)
}

print(len(vocab.items()))  # items() 返回所有 (token, ID) 组合。

# 查看词表末尾的几个条目，确认特殊 token 已加入。
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# V2 可以把未知 token 替换成 <|unk|>，因此更适合实际文本。
tokenizer_v2 = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

# 用 <|endoftext|> 把两段文本连接起来。
text = " <|endoftext|> ".join((text1, text2))

print(text)

# 编码：文本 -> token ID 列表。
print(tokenizer_v2.encode(text))

# 解码：token ID 列表 -> 文本。
print(tokenizer_v2.decode(tokenizer_v2.encode(text)))

# importlib.metadata 用来读取已安装 Python 包的版本号。
from importlib.metadata import version
import tiktoken

# 确认当前环境中 tiktoken 的版本，便于复现实验结果。
print("tiktoken version:", version("tiktoken"))

# GPT-2 使用的 BPE（Byte Pair Encoding）分词器。
# tokenizer.encode() 会把字符串转换为 token ID 列表，
# tokenizer.decode() 则执行相反的操作。
tokenizer = tiktoken.get_encoding("gpt2")

# <|endoftext|> 是 GPT-2 词表中的特殊 token，
# 常用于表示一段文本的结束。
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)

# 将文本编码为整数 ID。
# allowed_special 显式允许分词器把 <|endoftext|> 当作特殊 token 处理。
integers = tokenizer.encode(
    text,
    allowed_special={"<|endoftext|>"}

)

print(integers)

# 把 token ID 列表还原成文本，用来检查编码和解码是否一致。
decoded_text = tokenizer.decode(integers)
print(decoded_text)

# 逐个查看 token ID 对应的文本片段。
# 一个词不一定对应一个 token；BPE 可能把词拆成多个子词或字节片段。
for token_id in integers:
    token_text = tokenizer.decode([token_id])
    print(token_id, repr(token_text))


# 练习：观察一个看起来不像常见单词的字符串会如何被 BPE 拆分。
exercise_text = "Akwirw ier"
exercise_ids = tokenizer.encode(exercise_text)

print(exercise_ids)

# 查看每个子 token 的 ID 和解码结果。
for token_id in exercise_ids:
    print(token_id, repr(tokenizer.decode([token_id])))

# 重新解码整个 ID 序列。
print(tokenizer.decode(exercise_ids))

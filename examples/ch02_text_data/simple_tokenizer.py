import re


# 版本 1：构造一个最简单的“字符串 token <-> 整数 ID”双向映射。
# 它假设输入文本中的每个 token 都已经存在于词表中。
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # str_to_int: token 字符串 -> token ID。
        self.str_to_int = vocab

        # int_to_str: token ID -> token 字符串。
        # 字典推导式会遍历 vocab 中的每一个 (token, integer)。
        self.int_to_str = {
            integer: token
            for token, integer in vocab.items()
        }

    def encode(self, text):
        # 使用正则表达式切分单词、标点、破折号和空白字符。
        preprocessed = re.split(
            r'([,.?_!"()\']|--|\s)',
            text
        )

        # strip() 去掉每个片段两端的空白；
        # if item.strip() 过滤掉切分产生的空字符串。
        preprocessed = [
            item.strip()
            for item in preprocessed
            if item.strip()
        ]

        # 将每个 token 字符串转换为词表中的整数 ID。
        # V1 不处理未知 token，因此 token 不在词表中时会抛出 KeyError。
        ids = [
            self.str_to_int[token]
            for token in preprocessed
        ]

        # 返回 token ID 列表。
        return ids

    def decode(self, ids):
        # 先把每个整数 ID 映射回 token，再用空格连接。
        text = " ".join(
            self.int_to_str[token_id]
            for token_id in ids
        )

        # 连接后标点前可能多出空格，这里把它删除。
        text = re.sub(
            r'\s+([,.?!"()\'])',
            r'\1',
            text,
        )

        return text


# 版本 2：在 V1 的基础上增加未知 token 处理。
class SimpleTokenizerV2:
    def __init__(self, vocab):
        # 保存 token -> ID 的正向映射。
        self.str_to_int = vocab

        # 创建 ID -> token 的反向映射，供 decode() 使用。
        self.int_to_str = {
            integer: token
            for token, integer in vocab.items()

        }

    def encode(self, text):
        # V2 额外把冒号和分号也作为独立的标点 token。
        preprocessed = re.split(
            r'([,.:;?_!"()\']|--|\s)',
            text
        )

        # 删除空白片段。
        preprocessed = [
            item.strip()
            for item in preprocessed
            if item.strip()
        ]

        # 如果 token 不在词表中，用统一的 <|unk|> 代替。
        # 这样未知词不会导致查表失败。
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>"
            for item in preprocessed
        ]

        # 把处理后的 token 转为整数 ID。
        ids = [
            self.str_to_int[token]
            for token in preprocessed
        ]

        return ids

    def decode(self, ids):
        # 将 token ID 还原为 token 字符串并连接。
        text = " ".join(
            self.int_to_str[token_id]
            for token_id in ids
        )

        # 恢复标点与前一个 token 之间的紧凑格式。
        text = re.sub(
            r'\s+([,.:;?!"()\'])',
            r'\1',
            text
        )

        return text

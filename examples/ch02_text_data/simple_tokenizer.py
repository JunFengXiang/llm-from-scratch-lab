import re

class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {
            integer: token
            for token,integer in vocab.items()
        }
    def encode(self,text):
        preprocessed = re.split(
            r'([,.?_!"()\']|--|\s)',
            text
        )#按照原始规则分词
        preprocessed = [
            item.strip()
            for item in preprocessed
            if item.strip()
        ]#删除空白项
        ids = [
            self.str_to_int[token]
            for token in preprocessed
        ]#用str_to_int查询每个词元
        return ids#返回id列表
    def decode(self,ids):
        text = " ".join(
            self.int_to_str[token_id]
            for token_id in ids
        )
        text = re.sub(
            r'\s+([,.?!"()\'])',
            r'\1',
            text,
        )
        return text
class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {
            integer: token
            for token,integer in vocab.items()

        }
    def encode(self,text):
        preprocessed = re.split(
            r'([,.:;?_!"()\']|--|\s)',
            text
        )
        preprocessed = [
            item.strip()
            for item in preprocessed
            if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int
            else"<|unk|>"
            for item in preprocessed
        ]
        ids = [
            self.str_to_int[token]
            for token in preprocessed
        ]
        return ids
    def decode(self,ids):
        text = " ".join(
            self.int_to_str[token_id]
            for token_id in ids
        )
        text = re.sub(
            r'\s+([,.:;?!"()\'])',
            r'\1',
            text
        )
        return text
    
# 02 - 处理文本数据

## 实现清单

- [ ] 文本清洗和切分的最小示例
- [ ] 词元与词元 ID 的双向映射
- [ ] BPE 的手算示例和 API 实验
- [ ] 滑动窗口 next-token 数据集
- [ ] token embedding 与位置 embedding

## 测试清单

- `decode(encode(text))` 在约定范围内往返一致；
- 未知词元行为明确；
- 输入与目标相差一个位置；
- batch shape 为 `(batch_size, context_length)`；
- 越界和空文本有清晰报错。

## 门禁

闭卷实现滑动窗口 Dataset，并逐维解释一个 batch。

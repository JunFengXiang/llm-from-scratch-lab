# 04 - 从头实现 GPT 模型

## 组件

- [ ] LayerNorm
- [ ] GELU
- [ ] FeedForward
- [ ] residual/shortcut connection
- [ ] TransformerBlock
- [ ] GPTModel
- [ ] greedy generation

## 工程要求

- 每个组件有独立形状测试；
- 配置对象不散落魔法数字；
- 能统计参数量；
- 固定种子下前向可复现；
- 错误的 context length 有明确提示。

## 门禁

从 `(B, T)` token IDs 推导到 `(B, T, vocab_size)` logits 的全部形状。

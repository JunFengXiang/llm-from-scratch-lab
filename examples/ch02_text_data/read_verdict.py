import urllib.request

# 教材使用的《The Verdict》示例文本地址。
# 这里通过 GitHub raw 文件地址下载纯文本内容。
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

# 下载后的本地保存位置。
# 运行前需要确保 data/ch02 目录已经存在。
file_path = "data/ch02/the-verdict.txt"

# 下载文件并保存到本地。
urllib.request.urlretrieve(url, file_path)

# 以 UTF-8 编码读取文本。
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# len(raw_text) 统计的是字符数，不是 token 数。
print("Total number pf character:", len(raw_text))

# 只打印开头的一部分，检查文件是否下载和读取成功。
print(raw_text[:22000])

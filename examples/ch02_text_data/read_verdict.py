import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

file_path = "data/ch02/the-verdict.txt"

urllib.request.urlretrieve(url,file_path)
with open(file_path,"r",encoding ="utf-8") as f:
    raw_text = f.read()

print("Total number pf character:",len(raw_text))
print(raw_text[:22000])

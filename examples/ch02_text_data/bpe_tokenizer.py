from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)

integers = tokenizer.encode(
    text,
    allowed_special={"<|endoftext|>"}

)

print(integers)

decoded_text = tokenizer.decode(integers)
print(decoded_text)

for token_id in integers:
    token_text = tokenizer.decode([token_id])
    print(token_id, repr(token_text))


exercise_text = "Akwirw ier"
exercise_ids = tokenizer.encode(exercise_text)

print(exercise_ids)

for token_id in exercise_ids:
    print(token_id,repr(tokenizer.decode([token_id])))

print(tokenizer.decode(exercise_ids))

from tokenizer import (
    merge_pair,
    train_bpe,
    build_vocab,
    encode,
    decode,
    get_special_tokens,
    decode_with_special_tokens,
    save_tokenizer,
    load_tokenizer,
)

text = "Hello, world! 你好！"

# Test 1: Encode and decode
merges = train_bpe(text, vocab_size=270)
vocab = build_vocab(merges)

ids = encode(text, merges)
assert decode(ids, vocab) == text

print("Test 1 passed!")

# Test 2: Special tokens
special_tokens = get_special_tokens(merges)

ids_with_special = ids + [special_tokens["<|endoftext|>"]]

decoded = decode_with_special_tokens(
    ids_with_special, vocab, special_tokens
)

assert decoded == text + "<|endoftext|>"

print("Test 2 passed!")

# Test 3: Save and load
save_tokenizer(merges, "test_tokenizer.pkl")

loaded_merges = load_tokenizer("test_tokenizer.pkl")

assert loaded_merges == merges

print("Test 3 passed!")

# Test 4: Unicode
vocab = {
        0: b"\xe4",
        1: b"\xbd\xa0"
    }

assert decode([0, 1], vocab) == "你"

print("Test 4: Unicode passed!")

# Test 5: BPE merge
ids = [1, 2, 1, 2, 3]

result = merge_pair(
        ids,
        pair=(1, 2),
        new_id=256
    )

assert result == [256, 256, 3]

print("Test 5: BPE merge passed!")

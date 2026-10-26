import regex as re
import pickle
from collections import Counter
PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,2}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)
def pretokenize(text):
    return re.findall(PATTERN, text)

def text_to_bytes(text):
    chunks=pretokenize(text)
    return [
        tuple(chunk.encode("utf-8"))
        for chunk in chunks
    ]
def get_pairs(byte_chunks):
    counts=Counter()
    for ids, frequency in byte_chunks.items():
        for i in range(len(ids)-1):
            pair=(ids[i], ids[i+1])
            counts[pair]+=frequency
    return counts

def merge_pair(ids, pair, new_id):
    result=[]
    i=0
    while i<len(ids):
        if(
            i<len(ids)-1
            and (ids[i], ids[i+1])==pair
        ):
            result.append(new_id)
            i+=2
        else:
            result.append(ids[i])
            i+=1
    return result
def train_bpe(text, vocab_size=300):
    byte_chunks=Counter(text_to_bytes(text))
    merges={}
    num_merges=vocab_size-256
    for i in range(num_merges):
        pair_counts = get_pairs(byte_chunks)

        if not pair_counts:
            break

        best_pair = pair_counts.most_common(1)[0][0]

        new_id = 256 + i
        new_chunks=Counter()
        for ids, frequency in byte_chunks.items():
            merged=tuple(merge_pair(ids, best_pair, new_id))
            new_chunks[merged]+=frequency
        byte_chunks=new_chunks

        merges[best_pair] = new_id

    return merges
def encode(text, merges):
    byte_chunks=text_to_bytes(text)
    token_ids=[]
    for ids in byte_chunks:
        ids=list(ids)
        for pair, new_id in merges.items():
            ids=merge_pair(ids, pair, new_id)

        token_ids.extend(ids)
    return token_ids
def build_vocab(merges):
    vocab={i:bytes([i])for i in range(256)}
    for pair, new_id in merges.items():
        left, right=pair
        vocab[new_id]=vocab[left]+vocab[right]
    return vocab
def decode(token_ids, vocab):
    byte_sequence=b"".join(
        vocab[token_id]
        for token_id in token_ids
    )
    return byte_sequence.decode("utf-8")
def save_tokenizer(merges, path):
    with open(path, "wb") as f:
        pickle.dump(merges, f)
def load_tokenizer(path):
    with open(path, "rb") as f:
        merges=pickle.load(f)
    return merges
def get_special_tokens(merges):
    endoftext_id = 256 + len(merges)
    return{
     "<|endoftext|>": endoftext_id
    }
def decode_with_special_tokens(token_ids, vocab, special_tokens):
    endoftext_id=special_tokens["<|endoftext|>"]
    parts=[]
    current_bytes=bytearray()
    for token_id in token_ids:
        if token_id==endoftext_id:
            if current_bytes:
                parts.append(bytes(current_bytes).decode('utf-8'))
                current_bytes.clear()
            parts.append("<|endoftext|>")
        else:current_bytes.extend(vocab[token_id])
    if current_bytes:
        parts.append(bytes(current_bytes).decode("utf-8"))
    return "".join(parts)

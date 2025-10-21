# %%

import argparse
import cProfile
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import regex as re
from tqdm import tqdm
from cs336_basics.pre_tokenize import pre_tokenize


# %%
def compute_all_pair_freqs(pre_tokens):
    pair_freqs = defaultdict(int)
    pre_token_pairs = defaultdict(set)
    for pre_token, freq in pre_tokens.items():
        for byte_pair in zip(pre_token[:-1], pre_token[1:]):
            pair_freqs[byte_pair] += freq
            pre_token_pairs[pre_token] |= {byte_pair}
    return pair_freqs, pre_token_pairs


def update_pair_freqs(pre_token, merged, pair_freqs, i, pre_token_freq):
    """When merging two tokens, update the pair frequencies accordingly.

    Remove one occurrence of pairs that involve the merged tokens (3 pairs in general,
    two on the edges of the word). Add new pairs involving the merged pair.
    """
    n = len(pre_token)
    pair_freqs[pre_token[i], pre_token[i + 1]] -= pre_token_freq  # pair replaced
    if i > 0:
        pair_freqs[pre_token[i - 1], pre_token[i]] -= pre_token_freq  # pair before
        pair_freqs[(pre_token[i - 1], merged)] += pre_token_freq  # new pair before
    if i + 2 < n:
        pair_freqs[pre_token[i + 1], pre_token[i + 2]] -= pre_token_freq  # pair after
        pair_freqs[(merged, pre_token[i + 2])] += pre_token_freq  # new pair after
    return pair_freqs


def replace_pair(
    pre_token: tuple[bytes],
    pair: tuple[bytes],
    pair_freqs: dict[tuple[bytes], int],
    pre_token_freq: int,
) -> tuple[bytes]:
    """Apply the merging of the most frequent pair to a pre-token and return it.

    Also update the pair frequencies for pairs involving the merged pair."""
    result = []
    i = 0
    n = len(pre_token)
    while i < len(pre_token):
        if i + 1 < n and (pre_token[i], pre_token[i + 1]) == pair:
            merged = pre_token[i] + pre_token[i + 1]
            result.append(merged)
            pair_freqs = update_pair_freqs(
                pre_token,
                merged,
                pair_freqs,
                i,
                pre_token_freq,
            )
            i += 2
        else:
            result.append(pre_token[i])
            i += 1
    return tuple(result)


def merge_most_frequent_pair(pre_tokens, pair, pair_freqs, pre_token_pairs):
    """Merge the most frequent bytes pair in all the pre-tokens dict."""
    new_pre_tokens = defaultdict(int)
    for pre_token, freq in pre_tokens.items():
        # Skip if pair not in token
        if pair not in pre_token_pairs[pre_token]:
            new_pre_tokens[pre_token] += freq
            continue
        pre_token_new = replace_pair(
            pre_token=pre_token,
            pair=pair,
            pair_freqs=pair_freqs,
            pre_token_freq=freq,
        )
        new_pre_tokens[pre_token_new] += freq
        pre_token_pairs[pre_token_new] = {byte_pair for byte_pair in zip(pre_token_new[:-1], pre_token_new[1:])}
    return new_pre_tokens, pre_token_pairs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """_summary_

    Args:
        input_path (str): _description_
        vocab_size (int): _description_
        special_tokens (list[str]): _description_

    Returns:
        dict[int,bytes],list[tuple[bytes,bytes]]: _description_
    """
    pre_tokens = pre_tokenize(
        input_path,
        special_tokens_pattern="|".join([re.escape(t) for t in special_tokens]),
    )
    with cProfile.Profile() as pr:
        pair_freqs, pre_token_pairs = compute_all_pair_freqs(pre_tokens)

        vocab = {0: b"<|endoftext|>"} | {i + 1: bytes([i]) for i in range(256)}
        next_vocab_index = 257
        merges = []
        for vocab_index in tqdm(range(next_vocab_index, vocab_size), desc="Increasing vocab size"):
            most_frequent_pair = max(
                pair_freqs, key=lambda x: (pair_freqs[x], x)
            )  # lexico order with freq first then pair content
            # logger.info(f"Freq of most frequent = {pair_freqs[most_frequent_pair]}")
            vocab |= {vocab_index: most_frequent_pair[0] + most_frequent_pair[1]}
            merges.append(most_frequent_pair)
            pre_tokens, pre_token_pairs = merge_most_frequent_pair(
                pre_tokens=pre_tokens,
                pair=most_frequent_pair,
                pair_freqs=pair_freqs,
                pre_token_pairs=pre_token_pairs,
            )
            pair_freqs[most_frequent_pair] = 0
        pr.print_stats("cumtime")
    return vocab, merges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--vocab-size", type=int)
    args = parser.parse_args()
    # text = "TinyStoriesV2-GPT4"
    input_path = Path(__file__).parent.parent / f"data/{args.text}-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = args.vocab_size
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    save_dir = args.text
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + "/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(save_dir + "/merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    with open(save_dir + "/vocab.json", "w") as f:
        json.dump({i: str(v) for i, v in vocab.items()}, f, indent=4)
    with open(save_dir + "/merges.json", "w") as f:
        json.dump([[str(m) for m in merge] for merge in merges], f, indent=4)

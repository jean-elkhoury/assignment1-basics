# %%
import pickle
from collections.abc import Iterable, Iterator

import regex as re
from tqdm import tqdm


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.vocab_reverse = {value: key for key, value in vocab.items()}
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Class
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:"""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        pre_tokens = []

        if self.special_tokens:
            special_tokens_pattern = "|".join([re.escape(t) for t in self.special_tokens])
            special_token_occurrences = re.findall(string=text, pattern=special_tokens_pattern)
            strings = re.split(string=text, pattern=special_tokens_pattern)
        else:
            strings = [text]
            special_token_occurrences = []

        for i, string in tqdm(enumerate(strings), desc="Listing strings delimited by special tokens"):
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for word in re.finditer(pattern=PAT, string=string):
                pre_token = tuple(bytes([b]) for b in word[0].encode("utf-8"))
                for merge in self.merges:
                    pre_token = self.apply_merge(pre_token=pre_token, merge=merge)
                pre_tokens.extend(self.vocab_reverse[p] for p in pre_token)
            if special_token_occurrences and i < len(strings) - 1:
                pre_tokens.append(self.vocab_reverse[special_token_occurrences[i].encode("utf-8")])
        return pre_tokens

    def apply_merge(self, pre_token: tuple[bytes], merge: tuple[bytes]):
        """Transform a pre-token using saved merges."""
        result = []
        i = 0
        n = len(pre_token)
        while i < len(pre_token):
            if i + 1 < n and (pre_token[i], pre_token[i + 1]) == merge:
                merged = pre_token[i] + pre_token[i + 1]
                result.append(merged)
                i += 2
            else:
                result.append(pre_token[i])
                i += 1
        return tuple(result)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.

        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory."""
        for str in iterable:
            yield from self.encode(str)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode("utf-8", errors="replace")


# %%
# vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
# merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
# string = "the cat ate"
# tokenizer = Tokenizer(vocab=vocab, merges=merges)
tokenizer = Tokenizer.from_files(
    vocab_filepath="cs336_basics/tinystories/vocab.pkl",
    merges_filepath="cs336_basics/tinystories/merges.pkl",
    special_tokens=["<|endoftext|>"],
)
string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
encoding = tokenizer.encode(string)
print(encoding)
decoded = tokenizer.decode(encoding)
print(decoded)
decoded
# %%

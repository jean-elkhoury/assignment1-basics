import multiprocessing
import os
from collections import defaultdict
from typing import BinaryIO

import regex as re
from loguru import logger


# %%
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_pre_tokens(pre_tokens: dict[bytes, int], chunk_strings: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for string in chunk_strings:
        for word in re.finditer(pattern=PAT, string=string):
            byte_string = tuple(bytes([b]) for b in word[0].encode("utf-8"))
            pre_tokens[byte_string] += 1
    return pre_tokens


def pre_tokenize_string(pre_tokens, string, pattern):
    chunk_strings = re.split(pattern=pattern, string=string)
    pre_tokens = get_pre_tokens(pre_tokens, chunk_strings)
    return pre_tokens


def process_chunk(i, boundaries, input_path, special_tokens_pattern):
    pre_tokens = defaultdict(int)
    with open(input_path, "rb") as f:
        start, end = boundaries[i], boundaries[i + 1]
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")
        pre_tokenize_string(pre_tokens=pre_tokens, string=chunk, pattern=special_tokens_pattern)
    return pre_tokens


def pre_tokenize(input_path, special_tokens_pattern):
    with open(input_path, "rb") as f:
        num_processes = 8
        logger.info("Start chunking")
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    logger.info("Finished chunking. Starting multiprocessed chunk pretokenization")

    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(i, boundaries, input_path, special_tokens_pattern) for i in range(len(boundaries) - 1)]
        results = pool.starmap(process_chunk, args)
    logger.info("Finished chunked pretokenization. Merging chunks")
    pre_tokens = defaultdict(int)
    for pre_tokens_chunk in results:
        for token, freq in pre_tokens_chunk.items():
            pre_tokens[token] += freq
    logger.info("Finished merging chunks")
    return pre_tokens

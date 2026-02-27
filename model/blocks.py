"""Block-related helpers and simple combinatorial utilities."""

from typing import List


def blocks_to_block_index(blocks: List[List[int]], n: int, *, validate: bool = True) -> List[int]:
    """Returns block index for each item in 0..n-1."""
    block_idx = [-1] * n
    for b, block in enumerate(blocks):
        for item in block:
            if validate and (item < 0 or item >= n):
                raise ValueError(f"Item {item} out of range 0..{n-1}.")
            if block_idx[item] != -1:
                raise ValueError("Item appears twice in blocks.")
            block_idx[item] = b
    if validate and any(x < 0 for x in block_idx):
        raise ValueError("Blocks do not cover all items.")
    return block_idx


def T_of_sizes(sizes: List[int]) -> int:
    return sum(s * (s - 1) // 2 for s in sizes)


def T_of_blocks(blocks: List[List[int]]) -> int:
    return T_of_sizes([len(b) for b in blocks])

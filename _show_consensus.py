import json
import pandas as pd
from pathlib import Path
from collections import Counter

run_dir = sorted(Path("movielens/runs").glob("ml_partial_*"))[-1]
with open(run_dir / "map_result.json") as f:
    mp = json.load(f)

movies_meta  = pd.read_csv("movielens/ml-latest-small/movies.csv")
ml_movie_ids = pd.read_csv(run_dir / "movies.csv")["movieId"].tolist()

sizes = Counter(mp["z"])
print("Cluster sizes (non-zero):", {k: v for k, v in sorted(sizes.items()) if v > 0})

for c_idx, cl in enumerate(mp["clusters"]):
    n = sizes.get(c_idx, 0)
    if n == 0:
        continue
    theta = cl["theta"]
    print(f"\n=== Cluster {c_idx} ({n} users, theta={theta:.3f}) — top 30 ===")
    rows = []
    rank = 1
    for block in cl["blocks"]:
        tied = []
        for idx in block:
            mid = ml_movie_ids[idx]
            row = movies_meta[movies_meta["movieId"] == mid]
            title  = row["title"].values[0]  if len(row) else str(mid)
            genres = row["genres"].values[0] if len(row) else ""
            tied.append((rank, title, genres))
        rows.extend(tied)
        rank += len(block)
        if rank > 30:
            break
    print(f"  {'rank':>4}  {'title':<55}  genres")
    print("  " + "-" * 90)
    for rank_val, title, genres in rows[:30]:
        print(f"  {rank_val:>4}  {title:<55}  {genres}")

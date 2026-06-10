import pandas as pd
from itertools import combinations as comb
from collections import defaultdict

def load(file_name):
    data = pd.read_parquet(file_name,
        columns=["Game ID", "Word", "Group Name", "Group Level"])
    data = data.dropna(subset=["Game ID", "Word", "Group Name", "Group Level"])
    return data

def save(dataset, file_name):
    dataset.to_parquet(file_name, index=False)

def main():
    input_file = "data/AI_Slop_15000.parquet"
    output_file = "data/Zweights15k.parquet"

    games = load(input_file)
    groupings = games.groupby(["Game ID","Group Name"])
    Z_table = defaultdict(int)

    # For each unique group
    for _, group in groupings:
        # Get a list of the (4) words
        words = group["Word"].tolist()
        # Grab the 'Group Level' as a stand-in value
        value = group["Group Level"].iloc[0]
        # For each combination of two words
        for word1, word2 in comb(words, 2):
            # Force them to be alphabetical
            a,b = sorted([word1, word2])
            # Add the value into their stored pairing
            Z_table[(a,b)] += 1 + value

    Z_weights = pd.DataFrame([
        (a, b, weight)
        for (a, b), weight in Z_table.items()
    ], columns=["Word1", "Word2", "Weight"])

    save(Z_weights, output_file)


if __name__ == "__main__":
    main()
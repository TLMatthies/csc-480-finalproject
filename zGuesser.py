import pandas as pd
from itertools import combinations as comb
from z3 import Solver, Optimize, Int, If, Sum, sat, unsat

input_file = "data/Zweights.parquet"
zWeights = pd.read_parquet(input_file)
zWlookup = {(row.Word1, row.Word2): row.Weight
            for row in zWeights.itertuples(index=False)}


def groupScore(words):
    total = 0
    for w1, w2 in comb(words,2):
        a,b = sorted([w1,w2])
        total += zWlookup.get((a,b),0) #type:ignore
    return total

def makeGuess(wordList):
    if not len(wordList) == 16:
        print("You need to give me exactly 16 words.")
        return
    
    values = []
    for w1, w2 in comb(wordList, 2):
        a,b = sorted([w1, w2])
        # If word pair doesn't exist, assign val = 0
        val = zWlookup.get((a,b), 0)
        values.append((a, b, val))

    opt = Optimize()
    words = {w: Int(f"group_{w}") for w in wordList}
    # Force first word into the 0th group
    opt.add(words[wordList[0]] == 0)

    # Groups are 0,1,2,3
    for w in wordList:
        opt.add(words[w] >= 0)
        opt.add(words[w] < 4)
    
    # Only 4 words to a group
    for i in range(4):
        opt.add(Sum([If(words[w] == i, 1, 0)
                  for w in wordList]) == 4)
        
    scores = []
    for a, b, val in values:
        scores.append(If(words[a] == words[b], val, 0))
    
    opt.maximize(Sum(scores))

    if opt.check() == sat:
        model = opt.model()
        groups = {0: [], 1: [], 2: [], 3: []}
        for w in wordList:
            g = model[words[w]].as_long() #type:ignore
            groups[g].append(w)
        groupScores = []
        totalScore = 0
        for g, grouped in groups.items():
            score = groupScore(grouped)
            groupScores.append(score)
            totalScore += score #type:ignore

        print("I think you should guess:\n")
        for g, grouped in groups.items():
            score = groupScores[g]
            percent = (100*score/totalScore if totalScore > 0 else 0) #type:ignore
            print(
                f"{g} - "
                f"{score}"
                f"({percent}%): "
                f"{', '.join(grouped)}"
            )
    else:
        print("Oh dear, I can't figure out what I should guess!")
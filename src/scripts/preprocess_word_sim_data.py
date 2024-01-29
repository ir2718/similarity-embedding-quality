import pandas as pd

rg = pd.read_csv("./datasets/word_sim/rg-65.csv")
rg["similarity"] = rg["similarity"] / 4.

simlex = pd.read_csv("./datasets/word_sim/SimLex-999.txt", delimiter="\t")
simlex["SimLex999"] = simlex["SimLex999"] / 10.
simlex.rename({"SimLex999":"similarity"}, inplace=True, axis=1)

simverb = pd.read_csv(
    "./datasets/word_sim/SimVerb-3500.txt", 
    delimiter="\t", 
    header=None,
    names="word1,word2,type,similarity,word_type".split(",")
)
simverb["similarity"] = simverb["similarity"] / 10.

wordsim353 = pd.read_csv("./datasets/word_sim/wordsim353-sim.csv", index_col=0)
wordsim353["similarity"] = wordsim353["similarity"] / 10.

c = "word1,word2,similarity".split(",")
final_df = pd.concat(
    (rg[c], simlex[c], simverb[c], wordsim353[c]),
)
final_df['sorted_pair'] = final_df[['word1', 'word2']].apply(lambda x: tuple(sorted(x)), axis=1)
final_df = final_df.drop_duplicates(subset='sorted_pair', keep='first')
final_df = final_df.drop(columns=['sorted_pair'])
final_df.reset_index(inplace=True, drop=True)

final_df.to_csv("./datasets/word_similarity_dataset.csv")
print(final_df.shape)
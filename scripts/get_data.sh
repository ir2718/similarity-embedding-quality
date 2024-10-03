mkdir -p datasets
cd datasets

# get korean dataset
git clone git@github.com:kakaobrain/kor-nlu-datasets.git

# get word similarity datasets
mkdir word_sim

# rg65
wget https://raw.githubusercontent.com/vecto-ai/word-benchmarks/refs/heads/master/word-similarity/monolingual/en/rg-65.csv
mv rg-65.csv word_sim/

# wordsim353
wget https://raw.githubusercontent.com/vecto-ai/word-benchmarks/refs/heads/master/word-similarity/monolingual/en/wordsim353-sim.csv
mv wordsim353-sim.csv word_sim/

# simlex999 and cleanup
wget https://fh295.github.io/SimLex-999.zip
mv SimLex-999.zip word_sim/
unzip word_sim/SimLex-999.zip -d word_sim/
mv word_sim/SimLex-999/SimLex-999.txt word_sim/
rm word_sim/SimLex-999.zip
rm -r word_sim/SimLex-999/

# simverb3500
wget https://raw.githubusercontent.com/benathi/word2gm/refs/heads/master/evaluation_data/simverb/data/SimVerb-3500.txt
mv SimVerb-3500.txt word_sim/
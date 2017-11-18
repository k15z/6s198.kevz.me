import json
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups

dataset = defaultdict(list)

for subset in ["train", "test"]:
    newsgroups = fetch_20newsgroups(subset=subset, categories=["sci.crypt", "sci.electronics", "sci.med", "sci.space"])
    for target, text in zip(newsgroups.target, newsgroups.data):
        target = newsgroups.target_names[target]
        dataset["%sX" % subset].append(text)
        dataset["%sY" % subset].append(target)

with open("dataset.json", "wt") as fout:
    json.dump(dataset, fout)

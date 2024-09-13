import faiss
import sys
import json
import numpy as np
import bz2

most_freq_words_idxs = json.load(open('most_freq_words.json', 'r'))

file_ids = json.load(open('file_ids.json', 'r'))

docvec = np.zeros(len(most_freq_words_idxs))
file_to_check = sys.argv[1]
with bz2.open(file_to_check, 'rt') as f:
    data = json.load(f)
    for page in data['features']['pages']:
        if 'body' in page and page['body'] is not None and 'tokenPosCount' in page['body'] and page['body']['tokenPosCount'] is not None:
            for word in page['body']['tokenPosCount']:
                if word in most_freq_words_idxs:
                    docvec[most_freq_words_idxs[word]] = 1

faiss_index = faiss.read_index('index.faiss')

k = 5
D, I = faiss_index.search(np.array([docvec]), k)
for i in range(k):
    print("Score:", D[0][i], file_ids[str(I[0][i])])



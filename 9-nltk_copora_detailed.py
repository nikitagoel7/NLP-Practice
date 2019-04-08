from nltk.corpus.reader import TwitterCorpusReader
from nltk.tokenize import sent_tokenize
import os

root = os.environ['TWITTER']
reader = TwitterCorpusReader(root, '.*\.json')
t = twitter.TwitterCorpusReader()
#sample = t.raw("sample.txt")

#tok = sent_tokenize(sample)

#print (tok[5:15])
import json
for tweet in reader.docs():
    print(json.dumps(tweet, indent=1, sort_keys=True))

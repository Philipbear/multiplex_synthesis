import os


files = os.listdir('/home/shipei/projects/microbe_masst/output/all')

files = [f for f in files if f.endswith('_matches.tsv')]

print('unique files:', len(files))



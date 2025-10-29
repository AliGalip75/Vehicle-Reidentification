import pandas as pd
q = pd.read_csv("datasets/annot/query_fixed.csv")
g = pd.read_csv("datasets/annot/gallery_fixed.csv")
print(len(q), len(g))


from scipy.io import loadmat
mat = loadmat("pytorch_result.mat")
print(mat["query_f"].shape, mat["gallery_f"].shape)

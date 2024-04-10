import hnswlib
import numpy as np
import pickle
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="generate index file from itemid vectors")
parser.add_argument("--file", help="filepath e.g. output/xxx.txt",  required=True)
parser.add_argument("--output", help="output filepath e.g. indices/xxx.bin",  required=True)
parser.add_argument("--dim", help="vector dimensions", type=int, default=200)
parser.add_argument("--space", help="index space e.g. l2/cosine", default="cosine")
parser.add_argument("--ef", help="ef construction", type=int, default=200)
parser.add_argument("--m", help="index m", type=int, default=16)
parser.add_argument("--k", help="k nearest neighbors", type=int, default=1000)
args = parser.parse_args()

print("open file {}".format(args.file))

df = pd.read_parquet(args.file, engine='pyarrow')

ids = df.iloc[:,0].to_numpy()
vecs = df.iloc[:,1].to_numpy()
num_elements = len(ids)
ids = np.uint64(ids).reshape((num_elements,))


v = pd.Series(map(np.array,map(list,vecs)))
ve = np.stack(v.values)
vecs = np.float32(ve)


print("dim: ",args.dim)


# Declaring index
p = hnswlib.Index(space = args.space, dim = args.dim) # possible options are l2, cosine or ip

# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 2000, M = 16, random_seed = 100)

# Element insertion (can be called several times):
p.add_items(vecs, ids)

# Controlling the recall by setting ef:
p.set_ef(1000) # ef should always be > k

p.save_index(args.output)
print("save complete")
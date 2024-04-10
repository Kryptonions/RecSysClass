from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec

import argparse
import time
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="word2vec test file")
    parser.add_argument("--input", help="input filepath e.g. hdfs://R2/input/*", required=True)
    parser.add_argument("--output", help="output filepath e.g. example_model", default="example_model")
    parser.add_argument("--dim", help="vector dimensions", type=int, default=32)
    parser.add_argument("--space", help="index space e.g. l2/cosine", default="cosine")
    parser.add_argument("--window", help="window size", type=int, default=10)
    parser.add_argument("--k", help="k nearest neighbors", type=int, default=5)
    parser.add_argument("--iter", help="num iterations", type=int, default=15)
    parser.add_argument("--partitions", help="num partitions", type=int, default=15)
    parser.add_argument("--test_word", help="test word", default="3615338480")
    args = parser.parse_args()

    print("start training at time {}".format(time.time()))
    sc = SparkContext(appName="word2vec_train".format())  # SparkContext
    print("spark settings \n{}".format(sc.getConf().getAll()))

    inp = sc.textFile(args.input).map(lambda row: row.split(" "))

    word2vec = Word2Vec()
    word2vec.setVectorSize(args.dim)
    word2vec.setWindowSize(args.window)
    word2vec.setNumIterations(args.iter)
    word2vec.setNumPartitions(args.partitions)

    model = word2vec.fit(inp)

    output_path = args.output
    model.save(sc, output_path)
    print("saved, complete at time {}".format(time.time()))

    sc.stop()
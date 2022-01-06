#!/usr/bin/env python3

## import required libraries and set random number generator seeds:
import numpy as np; np.random.seed(1)
import tensorflow as tf; tf.random.set_seed(2)
from io import StringIO
import itertools
import pandas as pd
from subprocess import check_output
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout

def kmerCounts(seq, k):
    nucs = ["A", "C", "G", "T"]
    kms = ["".join(el) for el in itertools.product(*[nucs]*k)]
    out = np.zeros((len(seq), len(kms)), int)
    maxLen = seq.str.len().max()
    for i in range(maxLen-k+1):
        ikm = seq.str[i:(i+k)].values
        for (j, km) in enumerate(kms):
            out[ikm == km, j] += 1
    return pd.DataFrame(out, index=list(seq.index), columns=kms)

## -----------------------------------------------------------------------------
## read in sequence scores from training and test set files on github:
dataUrl = "https://raw.githubusercontent.com/KosuriLab/" + \
          "ecoli_promoter_mpra/master/processed_data/combined/"
trainScoreFile =\
    "tss_scramble_peak_expression_model_format_floored_train_genome_split.txt"
train = pd.read_csv(dataUrl + trainScoreFile,
                    sep="\t", header=None, index_col=None)
testScoreFile =\
    "tss_scramble_peak_expression_model_format_floored_test_genome_split.txt"
test = pd.read_csv(dataUrl + testScoreFile,
                   sep="\t", header=None, index_col=None)

## write training sequences to fasta file:
trainSeqsFile = "train_sequences.fa"
with open(trainSeqsFile, "w") as writer:
    for i in range(train.shape[0]):
        garbage = writer.write(">" + str(i) + "\n" + train.iloc[i, 0] + "\n")

## write test sequences to fasta file:
testSeqsFile = "test_sequences.fa"
with open(testSeqsFile, "w") as writer:
    for i in range(test.shape[0]):
        garbage = writer.write(">" + str(i) + "\n" + test.iloc[i, 0] + "\n")

## -----------------------------------------------------------------------------
## use Sarkes.jl to train an eigenboosted sarks model for scoring sequences:
sarksFile = "eigenboostsarks_hw250_b10_s0p75_pc3.jld2"
nPCs = 3
check_output("julia train_eigenboostsarks.jl" +
             " --trainscores " + dataUrl + trainScoreFile + " --noheader" +
             " --halfwindow " + str(250) +
             " --boost " + str(10) +
             " --shrinkage " + str(0.75) +
             " --pc " + str(nPCs) +
             " --out " + sarksFile,
             shell = True)

## use trained eigenboostsarks model (stored in jld2-format sarksFile)
## to score training sequences:
trainScores = pd.read_csv(StringIO(check_output(
    "julia score_eigenboostsarks.jl" +
    " --sarks " + sarksFile +
    " --seqs " + trainSeqsFile +
    " --k " + str(15),
    shell = True
).decode("utf-8")), sep="\t", header=None, index_col=None)

## calculate means and standard deviations of eigenboostsarks features
## in training set for centering and scaling of data for tensorflow:
nFeatsPerPC = int(trainScores.shape[1] / nPCs)
mu = pd.concat([
    pd.Series([trainScores.iloc[:, range((i*nFeatsPerPC),
                                         ((i+1)*nFeatsPerPC))].mean().mean()
              ] * nFeatsPerPC) for i in range(nPCs)
], ignore_index=True)
mu2 = pd.concat([
    pd.Series([(trainScores.iloc[:, range((i*nFeatsPerPC),
                                          ((i+1)*nFeatsPerPC))]**2).mean().mean()
              ] * nFeatsPerPC) for i in range(nPCs)
], ignore_index=True)
sig = np.sqrt((mu2 - mu**2) * (trainScores.shape[0] / (trainScores.shape[0]-1)))

## center and scale training data matrix for tensorflow:
trainScores = (trainScores - mu) / sig

## use trained eigenboostsarks model (stored in jld2-format sarksFile)
## to score test sequences:
testScores = pd.read_csv(StringIO(check_output(
    "julia score_eigenboostsarks.jl" +
    " --sarks " + sarksFile +
    " --seqs " + testSeqsFile +
    " --k " + str(15),
    shell = True
).decode("utf-8")), sep="\t", header=None, index_col=None)

## now center and scale test data matrix for tensorflow
## (using means and stdevs calculated from training data):
testScores = (testScores - mu) / sig

## -----------------------------------------------------------------------------
## calculate pentamer count matrix for training sequences:
train5mer = kmerCounts(train.iloc[:, 0], 5)
## calculate training set means and stdevs of pentamer counts:
mu5mer = train5mer.mean(axis=0)
sig5mer = train5mer.std(axis=0)
## center and scale training set pentamer count matrix:
train5mer = (train5mer - mu5mer) / sig5mer

## calculate pentamer count matrix for test sequences:
test5mer = kmerCounts(test.iloc[:, 0], 5)
## center and scale test set pentamer count matrix
## (using means and stdevs calculated from training data):
test5mer = (test5mer - mu5mer) / sig5mer

## -----------------------------------------------------------------------------
## concatenate eigenboostsarks and pentamer feature sets:
xtrain = pd.concat([trainScores, train5mer], axis=1)
xtest = pd.concat([testScores, test5mer], axis=1)

## construct and train ANN model with 3 hidden layers
## (of 50, 25, and 10 nodes, respectively)
## using concatenated training set feature matrix:
model = keras.Sequential([
    Dense(units=50), BatchNormalization(), Activation("elu"), Dropout(0.5),
    Dense(units=25), BatchNormalization(), Activation("elu"),
    Dense(units=10), BatchNormalization(), Activation("elu"),
    Dense(units=1)
])
model.compile(optimizer="Adam", loss="mse", metrics=["mse"])
model.fit(xtrain.values, train.loc[:, 1].values,
          epochs=20, batch_size=512, validation_split=0.10)

## use trained ANN model to score test-set sequences:
testPred = model.predict(xtest.values)
np.corrcoef(testPred[:, 0], test.loc[:, 1])[0, 1]**2  ## 0.347973

## repeat ANN model training 100 times
## (to average over stochasticity in deep learning algorithms;
##  note sarks itself is deterministic, so no need to re-run it):
r2s = []
for rep in range(100):
    model = keras.Sequential([
        Dense(units=50), BatchNormalization(), Activation("elu"), Dropout(0.5),
        Dense(units=25), BatchNormalization(), Activation("elu"),
        Dense(units=10), BatchNormalization(), Activation("elu"),
        Dense(units=1)
    ])
    model.compile(optimizer="Adam", loss="mse", metrics=["mse"])
    model.fit(xtrain.values, train.loc[:, 1].values,
              epochs=20, batch_size=512, validation_split=0.10)
    testPred = model.predict(xtest.values)
    r2s.append(np.corrcoef(testPred[:, 0], test.loc[:, 1])[0, 1]**2)

np.mean(r2s)  ## 0.3562902080507185
np.std(r2s)   ## 0.009238788083716365

## =============================================================================
## assemble results into DataFrame and save as tsv file:
out = pd.concat([pd.DataFrame({
                     "set" : "train",
                     "expression" : train.iloc[:, 1].values,
                     "prediction" : model.predict(xtrain.values)[:, 0],
                     "sequence" : train.iloc[:, 0].values
                 })[["set", "expression", "prediction", "sequence"]],
                 pd.DataFrame({
                     "set" : "test",
                     "expression" : test.iloc[:, 1].values,
                     "prediction" : model.predict(xtest.values)[:, 0],
                     "sequence" : test.iloc[:, 0].values
                 })[["set", "expression", "prediction", "sequence"]]],
                axis = 0)

out.to_csv("urtecho_2020_deepsarks.tsv.gz", compression="gzip",
           sep="\t", header=True, index=False)

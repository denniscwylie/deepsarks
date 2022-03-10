using ArgParse
using CSV
using DataFrames
using DataStructures
using HTTP
using JLD2
using Sarkses
using Statistics

## =============================================================================
s = ArgParseSettings()
@add_arg_table s begin
    "--trainseqs"
        help = "fasta file containing training set sequences"
    "--trainscores"
        help = "two column tsv file (with header): 1st column seq ids, 2nd column scores"
        required = true
    "--noheader"
        help = "flag to indicate that trainscores file has no header line"
        action = :store_true
    "--halfwindow", "-w"
        help = "half window width (kappa) for suffix array kernel smoothing"
        arg_type = Int
        required = true
    "--boost", "-b"
        help = "number of gradient-boosted sarks models to fit"
        arg_type = Int
        default = 1
    "--shrinkage", "-s"
        help = "shrinkage applied to predictions for pseudoresidual calculation"
        arg_type = Float64
        default = 0.0
    "--pc", "-p"
        help = "how many principal component sarks models to retain"
        arg_type = Int
        default = 1
    "--out", "-o"
        help = "name of jld2 file in which to save trained sarks object"
        required = true
end
parsed_args = parse_args(ARGS, s);

trainSeqsFile = parsed_args["trainseqs"];
trainScoresFile = parsed_args["trainscores"];
hw = parsed_args["halfwindow"];
nBoost = parsed_args["boost"];
shrinkage = parsed_args["shrinkage"];
nPC = parsed_args["pc"];

## =============================================================================
trainScoreDF = if match(r"^https?://", lowercase(trainScoresFile)) !== nothing
    CSV.read(HTTP.get(trainScoresFile).body,
             DataFrame, delim="\t", header=!parsed_args["noheader"])
else
    CSV.read(trainScoresFile,
             DataFrame, delim="\t", header=!parsed_args["noheader"])
end;

trainSeqs = nothing;
if trainSeqsFile != nothing
    trainSeqs = Sarkses.readFasta(trainSeqsFile);
    trainScores = OrderedDict([string(trainScoreDF[i, 1]) => trainScoreDF[i, 2]
                               for i in 1:size(trainScoreDF, 1)]);
else
    trainSeqs = OrderedDict([string(i) => trainScoreDF[i, 1]
                             for i in 1:size(trainScoreDF, 1)]);
    trainScores = OrderedDict([string(i) => trainScoreDF[i, 2]
                               for i in 1:size(trainScoreDF, 1)]);
end

## =============================================================================
## -- boosting -----------------------------------------------------------------
meanScore = mean(values(trainScores));
centScores = OrderedDict([seqName => (score-meanScore)
                          for (seqName, score) in trainScores])
boosted = Sarkses.boostSarks(nBoost, trainSeqs, centScores,
                             hw, 0.0, shrinkage=shrinkage);
baseSarks = boosted.sarkses[1];

## =============================================================================
## -- eigensarksing ------------------------------------------------------------
##    construct fewer, less correlated, principal component sarks models
##    capturing most of variation from boosted sarks models;
## first compute required eigendecomposition:
eigScores = Sarkses.eigenScores(boosted.sarkses, retainFirst=true);
## extract nPC principal component sarks models from the nBoost sarks models:
pcSarkses = [Sarkses.rescore(baseSarks, eigScores[i]) for i in 1:nPC];

# @save "boosted.jld2" boosted
@save parsed_args["out"] pcSarkses;

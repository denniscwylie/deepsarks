using ArgParse
using CSV
using DataFrames
using JLD2
using Sarkses

## =============================================================================
s = ArgParseSettings()
@add_arg_table s begin
    "--sarks"
        help = "jld2 file containing saved trained sarks model"
        required = true
    "--seqs"
        help = "fasta file containing sequences to be scored"
        required = true
    "--k", "-k"
        help = "maximum k-mer length to base smoothed scores on"
        arg_type = Int
        default = 15
end
parsed_args = parse_args(ARGS, s);

sarksFile = parsed_args["sarks"];
seqsFile = parsed_args["seqs"];
k = parsed_args["k"];

## =============================================================================
seqs = Sarkses.readFasta(seqsFile);
@load sarksFile pcSarkses;
mats = [Sarkses.sarksPositionalScoreMat(pcSarks, seqs, k=k)
        for pcSarks in pcSarkses];
CSV.write(stdout,
          DataFrame(hcat(mats...), :auto),
          delim = "\t",
          writeheader = false);

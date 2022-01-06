# deepsarks
Examples of usage of
[Sarkses.jl][https://github.com/denniscwylie/Sarkses.jl] feature
extraction functionality in predictive modeling of numeric scores
associated with sequence data.

## Urtecho 2020 example
[Urtecho et al.][https://www.biorxiv.org/content/10.1101/2020.01.04.894907v1]
measured gene expression levels resulting from many distinct promoter
sequences in *E. coli* using a massively parallel reporter assay. In
addition, they split many of these sequences into a separate training
and test set for benchmarking various different machine learning
approaches for predicting promoter activity based on promoter sequence.

The file [**urtecho_2020_deepsarks.py**][urtecho_2020_deepsarks.py]
applies a novel deep learning approach in which the features input to
an ANN (with 3 hidden layers) consist of:
- SArKS-based extracted feature scores
  - employing PCA-summarization (top 3 PCs) of
  - gradient-boosted SArKS models (10 boosting stages, each resulting
    in a separate SArKS model),
  combined with:
- counts of all 1024 distinct pentamers

The feature extraction steps performed in this approach are performed
during the execution of `urtecho\_2020\_deepsarks.py` by running (as
subprocesses) the two julia scripts:
- [**train_eigenboostsarks.jl**][train_eigenboostsarks.jl] and
- [**score_eigenboostsarks.jl**][score_eigenboostsarks.jl]

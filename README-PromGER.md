# PromGER
## Description
PromGER is a predictor for eukaryotic promoter identification utilizing sequence characteristics and graph embedding methods.

## Requirements
The code is based on python 3.7 It depends on several python packages, such as numpy, scikit-learn, networkx, pandas, catboost, karateclub.
* conda install numpy, pandas, scikit-learn, biopython, networkx
* pip install catboost, karateclub

## Usage

Command line usage:
```
$python main.py [-pos_fa pos_fa] [-neg_fa neg_fa] [-test_fa test_fa]
         [-dataset dataset] [-out out]
```
for example:
```
$python main.py -pos_fa ./datasets/hs_TATA_251bp_1/hs_TATA_251bp_1_P.fasta -neg_fa ./datasets/hs_TATA_251bp_1/hs_TATA_251bp_1_N.fasta -test_fa ./datasets/hs_TATA_251bp_1/hs_TATA_251bp_1_Test.fasta -dataset Hs -out ./results/Hs_TATA_251_1_results.csv
```

## Note
Please set your own optimal parameters according to the resource configuration.
To download most of the external datasets used in the paper (i.e., MNIST, QSAR, ENRON, and scRNA-Seq data), run `python simulations/download.py` from the command line. This will create a subdirectory `data` in the main directory containing the relevant files. 

To obtain the preliminary data for the micro-credit study (in `data/microcredit.csv`) you will need to run the following Quarto document.

### Reproducing the micro-credit example from Broderick et al. (2023)

1.  Download and extract the [Meager (2019) replication data](https://doi.org/10.3886/E116357V1) into the `./simulations/116357-V1` folder.
2.  Download [Quarto](https://quarto.org/docs/get-started/) and run the command

```
quarto render simulations/Reproduce-Broderick-et-al-2023-outlier.qmd
```
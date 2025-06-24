# UniProt Dataset

The UniProt Knowledgebase (UniProtKB) is an open database of protein sequences curated from translated genomic data [1].
The UniProt Reference Cluster (UniRef) databases provide clustered sets of sequences from UniProtKB [2], which have been
used in previous large language model training studies to improve diversity in protein training data. UniRef clusters
proteins hierarchically. At the highest level, UniRef100 groups proteins with identical primary sequences from the
UniProt Archive (UniParc). UniRef90 clusters these unique sequences into buckets with 90% sequence similarity, selecting
a single sequence from within each cluster as the representative sequence. UniRef50 is then built by clustering these
UniRef90 representative sequences into groups with 50% sequence similarity.

## Data Used for ESM-2 Pre-training

Since the original train/test splits from ESM-2 were not available [3], we replicated the ESM-2 pre-training experiments
with UniProt's 2024_03 release. Following the approach described by the ESM-2 authors, we removed artificial sequences
and reserved 0.5% of UniRef50 clusters for validation. From the 65,672,139 UniRef50 clusters, this resulted in 328,360
validation sequences. We then ran MMSeqs to further ensure no contamination of the training set with sequences similar
to the validation set. This resulted in 65,182,365 training UniRef50 clusters comprising 187,382,018 UniRef90 sequences.

Pretraining batches were formed by uniformly sampling each UniRef50 cluster from the training database, taking a
randomly chosen UniRef90 sequence from each.

## Data Availability

Two versions of the dataset are distributed, a full training dataset (~80Gb) and a 10,000 UniRef50 cluster random slice
(~150Mb). To load and use the sanity dataset, the [bionemo.core.data.load][bionemo.core.data.load.load] function
can be used to materialize the sanity dataset in the BioNeMo2 cache directory:

```python
from bionemo.core.data.load import load

sanity_data_dir = load("esm2/testdata_esm2_pretrain:2.0")
```

### NGC Resource Links

* [Sanity Dataset](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/esm2_pretrain_nemo2_testdata/files)
* [Full Dataset]

## Reference

1. UniProt Consortium. (2023). UniProt: The universal protein knowledgebase in 2023. Nucleic Acids Research, 51(D1),
   D523–D531. doi:10.1093/nar/gkac1052

2. Suzek, B. E., Wang, Y., Huang, H., McGarvey, P. B., Wu, C. H., & UniProt Consortium. (2015). UniRef clusters: a
   comprehensive and scalable alternative for improving sequence similarity searches. Bioinformatics (Oxford, England),
   31(6), 926–932. doi:10.1093/bioinformatics/btu739

3. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., … Rives, A. (2023). Evolutionary-scale prediction of
   atomic-level protein structure with a language model. Science (New York, N.Y.), 379(6637), 1123–1130.
   doi:10.1126/science.ade2574

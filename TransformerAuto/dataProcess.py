import pandas as pd
import numpy as np
from itertools import permutations
def rebuiltDataset(goldPath, expPath):
    goldStandard = pd.read_csv(goldPath, sep='\t', header=None)
    geneExpression = pd.read_csv(expPath, sep='\t')
    gene_pairs = list(permutations(geneExpression.columns, 2))
    trainList = [[geneExpression[gene1].values, geneExpression[gene2].values] for gene1, gene2 in gene_pairs]
    gold_dict = {(row[0], row[1]): 1 for row in goldStandard.to_numpy()}
    targets = np.array([gold_dict.get((gene1, gene2), 0) for gene1, gene2 in gene_pairs])
    return np.array(trainList).reshape((len(gene_pairs),-1 , 2)), targets, gene_pairs
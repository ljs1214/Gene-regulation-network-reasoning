# An sparse opyimization toolbox contains test data generation and network reasoning

## Test Data Generation

### **Import**

```{python}
from sparsetools import matCreater
```

### **Data generation**

```{python}
matCreater.matCreater(tfLen=10, sampleNums=200, geneNums=2000, normalLoc=0, normalScale=0.1)
```
| Parameter  |      Type      |  Explanation|
|----------|:-------------:|------:|
| tfLen|  int | The numbers of transcribe factors|
| sampleNums|    int  |  The numbers of transcribe samples  |
|geneNums | int|   The numbers of target genes  |
|normalLoc | float: recommond use 0|  Mean value of Gaussian noise  |
|normalScale | float|   Variance of Gaussian noise|

### **Return:**
| Parameter  |      Type      | Shapes| Explanation|
|----------|:-------------:|------:|------:|
| W_d|  np.array | (tfLen, sampleNums)|Over complete dictionary|
| zNetwork|    np.array  | (geneNums, tfLen)| Sparse matrix  |
|xTargetGene | np.array| (geneNums,sampleNums)|  Target   |
  

## Network reasoning  

```{python}
from sparsetools import Optimization
Optimization.networkreasoning(expre, HGS, tf_names, gene_names)
```
| Parameter  |      Type      |  Explanation|
|----------|:-------------:|------:|
| expre|  np.array | The expresion matrix of genes|
| HGS|    np.array  |  The matrix of network, first row is the names of TF and second row is the names of target genes.|
|tf_names| np.array|   Names of tf |
|gene_names| np.array|  Name of all genes(including TF)  |

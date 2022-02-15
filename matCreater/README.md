# An sparse opyimization toolbox contains test data generation and network reasoning

## Test Data Generation

### **Import**

```{python}
import matCreater
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


Relationship between parameters:
$$W_d*z_{Network} = x_{TargetGene}+\mu$$
$$\mu \sim N(normalLoc, normalScale)$$
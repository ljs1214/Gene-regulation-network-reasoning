from sparsetools import matCreater
from sparsetools import Optimization


expre = pd.read_csv(
    "/Users/nianhua/Nutstore\ Files/我的坚果云/jupyter/gene/network1/net1_expression_data.tsv", sep="\t")
HGS = pd.read_csv(
    "/Users/nianhua/Nutstore\ Files/我的坚果云/jupyter/gene/network1/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv", sep="\t", header=None)
tf_names = pd.read_csv(
    "/Users/nianhua/Nutstore\ Files/我的坚果云/jupyter/gene/network1/net1_transcription_factors.tsv", sep=",", header=None)
gene_names = pd.read_csv(
    "/Users/nianhua/Nutstore\ Files/我的坚果云/jupyter/gene/network1/net1_gene_ids.tsv", sep="\t")
expre = expre.T
del gene_names["Name"]
Optimization.networkreasoning(expre, HGS, tf_names, gene_names)

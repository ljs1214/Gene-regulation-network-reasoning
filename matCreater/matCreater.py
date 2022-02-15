def matCreater(tfLen=10, sampleNums=200, geneNums=2000, normalLoc=0.0, normalScale=0.1):

    zNetwork = []
    W_d = np.random.rand(tfLen, sampleNums)
    for i in range(geneNums):
        tempNetwork = [0 for _ in range(tfLen)]
        noneZeroIndex = np.random.choice(range(tfLen), 2, replace=False)
        for j in noneZeroIndex:
            tempNetwork[j] = random.uniform(-1, 1)
        zNetwork.append(tempNetwork)
    zNetwork = np.array(zNetwork)
    W_d = W_d.T
    zNetwork = zNetwork.T
    xTargetGene = np.dot(W_d, zNetwork)
    for i in range(len(xTargetGene)):
        for j in range(len(xTargetGene[i])):
            xTargetGene[i][j] += np.random.normal(loc=0.0, scale=0.1)
    return W_d, zNetwork, xTargetGene

def zNetworkTranslator(z):
    for i in range(len(z)):
        for j in range(len(z[i])):
            if z[i][j] != 0.0:
                z[i][j] = 1
    return z
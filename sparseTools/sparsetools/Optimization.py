def shrinkage(x, theta):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))


def ista(X, W_d, a, L, max_iter=100, eps=0.0001):
    # print(X.shape)
    # print(W_d.shape)
    eig, eig_vector = np.linalg.eig(W_d.T * W_d)
    print(np.max(eig))
    L = np.max(eig)
    del eig, eig_vector
    W_e = W_d.T / L
    recon_errors = []
    # print(W_d.shape[1])
    Z_old = np.zeros((W_d.shape[1], 1))
    # print(Z_old.shape)
    for i in range(max_iter):
        temp = W_d * Z_old - X
        #print(W_e * temp)
        Z_new = shrinkage(Z_old - W_e * temp, a / L)
        if np.sum(np.abs(Z_new - Z_old)) <= eps:
            break
        Z_old = Z_new
        recon_error = np.linalg.norm(X - W_d * Z_new, 2) ** 2
        recon_errors.append(recon_error)
        # print(recon_error)
    return Z_new, recon_errors


def networkreasoning(expre, HGS, tf_names, gene_names):
    network_tf = np.zeros((len(tf_names), len(gene_names)))

    # Change HGS into network matrix
    for i in range(len(HGS)):
        if HGS[2].iloc[i] == 1:
            try:
                network_tf[tf_names_list.index(
                    HGS[0].iloc[i]), name_list.index(HGS[1].iloc[i])] = 1
            except:
                pass
        else:
            try:
                network_tf[tf_names_list.index(
                    HGS[0].iloc[i]), name_list.index(HGS[1].iloc[i])] = 1
            except:
                pass
    new_expre = expre.T
    network_predict_tf = np.zeros((len(gene_names), len(gene_names)))
    temp_list = np.zeros((len(name_list), len(tf_names_list)))

    for i in tqdm(range(len(new_expre.iloc[0]))):  # len(new_expre.iloc[0])
        flag = False
        y = new_expre[gene_names["#ID"]].T.iloc[i]
        copy_new_expre = tf_exp.copy()
        try:
            del copy_new_expre[gene_names["#ID"][i]]
            print(gene_names["#ID"][i])
            temp_index = tf_names_list.index(gene_names["#ID"][i])
            flag = True
            print("T")  # 处理tf作为被调控因子的情况
        except:
            pass
        X = y.iloc[i]
        W_d = copy_new_expre
        Z_recon, recon_errors = ista(np.mat(X), np.mat(
            W_d.values), 0.01, 20000000, 100, 0.00001)
        par_lists = sum(Z_recon.T.A)
        if flag:
            print(len(par_lists))
            par_lists = np.insert(par_lists, temp_index, 0)
            print(len(par_lists))
        for j in range(len(temp_list[i])):
            predict_tf[i][j] = par_lists[j]
    return predict_tf

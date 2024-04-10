import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

def recogNet(transformer_autoencoder, trainList, targets):
    x_test_pred = transformer_autoencoder.predict(trainList)
    reconstruction_error = np.mean(np.power(trainList - x_test_pred, 2), axis=(1, 2))
    threshold = np.quantile(reconstruction_error, 0.95)

    # 如果重构误差大于阈值，则标记为异常（1），否则为正常（0）
    predicted_labels = np.where(reconstruction_error > threshold, 1, 0)
    
    # 假设true_labels是真实的标签数组，形状与x_test的第一维相同
    auc_score = roc_auc_score(targets, predicted_labels)

    print(f"AUC: {auc_score}")

    recall = recall_score(targets, predicted_labels, average='binary')

    print(f'Recall: {recall}')

    

    cm = confusion_matrix(targets, predicted_labels)
    TP = cm[1, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    FP = cm[0, 1]

    # 计算真正率（TPR）和假正率（FPR）
    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)

    print(f'True Positive Rate (TPR): {TPR}')
    print(f'False Positive Rate (FPR): {FPR}')

    import matplotlib.pyplot as plt
    reconstruction_error = np.mean(np.power(trainList - x_test_pred, 2), axis=(1, 2))
    # 分别收集蓝色和橙色点的索引
    blue_indices = [i for i, target in enumerate(targets) if target != 1]
    orange_indices = [i for i, target in enumerate(targets) if target == 1]

    # 先绘制蓝色点
    plt.scatter([blue_indices[i] for i in range(len(blue_indices))], 
                [reconstruction_error[i] for i in blue_indices], label = '0')

    for i in orange_indices:
        reconstruction_error[i] += 0.05

    # 再绘制橙色点，确保橙色在蓝色上层
    plt.scatter([orange_indices[i] for i in range(len(orange_indices))], 
                [reconstruction_error[i] for i in orange_indices], color='orange', label = '1')

    plt.xlabel('Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error by Target')
    plt.legend()
    plt.savefig('MSE of target.png')
    plt.show()
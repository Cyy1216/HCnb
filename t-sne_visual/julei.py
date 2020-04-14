#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz

true_img_features_path = 'file/img_fake_inception_v1.npy'
fake_img_features_path = 'file/img_true_inception_v1.npy'
file_path = '../DM-GAN/code/file_6000.npz'

way = 'kmeans' # kmeans or fzcmeans
class_num = 50
 
def cluster(img, cluster_num, way):
    if way == 'kmeans':
        return KMeans(n_clusters=cluster_num).fit(img)
    if way == 'fzcmeans':
        return fuzz.cmeans(img.T,c=cluster_num,m=2,error=0.0000005, maxiter=3000)[0]

def get_pred(img, model, way):
    if way == 'kmeans':
        return model.predict(img)
    if way == 'fzcmeans':
        u = fuzz.cluster.cmeans_predict(img.T, model, 2, error=0.0000005, maxiter=3000)[0]
        return np.argmax(u, axis=0)

def find_class_id(i, label_name, label_cluster, true_label_name):
    num = np.bincount(label_cluster[i])
    c_num = np.argmax(num)
    while c_num in label_name:
        max_num_a = np.max(np.bincount(label_cluster[label_name.index(c_num)]))
        max_num_b = np.max(num)
        if max_num_a >= max_num_b:
            num[c_num] = 0
            c_num = np.argmax(num)
            if np.sum(num) == 0:
                for j in true_label_name:
                    if j not in label_name:
                        label_name[i] = j
                        break
                break
        else:
            t_index = label_name.index(c_num)
            label_name[t_index] = -1
            label_name[i] = c_num
            label_name = find_class_id(t_index, label_name, label_cluster, true_label_name)
            break
    if c_num not in label_name: label_name[i]=c_num
    return label_name

def get_cluster_name(pred, label):
    true_label_name = np.unique(label)
    label_cluster =  {}  #统计每个簇包含的类别

    for i in range(len(pred)):
        if pred[i] not in label_cluster:
            label_cluster[pred[i]] = [label[i]]
        else:
            label_cluster[pred[i]].append(label[i])

    label_name = []
    for i in range(len(label_cluster)):
        label_name.append(-1)
        label_name = find_class_id(i, label_name, label_cluster, true_label_name)
    return label_name

def main():
    fake_img = np.load(true_img_features_path)
    true_img = np.load(fake_img_features_path)
    file_img = np.load(file_path)
    label = file_img['class_id']
    
    #生成按真实标签划分颜色的图片
    plt.scatter(true_img[:, 0], true_img[:, 1], c=label[:len(true_img)], s=3)
    plt.savefig('by_class.png')
    plt.close()

    #聚类
    model = cluster(true_img, class_num, way)
    pred_true = get_pred(true_img, model, way)
    #为每个簇分配标签
    label_name = get_cluster_name(pred_true, label)
    #获取生成图片的预测
    pred_false = get_pred(fake_img, model, way)

    color = []
    for i in range(len(pred_false)):
        if label_name[pred_false[i]] != label[i]:
            color.append('#FF0000')
        else:
            color.append('#0000FF')

    plt.scatter(true_img[:, 0], true_img[:, 1], c='#000000', s=3)
    plt.scatter(fake_img[:, 0], fake_img[:, 1], c=color, s=3)
    plt.savefig('result.png')


if __name__ == "__main__":
    main()


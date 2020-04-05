from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 
import numpy as np

import os
import sys
import scipy.misc
sys.path.append('inception')
sys.path.append('/home/huxinjian/workspace/inception_cub')
from extract_img import extract_img, preprocess


path = "../DM-GAN/code/file_6000.npz"
PCA_dim = 100
TSNE_dim = 2
T_perplexity = 120

def load_data(fullpath):
    print(fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    images.append(img)
        if len(images)>500:
            break
    print('images', len(images), images[0].shape)
    return images

def load_img(fake_img, true_img):
    fake_images = []
    true_images = []
    for path in fake_img:
        path = '../DM-GAN/' + path[path.find('model'):]
        img = scipy.misc.imread(path)
        fake_images.append(img)
    print(len(fake_images))
    for path in true_img:
        path = '../DM-GAN/' + path[path.find('model'):]
        img = scipy.misc.imread(path)
        true_images.append(img)
    print(len(true_images))
    return [fake_images, true_images] 

def main():
    pca_fake=PCA(n_components=PCA_dim)
    pca_true=PCA(n_components=PCA_dim)

    img_path  = np.load(path)
    class_ids, fake_path, true_path = img_path['class_id'], img_path['fake_path'], img_path['true_path']

    images = load_img(fake_path, true_path)

    true_img = []
    fake_img = []

    i=0
    while True:
        if len(images[0]) - i < 1000:
            img = images[0][i:]
        else:
            img = images[0][i:i+1000]
        i += 1000
        img = extract_img(img)
        true_img.append(img)
        if i >= len(images[0]):break
    
    i = 0
    while True:
        if len(images[1]) - i < 1000:
            img = images[1][i:]
        else:
            img = images[1][i:i+1000]
        i += 1000
        img = extract_img(img)
        fake_img.append(img)
        if i>= len(images[1]):break
    
    true_img = np.vstack(true_img)
    fake_img = np.vstack(fake_img)
    print(true_img.shape, fake_img.shape)

    img_pca = pca_fake.fit_transform(true_img)
    img_X = TSNE(n_components=TSNE_dim, perplexity=TSNE_dim).fit_transform(img_pca)
 
    img_pca = pca_true.fit_transform(fake_img)
    img_Y = TSNE(n_components=TSNE_dim, perplexity=TSNE_dim).fit_transform(img_pca)

    np.save('img_fake.npy', img_X)
    np.save('img_true.npy', img_Y)

    plt.scatter(img_X[:,0],img_X[:,1],s=8,color=(0.,0.5,0.))
    plt.scatter(img_Y[:,0],img_Y[:,1],s=8,color=(0.8,0,0.))
    plt.show()

if __name__ == '__main__':
    main()

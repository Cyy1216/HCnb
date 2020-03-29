from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 

import os
import sys
import scipy.misc
sys.path.append('inception_model')
from extract_img import extract_img


img_path = "E:\\dataset\\bird\\CUB_200_2011\\images"

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
        if len(images)>5000:
            break
    print('images', len(images), images[0].shape)
    return images

def main():
    pca=PCA(n_components=100)
    images = load_data(img_path)
    img_pre = extract_img(images)
    img_pca = pca.fit_transform(img_pre)
    img_X = TSNE(n_components=2).fit_transform(img_pca)
    # print(img_X.shape)
    ax = plt.gca()
    ax.scatter(img_X[:,0],img_X[:,1])
    plt.savefig('./test.jpg')

if __name__ == '__main__':
    main()
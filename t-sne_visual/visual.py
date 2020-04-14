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
from extract_model import extract_image, INIT_model

path = "../DM-GAN/code/file_3000_text.npz"
extract_img_way = 'CNN' #'inception_v1'
PCA_dim = 50
TSNE_dim = 2
T_perplexity = 60

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

def extract(images, way):
    all_img_features = []
    i=0
    while i < len(images):
        img = images[i:i+1000]
        i += 1000
        if way == 'inception_v3':
            img = extract_img(img)
        else:
            img = extract_image(img,way)

        all_img_features.append(img)
        if i >= len(images):break
    return np.vstack(all_img_features)

def main():
    pca_fake=PCA(n_components=PCA_dim)
    pca_true=PCA(n_components=PCA_dim)
    pca_text=PCA(n_components=50)

    img_path  = np.load(path, allow_pickle=True)
    class_ids, fake_path, true_path, text_features = img_path['class_id'], img_path['fake_path'],\
                                     img_path['true_path'], img_path['text_feature']
    text_features = np.vstack(text_features)

    images = load_img(fake_path, true_path)

    fake_img = extract(images[0], extract_img_way)
    true_img = extract(images[1], extract_img_way)
    # print(true_img.shape, fake_img.shape)

    img_pca = pca_true.fit_transform(true_img)
    img_X = TSNE(n_components=TSNE_dim, perplexity=T_perplexity).fit_transform(img_pca)
 
    img_pca = pca_fake.fit_transform(fake_img)
    img_Y = TSNE(n_components=TSNE_dim, perplexity=T_perplexity).fit_transform(img_pca)

    img_pca = pca_text.fit_transform(text_features)
    img_Z = TSNE(n_components=TSNE_dim, perplexity=T_perplexity).fit_transform(img_pca)

    np.save('file/img_true_{}.npy'.format(extract_img_way), img_X)
    np.save('file/img_fake_{}.npy'.format(extract_img_way), img_Y)
    np.save('file/text_{}.npy'.format(extract_img_way), img_Z)

    plt.scatter(img_X[:,0],img_X[:,1],s=8,color=(0.8,0.,0.))
    plt.scatter(img_Y[:,0],img_Y[:,1],s=8,color=(0.,0.5,0.))
    plt.scatter(img_Z[:,0],img_Z[:,1],s=8,color=(0.,0,0.2))
    plt.show()
    
if __name__ == '__main__':
    main()

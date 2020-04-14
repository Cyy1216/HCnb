import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import softmax 
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../DM-GAN/code/')

model_dir = {'inception_v1':'file/cub_inception_80.pkl','CNN':'../DM-GAN/DAMSMencoders/bird/image_encoder200.pth'} #
batch_size = 4

def INIT_model(way):
    if way == 'inception_v1':
        net = GoogleNet()
        state_dict = \
            torch.load(model_dir[way])
        net.load_state_dict(state_dict)
    if way == 'CNN':
        net = CNN_ENCODER()
        state_dict = \
            torch.load(model_dir[way], map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict)
    return net.eval().cuda()

def extract_image(img, way):
    net = INIT_model(way)
    img = np.array(img).transpose(0,3,1,2)
    print(img.shape)
    image_features = []
    i = 0
    while i < len(img):
        img_f = net(torch.FloatTensor(img[i:i+batch_size]).cuda())
        i += batch_size
        image_features.append(img_f.cpu().detach().numpy())
    return np.vstack(image_features)


class GoogleNet(nn.Module):
    def __init__(self,num_classes=200):
        super(GoogleNet,self).__init__()
        net = models.googlenet(pretrained=True)
        net.fc = nn.Sequential()
        self.feature = net
        self.fc = nn.Linear(1024, num_classes)

    def forward(self,x):
        return self.feature(x)

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

class CNN_ENCODER(nn.Module):
    def __init__(self):
        super(CNN_ENCODER, self).__init__()
        self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        return x
"""
add attention in feature extract model, just like binary vqa task. no scaler attribute
"""

from scipy import io
import os
from os.path import join
import numpy as np
from numpy import genfromtxt
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter


def build_data():
    data_name = 'CUB'

    mat_root = "/HDD-1_data/arlen/dataset/xlsa17/data"
    image_root = "../../dataset/CUB/CUB_200_2011/"
    image_embedding = "res101"
    class_embedding = "att_splits"

    # res101.mat
    mat_content = io.loadmat(mat_root + "/" + data_name + "/" + image_embedding + ".mat")
    img_files = mat_content['image_files'].squeeze()
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        if data_name == 'CUB':
            img_path = join(image_root, '/'.join(img_path.split('/')[7:]))
        new_img_files.append(img_path)

    image_files = np.array(new_img_files)
    image_label = mat_content['labels'].astype(int).squeeze() - 1
    feature = mat_content['features'].T

    scaler = preprocessing.MinMaxScaler()

    # att_splits.mat
    mat_content = io.loadmat(mat_root + "/" + data_name + "/" + class_embedding + ".mat")
    attribute = mat_content["att"].T

    test_seen_loc = mat_content['test_seen_loc'].squeeze() - 1
    test_unseen_loc = mat_content['test_unseen_loc'].squeeze() - 1
    trainval_loc = mat_content['trainval_loc'].squeeze() - 1

    _train_feature = scaler.fit_transform(feature[trainval_loc])
    _test_seen_feature = scaler.transform(feature[test_seen_loc])
    _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

    train_img = image_files[trainval_loc]
    train_feature = torch.from_numpy(_train_feature).float()
    mx = train_feature.max()
    train_feature.mul_(1/mx)
    train_label = image_label[trainval_loc].astype(int)
    train_att = torch.from_numpy(attribute[train_label]).float()

    test_image_unseen = image_files[test_unseen_loc]
    test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
    test_unseen_feature.mul_(1/mx)
    test_label_unseen = image_label[test_unseen_loc].astype(int)
    test_unseen_att = torch.from_numpy(attribute[test_label_unseen]).float()

    test_image_seen = image_files[test_seen_loc]
    test_seen_feature = torch.from_numpy(_test_seen_feature).float()
    test_seen_feature.mul_(1/mx)
    test_label_seen = image_label[test_seen_loc].astype(int)
    test_seen_att = torch.from_numpy(attribute[test_label_seen]).float()

    res = {
        'all_attribute': attribute,
        'train_image': train_img,
        'train_label': train_label,
        'train_attribute': train_att,
        'train_feature': train_feature,
        'test_unseen_image': test_image_unseen,
        'test_unseen_label': test_label_unseen,
        'test_unseen_attribute': test_unseen_att,
        'test_unseen_feature': test_unseen_feature,
        'test_seen_image': test_image_seen,
        'test_seen_label': test_label_seen,
        'test_seen_attribute': test_seen_att,
        'test_seen_feature': test_seen_feature
    }

    return res


class Data(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        data = build_data()
        self.image_path = data[mode + '_image']
        self.image_label = data[mode + '_label']
        self.image_attribute = data[mode + '_attribute']
        self.feature = data[mode + '_feature']

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.image_label[index]
        attribute = self.image_attribute[index]
        feature = self.feature[index]

        return image, label, attribute, feature

    def __len__(self):
        return len(self.image_label)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        dist = (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        dist[dist == 0.] = 1.
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor


class Res101(nn.Module):
    def __init__(self):
        super(Res101, self).__init__()

        res101 = models.resnet101(pretrained=True)
        res101.layer3[0] = nn.Sequential(*list(res101.layer3[0].children())[:-1])
        res101.layer4[0] = nn.Sequential(*list(res101.layer4[0].children())[:-1])
        modules = list(res101.children())[:-2]

        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        feature_map = self.resnet(x).squeeze()
        B, C, W, H = feature_map.shape
        feature_map = feature_map.view(B, C, W*H)

        return feature_map


class classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, nclass)

        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        out = self.logic(x)
        return out


def cal_accuracy(net_cls, net_F, net_linear, data_loader, device):
    scores = []
    labels = []
    cpu = torch.device('cpu')

    # load attribute map
    data_set_path = '../../dataset/'
    glove_path = data_set_path + "glove_embedding.csv"
    glove_map = genfromtxt(glove_path, delimiter=',', skip_header=0)

    for iteration, (img, label, attribute, feature) in enumerate(data_loader):
        image = img.to(device)

        feature_map = net_F(image)  # (B, 2048, 64)
        attribute_map = torch.from_numpy(glove_map).float().to(device)  # (312, 300)
        attribute_map = net_linear(attribute_map)  # (312, 64)

        key = feature_map
        query = attribute_map

        attention_map = torch.einsum('we,bem->bwm', query, key.permute(0, 2, 1))  # (B, 312, 2048)
        B, N1, N2 = attention_map.shape
        feature_attribute = F.max_pool1d(attention_map, 2048)
        feature_attribute = feature_attribute.view(B, N1)

        score = net_cls(feature_attribute)
        scores.append(score)
        labels.append(label)

    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    _, pred = scores.max(dim=1)
    pred = pred.view(-1).to(cpu)
    outpred = np.array(pred, dtype='int')

    labels = labels.numpy()
    unique_labels = np.unique(labels)

    acc = 0
    for i in unique_labels:
        idx = np.nonzero(labels == i)[0]
        acc += accuracy_score(labels[idx], outpred[idx])
    acc = acc / unique_labels.shape[0]

    return acc


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 500
    feature_shape = 312
    print(feature_shape)

    train_dir = 'tb_main_ver10_0611_2'
    train_writer = SummaryWriter(log_dir=train_dir)

    # load attribute map
    data_set_path = '../../dataset/'

    glove_path = data_set_path + "glove_embedding.csv"
    fasttext_path = data_set_path + "fasttext_embedding.csv"

    glove_map = genfromtxt(glove_path, delimiter=',', skip_header=0)
    fasttext_map = genfromtxt(fasttext_path, delimiter=',', skip_header=0)

    tfs = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_ds = Data(transforms_=tfs, mode='train')
    seen_ds = Data(transforms_=tfs, mode='test_seen')

    train_loader = DataLoader(
        train_ds,
        batch_size=96,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    seen_loader = DataLoader(
        seen_ds,
        batch_size=96,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )

    net_linear = nn.Linear(300, 64).to(device)

    net_cls = classifier(feature_shape, 200).to(device)
    net_F = Res101().to(device)

    opt = optim.Adam(itertools.chain(net_linear.parameters(), net_F.parameters(), net_cls.parameters()), lr=1e-3)

    loss_cls = nn.NLLLoss()

    for epoch in range(epochs):
        tqmd_iter = tqdm(list(enumerate(train_loader)))
        for iter, (image, label, attribute, feature) in tqmd_iter:
            # real_feature = feature.to(device)
            attribute = attribute.to(device)
            image = image.to(device)
            label = label.to(device)
            net_linear.zero_grad()
            net_F.zero_grad()
            net_cls.zero_grad()

            feature_map = net_F(image)  # (B, 2048, 64)
            attribute_map = torch.from_numpy(glove_map).float().to(device)  # (312, 300)
            attribute_map = net_linear(attribute_map)  # (312, 64)

            key = feature_map
            query = attribute_map

            attention_map = torch.einsum('we,bem->bwm', query, key.permute(0, 2, 1))  # (B, 312, 2048)
            B, N1, N2 = attention_map.shape
            feature_attribute = F.max_pool1d(attention_map, 2048)
            feature_attribute = feature_attribute.view(B, N1)

            pred_out = net_cls(feature_attribute)
            loss_classifier = loss_cls(pred_out, label)
            tqmd_iter.set_description(f'{loss_classifier.item():.4}')
            loss_classifier.backward()
            opt.step()

        print("epoch: %d/%d, cls loss: %.4f" % (epoch, epochs, loss_classifier.item()), end=" ")
        with torch.no_grad():
            train_acc = cal_accuracy(net_cls, net_F, net_linear, train_loader, device)
            val_acc = cal_accuracy(net_cls, net_F, net_linear, seen_loader, device)
        print("train acc: %.4f, val acc: %.4f" % (train_acc, val_acc))
        train_writer.add_scalar("CLS loss", loss_classifier.item(), epoch)
        train_writer.add_scalar("train acc", train_acc, epoch)
        train_writer.add_scalar("val acc", val_acc, epoch)


if __name__ == "__main__":
    train()

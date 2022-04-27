from tkinter import Variable
from scipy import io
from os.path import join
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from numpy import genfromtxt

import torch
import torch.nn as nn
import torchvision.models as models

import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


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

    # att_splits.mat
    mat_content = io.loadmat(mat_root + "/" + data_name + "/" + class_embedding + ".mat")
    attribute = mat_content["att"].T

    test_seen_loc = mat_content['test_seen_loc'].squeeze() - 1
    test_unseen_loc = mat_content['test_unseen_loc'].squeeze() - 1
    trainvalloc = mat_content["trainval_loc"].squeeze() - 1

    train_img = image_files[trainvalloc]
    train_label = image_label[trainvalloc].astype(int)
    train_att = attribute[train_label]

    test_image_unseen = image_files[test_unseen_loc]
    test_label_unseen = image_label[test_unseen_loc]
    test_unseen_att = attribute[test_label_unseen]

    test_image_seen = image_files[test_seen_loc]
    test_label_seen = image_label[test_seen_loc]
    test_seen_att = attribute[test_label_seen]

    res = {
        'all_attribute': attribute,
        'train_image': train_img,
        'train_label': train_label,
        'train_attribute': torch.from_numpy(train_att).float(),
        'test_unseen_image': test_image_unseen,
        'test_unseen_label': test_label_unseen,
        'test_unseen_attribute': torch.from_numpy(test_unseen_att).float(),
        'test_seen_image': test_image_seen,
        'test_seen_label': test_label_seen,
        'test_seen_attribute': torch.from_numpy(test_seen_att).float()
    }

    return res


class Data(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        data = build_data()
        self.image_path = data[mode + '_image']
        self.image_label = data[mode + '_label']
        self.image_attribute = data[mode + '_attribute']

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.image_label[index]
        attribute = self.image_attribute[index]

        return image, label, attribute

    def __len__(self):
        return len(self.image_label)


class Resnet101(nn.Module):
    def __init__(self, finetune=False):
        super(Resnet101, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        modules = list(resnet101.children())[:-1]

        self.model = nn.Sequential(*modules)
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x).squeeze()


class Generator(nn.Module):
    def __init__(self, z_dim, attr_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # CUB 300+312 -> 4096
            nn.Linear(z_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim, attr_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # CUB 2048+312 -> 4096
            nn.Linear(x_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(classifier, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.logic(self.fc(x))

        return out


def cal_accuracy(feature_extract, classifier, data_loader, device, bias=None):
    scores = []
    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label, attribute) in enumerate(data_loader):
        img = img.to(device)
        # feature.shape == (B, 300)
        feature = feature_extract(img)
        score = classifier(feature)
        scores.append(score)
        labels.append(label)

    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    if bias is not None:
        scores = scores-bias

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


def generate_syn_feature(generator, classes, attribute, num, device):
    nclass = classes.shape[0]
    syn_feature = torch.FloatTensor(nclass*num, 2048).to(device)
    syn_label = torch.LongTensor(nclass*num).to(device)
    syn_att = torch.FloatTensor(num, 312).to(device)
    syn_noise = torch.FloatTensor(num, 300).to(device)

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]

        syn_att.copy_(torch.Tensor(np.broadcast_to(iclass_att, (num, iclass_att.shape[0]))))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        z_cat = torch.cat((syn_noisev, syn_attv), dim=1)
        fake = generator(z_cat)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    GAN_epochs = 20
    CLS_epochs = 10

    tfs = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_ds = Data(transforms_=tfs, mode='train')
    seen_ds = Data(transforms_=tfs, mode='test_seen')
    unseen_ds = Data(transforms_=tfs, mode='test_unseen')

    train_loader = DataLoader(
        train_ds,
        batch_size=100,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    seen_loader = DataLoader(
        seen_ds,
        batch_size=100,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    unseen_loader = DataLoader(
        unseen_ds,
        batch_size=100,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )

    net_feat = Resnet101().to(device)
    net_G = Generator(300, 312).to(device)
    net_D = Discriminator(2048, 312).to(device)
    # 2048: resnet feature; 200: cub data classes
    net_cls = classifier(2048, 200).to(device)

    opt_G = optim.Adam(net_G.parameters(), lr=1e-3)
    opt_D = optim.Adam(net_D.parameters(), lr=1e-3)
    opt_cls = optim.Adam(net_cls.parameters(), lr=1e-3)

    loss_adv = nn.BCELoss()
    loss_cls = nn.NLLLoss()

    #  train generator
    for epoch in range(GAN_epochs):
        for iter, (image, label, attribute) in enumerate(train_loader):
            image = image.to(device)
            feature = net_feat(image)
            # real_feature.shape == (B, 2048)
            real_feature = Variable(feature.float())
            # attribute.shape == (B, 312)
            attribute = Variable(attribute.float()).to(device)
            label = label.to(device)

            batch = image.shape[0]

            # Adversarial ground truths
            valid = Variable(torch.Tensor(batch, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.Tensor(batch, 1).fill_(0.0), requires_grad=False).to(device)

            # sample noise as generator input
            z_feature = Variable(torch.Tensor(np.random.normal(0, 1, (batch, 300)))).to(device)
            z_attribute = Variable(torch.Tensor(np.random.normal(0, 1, (batch, 312)))).to(device)
            # z_cat.shape == (B, 612)
            z_cat = torch.cat((z_feature, z_attribute), dim=1)

            # syn_feature.shape == (B, 2048)
            syn_feature = net_G(z_cat)

            # real_cat.shape == (B, 2360)
            real_cat = torch.cat((real_feature, attribute), dim=1)
            # syn_cat.shape == (B, 2360)
            syn_cat = torch.cat((syn_feature, attribute), dim=1)

            loss_G = loss_adv(net_D(syn_cat), valid)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            real_loss = loss_adv(net_D(real_cat), valid)
            fake_loss = loss_adv(net_D(syn_cat.detach()), fake)
            loss_D = (real_loss + fake_loss) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            print("Epoch: {}/{}, Batch: {}/{}, G loss: {:.4f}, D loss: {:.4f}".format(
                    epoch, GAN_epochs, iter, len(train_loader), loss_G.item(), loss_D.item()
                )
            )

    # train classifier
    res = build_data()
    unseenclasses = res['test_unseen_label']
    # unseenattribute.shape == (200, 312); 200: all classes, 312: all attrubutes
    unseenattribute = res['all_attribute']
    # syn_feature.shape == (148350, 2048)
    # syn_label.shape == (148350)
    syn_feature, syn_label = generate_syn_feature(net_G, unseenclasses, unseenattribute, 50, device)

    net_G.eval()
    for epoch in range(CLS_epochs):
        for iter, (image, label, attribute) in enumerate(train_loader):
            image = image.to(device)
            feature = net_feat(image)
            label = label.to(device)
            # train_X.shape == (148407, 2048)
            train_X = torch.cat((feature, syn_feature), 0)
            # train_Y.shape == (148407)
            train_Y = torch.cat((label, syn_label), 0)

            pred_out = net_cls(train_X)

            loss = loss_cls(pred_out, train_Y)

            opt_cls.zero_grad()
            loss.backward()
            opt_cls.step()

            print("Epoch: {}/{}, Batch: {}/{}, CLS loss: {:.4f}".format(
                    epoch, CLS_epochs, iter, len(train_loader), loss.item()
                )
            )

    with torch.no_grad():
        seen_acc = cal_accuracy(net_feat, net_cls, seen_loader, device)
        unseen_acc = cal_accuracy(net_feat, net_cls, unseen_loader, device)
    H = 2*seen_acc*unseen_acc / (seen_acc+unseen_acc)
    print("seen: {:.4f}, unseen: {:.4f}, H: {:.4f}".format(seen_acc, unseen_acc, H))


if __name__ == "__main__":
    train()

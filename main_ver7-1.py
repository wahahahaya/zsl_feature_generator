"""
let the feature extract model output different shape 2048 and 300
"""

from scipy import io
from os.path import join
import numpy as np
from numpy import genfromtxt, real
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
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
    feature = mat_content['features'].T

    # att_splits.mat
    mat_content = io.loadmat(mat_root + "/" + data_name + "/" + class_embedding + ".mat")
    attribute = mat_content["att"].T

    test_seen_loc = mat_content['test_seen_loc'].squeeze() - 1
    test_unseen_loc = mat_content['test_unseen_loc'].squeeze() - 1
    trainval_loc = mat_content['trainval_loc'].squeeze() - 1

    scaler = preprocessing.MinMaxScaler()

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
    def __init__(self, feature_shape):
        super(Res101, self).__init__()

        res101 = models.resnet101(pretrained=True)
        self.feature_shape = feature_shape
        if self.feature_shape == 2048:
            modules = list(res101.children())[:-1]
        elif self.feature_shape == 300:
            modules = list(res101.children())[:-2]

        self.resnet = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(2048, 300, 1, 1)
        self.scaler = PyTMinMaxScalerVectorized()

    def forward(self, x):
        if self.feature_shape == 2048:
            feature_vector = self.resnet(x).squeeze()
        elif self.feature_shape == 300:
            # feature_map.shape == (B, 2048, 8, 8)
            feature_map = self.resnet(x)
            # feature_map.shape == (B, 300, 8, 8)
            feature_map = self.conv(feature_map)
            # feature_vector.shape == (B, 300)
            feature_vector = self.avgpool(feature_map).squeeze()

        out = self.scaler(feature_vector)

        return out


class Encoder(nn.Module):
    def __init__(self, feature_shape):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(feature_shape+312, 1024)
        self.fc3 = nn.Linear(1024, 600)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(600, 300)
        self.linear_log_var = nn.Linear(600, 300)
        self.apply(weights_init)

    def forward(self, x, att):
        x = torch.cat((x, att), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Generator(nn.Module):
    def __init__(self, feature_shape):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(612, 1024)
        self.fc2 = nn.Linear(1024, feature_shape)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, z, att):
        z = torch.cat((z, att), dim=-1)
        x = self.lrelu(self.fc1(z))
        out = self.sigmoid(self.fc2(x))
        return out


class Discriminator(nn.Module):
    def __init__(self, feature_shape):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(feature_shape+312, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h


class classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, nclass)

        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        out = self.logic(x)
        return out


def cal_accuracy(classifier, feature_net, data_loader, device):
    scores = []
    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label, attribute, feature) in enumerate(data_loader):
        img = img.to(device)
        feature = feature_net(img)
        # feature.shape == (B, 300)
        # feature = feature.to(device)
        score = classifier(feature)
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


def generate_syn_feature(generator, classes, attribute, feature_shape, num, device):
    nclass = classes.shape[0]
    syn_feature = torch.FloatTensor(nclass*num, feature_shape).to(device)
    syn_label = torch.LongTensor(nclass*num).to(device)
    syn_att = torch.FloatTensor(num, 312).to(device)
    syn_att.requires_grad_(False)
    syn_noise = torch.FloatTensor(num, 300).to(device)
    syn_noise.requires_grad_(False)
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = torch.Tensor(attribute[iclass])
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            fake = generator(syn_noise, syn_att)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


def compute_gradient_penalty(D, real_data, fake_data, attribute, device, lambda1):
    alpha = torch.rand(real_data.size(0), 1).to(device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True).to(device)

    disc_interpolates = D(interpolates, Variable(attribute))
    ones = torch.ones(disc_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), reduction='none')
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 500
    feature_shape = 300
    print(feature_shape)
    # # load attribute map
    # data_set_path = '../../dataset/'

    # glove_path = data_set_path + "glove_embedding.csv"
    # fasttext_path = data_set_path + "fasttext_embedding.csv"

    # glove_map = genfromtxt(glove_path, delimiter=',', skip_header=0)
    # fasttext_map = genfromtxt(fasttext_path, delimiter=',', skip_header=0)

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
        batch_size=32,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    seen_loader = DataLoader(
        seen_ds,
        batch_size=64,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    unseen_loader = DataLoader(
        unseen_ds,
        batch_size=64,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )

    net_E = Encoder(feature_shape).to(device)
    net_G = Generator(feature_shape).to(device)
    net_D = Discriminator(feature_shape).to(device)
    net_cls = classifier(feature_shape, 200).to(device)
    net_F = Res101(feature_shape).to(device)

    opt_E = optim.Adam(net_E.parameters(), lr=1e-3)
    opt_G = optim.Adam(net_G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_D = optim.Adam(net_D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_cls = optim.Adam(net_cls.parameters(), lr=1e-3)
    # opt_F = optim.Adam(net_F.parameters(), lr=1e-3)

    loss_cls = nn.NLLLoss()

    best_gzsl_acc = 0
    best_train_acc = 0
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = (-1*one).to(device)
    lambda1 = 10
    net_F.eval()
    res = build_data()
    for epoch in range(epochs):
        train_feature = np.zeros((1, feature_shape))
        for iter, (image, label, attribute, feature) in enumerate(train_loader):
            # real_feature = feature.to(device)
            attribute = attribute.to(device)
            image = image.to(device)
            label = label.to(device)
            batch = feature.shape[0]

            # real_feature.shape == (B, 2048)
            real_feature = net_F(image)

            train_feature = np.vstack((train_feature, real_feature.cpu().detach().numpy()))

            for p in net_D.parameters():
                p.requires_grad = True

            gp_sum = 0
            for _ in range(5):
                net_D.zero_grad()
                # real_feature = net_F(image)
                input_resv = Variable(real_feature)
                input_attv = Variable(attribute)

                criticD_real = net_D(input_resv, input_attv)
                criticD_real = 10*criticD_real.mean()
                criticD_real.backward(mone)

                means, log_var = net_E(input_resv, input_attv)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn([batch, 300]).to(device)
                eps = Variable(eps)
                z = eps * std + means

                fake = net_G(z, input_attv)

                criticD_fake = net_D(fake.detach(), input_attv)
                criticD_fake = 10*criticD_fake.mean()
                criticD_fake.backward(one)

                # gradient penalty
                gradient_penalty = 10*compute_gradient_penalty(net_D, real_feature, fake.data, attribute, device, lambda1)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty  # add Y here and #add vae reconstruction loss
                opt_D.step()

            gp_sum /= (10*lambda1*5)
            if (gp_sum > 1.05).sum() > 0:
                lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                lambda1 /= 1.1

            # ############ Generator training ##############
            # Train Generator and Decoder
            for p in net_D.parameters():  # freeze discrimator
                p.requires_grad = False

            net_E.zero_grad()
            net_G.zero_grad()
            # net_F.zero_grad()
            # real_feature = net_F(image)
            input_resv = Variable(real_feature)
            input_attv = Variable(attribute)

            means, log_var = net_E(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch, 300]).to(device)
            eps = Variable(eps)
            z = eps * std + means

            fake = net_G(z, input_attv)

            vae_loss_seen = loss_fn(fake, input_resv, means, log_var)

            criticG_fake = net_D(fake, input_attv).mean()
            errG = vae_loss_seen - 10*criticG_fake
            errG.backward()

            G_cost = -criticG_fake
            opt_G.step()
            opt_E.step()
            # opt_F.step()
        print("Epoch: %d/%d, G loss: %.4f, D loss: %.4f, VEA loss: %.4f, Wasserstein_D: %.4f" % (
                    epoch, epochs, G_cost.item(), D_cost.item(), vae_loss_seen.item(), Wasserstein_D
            ), end=" "
        )
        net_G.eval()
        # train classifier
        unseenclasses = np.unique(res['test_unseen_label'])
        # unseenattribute.shape == (200, 312); 200: all classes, 312: all attrubutes
        all_attribute = res['all_attribute']
        # train_feature.shape == (7057, 2048)
        # train_label == (7057)
        train_feature = torch.from_numpy(train_feature[1:]).float().to(device)
        train_label = torch.from_numpy(res['train_label']).to(device)
        # syn_feature.shape == (5000, 2048)
        # syn_label.shape == (5000)
        syn_feature, syn_label = generate_syn_feature(net_G, unseenclasses, all_attribute, feature_shape, 100, device)
        train_X = torch.cat((train_feature, syn_feature), 0)
        train_Y = torch.cat((train_label, syn_label), 0)

        pred_out = net_cls(train_X)
        loss_classifier = loss_cls(pred_out, train_Y)
        net_cls.zero_grad()
        loss_classifier.backward()
        opt_cls.step()

        with torch.no_grad():
            train_acc = cal_accuracy(net_cls, net_F, train_loader, device)
            seen_acc = cal_accuracy(net_cls, net_F, seen_loader, device)
            unseen_acc = cal_accuracy(net_cls, net_F, unseen_loader, device)
        H = 2*seen_acc*unseen_acc / (seen_acc+unseen_acc)
        if best_gzsl_acc < H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = seen_acc, unseen_acc, H
        if best_train_acc < train_acc:
            best_train_acc = train_acc
        print("train: %.4f, seen: %.4f, unseen: %.4f, H: %.4f" % (train_acc, seen_acc, unseen_acc, H))
    print('the best GZSL seen accuracy is %.4f' % best_acc_seen)
    print('the best GZSL unseen accuracy is %.4f' % best_acc_unseen)
    print('the best GZSL H is %.4f' % best_gzsl_acc)
    print('the best train acc is %.4f' % best_train_acc)


if __name__ == "__main__":
    train()

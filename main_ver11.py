"""
traditional zsl training, by using the feature from the res101.mat, add the contrastive learning after generator
"""

from scipy import io
from os.path import join
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tqdm import tqdm
import random

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
    # mx = train_feature.max()
    # train_feature.mul_(1/mx)
    train_label = image_label[trainval_loc].astype(int)
    train_att = torch.from_numpy(attribute[train_label]).float()

    test_image_unseen = image_files[test_unseen_loc]
    test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
    # test_unseen_feature.mul_(1/mx)
    test_label_unseen = image_label[test_unseen_loc].astype(int)
    test_unseen_att = torch.from_numpy(attribute[test_label_unseen]).float()

    test_image_seen = image_files[test_seen_loc]
    test_seen_feature = torch.from_numpy(_test_seen_feature).float()
    # test_seen_feature.mul_(1/mx)
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


class Res101(nn.Module):
    def __init__(self):
        super(Res101, self).__init__()

        res101 = models.resnet101(pretrained=True)
        modules = list(res101.children())[:-1]

        self.resnet = nn.Sequential(*modules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out.shape == (B, 2048, W, H)
        out = self.resnet(x).squeeze()
        out = self.sigmoid(out)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(2360, 1024)
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
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(612, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, z, att):
        z = torch.cat((z, att), dim=-1)
        x = self.lrelu(self.fc1(z))
        out = self.sigmoid(self.fc2(x))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2048 + 312, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h


class projection(nn.Module):
    def __init__(self):
        super(projection, self).__init__()

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding = self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding, out_z


class classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, nclass)

        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        out = self.logic(x)
        return out


def cal_accuracy(classifier, net_proj, data_loader, device):
    scores = []
    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label, attribute, feature) in enumerate(data_loader):
        feature = feature.to(device)
        embed, _ = net_proj(feature)
        score = classifier(embed)
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


def generate_syn_feature(generator, classes, attribute, num, device):
    nclass = classes.shape[0]
    syn_feature = torch.FloatTensor(nclass*num, 2048).to(device)
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
    # recon_x == fake
    # x == real
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), reduction='none')
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = 1
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # (12057, 12057)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 500
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    tfs = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_ds = Data(transforms_=tfs, mode='train')
    seen_ds = Data(transforms_=tfs, mode='test_seen')
    unseen_ds = Data(transforms_=tfs, mode='test_unseen')

    train_loader = DataLoader(
        train_ds,
        batch_size=1024,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    seen_loader = DataLoader(
        seen_ds,
        batch_size=1024,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    unseen_loader = DataLoader(
        unseen_ds,
        batch_size=1024,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )

    net_E = Encoder().to(device)
    net_G = Generator().to(device)
    net_D = Discriminator().to(device)
    net_proj = projection().to(device)
    net_cls = classifier(2048, 200).to(device)

    opt_E = optim.Adam(net_E.parameters(), lr=1e-3)
    opt_G = optim.Adam(net_G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_D = optim.Adam(net_D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_P = optim.Adam(net_proj.parameters(), lr=1e-3)
    opt_cls = optim.Adam(net_cls.parameters(), lr=1e-3)

    loss_cls = nn.NLLLoss()
    loss_con = SupConLoss()

    best_gzsl_acc = 0
    best_train_acc = 0
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = (-1*one).to(device)
    lambda1 = 10
    for epoch in range(epochs):
        for iter, (image, label, attribute, feature) in tqdm(list(enumerate(train_loader))):
            real_feature = feature.to(device)
            attribute = attribute.to(device)
            label = label.to(device)
            batch = feature.shape[0]

            for p in net_D.parameters():
                p.requires_grad_(True)

            gp_sum = 0
            for _ in range(5):
                net_D.zero_grad()
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
                p.requires_grad_(False)

            net_E.zero_grad()
            net_G.zero_grad()

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

        print("Epoch: %d/%d, G loss: %.4f, D loss: %.4f, VEA loss: %.4f, Wasserstein_D: %.4f" % (
                    epoch, epochs, G_cost.item(), D_cost.item(), vae_loss_seen.item(), Wasserstein_D
            ), end=" "
        )

        net_G.eval()
        # train classifier
        res = build_data()
        unseenclasses = np.unique(res['test_unseen_label'])
        # unseenattribute.shape == (200, 312); 200: all classes, 312: all attrubutes
        all_attribute = res['all_attribute']
        # train_feature.shape == (7057, 2048)
        train_feature = res['train_feature'].to(device)
        train_label = torch.from_numpy(res['train_label']).to(device)
        # syn_feature.shape == (5000, 2048)
        # syn_label.shape == (5000)
        syn_feature, syn_label = generate_syn_feature(net_G, unseenclasses, all_attribute, 100, device)
        train_X = torch.cat((train_feature, syn_feature), 0)
        train_Y = torch.cat((train_label, syn_label), 0)

        for p in net_proj.parameters():
            p.requires_grad_(True)
        net_proj.zero_grad()
        # h.shape == (12057, 2048)
        # z.shape == (12057, 512)
        h, z = net_proj(train_X)
        sim_loss = loss_con(z, train_Y)
        sim_loss.backward()
        opt_P.step()

        for p in net_proj.parameters():  # freeze projection
            p.requires_grad_(False)

        net_cls.zero_grad()
        h, z = net_proj(train_X)
        pred_out = net_cls(h)
        loss_classifier = loss_cls(pred_out, train_Y)
        loss_classifier.backward()
        opt_cls.step()

        print("SimCLR loss: %.4f, CLS loss: %.4f" % (
                    sim_loss.item(), loss_classifier.item()
            ), end=" "
        )

        with torch.no_grad():
            train_acc = cal_accuracy(net_cls, net_proj, train_loader, device)
            seen_acc = cal_accuracy(net_cls, net_proj, seen_loader, device)
            unseen_acc = cal_accuracy(net_cls, net_proj, unseen_loader, device)
        H = 2*seen_acc*unseen_acc / (seen_acc + unseen_acc)
        if best_gzsl_acc < H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = seen_acc, unseen_acc, H
        if best_train_acc < train_acc:
            best_train_acc = train_acc
        print("train: %.4f, seen: %.4f, unseen: %.4f, H: %.4f" % (train_acc, seen_acc, unseen_acc, H))
    print('the best GZSL seen accuracy is %.4f' % best_acc_seen)
    print('the best GZSL unseen accuracy is %.4f' % best_acc_unseen)
    print('the best GZSL H is %.4f' % best_gzsl_acc)
    print('the best train acc is %.4f' % best_train_acc)
    print("Random Seed: ", manualSeed)


if __name__ == "__main__":
    train()

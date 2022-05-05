from scipy import io
from os.path import join
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

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
    trainvalloc = mat_content["trainval_loc"].squeeze() - 1

    train_img = image_files[trainvalloc]
    train_label = image_label[trainvalloc].astype(int)
    train_att = attribute[train_label]
    train_feature = feature[trainvalloc]

    test_image_unseen = image_files[test_unseen_loc]
    test_label_unseen = image_label[test_unseen_loc]
    test_unseen_att = attribute[test_label_unseen]
    test_unseen_feature = feature[test_unseen_loc]

    test_image_seen = image_files[test_seen_loc]
    test_label_seen = image_label[test_seen_loc]
    test_seen_att = attribute[test_label_seen]
    test_seen_feature = feature[test_seen_loc]

    res = {
        'all_attribute': attribute,
        'train_image': train_img,
        'train_label': train_label,
        'train_attribute': torch.from_numpy(train_att).float(),
        'train_feature': torch.from_numpy(train_feature).float(),
        'test_unseen_image': test_image_unseen,
        'test_unseen_label': test_label_unseen,
        'test_unseen_attribute': torch.from_numpy(test_unseen_att).float(),
        'test_unseen_feature': torch.from_numpy(test_unseen_feature).float(),
        'test_seen_image': test_image_seen,
        'test_seen_label': test_label_seen,
        'test_seen_attribute': torch.from_numpy(test_seen_att).float(),
        'test_seen_feature': torch.from_numpy(test_seen_feature).float()
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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(612, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, z, att):
        z = torch.cat((z, att), dim=1)
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


class classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(classifier, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.logic(self.fc(x))
        return out


def cal_accuracy(classifier, data_loader, device):
    scores = []
    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label, attribute, feature) in enumerate(data_loader):
        img = img.to(device)
        # feature.shape == (B, 300)
        feature = feature.to(device)
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


def compute_gradient_penalty(D, real_data, fake_data, attribute, device):
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 500

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

    net_G = Generator().to(device)
    net_D = Discriminator().to(device)
    # 2048: resnet feature; 200: cub data classes
    net_cls = classifier(2048, 200).to(device)

    opt_G = optim.Adam(net_G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_D = optim.Adam(net_D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_cls = optim.Adam(net_cls.parameters(), lr=1e-3)

    loss_cls = nn.NLLLoss()

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    best_gzsl_acc = 0
    for epoch in range(epochs):
        for iter, (image, label, attribute, feature) in enumerate(train_loader):
            real_feature = feature.to(device)
            attribute = attribute.to(device)
            label = label.to(device)

            batch = feature.shape[0]

            for _ in range(5):
                net_D.zero_grad()

                z_feature = torch.Tensor(np.random.normal(0, 1, (batch, 300))).to(device)
                z_feature.requires_grad_(False)

                syn_feature = net_G(z_feature, attribute)

                L_D_fake = 10*net_D(syn_feature, attribute).mean()
                L_D_fake.backward(one)

                L_D_real = 10*net_D(real_feature, attribute).mean()
                L_D_real.backward(mone)

                gp = 10*compute_gradient_penalty(net_D, real_feature, syn_feature, attribute, device)
                gp.backward()

                L_D = L_D_fake - L_D_real + gp
                opt_D.step()

            net_G.zero_grad()
            z_feature = torch.Tensor(np.random.normal(0, 1, (batch, 300))).to(device)
            z_feature.requires_grad_(False)

            syn_feature = net_G(z_feature, attribute)

            L_G_fake = (net_D(syn_feature, attribute)).mean()
            L_G = -10*L_G_fake

            L_G.backward()
            opt_G.step()

        print("Epoch: %d/%d, G loss: %.4f, D loss: %.4f" % (
                    epoch, epochs, L_G.item(), L_D.item()
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
        pred_out = net_cls(train_X)
        loss_classifier = loss_cls(pred_out, train_Y)
        opt_cls.zero_grad()
        loss_classifier.backward()
        opt_cls.step()

        with torch.no_grad():
            seen_acc = cal_accuracy(net_cls, seen_loader, device)
            unseen_acc = cal_accuracy(net_cls, unseen_loader, device)
        H = 2*seen_acc*unseen_acc / (seen_acc+unseen_acc)
        if best_gzsl_acc < H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = seen_acc, unseen_acc, H
        print("seen: %.4f, unseen: %.4f, H: %.4f" % (seen_acc, unseen_acc, H))
    print('the best GZSL seen accuracy is %.4f' % best_acc_seen)
    print('the best GZSL unseen accuracy is %.4f' % best_acc_unseen)
    print('the best GZSL H is %.4f' % best_gzsl_acc)


if __name__ == "__main__":
    train()

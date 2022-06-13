"""
base on clip model
"""


from scipy import io
from os.path import join
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tqdm import tqdm
import clip

import torch
import torch.nn as nn

import torch.optim as optim


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


class classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, nclass)

        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        out = self.logic(x)
        return out


def cal_accuracy(net_cls, model, data_loader, device):
    scores = []
    labels = []
    cpu = torch.device('cpu')

    # load attribute map
    att = []
    with open('/HDD-1_data/arlen/dataset/CUB/CUB_200_2011/attributes.txt') as f:
        for line in f:
            att.append(line.split(' ')[1].strip())

    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in att])

    for iteration, (img, label, attribute, feature) in enumerate(data_loader):
        image = img.to(device)

        text = text.to(device)
        net_cls.zero_grad()

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1).float()

        score = net_cls(similarity)
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

    # load attribute map
    att = []
    with open('/HDD-1_data/arlen/dataset/CUB/CUB_200_2011/attributes.txt') as f:
        for line in f:
            att.append(line.split(' ')[1].strip())

    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in att])

    tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_ds = Data(transforms_=tfs, mode='train')
    seen_ds = Data(transforms_=tfs, mode='test_seen')

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )
    seen_loader = DataLoader(
        seen_ds,
        batch_size=32,
        num_workers=23,
        shuffle=False,
        pin_memory=True
    )

    net_cls = classifier(312, 200).to(device)
    model, preprocess = clip.load('ViT-B/32', device)

    opt = optim.Adam(net_cls.parameters(), lr=1e-3)

    loss_cls = nn.NLLLoss()

    for epoch in range(epochs):
        tqmd_iter = tqdm(list(enumerate(train_loader)))
        for iter, (image, label, attribute, feature) in tqmd_iter:
            image = image.to(device)
            label = label.to(device)

            text = text.to(device)

            net_cls.zero_grad()

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).softmax(dim=-1).float()

            pred_out = net_cls(similarity)
            loss_classifier = loss_cls(pred_out, label)
            tqmd_iter.set_description(f'{loss_classifier.item():.4}')
            loss_classifier.backward()
            opt.step()

        print("epoch: %d/%d, cls loss: %.4f" % (epoch, epochs, loss_classifier.item()), end=" ")
        with torch.no_grad():
            train_acc = cal_accuracy(net_cls, model, train_loader, device)
            val_acc = cal_accuracy(net_cls, model, seen_loader, device)
        print("train acc: %.4f, val acc: %.4f" % (train_acc, val_acc))


if __name__ == "__main__":
    train()

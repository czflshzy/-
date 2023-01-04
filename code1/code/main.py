import argparse
import os
import time

import pandas as pd
import torch
from torch import nn, optim

from ResNet import resnet50, resnet34
from dataset import BitmojiDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args, train_loader, test_loader):
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        index = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = cost(outputs, labels)

            if index % 10 == 0:
                print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += 1

        if epoch % 1 == 0:
            end = time.time()
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss.item(), (end - start) * 2))

            model.eval()

            correct_prediction = 0.
            total = 0
            for images, labels in test_loader:
                # to GPU
                images = images.to(device)
                labels = labels.to(device)
                # print prediction
                outputs = model(images)
                # equal prediction and acc

                _, predicted = torch.max(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            print("Acc: %.4f" % (correct_prediction / total))

        # Save the model checkpoint
        torch.save(model, os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch)))
    print("Model save to %s." % (os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch))))


def test(args, test_loader):
    model_path = os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, args.model_test_epoch))
    model_test = torch.load(model_path)
    model_test.eval()
    result_list = []
    col = ["image_id", "is_male"]
    for img_path, images in test_loader:
        # to GPU
        images = images.to(device)
        # print prediction
        outputs = model_test(images)
        _, predicted = torch.max(outputs.data, 1)
        img_name = img_path[0].split("/")[-1]
        if predicted == 0:
            predicted = -1
        else:
            predicted = 1
        result_list.append([img_name, predicted])

    test_df = pd.DataFrame(columns=col, data=result_list)
    test_df.to_csv('./test.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    Train = False
    parser = argparse.ArgumentParser(description='training hyper-parameter')
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='bitmoji', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    parser.add_argument("--train_rate", default=0.7, type=float)
    parser.add_argument("--model_data_num", default=3000, type=int)
    parser.add_argument("--model_test_num", default=1084, type=int)
    parser.add_argument("--model_test_epoch", default=25, type=int)

    args = parser.parse_args()

    if Train:

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        total_idx = [i for i in range(0, args.model_data_num)]
        train_idx = sample(total_idx, int(args.train_rate * args.model_data_num))
        valid_idx = list(set(total_idx) - set(train_idx))

        mojiDataset_train = BitmojiDataset("./data/BitmojiDataset/trainimages/", "./data/train.csv", train_idx,
                                           train=True)
        mojiDataLoader_train = torch.utils.data.DataLoader(dataset=mojiDataset_train,
                                                           batch_size=args.batch_size,
                                                           shuffle=True)

        mojiDataset_valid = BitmojiDataset("./data/BitmojiDataset/trainimages/", "./data/train.csv", valid_idx,
                                           valid=True)
        mojiDataLoader_valid = torch.utils.data.DataLoader(dataset=mojiDataset_valid,
                                                           batch_size=args.batch_size,
                                                           shuffle=True)

        print("Train numbers:{:d}".format(len(mojiDataset_train)))
        print("Valid numbers:{:d}".format(len(mojiDataset_valid)))
        print(f"Train device:{device}")
        model = resnet34(num_classes=args.num_class)
        # model = resnet50(num_classes=args.num_class)
        model = model.to(device)
        cost = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

        train(args, mojiDataLoader_train, mojiDataLoader_valid)
    else:
        test_idx = [i for i in range(args.model_data_num, args.model_data_num + args.model_test_num)]
        mojiDataset_test = BitmojiDataset("./data/BitmojiDataset/testimages/", None, test_idx,
                                          test=True)
        mojiDataLoader_train = torch.utils.data.DataLoader(dataset=mojiDataset_test,
                                                           batch_size=1,
                                                           shuffle=False)
        test(args, mojiDataLoader_train)

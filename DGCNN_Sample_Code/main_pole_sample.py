from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from our_pole_sample import KITTIPoleSample

import pickle
import time as time
from tqdm import tqdm

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    # train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
    #                          batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    data_path = '/mmdetetcion3d/data/kitti/training/velodyne'
    pcd_file = '*.bin'
    lables_file = '*.txt'
    calib_file = '*.txt'
    idx_file_train = 'train.txt'
    idx_file_val = 'val.txt'
    nodes = 42

    dataset_train = KITTIPoleSample(data_path=data_path, nodes=nodes, pcd_file=pcd_file, lables_file=lables_file, calib_file=calib_file, idx_file=idx_file_train)
    dataset_val = KITTIPoleSample(data_path=data_path, nodes=nodes, pcd_file=pcd_file, lables_file=lables_file, calib_file=calib_file, idx_file=idx_file_val)
    # dataset_val = KITTIPoleSample(data_path=data_path, nodes=nodes, pcd_file=pcd_file, lables_file=lables_file, calib_file=calib_file)
    print("Training set %d" % len(dataset_train))
    print("Validation set %d" % len(dataset_val))
    train_size = int(len(dataset_train))
    val_size = int(len(dataset_val))
    # val_size = int(len(dataset_val))
    # train_dataset = torch.utils.data.random_split(dataset_train, [train_size])
    # val_dataset = torch.utils.data.random_split(dataset_val, [val_size])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, drop_last=True, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, drop_last=True, shuffle=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).cuda(0)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    #model = torch.load('./our_checkpoints/dgcnn_seg/pole_car_pw_21_label_1/pole_car_epoch_1.pth')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-4)
    
    # criterion = cal_loss
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4]).to(device))
    pos_weight = torch.tensor([5]).cuda(0)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    store = False
    best_test_acc = 0
    best_avg_per_class_acc = 0

    for epoch in range(args.epochs):
        big_sample_num = 0
        if epoch == args.epochs - 1:
            store = True

        
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        i = 0
        loss_sum = 0
        step_res = train_size
        time_sum = 0
        for data, label in train_loader:
            since = time.time()
            i += 1 
            data, label = data.to(device).squeeze(0), label.to(device).squeeze(0)
            #print("data", data.shape)
            #data_org = data.to('cpu')
            #label_org = label.to('cpu')
            # get front points
            data = data.permute(0, 2, 1).contiguous()
            batch_size = 2
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            #label_pred = logits.to('cpu')
            # ========================== Save point clouds =========================
            #store_path = './our_checkpoints/dgcnn_seg/front_train_vis'
            #pts_dict = {'data_org': data_org, 'label_org': label_org, 'label_pred': label_pred}
            #local_time = time.localtime()
            #file_name = '{}_{}_{}_{}_{}.npy'.format(local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)
            #os.system('mkdir -p {}'.format(store_path))
            #file_path = os.path.join(store_path, file_name)
            # # file_path = store_path
            # # file_path = open(file_path, file_name)
            #with open (file_path, 'wb') as f:
            #     pickle.dump(pts_dict, f)
        # ========================== End of save ================================
            # print("logits", logits.shape)
            # print("label", label.shape)
            loss = criterion(logits, label.float())
            loss_sum += loss
            time_sum += (time.time() - since)
            if i % 50 == 0:
                loss_avg = loss_sum / 50
                step_res -= 50
                time_elapsed = time_sum * (step_res // 50) * (args.epochs - epoch) # time_sum: 50 steps training time; step_res: rest training steps in one epoch; args.epochs - epoch: rest epochs
                # print("epoch %d, step (%d /5950), %.0fh %.0fm, lr: %.6f, loss: %.6f" % (epoch, i, time_elapsed//3600, time_elapsed%60, opt.state_dict()['param_groups'][0]['lr'], loss_avg))
                print("epoch %d, step (%d / %d), %.0fh %.0fm, lr: %.6f, loss: %.6f" % (epoch, i, int(len(dataset_train) / 4), time_elapsed//60, time_elapsed%60, scheduler.get_last_lr()[0], loss_avg))
                loss_sum = 0
                time_sum = 0
            loss.backward()
            opt.step()
           # preds = logits.max(dim=1)[1]
            preds = logits
            count += batch_size
            train_loss += loss.item() * batch_size
            #train_true.append(label.cpu().numpy().reshape(label.shape[1], 2))
            #label_max = label.max(dim=1)[1]
            train_true.append(label.cpu().numpy().reshape(label.shape[1], 2))
            train_pred.append(preds.detach().cpu().numpy().reshape(label.shape[1], 2))
        scheduler.step()
        train_true = np.concatenate(train_true, axis=0)
        #train_true = np.argmax(train_true, axis=1)
        #print(train_true.shape)
        train_pred = np.concatenate(train_pred, axis=0)
        train_pred_indices = np.argmax(train_pred, axis=1)
        train_pred_indices = np.expand_dims(train_pred_indices, axis=1)
        train_pred = np.zeros_like(train_pred)
        np.put_along_axis(train_pred, train_pred_indices, 1, axis=1)
        #train_true = np.argmax(train_true, axis=1)
        #print('pred {}'.format(train_pred.shape))
        #trian_pred = np.argmax(train_pred, axis=1)
        train_true = train_true.argmax(axis=1)
        train_pred = train_pred.argmax(axis=1)
       # c = train_pred[np.all((train_pred != [0, 0]) | (train_pred != [1, 0]) | (train_pred != [0, 1]), axis=1)]
       # c = train_pred[np.all((train_pred == [1, 0]), axis=1)]
       # print(c.shape)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                train_loss*1.0/count,
                                                                                metrics.accuracy_score(
                                                                                    train_true, train_pred),
                                                                                metrics.balanced_accuracy_score(
                                                                                train_true, train_pred))
        io.cprint(outstr)

        torch.save(model, '{}/pole_car_epoch_{}.pth'.format(args.exp_name, epoch))
        ####################
        # Val
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(val_loader):
            # data, label = data.to(device), label.to(device).squeeze()
            data, label = data.cuda(0).squeeze(), label.cuda(0).squeeze(0)
            # data_org = data.to('cpu') # save orignal points
            # label_org = label.to('cpu')
            data_org = data.to('cpu')
            label_org = label.to('cpu')
            # label_sample = label.to('cpu')
            data = data.unsqueeze(0).permute(0, 2, 1).contiguous()
            # data_sample = data.to('cpu') # save sampling points
            batch_size = data.size()[0]
            logits = model(data)
            label_pred = logits.to('cpu')

        # ========================== Save point clouds =========================
            if store:
                store_path = '%s' % args.exp_name
                pts_dict = {'data_org': data_org, 'label_org': label_org , 'label_pred': label_pred}
                local_time = time.localtime()
                file_name = '{}_{}_{}_{}_{}.npy'.format(local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)
                os.system('mkdir -p {}'.format(store_path))
                file_path = os.path.join(store_path, file_name)
            # # file_path = store_path
            # # file_path = open(file_path, file_name)
                with open (file_path, 'wb') as f:
                     pickle.dump(pts_dict, f)
        # ========================== End of save ================================
            loss = criterion(logits, label.float())
           # preds = logits.max(dim=1)[1]
            preds = logits
            count += batch_size
            test_loss += loss.item() * batch_size
            #test_true.append(label.cpu().numpy())
            test_true.append(label.cpu().numpy().reshape(-1, 2))
            #test_pred.append(preds.detach().cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy().reshape(-1, 2))
        #test_true = np.concatenate(test_true)
        #test_true = np.concatenate(test_true)
        test_true = np.concatenate(test_true, axis=0)
        #test_true = np.argmax(test_true, axis=1)
        test_pred = np.concatenate(test_pred, axis=0)
        test_pred_indices = np.argmax(test_pred, axis=1).reshape(test_pred.shape[0], 1)
        test_perd_indices = np.expand_dims(test_pred_indices, axis=1)
        test_pred = np.zeros_like(test_pred)
        np.put_along_axis(test_pred, test_pred_indices, 1, axis=1)
        
        test_true = test_true.argmax(axis=1)
        test_pred = test_pred.argmax(axis=1)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if avg_per_class_acc >= best_avg_per_class_acc:
            best_avg_per_class_acc = avg_per_class_acc
            print("best_avg_per_class__acc is {}".format(best_avg_per_class_acc))
            # torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            torch.save(model, '%s/best_model.pth' % args.exp_name)


# def test(args, io):
#     test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
#                              batch_size=args.test_batch_size, shuffle=True, drop_last=False)

#     device = torch.device("cuda" if args.cuda else "cpu")

#     #Try to load models
#     model = DGCNN(args).to(device)
#     model = nn.DataParallel(model)
#     model.load_state_dict(torch.load(args.model_path))
#     model = model.eval()
#     test_acc = 0.0
#     count = 0.0
#     test_true = []
#     test_pred = []
#     for data, label in test_loader:

#         data, label = data.to(device), label.to(device).squeeze()
#         data = data.permute(0, 2, 1)
#         batch_size = data.size()[0]
#         logits = model(data)
#         preds = logits.max(dim=1)[1]
#         test_true.append(label.cpu().numpy())
#         test_pred.append(preds.detach().cpu().numpy())
#     test_true = np.concatenate(test_true)
#     test_pred = np.concatenate(test_pred)
#     test_acc = metrics.accuracy_score(test_true, test_pred)
#     avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
#     outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
#     io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

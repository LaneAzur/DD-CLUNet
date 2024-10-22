import os, glob
import os.path as osp
import numpy as np
from natsort import natsorted

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from models.VoxelMorph import VoxelMorph
from models.u_denseu_cl_fusion import VoxelMorph as u_denseu_cl_fusion
from models.u_denseu_concat import VoxelMorph as u_denseu_concat
from models.u_denseu_cl_1con import VoxelMorph as u_denseu_cl_1con
from models.u_denseu_cl_3con import VoxelMorph as u_denseu_cl_3con

import utils.utils as utils
import utils.losses as losses
from utils.csv_logger import log_csv
from utils.train_utils import adjust_learning_rate, save_checkpoint

from data import datasets, trans
import argparse
import time

'''
parse the command line arg
'''
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=2)

parser.add_argument('--train_dir', type=str, default='data/LPBA40/train/')
parser.add_argument('--val_dir', type=str, default='data/LPBA40/test/')
parser.add_argument('--label_dir', type=str, default='data/LPBA40/label/')
parser.add_argument('--dataset', type=str, default='LPBA')
parser.add_argument('--atlas_dir', type=str, default='data/LPBA40/fixed.nii.gz')

parser.add_argument('--model', type=str, default='u_u_ecl')

parser.add_argument('--training_lr', type=float, default=1e-4)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=500)

parser.add_argument('--weight_model', type=float, default=0.02)
parser.add_argument('--weight_opt', type=float, default=1)
parser.add_argument('--model_idx', type=int, default=-1, help='the index of model loaded')

args = parser.parse_args()


def load_model(img_size):
    if args.model == 'VoxelMorph':
        model = VoxelMorph(img_size)
    elif args.model == 'u_denseu_cl_fusion_0.01':
        model = u_denseu_cl_fusion(img_size)
    elif args.model == 'u_denseu_cl_fusion_0.1':
        model = u_denseu_cl_fusion(img_size)
    elif args.model == 'u_denseu_cl_fusion_1':
        model = u_denseu_cl_fusion(img_size)
    elif args.model == 'u_denseu_cl_1con':
        model = u_denseu_cl_1con(img_size)
    elif args.model == 'u_denseu_cl_3con':
        model = u_denseu_cl_3con(img_size)
    elif args.model == 'u_denseu_cl_concat':
        model = u_denseu_concat(img_size)
    model.cuda()
    return model


def load_data():
    if args.dataset == "IXI":
        train_composed = transforms.Compose([trans.RandomFlip(0), trans.NumpyType((np.float32, np.float32))])
        val_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.IXIBrainDataset(glob.glob(args.train_dir + '*.pkl'), args.atlas_dir,
                                             transforms=train_composed)
        val_set = datasets.IXIBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), args.atlas_dir,
                                                transforms=val_composed)
    elif args.dataset == "OASIS":
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.OASISBrainDataset(glob.glob(args.train_dir + '*.pkl'), transforms=train_composed)
        val_set = datasets.OASISBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), transforms=val_composed)
    elif args.dataset == "LPBA":
        # train_composed = transforms.Compose([trans.RandomFlip(0), trans.NumpyType((np.float32, np.float32))])

        train_composed = transforms.Compose([trans.RandomFlip(),
                                             trans.RandomRotate(),
                                             trans.RandomCrop((112, 144, 112)),
                                             trans.NumpyType((np.float32, np.float32))])

        val_composed = transforms.Compose([trans.Seg_norm(),
                                           trans.CenterCrop((112, 144, 112)),
                                           trans.NumpyType((np.float32, np.int16))])

        train_set = datasets.LPBADataset(glob.glob(args.train_dir + '*.nii.gz'), args.atlas_dir,
                                         transforms=train_composed)

        val_set = datasets.LPBAInferDataset(glob.glob(args.val_dir + '*.nii.gz'), args.atlas_dir, args.label_dir,
                                            transforms=val_composed)
    elif args.dataset == "AbdomenCTCT":
        train_set = datasets.AbdomenCTCT(path=osp.join(args.train_dir, 'train.json'),
                                         img_dir=osp.join(args.train_dir, 'imagesTr/'),
                                         labels_dir=osp.join(args.train_dir, 'labelsTr/'))
        val_set = datasets.AbdomenCTCT(path=osp.join(args.train_dir, 'test.json'),
                                       img_dir=osp.join(args.train_dir, 'imagesTr/'),
                                       labels_dir=osp.join(args.train_dir, 'labelsTr/'))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

    return train_loader, val_loader


def main():

    epoch_start = args.epoch_start  # start epoch (use for continue training)
    lr = args.training_lr  # lr for model
    max_epoch = args.max_epoch
    model_idx = args.model_idx
    pre_load = False
    eval_time = utils.AverageMeter()

    weights_model = [1, args.weight_model]  # loss weighs of model loss
    weights_opt = [1, args.weight_opt]  # loss weights of optimizer

    save_dir = '{}_{}_bio/'.format(args.model, args.dataset)
    if not os.path.exists('checkpoints/' + save_dir):
        os.makedirs('checkpoints/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    if args.dataset == 'AbdomenCTCT':
        img_size = (192, 160, 256)
    else:
        # img_size = (160, 192, 160) if args.dataset == 'LPBA' else (160, 192, 224)
        img_size = (112, 144, 112) if args.dataset == 'LPBA' else (160, 192, 224)
    '''
    initialize model
    '''
    model = load_model(img_size)

    '''
    initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    '''
    if continue from previous training
    '''
    if epoch_start:
        model_dir = 'checkpoints/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)

        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[model_idx]))
        model.load_state_dict(best_model)
        for name, param in model.named_parameters():
            print(name)
    else:
        updated_lr = lr

    if pre_load:
        model_dict = model.state_dict()
        model_dir = 'checkpoints/VM.pth.tar'
        loaded_dict = torch.load(model_dir, map_location="cpu")['state_dict']
        # print(model.state_dict().keys())
        pretrained_dict = {k: v for k, v in loaded_dict.items() if k in model.state_dict()}
        # print(pretrained_dict.keys())
        model.load_state_dict(pretrained_dict, strict=False)
    # for name, param in model.named_parameters():
    # print(name)
    #  if "z" not in name:
    #     param.requires_grad = False

    '''
    initialize dataset
    '''
    train_loader, val_loader = load_data()

    '''
    initialize optimizer and loss functions
    '''
    adam = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')
    criterion_mse = nn.MSELoss()
    criterion_CL = losses.contrastive_loss()

    best_dsc = 0

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        training
        '''
        loss_all = utils.AverageMeter()
        idx = 0

        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(adam, epoch, max_epoch, lr)

            data = [t.cuda() for t in data]

            x = data[0]
            y = data[1]
            output = model(x, y)

            # x_in = torch.cat((x,y), dim=1)
            # output = model(x_in)

            if args.dataset == "AbdomenCTCT":
                loss_mse = criterion_mse(output[0], y) * weights_model[0]
                loss_reg = criterion_reg(output[1], y) * weights_model[1]
                loss = loss_mse + loss_reg
                loss_vals = [loss_mse, loss_reg]
                loss_all.update(loss.item(), y.numel())
            else:
                loss_ncc = criterion_ncc(output[0], y)
                loss_reg = criterion_reg(output[1], y)

                contrastive_features_x1 = output[2]
                contrastive_features_y1 = output[3]
                # contrastive_features_x2 = output[4]
                # contrastive_features_y2 = output[5]
                # contrastive_features_x3 = output[6]
                # contrastive_features_y3 = output[7]
                # loss_cl = 0.001 * (criterion_CL(contrastive_features_x1, contrastive_features_y1)+
                #                   criterion_CL(contrastive_features_x2, contrastive_features_y2)+
                #                   criterion_CL(contrastive_features_x3, contrastive_features_y3))
                loss_cl = 0.001 * (criterion_CL(contrastive_features_x1, contrastive_features_y1))
                loss = loss_ncc + loss_reg + loss_cl

                loss_vals = [loss_ncc, loss_reg, loss_cl]
                loss_all.update(loss.item(), y.numel())

            # compute gradient and do SGD step
            adam.zero_grad()
            loss.backward()
            adam.step()

            '''
            For OASIS dataset and AbdomenCTCT dataset
            '''
            if args.dataset == "OASIS" or args.dataset == "AbdomenCTCT":
                y_in = torch.cat((y, x), dim=1)
                output = model(y_in)

                '''initialize ofg'''


                if args.dataset == "AbdomenCTCT":
                    loss_mse = criterion_mse(output[0], x) * weights_model[0]
                    loss_reg = criterion_reg(output[1], x) * weights_model[1]
                    loss = loss_mse + loss_reg
                    loss_vals = [loss_mse, loss_reg]
                    loss_all.update(loss.item(), y.numel())
                else:
                    loss_ncc = criterion_ncc(output[0], x)
                    loss_reg = criterion_reg(output[1], x)
                    loss = loss_ncc + loss_reg
                    loss_vals = [loss_ncc, loss_reg]

                loss_all.update(loss.item(), x.numel())
                adam.zero_grad()
                loss.backward()
                adam.step()

            current_lr = adam.state_dict()['param_groups'][0]['lr']
            print('Epoch [{}/{}] Iter [{}/{}] - loss {:.4f}, Img Sim: {:.6f}, CL: {:.6f},  Reg: {:.6f}, lr: {:.6f}'
                  .format(epoch, max_epoch, idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_cl.item(),
                          loss_vals[1].item(), current_lr))
            #print('time: {}s'.format(eval_time.avg))

        '''
        validation
        '''
        eval_dsc = utils.AverageMeter()
        eval_det = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x, y, x_seg, y_seg = data
                time_start = time.time()
                output = model(x, y)
                # x_in = torch.cat((x, y), dim=1)
                # output = model(x_in)

                time_end = time.time()
                eval_time.update(time_end - time_start, x.size(0))
                print("{}s".format(time_end - time_start))
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])

                '''update DSC'''
                if args.dataset == "OASIS":
                    dsc = utils.dice_OASIS(def_out.long(), y_seg.long())
                elif args.dataset == "IXI":
                    dsc = utils.dice_IXI(def_out.long(), y_seg.long())
                elif args.dataset == "LPBA":
                    dsc = utils.dice_LPBA(y_seg.cpu().detach().numpy(), def_out[0, 0, ...].cpu().detach().numpy())
                elif args.dataset == "AbdomenCTCT":
                    dsc_1 = utils.dice_AbdomenCTCT(y_seg.contiguous(), def_out.contiguous(), 14).cpu()
                    dsc_ident = utils.dice_AbdomenCTCT(y_seg.contiguous(), y_seg.contiguous(),
                                                       14).cpu() * utils.dice_AbdomenCTCT(x_seg.contiguous(),
                                                                                          x_seg.contiguous(), 14).cpu()
                    dsc = dsc_1.sum() / (dsc_ident > 0.1).sum()
                eval_dsc.update(dsc.item(), x.size(0))

                '''update Jdet'''
                jac_det = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :, :])
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]
                eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

        '''save model'''
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': adam.state_dict(),
        }, save_dir='checkpoints/' + save_dir,
            filename='dsc{:.3f}_epoch{:d}.pth.tar'.format(eval_dsc.avg, epoch))

        print('\nEpoch [{}/{}] - DSC: {:.6f}, Jdet: {:.8f}, loss: {:.6f}, lr: {:.6f}\n'.format(
            epoch, max_epoch, eval_dsc.avg, eval_det.avg, loss_all.avg, current_lr))
        log_csv(save_dir, epoch, eval_dsc.avg, eval_det.avg, loss_all.avg, current_lr)

        loss_all.reset()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    '''
      GPU configuration
      '''
    GPU_iden = 5
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()

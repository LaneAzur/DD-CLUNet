import os, utils.utils as utils, glob, argparse, time, torch
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
from torchvision import transforms
from natsort import natsorted


from models.VoxelMorph import VoxelMorph

from models.u_denseu_cl_fusion import VoxelMorph as u_denseu_cl_fusion
from models.u_denseu_concat import VoxelMorph as u_denseu_cl_concat
from models.u_denseu_cl_1con import VoxelMorph as u_denseu_cl_1con
from models.u_denseu_cl_3con import VoxelMorph as u_denseu_cl_3con


import argparse
import nibabel as nib
import time
from utils.csv_logger import infer_csv

# parse the commandline
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='LPBA')
parser.add_argument('--test_dir', type=str, default='data/LPBA40/test/')
parser.add_argument('--label_dir', type=str, default='data/LPBA40/label/')
parser.add_argument('--atlas_dir', type=str, default='data/LPBA40/fixed.nii.gz')
parser.add_argument('--save_dir', type=str, default='./infer_results/', help='The directory to save infer results')
parser.add_argument('--model', type=str, default='u_denseu_cl_concat')
parser.add_argument('--model_dir', type=str, default='checkpoints/', help='The directory path that saves model weights')
parser.add_argument('--ofg', action='store_true', help="use ofg or not")
args = parser.parse_args()


def main():
    if args.ofg:
        csv_name = args.model + '_opt.csv'
    else:
        csv_name = args.model + '.csv'

    save_dir = args.save_dir + args.dataset + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """Initialize model"""
    # img_size = (160, 192, 160) if args.dataset == "LPBA" else (160, 192, 224)
    img_size = (112, 144, 112) if args.dataset == "LPBA" else (160, 192, 224)
    if args.model == "VoxelMorph":
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
        model = u_denseu_cl_concat(img_size)
    else:
        raise ValueError("{} doesn't exist!".format(args.model))

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    """Load model weights"""
    if args.model_dir is None:
        raise ValueError("model_dir is None")
    else:
        model_dir = args.model_dir
    model_idx = -1

    model_dir_id = args.model + str('_') + str('LPBA_bio') + str('/')

    model_dir = os.path.join(model_dir, model_dir_id)

    print(model_dir)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    """load test dataset"""
    test_dir = args.test_dir
    atlas_dir = args.atlas_dir
    if args.dataset == 'IXI':
        test_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16)), ])
        test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    elif args.dataset == 'OASIS':
        test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)), ])
        test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    elif args.dataset == "LPBA":
        test_composed = transforms.Compose([trans.Seg_norm(),
                                            trans.CenterCrop((112, 144, 112)),
                                            trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.LPBAInferDataset(glob.glob(test_dir + '*.nii.gz'), atlas_dir, args.label_dir,
                                             transforms=test_composed)
    else:
        raise ValueError("Dataset name is wrong!")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

    """start infering"""
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    eval_time = utils.AverageMeter()

    print("Start Inferring\n")
    with torch.no_grad():
        idx = 0
        for data in test_loader:
            idx += 1
            print(idx)
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]


            # evaluate infer time
            time_start = time.time()

            x_def, flow, _, _ = model(x, y)
            # x_def, flow = model(x_in)
            time_end = time.time()
            eval_time.update(time_end - time_start, x.size(0))

            # ! more accurate
            # x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            # x_seg_oh = torch.squeeze(x_seg_oh, 1)
            # x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            # def_out = model.spatial_trans(x_seg.float(), flow.float())
            # x_segs = []
            # for i in range(46):
            #     def_seg = reg_model([x_seg_oh[:, i:i + 1, ...].float(), flow.float()])
            #     x_segs.append(def_seg)
            # x_segs = torch.cat(x_segs, dim=1)
            # def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            # del x_segs, x_seg_oh

            # evaluate Jdet
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            Jdet = np.sum(jac_det <= 0) / np.prod(tar.shape)
            eval_det.update(Jdet, x.size(0))
            print('det < 0: {}'.format(Jdet))

            # evaluate DSC
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            if args.dataset == "OASIS":
                dsc_trans = utils.dice_OASIS(def_out.long(), y_seg.long())
                dsc_raw = utils.dice_OASIS(x_seg.long(), y_seg.long())
            elif args.dataset == "IXI":
                dsc_trans = utils.dice_IXI(def_out.long(), y_seg.long())
                dsc_raw = utils.dice_IXI(x_seg.long(), y_seg.long())
            elif args.dataset == "LPBA":
                print('y_seg', y_seg.type(), 'def_out', def_out.type())
                print('y_seg', y_seg.shape, 'def_out', def_out.shape)
                print(y_seg.cpu().detach().numpy().shape, def_out[0, 0, ...].cpu().detach().numpy().shape)
                dsc_trans = utils.dice_LPBA(y_seg.cpu().detach().numpy(), def_out[0, 0, ...].cpu().detach().numpy())
                dsc_raw = utils.dice_LPBA(y_seg.cpu().detach().numpy(), x_seg.cpu().detach().numpy())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

            infer_csv(save_dir, csv_name, idx, dsc_raw.item(), dsc_trans.item(), Jdet, time_end - time_start)
            print()

        print('Average:')
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('time: {}s'.format(eval_time.avg))
        infer_csv(save_dir, csv_name, 'avg', eval_dsc_raw.avg, eval_dsc_def.avg, eval_det.avg, eval_time.avg)


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

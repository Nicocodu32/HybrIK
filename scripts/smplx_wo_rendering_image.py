import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLX
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

det_transform = T.Compose([T.ToTensor()])

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

parser = argparse.ArgumentParser(description='HybrIK Demo')
parser.add_argument('--gpu', help='gpu', default='0', type=str)
parser.add_argument('--image-name', help='image name', required=True, type=str)
parser.add_argument('--out-dir', help='output folder', default='output', type=str)
parser.add_argument('--save-pk', default=True, dest='save_pk', help='save prediction', action='store_true')

opt = parser.parse_args()
if opt.gpu == '0':
    opt.gpu = 0

cfg_file = './configs/smplx/256x192_hrnet_rle_smplx_kid.yaml'
CKPT = './pretrained_models/hybrikx_rle_hrnet.pth'

cfg = update_config(cfg_file)
cfg['MODEL']['EXTRA']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)
cfg['LOSS']['ELEMENTS']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

res_keys = [
    'pred_phi',
    'pred_shape_full',
    'pred_beta',
    'pred_expression',
    'pred_theta_quat',
    'pred_theta_mat',
    'pred_lh_uvd',
    'pred_rh_uvd',
    'pred_uvd_jts',
    'pred_xyz_hybrik',
    'pred_xyz_hybrik_struct',
    'pred_xyz_full',
    'pred_uv_full',
    'pred_vertices',
    'pred_sigma',
    'scores',
    'maxvals',
    'cam_scale',
    'cam_root',
    'transl',
    'img_feat',
    'pred_camera',
    'gt_output',
    'bbox',
    'height',
    'width',
    'img_path'
]

res_db = {k: [] for k in res_keys}

transformation = SimpleTransform3DSMPLX(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

det_model = fasterrcnn_resnet50_fpn(pretrained=True)
hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

if opt.gpu == 0:
    det_model.cuda(opt.gpu)
    hybrik_model.cuda(opt.gpu)
else:
    det_model.to(opt.gpu)
    hybrik_model.to(opt.gpu)
det_model.eval()
hybrik_model.eval()

image_basename = os.path.basename(opt.image_name).split('.')[0]

# Create output directories
if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

with torch.no_grad():
    # Load and process the image
    input_image = cv2.cvtColor(cv2.imread(opt.image_name), cv2.COLOR_BGR2RGB)
    det_input = det_transform(input_image).to(opt.gpu)
    det_output = det_model([det_input])[0]

    tight_bbox = get_one_box(det_output)
    if tight_bbox is None:
        raise ValueError("No bounding box detected in the image.")

    # Run HybrIK
    pose_input, bbox, img_center = transformation.test_transform(input_image.copy(), tight_bbox)
    pose_input = pose_input.to(opt.gpu)[None, :, :, :]

    pose_output = hybrik_model(
        pose_input, flip_test=True,
        bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
        img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
    )

    transl = pose_output.transl.detach()

    if opt.save_pk:
        assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

        pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
        pred_shape_full = pose_output.pred_shape_full.squeeze(dim=0).cpu().numpy()
        pred_beta = pose_output.pred_beta.squeeze(dim=0).cpu().data.numpy()
        pred_expression = pose_output.pred_expression.squeeze(dim=0).cpu().numpy()
        pred_theta_quat = pose_output.pred_theta_quat.squeeze(dim=0).cpu().data.numpy()
        pred_theta_mat = pose_output.pred_theta_mat.squeeze(dim=0).cpu().data.numpy()
        pred_lh_uvd = pose_output.pred_lh_uvd.squeeze(dim=0).cpu().numpy()
        pred_rh_uvd = pose_output.pred_rh_uvd.squeeze(dim=0).cpu().numpy()
        pred_uvd_jts = pose_output.pred_uvd_jts.reshape(-1, 3).cpu().data.numpy()
        pred_xyz_hybrik = pose_output.pred_xyz_hybrik.reshape(-1, 3).cpu().data.numpy()
        pred_xyz_hybrik_struct = pose_output.pred_xyz_hybrik_struct.reshape(-1, 3).cpu().data.numpy()
        pred_xyz_full = pose_output.pred_xyz_full.reshape(-1, 3).cpu().data.numpy()
        pred_uv_full = pose_output.pred_uv_full.reshape(-1, 2).cpu().data.numpy()
        pred_vertices = pose_output.pred_vertices.reshape(-1, 3).cpu().data.numpy()
        pred_sigma = pose_output.pred_sigma.squeeze(dim=0).cpu().data.numpy()
        scores = pose_output.scores.squeeze(dim=0).cpu().numpy()
        maxvals = pose_output.maxvals.squeeze(dim=0).cpu().numpy()
        cam_scale = pose_output.cam_scale.squeeze(dim=0).cpu().numpy()
        cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
        transl = transl[0].cpu().data.numpy()
        img_feat = pose_output.img_feat.squeeze(dim=0).cpu().data.numpy()
        pred_camera = pose_output.pred_camera.squeeze(dim=0).cpu().numpy()
        gt_output = pose_output.gt_output
        img_size = np.array((input_image.shape[0], input_image.shape[1]))

        res_db['pred_phi'].append(pred_phi)
        res_db['pred_shape_full'].append(pred_shape_full)
        res_db['pred_beta'].append(pred_beta)
        res_db['pred_expression'].append(pred_expression)
        res_db['pred_theta_quat'].append(pred_theta_quat)
        res_db['pred_theta_mat'].append(pred_theta_mat)
        res_db['pred_lh_uvd'].append(pred_lh_uvd)
        res_db['pred_rh_uvd'].append(pred_rh_uvd)
        res_db['pred_uvd_jts'].append(pred_uvd_jts)
        res_db['pred_xyz_hybrik'].append(pred_xyz_hybrik)
        res_db['pred_xyz_hybrik_struct'].append(pred_xyz_hybrik_struct)
        res_db['pred_xyz_full'].append(pred_xyz_full)
        res_db['pred_uv_full'].append(pred_uv_full)
        res_db['pred_vertices'].append(pred_vertices)
        res_db['pred_sigma'].append(pred_sigma)
        res_db['scores'].append(scores)
        res_db['maxvals'].append(maxvals)
        res_db['cam_scale'].append(cam_scale)
        res_db['cam_root'].append(cam_root)
        res_db['transl'].append(transl)
        res_db['img_feat'].append(img_feat)
        res_db['pred_camera'].append(pred_camera)
        res_db['gt_output'].append(gt_output)

        res_db['bbox'].append(np.array(bbox))
        res_db['height'].append(img_size[0])
        res_db['width'].append(img_size[1])
        res_db['img_path'].append(opt.image_name)

if opt.save_pk:
    n_frames = len(res_db['img_path'])
    for k in res_db.keys():
        res_db[k] = np.stack(res_db[k])
        assert res_db[k].shape[0] == n_frames

    with open(os.path.join(opt.out_dir, f'res_{image_basename}_x.pk'), 'wb') as fid:
        pk.dump(res_db, fid)

    print(f"Results saved to {os.path.join(opt.out_dir, f'res_{image_basename}_x.pk')}")

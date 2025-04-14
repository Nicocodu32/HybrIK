import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_max_iou_box, get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

det_transform = T.Compose([T.ToTensor()])

halpe_wrist_ids = [94, 115]
halpe_left_hand_ids = [
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
]
halpe_right_hand_ids = [
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
]

halpe_lhand_leaves = [
    8, 12, 20, 16, 4
]
halpe_rhand_leaves = [
    8, 12, 20, 16, 4
]

halpe_hand_ids = [i + 94 for i in halpe_left_hand_ids] + [i + 115 for i in halpe_right_hand_ids]
halpe_hand_leaves_ids = [i + 94 for i in halpe_lhand_leaves] + [i + 115 for i in halpe_rhand_leaves]

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()
    return stream, videoinfo, datalen

def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

parser = argparse.ArgumentParser(description='HybrIK Demo')
parser.add_argument('--gpu', help='gpu', default=0, type=int)
parser.add_argument('--video-name', help='video name', default='', type=str)
parser.add_argument('--out-dir', help='output folder', default='', type=str)
parser.add_argument('--save-pk', default=False, dest='save_pk', help='save prediction', action='store_true')

opt = parser.parse_args()

cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = './pretrained_models/hybrik_hrnet.pth'

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
    'pred_uvd',
    'pred_xyz_17',
    'pred_xyz_29',
    'pred_xyz_24_struct',
    'pred_scores',
    'pred_camera',
    # 'f',
    'pred_betas',
    'pred_thetas',
    'pred_phi',
    'pred_cam_root',
    # 'features',
    'transl',
    'bbox',
    'height',
    'width',
    'img_path',

    'pred_delta_shape',
    'pred_xyz_24',
    'pred_vertices',
    'maxvals',
    'cam_scale',
    'pred_sigma',
    'img_feat'
]

res_db = {k: [] for k in res_keys}

transformation = SimpleTransform3DSMPLCam(
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

det_model.to('cpu')
hybrik_model.to('cpu')
det_model.eval()
hybrik_model.eval()

print('### Extract Image...')
video_basename = os.path.basename(opt.video_name).split('.')[0]

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))

_, info, _ = get_video_info(opt.video_name)
video_basename = os.path.basename(opt.video_name).split('.')[0]

os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png')

files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None
smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

print('### Run Model...')
idx = 0
for img_path in tqdm(img_path_list):
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    with torch.no_grad():
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to('cpu')
        det_output = det_model([det_input])[0]

        if prev_box is None:
            tight_bbox = get_one_box(det_output)
            if tight_bbox is None:
                continue
        else:
            tight_bbox = get_one_box(det_output)

        if tight_bbox is None:
            tight_bbox = prev_box

        prev_box = tight_bbox

        pose_input, bbox, img_center = transformation.test_transform(
            input_image.copy(), tight_bbox)
        pose_input = pose_input.to('cpu')[None, :, :, :]

        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
        )

        transl = pose_output.transl.detach()

        if opt.save_pk:
            assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

            pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
                17, 3).cpu().data.numpy()
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
                24, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu(
            ).data[:, :29].reshape(29).numpy()
            pred_camera = pose_output.pred_camera.squeeze(
                dim=0).cpu().data.numpy()
            pred_betas = pose_output.pred_shape.squeeze(
                dim=0).cpu().data.numpy()
            pred_theta = pose_output.pred_theta_mats.squeeze(
                dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            pred_delta_shape = pose_output.pred_delta_shape.squeeze(
                dim=0).cpu().data.numpy()
            pred_xyz_jts_24 = pose_output.pred_xyz_jts_24.reshape(
                -1, 3).cpu().data.numpy()
            pred_vertices = pose_output.pred_vertices.reshape(
                -1, 3).cpu().data.numpy()
            maxvals = pose_output.maxvals.cpu().data.numpy()
            cam_scale = pose_output.cam_scale.cpu().data.numpy()
            pred_sigma = pose_output.pred_sigma.squeeze(
                dim=0).cpu().data.numpy()
            img_feat = pose_output.img_feat.cpu().data.numpy()


            res_db['pred_xyz_17'].append(pred_xyz_jts_17)
            res_db['pred_uvd'].append(pred_uvd_jts)
            res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            res_db['pred_xyz_24_struct'].append(pred_xyz_jts_24_struct)
            res_db['pred_scores'].append(pred_scores)
            res_db['pred_camera'].append(pred_camera)
            res_db['pred_betas'].append(pred_betas)
            res_db['pred_thetas'].append(pred_theta)
            res_db['pred_phi'].append(pred_phi)
            res_db['pred_cam_root'].append(pred_cam_root)
            res_db['transl'].append(transl[0].cpu().data.numpy())
            res_db['bbox'].append(np.array(bbox))
            res_db['height'].append(img_size[0])
            res_db['width'].append(img_size[1])
            res_db['img_path'].append(img_path)

            res_db['pred_delta_shape'].append(pred_delta_shape)
            res_db['pred_xyz_24'].append(pred_xyz_jts_24)
            res_db['pred_vertices'].append(pred_vertices)
            res_db['maxvals'].append(maxvals)
            res_db['cam_scale'].append(cam_scale)
            res_db['pred_sigma'].append(pred_sigma)
            res_db['img_feat'].append(img_feat)

if opt.save_pk:
    n_frames = len(res_db['img_path'])
    for k in res_db.keys():
        print(k)
        res_db[k] = np.stack(res_db[k])
        assert res_db[k].shape[0] == n_frames

    with open(os.path.join(opt.out_dir, f'res_{video_basename}.pk'), 'wb') as fid:
        pk.dump(res_db, fid)

os.system(f'rm -rf {opt.out_dir}/raw_images')

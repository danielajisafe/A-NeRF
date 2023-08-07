# MIRROR ANERF

#MIRROR ANERF


import os
import cv2
import time
import ipdb
import h5py
#from regex import P
import torch
import shutil
import imageio
import datetime
import numpy as np
import deepdish as dd
from tqdm import trange
import matplotlib.pyplot as plt
from run_nerf import render_path
from run_nerf import config_parser as nerf_config_parser

from core.pose_opt import load_poseopt_from_state_dict, pose_ckpt_to_pose_data
from core.load_data import generate_bullet_time, get_dataset
from core.raycasters import create_raycaster

from core.utils.evaluation_helpers import txt_to_argstring
from core.utils.dan_compute_eval import eval_opt_kp
from core.utils.skeleton_utils import CMUSkeleton, smpl_rest_pose, get_smpl_l2ws, nerf_c2w_to_extrinsic, swap_mat, get_per_joint_coords
from core.utils.skeleton_utils import draw_skeletons_3d, rotate_x, rotate_y, skeleton3d_to_2d, axisang_to_rot, rot_to_axisang
from pytorch_msssim import SSIM

import sys
sys.path.append("/scratch/dajisafe/smpl/A_temp_folder/A-NeRF/core/utils")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GT_PREFIXES = {
    'h36m': 'data/h36m/',
    'surreal': None,
    'perfcap': 'data/',
    'mixamo': 'data/mixamo',
    'mirror':'data/mirror',
}

n_joints = 26

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # nerf config
    parser.add_argument('--nerf_args', type=str, required=True,
                        help='path to nerf configuration (args.txt in log)')
    parser.add_argument('--ckptpath', type=str, required=True,
                        help='path to ckpt')

    # render config
    parser.add_argument('--render_res', nargs='+', type=int, default=[1080, 1920],
                        help='tuple of resolution in (H, W) for rendering')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset to render')
    parser.add_argument("--data_path", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument('--n_bullet', type=int, default=3,
                        help='no of bullet views to render')
    parser.add_argument('--n_bubble', type=int, default=10,
                        help='no of bubble views to render')
    parser.add_argument('--bullet_ang', type=int, default=360,
    help='angle of novel bullet view to render') 
    parser.add_argument('--switch_cam', action='store_true', default=False,
                        help='replace (have both) real camera and virtual camera')
    parser.add_argument('--evaluate_pose', action='store_true', default=False,
                        help='evaluate the [refined/initial] pose for e.g trainset or images')
    parser.add_argument('--psnr_images', action='store_true', default=False,
                        help='evaluate images for psnr - use same img ids or offsets')
    parser.add_argument('--generate_skeleton_overlay', action='store_true', default=False,
                        help='overlay skeleton (initial or refined) on GT images') 
    parser.add_argument("--use_mirr", action='store_true',
                        help='use both masks')
    # parser.add_argument("--no_reload", action='store_true',
    #                     help='do not reload weights from saved ckpt?')
    parser.add_argument("--render_folder", type=str, default='',
                        help='use existing render folder')

    # used to updated the same parameters copied/passed over from run_nerf 
    parser.add_argument("--train_size", type=int, default=None,
                        help='no of samples for training')
    parser.add_argument("--data_size", type=int, default=None,
                        help='no of samples for training')

    parser.add_argument('--entry', type=str, required=True,
                        help='entry in the dataset catalog to render')
    parser.add_argument('--white_bkgd', action='store_true',
                        help='render with white background')
    parser.add_argument('--render_type', type=str, default='retarget',
                        help='type of rendering to conduct')
    parser.add_argument('--save_gt', action='store_true',
                        help='save gt frames')
    parser.add_argument('--fps', type=int, default=14,
                        help='fps for video')
    parser.add_argument('--mesh_res', type=int, default=255,
                        help='resolution for marching cubes')
    # kp-related
    parser.add_argument('--render_refined', action='store_true', default=False,
                        help='render from refined poses')
    parser.add_argument('--subject_idx', type=int, default=0,
                        help='which subject to render (for MINeRF)')

    # frame-related
    parser.add_argument('--train_len', type=int, default=0,
                        help='length of trainset')
    parser.add_argument('--selected_idxs', nargs='+', type=int, default=None,
                        help='hand-picked idxs for rendering')
    parser.add_argument('--selected_framecode', type=int, default=None,
                        help='hand-picked framecode for rendering')
    

    # saving
    parser.add_argument('--outputdir', type=str, default='render_output/',
                        help='output directory')
    parser.add_argument('--runname', type=str, required=True,
                        help='run name as an identifier ')

    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='to do evaluation at the end or not (only in bounding box)')

    parser.add_argument('--no_save', action='store_true',
                        help='no image saving operation')
    parser.add_argument('--apples14', action='store_true', default=False,
                        help='use common 14 joints between SMPL, alphapose and Mirror skeleton')
    parser.add_argument('--body15', action='store_true', default=False,
                        help='use common 15 joints between alphapose and Mirror skeleton - includes nose joint')

    return parser

def load_nerf(args, nerf_args, skel_type=CMUSkeleton):
    #import ipdb; ipdb.set_trace()

    ckptpath = args.ckptpath
    nerf_args.ft_path = args.ckptpath

    # some info are unknown/not provided in nerf_args
    # dig those out from state_dict directly
    nerf_sdict = torch.load(ckptpath)

    #import ipdb; ipdb.set_trace()
    # get data_attrs used for training the models
    data_attrs = get_dataset(nerf_args).get_meta()
    if 'framecodes.codes.weight' in nerf_sdict['network_fn_state_dict']:
        framecodes = nerf_sdict['network_fn_state_dict']['framecodes.codes.weight']
        data_attrs['n_views'] = framecodes.shape[0]

    # load poseopt_layer (if exist)
    popt_layer = None
    #import ipdb; ipdb.set_trace()
    if nerf_args.opt_pose:
        popt_layer = load_poseopt_from_state_dict(nerf_sdict)
    nerf_args.finetune = True

    render_kwargs_train, render_kwargs_test, _, grad_vars, _, _ = create_raycaster(nerf_args, data_attrs, device=device)

    # freeze weights
    for grad_var in grad_vars:
        grad_var.requires_grad = False

    render_kwargs_test['ray_caster'] = render_kwargs_train['ray_caster']
    render_kwargs_test['ray_caster'].eval()

    return render_kwargs_test, popt_layer

def load_render_data(args, nerf_args, poseopt_layer=None, opt_framecode=True):
    
    # TODO: note that for models trained on SPIN data, they may not react well
    catalog = init_catalog(args)[args.dataset][args.entry]
    render_data = catalog.get(args.render_type, {})
    data_h5 = catalog['data_h5']
    #import ipdb; ipdb.set_trace()

    if args.use_mirr or args.switch_cam:
        print("Also loading virtual camera!")
        # catalog = init_catalog(args)[args.dataset][args.entry]
        if 'data_h5_v' in catalog: data_h5_v = catalog['data_h5_v']


    # to real cameras (due to the extreme focal length they were trained on..)
    # TODO: add loading with opt pose option
    if poseopt_layer is not None:
        rest_pose = poseopt_layer.get_rest_pose().cpu().numpy()[0]
        print("Load rest pose for poseopt!")
    else:
        try:
            rest_pose = dd.io.load(data_h5, '/rest_pose')
            print("Load rest pose from h5!")
        except:
            rest_pose = smpl_rest_pose * dd.io.load(data_h5, '/ext_scale')
            print("Load smpl rest pose!")


    # import ipdb; ipdb.set_trace()
    if args.render_refined:

        if 'refined' in catalog:
            print(f"loading refined poses from {catalog['refined']}")
            #poseopt_layer = load_poseopt_from_state_dict(torch.load(catalog['refined']))
            kps, bones = pose_ckpt_to_pose_data(catalog['refined'], legacy=True)[:2]
        else:
            with torch.no_grad():
                bones = poseopt_layer.get_bones().cpu().numpy()
                kps = poseopt_layer(np.arange(len(bones)))[0].cpu().numpy()

        #import ipdb; ipdb.set_trace()
        
        if render_data is not None:
            render_data['refined'] = [kps, bones]
            render_data['idx_map'] = catalog.get('idx_map', None)
        else:
            render_data = {'refined': [kps, bones],
                           'idx_map': catalog['idx_map']}
    else:
        render_data['idx_map'] = catalog.get('idx_map', None)

    render_data['args'] = args
    pose_keys = ['/kp3d', '/bones']
    cam_keys = ['/c2ws', '/focals', '/centers']
    # Do partial load here!
    # Need:
    # 1. kps: for root location
    # 2. bones: for bones
    # 3. camera stuff: focals, c2ws, ... etc
    c2ws, focals, centers_n = dd.io.load(data_h5, cam_keys)
    #centers_n = centers_n.detach.numpy()

    if args.switch_cam:
        A_mirr =  dd.io.load(data_h5_v, '/A_dash')
        # A_mirr is duplicated. select id or pick one.
        # sel_id = catalog.get(args.render_type, {}) ['selected_idxs'] 
        # A_mirr = A_mirr[sel_id].reshape(4,4) 

        n_size = A_mirr.shape[0]
        # extract V cam from A mirror matrix. Do we add negative to V position?
        #v_cam_3_4 = np.concatenate((np.linalg.inv(A_mirr[...,:3,:3]), -A_mirr[...,:3,3:4]),2)
        #last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
        #last_row = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), [n_size,1,1])
        #v_cam =  np.concatenate([v_cam_3_4, last_row], axis=1)

        '''anerf-inspired'''
        #A_mirr_flip = nerf_c2w_to_extrinsic(A_mirr[...,:3,:3])
        #v_cam_3_4 = np.concatenate((A_mirr_flip, -A_mirr[...,:3,3:4]),2)
        #v_cam =  np.concatenate([v_cam_3_4, last_row], axis=1)

        #import ipdb; ipdb.set_trace()
        #root = render_data['root']
        #if root is not None:
        #    v_cam[..., :3, -1] -= root.reshape(-1,3)
        
        '''flip the camera about the x-axis, to simulate flipped image'''
        v_cam_3_4 = np.concatenate((swap_mat(A_mirr[...,:3,:3]), A_mirr[...,:3,3:4]),2)
        last_row = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), [n_size,1,1])
        v_cam =  np.concatenate([v_cam_3_4, last_row], axis=1)
        # Testing: switch real with a virtual camera
        print("c2ws", c2ws[0])
        print("v_cam", v_cam[0])

        '''swap cameras'''
        #c2ws = v_cam
    
    _, H, W, _ = dd.io.load(data_h5, ['/img_shape'])[0]
    
    #import ipdb; ipdb.set_trace()
    # handle resolution
    if args.render_res is not None:
        assert len(args.render_res) == 2, "Image resolution should be in (H, W)"
        H_r, W_r = args.render_res
        # TODO: only check one side for now ...
        scale = float(H_r) / float(H)
        focals *= scale
        H, W = H_r, W_r

    # Load data based on type:
    bones = None
    root = None
    bg_imgs, bg_indices = None, None


    if args.render_type in ['retarget', 'mesh']:
        print(f'Load data for retargeting!')
        kps, skts, c2ws, cam_idxs, focals, bones = load_retarget(data_h5, c2ws, focals,
                                                                 rest_pose, pose_keys,
                                                                 **render_data)
    elif args.render_type == 'bullet':
        #resp = input("do you want to evaluate, yes or no ?")
        if args.evaluate_pose:
            print(f'Load data for evaluation!')
            kps, skts, c2ws, cam_idxs, focals, bones = eval_bullettime_kps(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    centers=centers_n,
                                                                    **render_data)
        elif args.generate_skeleton_overlay:
            print(f'Load data to overlay skeleton on GT images.')
            _, _, _, _, _, _ = generate_skeleton_overlay(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    centers=centers_n,
                                                                    **render_data)
        
        elif args.psnr_images:
            print(f'Load data to generate images.')
            kps, skts, c2ws, cam_idxs, focals, bones = generate_psnr_imgs(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    centers=centers_n,
                                                                    **render_data)
        else:
            print(f'Load data for bullet time effect!')
            if args.switch_cam:
                '''only c2ws changes'''
                v_focals = focals.copy()
                kps, skts, c2ws, cam_idxs, focals, bones, root = load_bullettime(data_h5, c2ws, focals,
                                                                        rest_pose, pose_keys,
                                                                        centers=centers_n,
                                                                        **render_data)
                # ipdb.set_trace()
                _, _, c2ws_virt, _, _, _, _ = load_bullettime(data_h5, v_cam, v_focals,
                                                                        rest_pose, pose_keys,
                                                                        centers=centers_n,
                                                                        **render_data)
                # combined real and virt rendering
                #ipdb.set_trace()
                c2ws = np.concatenate([c2ws, c2ws_virt])

            else:
                kps, skts, c2ws, cam_idxs, focals, bones, root = load_bullettime(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    centers=centers_n,
                                                                    **render_data)

        #import ipdb; ipdb.set_trace()
    elif args.render_type == 'poserot':

        kps, skts, bones, c2ws, cam_idxs, focals = load_pose_rotate(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    **render_data)
    elif args.render_type == 'interpolate':
        print(f'Load data for pose interpolation!')
        kps, skts, c2ws, cam_idxs, focals = load_interpolate(data_h5, c2ws, focals,
                                                             rest_pose, pose_keys,
                                                             **render_data)

    elif args.render_type == 'animate':
        print(f'Load data for pose animate!')
        kps, skts, c2ws, cam_idxs, focals = load_animate(data_h5, c2ws, focals,
                                                         rest_pose, pose_keys,
                                                          **render_data)
    elif args.render_type == 'bubble':
        print(f'Load data for bubble camera!')
        kps, skts, c2ws, cam_idxs, focals = load_bubble(data_h5, c2ws, focals,
                                                        rest_pose, pose_keys,
                                                        **render_data)
    elif args.render_type == 'selected':
        selected_idxs = np.array(args.selected_idxs)
        kps, skts, c2ws, cam_idxs, focals = load_selected(data_h5, c2ws, focals,
                                                          rest_pose, pose_keys,
                                                          selected_idxs, **render_data)
    elif args.render_type.startswith('val'):
        print(f'Load data for valditation!')
        is_surreal = ('surreal_val' in data_h5) or (args.entry == 'hard' and 'surreal' in data_h5)
        is_neuralbody = 'neuralbody' in args.dataset
        if is_neuralbody:
            import h5py
            nb_h5 = h5py.File(data_h5, 'r')
            c2ws = c2ws[nb_h5['img_pose_indices']]
            focals = focals[nb_h5['img_pose_indices']]
            render_data['selected_idxs'] = np.arange(len(c2ws))
            nb_h5.close()
        kps, skts, c2ws, cam_idxs, focals, bones = load_retarget(data_h5, c2ws, focals,
                                                                 rest_pose, pose_keys,
                                                                 is_surreal=is_surreal,
                                                                 is_neuralbody=is_neuralbody,
                                                                 **render_data)
        if not is_surreal:
            try:
                bg_imgs = dd.io.load(data_h5, '/bkgds').astype(np.float32).reshape(-1, H, W, 3) / 255.
                bg_indices = dd.io.load(data_h5, '/bkgd_idxs')[cam_idxs].astype(np.int64)
            except:
                bg_imgs = np.zeros((1, H, W, 3), dtype=np.float32)
                bg_indices = np.zeros((len(cam_idxs),), dtype=np.int32)
            cam_idxs = cam_idxs * 0 -1

    elif args.render_type == 'correction':
        print(f'Load data for visualizing correction!')
        kps, skts, c2ws, cam_idxs, focals = load_correction(data_h5, c2ws, focals,
                                                            rest_pose, pose_keys,
                                                            **render_data)
        bg_imgs = dd.io.load(data_h5, '/bkgds').astype(np.float32).reshape(-1, H, W, 3) / 255.
        bg_indices = dd.io.load(data_h5, '/bkgd_idxs')[cam_idxs].astype(np.int64)

    else:
        raise NotImplementedError(f'render type {args.render_type} is not implemented!')


    #import ipdb; ipdb.set_trace()
    # added for rendering background image
    if not args.white_bkgd:
        bg_imgs = dd.io.load(data_h5, '/bkgds').astype(np.float32).reshape(-1, H, W, 3) / 255.
        bg_indices = dd.io.load(data_h5, '/bkgd_idxs')[cam_idxs].astype(np.int64)


    gt_paths, gt_mask_paths = None, None
    is_gt_paths = True
    if args.save_gt or args.eval:
        if args.render_type == 'retarget':
            extract_idxs = cam_idxs
        elif 'selected_idxs' in render_data:
            extract_idxs = render_data['selected_idxs']
        else:
            extract_idxs = selected_idxs
        # unique without sort
        unique_idxs = np.unique(extract_idxs, return_index=True)[1]
        extract_idxs = np.array([extract_idxs[idx] for idx in sorted(unique_idxs)])
        try:
            gt_to_mask_map = init_catalog(args)[args.dataset].get('gt_to_mask_map', ('', ''))
            gt_paths = np.array(dd.io.load(data_h5, '/img_path'))[extract_idxs]
            gt_prefix = GT_PREFIXES[args.dataset]
            gt_paths = [os.path.join(gt_prefix, gt_path) for gt_path in gt_paths]
            gt_mask_paths = [p.replace(*gt_to_mask_map) for p in gt_paths]
        except:
            print('gt path does not exist, grab images directly!')
            is_gt_paths = False
            # gt_paths = dd.io.load(data_h5, '/imgs')[extract_idxs]
            # gt_mask_paths = dd.io.load(data_h5, '/masks')[extract_idxs]

            gt_paths = dd.io.load(data_h5, '/imgs', sel=dd.aslice[extract_idxs, ...])
            gt_mask_paths = dd.io.load(data_h5, '/masks', sel=dd.aslice[extract_idxs, ...])

    # handle special stuff
    if args.selected_framecode is not None:
        cam_idxs[:] = args.selected_framecode

    # perfcap have a peculiar camera setting
    if args.dataset == 'perfcap':
        c2ws[..., :3, -1] /= 1.05

    subject_idxs = None
    if nerf_args.nerf_type.startswith('minerf'):
        subject_idxs = (np.ones((len(kps),)) * args.subject_idx).astype(np.int64)

    ret_dict = {'kp': kps, 'skts': skts, 'render_poses': c2ws,
                'cams': cam_idxs if opt_framecode else None,
                'hwf': (H, W, focals),
                'bones': bones,
                'bg_imgs': bg_imgs,
                'bg_indices': bg_indices,
                'subject_idxs': subject_idxs,
                #'centers': centers, # May 4 addition
                'root':root
                }

    gt_dict = {'gt_paths': gt_paths,
               'gt_mask_paths': gt_mask_paths,
               'is_gt_paths': is_gt_paths,
               'bg_imgs': bg_imgs,
               'bg_indices': bg_indices}

    return ret_dict, gt_dict, render_data['selected_idxs']

def init_catalog(args, n_bullet=3):
    n_bullet = args.n_bullet
    bullet_ang = args.bullet_ang
    #import ipdb; ipdb.set_trace()

    RenderCatalog = {
        'h36m': None,
        'surreal': None,
        'mirror': None,
        'perfcap': None,
        'mixamo': None,
        '3dhp': None,
    }

    def load_idxs(path):
        if not os.path.exists(path):
            print(f'Index file {path} does not exist.')
            return []
        return np.load(path)
    def set_dict(selected_idxs, **kwargs):
        return {'selected_idxs': np.array(selected_idxs), **kwargs}

    # H36M
    s9_idx = [121, 500, 1000, 1059, 1300, 1600, 1815, 2400, 3014, 3702, 4980]
    h36m_s9 = {
        'data_h5': 'data/h36m/S9_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/h36m/s9_sub64_500k.tar',
        'retarget': set_dict(s9_idx, length=5),
        'bullet': set_dict([0],#s9_idx,
                           n_bullet=n_bullet, undo_rot=False,
                           center_cam=True),
        'interpolate': set_dict(s9_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'correction': set_dict(load_idxs('data/h36m/S9_top50_refined.npy')[:1], n_step=30),
        'animate': set_dict([1000, 1059, 2400], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([17,19,21,23])),
        'bubble': set_dict(s9_idx, n_step=30),
        'poserot': set_dict(np.array([1000])),
        'val': set_dict(load_idxs('data/h36m/S9_val_idxs.npy'), length=1, skip=1),
    }
    s11_idx = [213, 656, 904, 1559, 1815, 2200, 2611, 2700, 3110, 3440, 3605]
    h36m_s11 = {
        'data_h5': 'data/h36m/S11_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/h36m/s11_sub64_500k.tar',
        'retarget': set_dict(s11_idx, length=5),
        'bullet': set_dict(s11_idx, n_bullet=n_bullet, undo_rot=True,
                           center_cam=True),
        'interpolate': set_dict(s11_idx, n_step=10, undo_rot=True,
                                center_cam=True),

        'correction': set_dict(load_idxs('data/h36m/S11_top50_refined.npy')[:1], n_step=30),
        'animate': set_dict([2507, 700, 900], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([3,6,9,12,15,16,18])),
        'bubble': set_dict(s11_idx, n_step=30),
        'val': set_dict(load_idxs('data/h36m/S11_val_idxs.npy'), length=1, skip=1),
    }

    # MIRROR
    #import ipdb; ipdb.set_trace()
    comb = args.data_path.split("/")[-2]
    view = int(comb.split("_cam_")[1])

    

    # if view ==3:
    #     easy_idx = [0, 473, 467, 1467]
    # elif view ==2:
    #     easy_idx = [30, 400, 700, 1100]
    # elif view ==5:
    #     easy_idx = [50, 200, 1000, 1200]
    # else:

    #     import ipdb; ipdb.set_trace()

    # [1,209,212,250,280,340,368,369,406,428,438,993]
    # [449,624,644,680,705,746,998,1170]

    #import ipdb; ipdb.set_trace()
    #rebuttal_tim = np.arange(800,1178) #[992,1027,1041,1087,1088,1133,1134,1172,1175] #[449,624,644,680,705,746,998,1170,1,209,212,250,280,340,368,369,406,428,438,993]  #np.arange(0, 500) #[500] #1177, 814]
    if args.evaluate_pose:
        easy_idx = np.arange(0, args.train_len)
        args.selected_idxs = easy_idx

    elif args.psnr_images:
        
        import pickle#5 as pickle
        with open(f'{args.data_path}/calib{view}_20000iter_May_11.pickle', 'rb') as handle:
            from_pickle = pickle.load(handle)

        # spin_pickle = f"{args.data_path}/output.pkl"
        # import deepdish as dd
        # import pickle as pkl
        # if spin_pickle.endswith(".pkl"):
        #     spin_raw_data = pkl.load(open(spin_pickle, "rb"))
        # else:
        #     spin_raw_data = dd.io.load(spin_pickle)

        mirrr_raw_idxs = from_pickle['chosen_frames'][:args.train_len]

        #""" please update code: hard-coded vanilla indices here for now"""
        # vanilla_raw_idxs = np.arange(1800) #spin_raw_data['valid_idxs']
        # common_eval_pts = list(set(mirrr_raw_idxs).intersection(set(vanilla_raw_idxs)))

        # mirr_sel_idxs = list(map(lambda x: mirrr_raw_idxs.index(x), common_eval_pts))
        # mirr_sel_idxs = mirr_sel_idxs[:args.train_len]

        # 20th images among common subset
        #mirr_sel_idxs = mirrr_raw_idxs[:args.train_len]
        #n = len(mirr_sel_idxs)
        #sel_idxs = mirr_sel_idxs[::20]
        
        n = len(mirrr_raw_idxs)
        offset_indices = []

        for i in range(0,n,20):
            index = mirrr_raw_idxs.index(i)
            offset_indices.append(index)

        easy_idx = offset_indices#[-1:]
        args.selected_idxs = easy_idx 
        # pick a subset that is common to both
        #import ipdb; ipdb.set_trace()

    elif generate_skeleton_overlay:
        # mirrr_raw_idxs = from_pickle['chosen_frames'][:args.train_len]

        # #""" please update code: hard-coded vanilla indices here for now"""
        # n = len(mirrr_raw_idxs)
        # offset_indices = []

        # for i in range(0,n,20):
        #     index = mirrr_raw_idxs.index(i)
        #     offset_indices.append(index)

        # easy_idx = offset_indices[-1:]

        print("did you consider offset in selected idxs?")
        time.sleep(3)
        easy_idx = args.selected_idxs
        args.selected_idxs = easy_idx 

    else: #render
        easy_idx = args.selected_idxs #[0] #rebuttal_tim #[406,466,340,600,900,814] # #[0, 465, 473, 467, 1467] # [10, 70, 350, 420, 490, 910, 980, 1050] #np.arange(0, args.train_len)
    
    # ipdb.set_trace()
    mirror_easy = {
        'data_h5': args.data_path + '/mirror_train_h5py.h5', #'/tim_train_h5py.h5',
        'data_h5_v': args.data_path + '/v_mirror_train_h5py.h5',
        'retarget': set_dict(easy_idx, length=25, skip=2, center_kps=True),
        'bullet': set_dict(easy_idx, n_bullet=args.n_bullet, bullet_ang=args.bullet_ang),
        'bubble': set_dict(easy_idx, n_step=args.n_bubble),
        'mesh': set_dict([0], length=1, skip=1),
    }

    # h5_path =  args.data_path + '/tim_train_h5py.h5'
    # dataset = h5py.File(h5_path, 'r', swmr=True)
    # ipdb.set_trace()

    # TODO: create validation indices
    #ipdb.set_trace()
    test_val_idx = [0] #np.arange(0, 1563) #[0] # required for rendering at training time?
    mirror_val = {
        'data_h5': args.data_path + '/mirror_val_h5py.h5',
        'data_h5_v': args.data_path + '/v_mirror_val_h5py.h5',
        'val': set_dict(test_val_idx, length=1, skip=1),
        'val2': set_dict(test_val_idx, length=1, skip=1),

        # dan
        #'bullet': set_dict(test_val_idx, n_bullet=args.n_bullet),

    #     'val': set_dict(load_idxs(args.data_path + '/mirror_val_idxs.npy'), length=1, skip=1),
    #     'val2': set_dict(load_idxs(args.data_path + '/mirror_val_idxs.npy')[:300], length=1, skip=1),
    }
    
    hard_idx = [140, 210, 280, 490, 560, 630, 700, 770, 840, 910]
    mirror_hard = {
        'data_h5': args.data_path + '/mirror_train_h5py.h5', #'/tim_train_h5py.h5', #1620
        'data_h5_v': args.data_path + '/v_mirror_train_h5py.h5', #1620
        'retarget': set_dict(hard_idx, length=60, skip=5, center_kps=True),
        #'bullet': set_dict([190,  210,  230,  490,  510,  530,  790,  810,  830,  910,  930, 950, 1090, 1110, 1130],
        #                   n_bullet=n_bullet, center_kps=True, center_cam=False),
        'bubble': set_dict(hard_idx, n_step=30),
        #'val': set_dict(np.array([1200 * i + np.arange(420, 700)[::5] for i in range(0, 9, 2)]).reshape(-1), length=1, skip=1),
        'mesh': set_dict([0], length=1, skip=1),
    }

    # SURREAL
    easy_idx = [10, 70, 350, 420, 490, 910, 980, 1050]
    surreal_val = {
        'data_h5': 'data/surreal/surreal_val_h5py.h5', 
        'val': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy'), length=1, skip=1),
        'val2': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy')[:300], length=1, skip=1),
    }
    surreal_easy = {
        'data_h5': 'data/surreal/surreal_train_h5py.h5',
        'retarget': set_dict(easy_idx, length=25, skip=2, center_kps=True),
        'bullet': set_dict(easy_idx, n_bullet=n_bullet),
        'bubble': set_dict(easy_idx, n_step=30),
    }
    hard_idx = [140, 210, 280, 490, 560, 630, 700, 770, 840, 910]
    surreal_hard = {
        'data_h5': 'data/surreal/surreal_train_h5py.h5',
        'retarget': set_dict(hard_idx, length=60, skip=5, center_kps=True),
        'bullet': set_dict([190,  210,  230,  490,  510,  530,  790,  810,  830,  910,  930, 950, 1090, 1110, 1130],
                           n_bullet=n_bullet, center_kps=True, center_cam=False),
        'bubble': set_dict(hard_idx, n_step=30),
        'val': set_dict(np.array([1200 * i + np.arange(420, 700)[::5] for i in range(0, 9, 2)]).reshape(-1), length=1, skip=1),
        'mesh': set_dict([930], length=1, skip=1),
    }

    # PerfCap
    weipeng_idx = [0, 50, 100, 150, 200, 250, 300, 350, 430, 480, 560,
                   600, 630, 660, 690, 720, 760, 810, 850, 900, 950, 1030,
                   1080, 1120]
    perfcap_weipeng = {
        'data_h5': 'data/MonoPerfCap/Weipeng_outdoor/Weipeng_outdoor_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/perfcap/weipeng_tv_500k.tar',
        'retarget': set_dict(weipeng_idx, length=30, skip=2),
        'bullet': set_dict(weipeng_idx, n_bullet=n_bullet),
        'interpolate': set_dict(weipeng_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(weipeng_idx, n_step=30),
        'val': set_dict(np.arange(1151)[-230:], length=1, skip=1),
        'animate': set_dict([300, 480, 700], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([1,4,7,10,17,19,21,23])),
    }

    nadia_idx = [0, 65, 100, 125, 230, 280, 410, 560, 600, 630, 730, 770,
                 830, 910, 1010, 1040, 1070, 1100, 1285, 1370, 1450, 1495,
                 1560, 1595]
    perfcap_nadia = {
        'data_h5': 'data/MonoPerfCap/Nadia_outdoor/Nadia_outdoor_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/perfcap/nadia_tv_500k.tar',
        'retarget': set_dict(nadia_idx, length=30, skip=2),
        'bullet': set_dict(nadia_idx, n_bullet=n_bullet),
        'interpolate': set_dict(nadia_idx, n_step=10, undo_rot=True,
                                center_cam=True, center_kps=True),
        'bubble': set_dict(nadia_idx, n_step=30),
        'animate': set_dict([280, 410, 1040], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([1,2,4,5,7,8,10,11])),
        'val': set_dict(np.arange(1635)[-327:], length=1, skip=1),
    }

    # Mixamo
    james_idx = [20, 78, 138, 118, 1149, 333, 3401, 2221, 4544]
    mixamo_james = {
        'data_h5': 'data/mixamo/James_processed_h5py.h5',
        'idx_map': load_idxs('data/mixamo/James_selected.npy'),
        'refined': 'neurips21_ckpt/trained/ours/mixamo/james_tv_500k.tar',
        'retarget': set_dict(james_idx, length=30, skip=2),
        'bullet': set_dict(james_idx, n_bullet=n_bullet, center_cam=True, center_kps=True),
        'interpolate': set_dict(james_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(james_idx, n_step=30),
        'animate': set_dict([3401, 1149, 4544], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([18,19,20,21,22,23])),
        'mesh': set_dict([20, 78], length=1, undo_rot=False),
    }

    archer_idx = [158, 672, 374, 414, 1886, 2586, 2797, 4147, 4465]
    mixamo_archer = {
        'data_h5': 'data/mixamo/Archer_processed_h5py.h5',
        'idx_map': load_idxs('data/mixamo/Archer_selected.npy'),
        'refined': 'neurips21_ckpt/trained/ours/mixamo/archer_tv_500k.tar',
        'retarget': set_dict(archer_idx, length=30, skip=2),
        'bullet': set_dict(archer_idx, n_bullet=n_bullet, center_cam=True, center_kps=True),
        'interpolate': set_dict(archer_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(archer_idx, n_step=30),
        'animate': set_dict([1886, 2586, 4465], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([18,19,20,21,22,23])),
    }

    # NeuralBody
    nb_subjects = ['315', '377', '386', '387', '390', '392', '393', '394']
    # TODO: hard-coded: 6 views
    nb_idxs = np.arange(len(np.concatenate([np.arange(1, 31), np.arange(400, 601)])) * 6)
    nb_dict = lambda subject: {'data_h5': f'data/zju_mocap/{subject}_test_h5py.h5',
                               'val': set_dict(nb_idxs, length=1, skip=1)}

    RenderCatalog['h36m'] = {
        'S9': h36m_s9,
        'S11': h36m_s11,
        'gt_to_mask_map': ('imageSequence', 'Mask'),
    }
    RenderCatalog['surreal'] = {
        'val': surreal_val,
        'easy': surreal_easy,
        'hard': surreal_hard,
    }
    RenderCatalog['mirror'] = {
        'val': mirror_val,
        'easy': mirror_easy,
        'hard': mirror_hard,
    }
    RenderCatalog['perfcap'] = {
        'weipeng': perfcap_weipeng,
        'nadia': perfcap_nadia,
        'gt_to_mask_map': ('images', 'masks'),
    }
    RenderCatalog['mixamo'] = {
        'james': mixamo_james,
        'archer': mixamo_archer,
    }
    RenderCatalog['neuralbody'] = {
        f'{subject}': nb_dict(subject) for subject in nb_subjects
    }

    return RenderCatalog

def find_idxs_with_map(selected_idxs, idx_map):
    if idx_map is None:
        return selected_idxs
    match_idxs = []
    for sel in selected_idxs:
        for i, m in enumerate(idx_map):
            if m == sel:
                match_idxs.append(i)
                break
    return np.array(match_idxs)

def load_correction(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, n_step=8, refined=None, center_kps=False,
                    idx_map=None):
    assert refined is not None
    c2ws = c2ws[selected_idxs]
    focals = focals[selected_idxs]

    init_kps, init_bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
    refined_kps, refined_bones = refined
    refined_kps = refined_kps[selected_idxs]
    refined_bones = refined_bones[selected_idxs]

    w = np.linspace(0, 1.0, n_step, endpoint=False).reshape(-1, 1, 1)
    interp_bones = []
    for i, (init_bone, refine_bone) in enumerate(zip(init_bones, refined_bones)):
        # interpolate from initial bone to refined bone
        interp_bone = init_bone[None] * (1-w) + refine_bone[None] * w
        interp_bones.append(interp_bone)
    interp_bones = np.concatenate(interp_bones, axis=0)
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in interp_bones])
    l2ws = l2ws.reshape(len(selected_idxs), n_step, 26, 4, 4)
    l2ws[..., :3, -1] += refined_kps[:, None, :1, :]
    l2ws = l2ws.reshape(-1, 26, 4, 4)
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    c2ws = c2ws[:, None].repeat(n_step, 1).reshape(-1, 4, 4)
    focals = focals[:, None].repeat(n_step, 1).reshape(-1,2)
    cam_idxs = selected_idxs[:, None].repeat(n_step, 1).reshape(-1)

    return kps, skts, c2ws, cam_idxs, focals

def load_retarget(pose_h5, c2ws, focals, rest_pose, pose_keys,
                  selected_idxs, length, skip=1, refined=None,
                  center_kps=False, idx_map=None, is_surreal=False,
                  undo_rot=False, is_neuralbody=False):

    l = length
    if skip > 1 and l > 1:
        selected_idxs = np.concatenate([np.arange(s, min(s+l, len(c2ws)))[::skip] for s in selected_idxs])
    #selected_idxs = np.clip(selected_idxs, a_min=0, a_max=len(c2ws)-1)

    c2ws = c2ws[selected_idxs]
    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]
    cam_idxs = selected_idxs

    if refined is None:
        if not is_surreal and not is_neuralbody:
            kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        elif is_neuralbody:
            kps, bones = dd.io.load(pose_h5, pose_keys)
            # TODO: hard-coded, could be problematic
            kps = kps.reshape(-1, 1, 26, 3).repeat(6, 1).reshape(-1, 26, 3)
            bones = bones.reshape(-1, 1, 26, 3).repeat(6, 1).reshape(-1, 26, 3)
        else:
            kps, bones = dd.io.load(pose_h5, pose_keys)
            kps = kps[None].repeat(9, 0).reshape(-1, 26, 3)[selected_idxs]
            bones = bones[None].repeat(9, 0).reshape(-1, 26, 3)[selected_idxs]
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]

    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    return kps, skts, c2ws, cam_idxs, focals, bones

def load_animate(pose_h5, c2ws, focals, rest_pose, pose_keys,
                     selected_idxs, joints, n_step=10, refined=None,
                     undo_rot=False, center_cam=False,
                     center_kps=False, idx_map=None):

    # TODO: figure out how to pick c2ws more elegantly?
    c2ws = c2ws[selected_idxs]
    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]

    # TODO: Update this part to (-1,2)
    #focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
    #cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]

    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root
    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    interp_bones= []
    w = np.linspace(0, 1.0, n_step, endpoint=False).reshape(-1, 1, 1)
    for i in range(len(bones)-1):
        bone = bones[i:i+1, joints]
        next_bone = bones[i+1:i+2, joints]
        interp_bone = bone * (1 - w) + next_bone * w
        interp_bones.append(interp_bone)

    interp_bones.append(bones[-1:, joints])
    interp_bones = np.concatenate(interp_bones, axis=0)
    base_bones = bones[:1].repeat(len(interp_bones), 0).copy()
    base_bones[:, joints] = interp_bones
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in base_bones])
    l2ws[..., :3, -1] += kps[:1, :1, :].copy() # only use the first pose as reference
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    c2ws = c2ws[:1].repeat(len(kps), 0)
    focals = focals[:1].repeat(len(kps), 0)
    cam_idxs = selected_idxs[:1].repeat(len(kps), 0)
    return kps, skts, c2ws, cam_idxs, focals


def load_pose_rotate(pose_h5, c2ws, focals, rest_pose, pose_keys,
                     selected_idxs, refined=None, n_bullet=30,
                     undo_rot=False, center_cam=True, center_kps=False,
                     idx_map=None):
    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]

    rots = torch.zeros(len(bones), 4, 4)
    rots_ = axisang_to_rot(torch.tensor(bones[..., :1, :]).reshape(-1, 3))
    rots[..., :3, :3] = rots_
    rots[..., -1, -1] = 1.
    rots = rots.cpu().numpy()
    rots_y = torch.tensor(generate_bullet_time(rots[0], n_views=n_bullet//3, axis='y'))
    rots_x = torch.tensor(generate_bullet_time(rots[0], n_views=n_bullet//3, axis='x'))
    rots_z = torch.tensor(generate_bullet_time(rots[0], n_views=n_bullet//3, axis='z'))
    rots = torch.cat([rots_y, rots_x, rots_z], dim=0)

    root_rotated = rot_to_axisang(rots[..., :3, :3]).cpu().numpy()
    bones = bones.repeat(len(root_rotated), 0)
    bones[..., 0, :] = root_rotated
    c2ws = c2ws[selected_idxs].repeat(len(root_rotated), 0)
    focals = focals[selected_idxs].repeat(len(root_rotated), 0)

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    cam_idxs = selected_idxs.repeat(len(root_rotated), 0)

    return kps, skts, bones, c2ws, cam_idxs, focals

def load_interpolate(pose_h5, c2ws, focals, rest_pose, pose_keys,
                     selected_idxs, n_step=10, refined=None,
                     undo_rot=False, center_cam=False,
                     center_kps=False, idx_map=None):

    # TODO: figure out how to pick c2ws more elegantly?
    c2ws = c2ws[selected_idxs]
    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]
    #focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
    #cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]

    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root
    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    interp_bones= []
    w = np.linspace(0, 1.0, n_step, endpoint=False).reshape(-1, 1, 1)
    for i in range(len(bones)-1):
        bone = bones[i:i+1]
        next_bone = bones[i+1:i+2]
        interp_bone = bone * (1 - w) + next_bone * w
        interp_bones.append(interp_bone)
    interp_bones.append(bones[-1:])
    interp_bones = np.concatenate(interp_bones, axis=0)
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in interp_bones])
    l2ws[..., :3, -1] += kps[:1, :1, :].copy() # only use the first pose as reference
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    c2ws = c2ws[:1].repeat(len(kps), 0)
    focals = focals[:1].repeat(len(kps), 0)
    cam_idxs = selected_idxs[:1].repeat(len(kps), 0)
    return kps, skts, c2ws, cam_idxs, focals



def eval_bullettime_kps(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30, bullet_ang=360,
                    #centers=None,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None, args=None):

    # undo_rot=False; center_cam=True; center_kps=True
    center_kps = False; center_cam= False
    #import ipdb; ipdb.set_trace()

    # prepare camera
    c2ws = c2ws[selected_idxs]
    # centers = centers[selected_idxs]

    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    # prepare pose
    # TODO: hard-coded for now so we can quickly view the outcomes!
    if refined is None:
        _ = input("using initial poses")
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)
    
    cam_id = int(args.data_path.split("_cam_")[1][0])

    #import ipdb; ipdb.set_trace()
    #resp = input("do you have the choosen frames, y/n?")
    
    if True:
        print()
        print("re-using hard-coded eval indices for evaluation")
        time.sleep(3)
        
    #import ipdb; ipdb.set_trace()
    """
    # special case - evaluate step mirror initial using 100% frames (protocol 1) 
    kps_raw, _ = dd.io.load(pose_h5, pose_keys) #, sel=dd.aslice[selected_idxs, ...])

    test_pose_h5 = pose_h5.replace("train", "test")
    kps_test, _ = dd.io.load(test_pose_h5, pose_keys)

    # combined
    kps_comb = np.concatenate([kps, kps_test])
    print(f"kps_raw {kps_raw.shape} kps selected {kps.shape}, kps combined {kps_comb.shape}")
    time.sleep(3)
    """

    """
    logic: evaluate every 100th in eval set (18 in total, most possible)
    find corresponding 18th in reconstruction 
    1. would be relative to reconstruction sequence
    2. the number of evaluations would depend on the size of the reconstruction sequence
    """
    
    """these are eval indices within alphapose valid frames (also called chosen frames)"""
    if cam_id==2:
        comb = args.data_path.split("/")[-2]
        #comb="fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2" # 2022-05-16-02-10-36
        rec_eval_pts = [0, 100, 200, 300, 400, 493, 593, 693, 793, 893, 993, 1093, 1193, 1293, 1393]
        gt_eval_pts =  [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    elif cam_id==3:
        comb = args.data_path.split("/")[-2]
        #comb="23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3" # 2022-05-16-02-01-28
        rec_eval_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
        gt_eval_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]

    elif cam_id==5:
        comb = args.data_path.split("/")[-2]
        #comb="c28e8104-b416-474c-914c-c911baa8540b_cam_5" # 2022-05-16-14-42-42
        rec_eval_pts = [0, 100, 200, 288, 388, 488, 588, 688, 788, 888, 988, 1088]
        gt_eval_pts = [0, 100, 200, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
    
    elif cam_id==6:
        comb = args.data_path.split("/")[-2]
        #comb="ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6" # 2022-05-16-02-23-39
        rec_eval_pts = [0, 100, 200, 300, 400, 500, 599, 699, 799, 899, 999, 1099, 1199, 1299, 1399, 1499, 1599, 1699]
        gt_eval_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
    
    elif cam_id==7:
        comb = args.data_path.split("/")[-2]    
        #comb="261970f0-e705-4546-a957-b719526cbc4a_cam_7" # 2022-05-16-14-12-18
        rec_eval_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1636]
        gt_eval_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700]
 
    else:
        print("what are your eval idx?")
        ipdb.set_trace()
    
    
    """ 
    print("******Protocol I: special case ******")
    eval_opt_kp(kps_comb, comb, rec_eval_pts, gt_eval_pts, args=args)
    """

    print("******Protocol II: ******")
    eval_opt_kp(kps, comb, rec_eval_pts, gt_eval_pts, args=args)
    
    #name = "pose_opt_eval_may16"
    #np.save(f"/scratch/dajisafe/smpl/mirror_project_dir/authors_eval_data/temp_dir/cam_trajectory_{name}.npy", np.array(c2ws))
    #np.save(f"/scratch/dajisafe/smpl/mirror_project_dir/authors_eval_data/temp_dir/kps_{name}.npy", np.array(kps))

    print("pose evaluation complete...")
    import ipdb; ipdb.set_trace()

def generate_skeleton_overlay(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30, bullet_ang=360,
                    centers=None,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None, args=None):

    import ipdb
    comb = args.data_path.split("/")[-2]
    view = comb.split("_cam_")[1]

    """we need the global root position for 3D skeleton overlay, so no root-centering, 
    except you center the camera using the root location as well"""

    undo_rot=False; center_cam=False; center_kps=True
    #center_kps = False; center_cam= False

    # prepare camera
    # c2ws = c2ws[selected_idxs]
    # centers = centers[selected_idxs]

    try:
        #ipdb.set_trace()
        c2ws = c2ws[selected_idxs]

        if isinstance(focals, float):
            focals = np.array([focals] * len(selected_idxs))
        else:
            focals = focals[selected_idxs]

        if isinstance(centers, float):
            centers = np.array([centers] * len(selected_idxs))
        else:
            if len(centers.shape) > 2:
                centers = centers[0][selected_idxs]
            else:
                centers = centers[selected_idxs]

        #ipdb.set_trace()
    except:
        # import h5py
        ipdb.set_trace()


    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    # prepare pose
    # TODO: hard-coded for now so we can quickly view the outcomes!
    # if refined is None:
    if not args.render_refined:
        print("using *** initial optimized mirror poses ***")
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        if args.switch_cam:
            print("are you using the right data path for pose_h5?")
            ipdb.set_trace()
    else:
        print("using *** refined mirror poses ***")
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)
    

    #import ipdb; ipdb.set_trace()
    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        # move kps to 0 and cam to -root
        kps[..., :, :] -= root
        c2ws[..., :3, -1] -= root.reshape(-1,3)

    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    #import ipdb; ipdb.set_trace()
    c2ws = generate_bullet_time(c2ws, n_views=n_bullet, bullet_ang=bullet_ang).transpose(1, 0, 2, 3).reshape(-1, 4, 4)
    

    # if isinstance(focals, float):
    #     focals = np.array([focals] * len(selected_idxs))
    # else:
    #     focals = focals[selected_idxs]

    #import ipdb; ipdb.set_trace()
    #focals = focals[:, None].repeat(n_bullet, 1).reshape(-1,2)
    if len(focals.shape) == 2 and focals.shape[1] == 2:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
    # print(f"focals {focals[0:1]}")

    if len(centers.shape) == 2 and centers.shape[1] == 2:
        centers = centers[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        centers = centers[:, None].repeat(n_bullet, 1).reshape(-1)

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)


    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    #import ipdb; ipdb.set_trace()
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    # prepare GT images
    gt_imgs = dd.io.load(pose_h5, '/imgs', sel=dd.aslice[selected_idxs, ...])
    centers = dd.io.load(pose_h5, '/centers', sel=dd.aslice[selected_idxs, ...])
    #gt_masks = dd.io.load(pose_h5, '/masks', sel=dd.aslice[selected_idxs, ...])
    img_shape = dd.io.load(pose_h5, '/img_shape')
    _,H,W,C = img_shape.tolist()
    #import ipdb; ipdb.set_trace()
    
    # expand shape for repeat
    kps = kps[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 3)
    skts = skts[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 4, 4)
    #centers = centers[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, 2)
    #import ipdb; ipdb.set_trace()

    """begin overlay"""
    
    for i in trange(len(selected_idxs)):
        idx = selected_idxs[i]
        from core.utils.skeleton_utils import draw_skeletons_3d
        # import imageio

        #img = dd.io.load(pose_h5, '/imgs', sel=dd.aslice[idx, ...])
        #center = dd.io.load(pose_h5, '/centers', sel=dd.aslice[idx, ...])
        img = gt_imgs[i].reshape(H, W, 3)
        #msk = gt_masks['sampling_masks'][idx].reshape(H, W, 1)
        img = img.copy() #* msk.copy()
        skel_img = draw_skeletons_3d(img[None], kps[i:i+1], c2ws[i][None], H, W, focals[i][None], centers[i][None])
        #skel_img = draw_skeletons_3d(img[None], kps[:1], c2w[None], H, W, focal[None], center[None])

        # ipdb.set_trace()
        if not args.render_refined:
            save_dir = f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/initial_overlay/mirror/{view}/init_2023_08_06/"
        else:
            save_dir = f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/refined_overlay/mirror/{view}/ref_2023_08_06"

        os.makedirs(save_dir, exist_ok=True)
        imageio.imwrite(f'{save_dir}/{idx:05d}.png', skel_img.reshape(H, W, 3))
    print(cam_idxs)

    print ("overlay complete...")
    ipdb.set_trace()

    #plt.savefig(f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/initial_overlay/init_SPIN/5/pixel_loc.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)

    return kps, skts, c2ws, cam_idxs, focals, bones

def generate_psnr_imgs(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30, bullet_ang=360,
                    #centers=None,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None, args=None):


    """we need the global root position for PSNR images, so no root-centering"""
    center_kps = False; center_cam= False
    #import ipdb; ipdb.set_trace()

    # prepare camera
    c2ws = c2ws[selected_idxs]
    # centers = centers[selected_idxs]

    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    # prepare pose
    # TODO: hard-coded for now so we can quickly view the outcomes!
    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        if args.switch_cam:
            print("are you using the right data path for pose_h5?")
            ipdb.set_trace()
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)
    

    #import ipdb; ipdb.set_trace()
    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        # move kps to 0 and cam to -root
        kps[..., :, :] -= root
        c2ws[..., :3, -1] -= root.reshape(-1,3)

    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    #import ipdb; ipdb.set_trace()
    c2ws = generate_bullet_time(c2ws, n_views=n_bullet, bullet_ang=bullet_ang).transpose(1, 0, 2, 3).reshape(-1, 4, 4)
    

    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]

    #import ipdb; ipdb.set_trace()
    #focals = focals[:, None].repeat(n_bullet, 1).reshape(-1,2)
    if len(focals.shape) == 2 and focals.shape[1] == 2:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
    print(f"focals {focals[0:1]}")

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    #import ipdb; ipdb.set_trace()
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    
    # expand shape for repeat
    kps = kps[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 3)
    skts = skts[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 4, 4)
    #centers = centers[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, 2)
    #import ipdb; ipdb.set_trace()

   
    return kps, skts, c2ws, cam_idxs, focals, bones#, root#, centers

def load_bullettime(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30, bullet_ang=360,
                    #centers=None,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None, args=None):

    '''center kps or center cam'''
    undo_rot=False; center_cam=False; center_kps=True
    #import ipdb; ipdb.set_trace()

    # prepare camera
    c2ws = c2ws[selected_idxs]
    # centers = centers[selected_idxs]

    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    # prepare pose
    # TODO: hard-coded for now so we can quickly view the outcomes!
    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        if args.switch_cam:
            print("are you using the right data path for pose_h5?")
            ipdb.set_trace()
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)
    

    #import ipdb; ipdb.set_trace()
    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        # move kps to 0 and cam to -root
        kps[..., :, :] -= root
        c2ws[..., :3, -1] -= root.reshape(-1,3)

    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    #import ipdb; ipdb.set_trace()
    c2ws = generate_bullet_time(c2ws, n_views=n_bullet, bullet_ang=bullet_ang).transpose(1, 0, 2, 3).reshape(-1, 4, 4)
    

    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]

    #import ipdb; ipdb.set_trace()
    #focals = focals[:, None].repeat(n_bullet, 1).reshape(-1,2)
    if len(focals.shape) == 2 and focals.shape[1] == 2:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
    print(f"focals {focals[0:1]}")

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    #import ipdb; ipdb.set_trace()
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    
    # expand shape for repeat
    kps = kps[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 3)
    skts = skts[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 4, 4)
    #centers = centers[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, 2)
    #import ipdb; ipdb.set_trace()

   
    return kps, skts, c2ws, cam_idxs, focals, bones, root#, centers

def load_selected(pose_h5, c2ws, focals, rest_pose, pose_keys,
                  selected_idxs, refined=None, idx_map=None):

    # get cameras
    c2ws = c2ws[selected_idxs]
    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]

    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    return kps, skts, c2ws, cam_idxs, focals

def load_bubble(pose_h5, c2ws, focals, rest_pose, pose_keys,
                selected_idxs, x_deg=15., y_deg=25., z_t=0.1,
                refined=None, n_step=5, center_kps=True, idx_map=None, args=None):
    x_rad = x_deg * np.pi / 180.
    y_rad = y_deg * np.pi / 180.

    # load poses
    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_step, 1).reshape(-1)

    # center camera
    c2ws = c2ws[selected_idxs]
    shift_x = c2ws[..., 0, -1].copy()
    shift_y = c2ws[..., 1, -1].copy()
    c2ws[..., :2, -1] = 0.

    # shift camera
    root = kps[..., :1, :].copy()
    c2ws[..., :3, -1] -= root.reshape(-1,3)

    z_t = z_t * c2ws[0, 2, -1]

    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]
    focals = focals[:, None].repeat(n_step, 1).reshape(-1,2)

    motions = np.linspace(0., 2 * np.pi, n_step, endpoint=True)
    x_motions = (np.cos(motions) - 1.) * x_rad
    y_motions = np.sin(motions) * y_rad
    z_trans = (np.sin(motions) + 1.) * z_t
    cam_motions = []
    for x_motion, y_motion in zip(x_motions, y_motions):
        cam_motion = rotate_x(x_motion) @ rotate_y(y_motion)
        cam_motions.append(cam_motion)

    bubble_c2ws = []
    for c2w in c2ws:
        bubbles = []
        for cam_motion, z_tran in zip(cam_motions, z_trans):
            c = c2w.copy()
            c[2, -1] += z_tran
            bubbles.append(cam_motion @ c)
        bubble_c2ws.append(bubbles)

    

    # undo rot
    #bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    # center kps
    root = kps[..., :1, :].copy()
    kps[..., :, :] -= root

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    # expand shape for repeat
    kps = kps[:, None].repeat(n_step, 1).reshape(len(selected_idxs) * n_step, -1, 3)
    skts = skts[:, None].repeat(n_step, 1).reshape(len(selected_idxs) * n_step, -1, 4, 4)

    c2ws = np.array(bubble_c2ws).reshape(-1, 4, 4)

    #import ipdb; ipdb.set_trace()
    return kps, skts, c2ws, cam_idxs, focals


def to_tensors(data_dict):
    tensor_dict = {}
    for k in data_dict:
        if k == "centers": # added May 4 
            tensor_dict[k] = data_dict[k]
        elif isinstance(data_dict[k], np.ndarray):
            if k == 'bg_indices' or k == 'subject_idxs':
                tensor_dict[k] = torch.tensor(data_dict[k]).long()
            else:
                tensor_dict[k] = torch.tensor(data_dict[k]).float()
        elif k == 'hwf' or k == 'cams' or k == 'bones': 
            tensor_dict[k] = data_dict[k]
        elif data_dict[k] is None:
            tensor_dict[k] = None
        else:
            raise NotImplementedError(f"{k}: only nparray and hwf are handled now!")
    return tensor_dict

def evaluate_metric(rgbs, accs, bboxes, gt_dict, basedir, time=""):
    '''
    Always evaluate in the box
    '''
    gt_paths = gt_dict['gt_paths']
    mask_paths = gt_dict['gt_mask_paths']
    is_gt_paths = gt_dict['is_gt_paths']

    bg_imgs = gt_dict['bg_imgs']
    bg_indices = gt_dict['bg_indices']

    psnrs, ssims = [], []
    fg_psnrs, fg_ssims = [], []
    ssim_eval = SSIM(size_average=False)

    for i, (rgb, acc, bbox, gt_path) in enumerate(zip(rgbs, accs, bboxes, gt_paths)):

        # track the last 150 results
        if (i + 1) % 150 == 0:
            import ipdb; ipdb.set_trace()
            np.save(os.path.join(basedir, time, 'scores.npy'),
                    {'psnr': psnrs, 'ssim': ssims, 'fg_psnr': fg_psnrs, 'fg_ssim': fg_ssims},
                     allow_pickle=True)

        tl, br = bbox
        if is_gt_paths:
            gt_img = imageio.imread(gt_path) / 255.
            gt_mask = None
            if mask_paths is not None:
                gt_mask = imageio.imread(mask_paths[i])
                gt_mask[gt_mask < 128] = 0
                gt_mask[gt_mask >= 128] = 1
                mask_cropped = gt_mask[tl[1]:br[1], tl[0]:br[0]].astype(np.float32)
        else:
            # TODO: hard-coded, fix this.
            gt_img = gt_path.reshape(*rgb.shape) / 255.
            gt_mask = mask_paths[i].reshape(*rgb.shape[:2], 1) if mask_paths is not None else None

        # h36m special case
        if gt_img.shape[0] == 1002:
            gt_img = gt_img[1:-1]

        if gt_mask is not None:
            if gt_mask.shape[0] == 1002:
                gt_mask = gt_mask[1:-1]
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask[..., :1]
            elif len(gt_mask.shape) == 2:
                gt_mask = gt_mask[:, :, None]
            if bg_imgs is not None:
                bg_img = bg_imgs[bg_indices[i]]
                gt_img = gt_img * gt_mask + (1.-gt_mask) * bg_img

            mask_cropped = gt_mask[tl[1]:br[1], tl[0]:br[0]].astype(np.float32)
            if mask_cropped.sum() < 1:
                print(f"frame {i} is empty, skip")
                continue

        gt_cropped = gt_img[tl[1]:br[1], tl[0]:br[0]].astype(np.float32)
        rgb_cropped = rgb[tl[1]:br[1], tl[0]:br[0]]

        se = np.square(gt_cropped - rgb_cropped)
        box_psnr = -10. * np.log10(se.mean())

        rgb_tensor = torch.tensor(rgb_cropped[None]).permute(0, 3, 1, 2)
        gt_tensor = torch.tensor(gt_cropped[None]).permute(0, 3, 1, 2)
        box_ssim = ssim_eval(rgb_tensor, gt_tensor).permute(0, 2, 3, 1).cpu().numpy()

        ssims.append(box_ssim.mean())
        psnrs.append(box_psnr)

        if gt_mask is not None:
            """se is the error at the size of the bbox * mask cropped at the size of the bbox -> error within the foreground"""
            denom = (mask_cropped.sum() * 3.)
            masked_mse = (se * mask_cropped).sum() / denom
            fg_psnr = -10. * np.log10(masked_mse)
            fg_ssim = (box_ssim[0] * mask_cropped).sum() / denom

            fg_ssims.append(fg_ssim.mean())
            fg_psnrs.append(fg_psnr)

    score_dict = {'psnr': psnrs, 'ssim': ssims, 'fg_psnr': fg_psnrs, 'fg_ssim': fg_ssims}

    np.save(os.path.join(basedir, time, 'scores.npy'), score_dict, allow_pickle=True)

    with open(os.path.join(basedir, time, 'score_final.txt'), 'w') as f:
        for k in score_dict:
            avg = np.mean(score_dict[k])
            f.write(f'{k}: {avg}\n')

@torch.no_grad()
def render_mesh(basedir, render_kwargs, tensor_data, chunk=1024, radius=1.80,
                res=255, threshold=10., time=""):
    import mcubes, trimesh
    ray_caster = render_kwargs['ray_caster']

    os.makedirs(os.path.join(basedir, time, 'meshes'), exist_ok=True)
    kps, skts, bones = tensor_data['kp'], tensor_data['skts'], tensor_data['bones']
    v_t_tuples = []
    for i in range(len(kps)):
        raw_d = ray_caster(kps=kps[i:i+1], skts=skts[i:i+1], bones=bones[i:i+1],
                           radius=radius, render_kwargs=render_kwargs['preproc_kwargs'],
                           res=res, netchunk=chunk, fwd_type='mesh')
        sigma = np.maximum(raw_d.cpu().numpy(), 0)
        vertices, triangles = mcubes.marching_cubes(sigma, threshold)
        mesh = trimesh.Trimesh(vertices / res - .5, triangles)
        mesh.export(os.path.join(basedir, time, 'meshes', f'{i:03d}.ply'))

def run_render():
    args = config_parser().parse_args()

    # check
    #import ipdb; ipdb.set_trace()
    # time = datetime.datetime.now().strftime("%Y-%m-%d-%H") # ("%Y-%m-%d-%H-%M-%S")

    if args.render_folder == '':
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # ("%Y-%m-%d-%H-%M-%S")
        print(f"new render folder: {time}")
    else:
        time = args.render_folder
        print(f"existing render folder: {time}")

    comb = args.data_path.split("/")[-2]
    view = comb.split("_cam_")[1]
    print(f"camera: {view} comb: {comb}")

    
    # update data_path
    if args.dataset== 'mirror':
        GT_PREFIXES[args.dataset] = args.data_path
    

    # parse nerf model args
    nerf_args = txt_to_argstring(args.nerf_args)
    #import ipdb; ipdb.set_trace()
    nerf_args, unknown_args = nerf_config_parser().parse_known_args(nerf_args)
    print(f'UNKNOWN ARGS: {unknown_args}')

    # update with parameters from run_render directly
    nerf_args.train_size =  args.train_size
    nerf_args.data_size = args.data_size
    # nerf_args.randomize_view_dir = args.randomize_view_dir

    # load nerf model
    render_kwargs, poseopt_layer = load_nerf(args, nerf_args)
    

    # prepare the required data
    render_data, gt_dict, selected_idxs = load_render_data(args, nerf_args, poseopt_layer, nerf_args.opt_framecode)

    # if args.switch_cam:
    #     print("switching real camera with virtual camera")
    #     # do you need data from other view? --------------------
    #     catalog = init_catalog(args)[args.dataset][args.entry]
    #     if 'data_h5_v' in catalog: data_h5_v = catalog['data_h5_v']
    #     # (['kp', 'skts', 'render_poses', 'cams', 'hwf', 'bones', 'bg_imgs', 'bg_indices', 'subject_idxs'])

    #     A_mirr =  dd.io.load(data_h5_v, '/A_dash')
    #     # A_mirr is duplicated. select id or pick one.
    #     sel_id = catalog.get(args.render_type, {}) ['selected_idxs'] 
    #     A_mirr = A_mirr[sel_id].reshape(4,4) 

    #     # extract V cam from A mirror matrix. Do we add negative to V position?
    #     v_cam_3_4 = np.concatenate((np.linalg.inv(A_mirr[:3,:3]), -A_mirr[:3,3:4]),1)
    #     last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
    #     v_cam =  np.concatenate([v_cam_3_4, last_row], axis=0)
    #     v_cam = v_cam.reshape(1,4,4)

    #     #import ipdb; ipdb.set_trace()
    #     root = render_data['root']
    #     if root is not None:
    #         v_cam[..., :3, -1] -= root.reshape(-1,3)
    #     # --- Testing: switch real with a virtual camera
    #     render_data['render_poses'] = v_cam
        # ------------------------------------------------------

    #import ipdb; ipdb.set_trace()
    tensor_data = to_tensors(render_data)

    basedir = os.path.join(args.outputdir, args.runname)
    os.makedirs(os.path.join(basedir, time), exist_ok=True)
    if args.render_type == 'mesh':
        render_mesh(basedir, render_kwargs, tensor_data, res=args.mesh_res, chunk=nerf_args.chunk, time=time)
        return

    #import ipdb; ipdb.set_trace()
    print(f"render_path called in run_render.py")
    rgbs, _, accs, valid_idxs, bboxes = render_path(render_kwargs=render_kwargs,
                                      chunk=nerf_args.chunk//8, #added May 4
                                      ext_scale=nerf_args.ext_scale,
                                      ret_acc=True,
                                      white_bkgd=args.white_bkgd,
                                      **tensor_data,
                                      args=args)

    #ipdb.set_trace()
    
    
    if gt_dict['gt_paths'] is not None:
        if args.eval:
            #print("are you using full or raw rgb?")
            #ipdb.set_trace()
            evaluate_metric(rgbs, accs, bboxes, gt_dict, basedir, time)
            pass

        elif args.save_gt and gt_dict['is_gt_paths']:
            gt_paths = gt_dict['gt_paths']
            os.makedirs(os.path.join(basedir, time, 'gt'), exist_ok=True)
            for i in range(len(gt_paths)):
                shutil.copyfile(gt_paths[i], os.path.join(basedir, time, 'gt',  f'{i:05d}.png'))

    if args.no_save:
        return

    # take to [0, 255] for skeleton plot
    rgbs = (rgbs * 255).astype(np.uint8)
    accs = (accs * 255).astype(np.uint8)

    # if args.switch_cam:
    #     #duplicate kp and focal twice
    #     render_data['kp'] = np.tile(render_data['kp'],[2,1,1])
    #     render_data['hwf'] = (render_data['hwf'][0], 
    #                           render_data['hwf'][1],
    #                           np.tile(render_data['hwf'][2],[2,1]))

    # # overlay on body reconstruction
    # skeletons = draw_skeletons_3d(rgbs, render_data['kp'],
    #                               render_data['render_poses'],
    #                               *render_data['hwf'])

    #ipdb.set_trace()

    os.makedirs(os.path.join(basedir, time, f'image_{view}'), exist_ok=True)
    os.makedirs(os.path.join(basedir, time, f'skel_{view}'), exist_ok=True)
    os.makedirs(os.path.join(basedir, time, f'acc_{view}'), exist_ok=True)

    '''post-rendering real-virt blending'''
    
    def blend(real_imgs, acc_map_real, virt_imgs, acc_map_virt):
        # ref https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
        h, w = render_data['hwf'][0], render_data['hwf'][1]
        n_size = real_imgs.shape[0]

        bkgd_img = render_data['bg_imgs'].reshape(1, h, w, 3)
        bkgd_img = np.tile(bkgd_img, [n_size,1,1,1])

        '''overlay rendering: order - virtual first, then real'''
        # acc_map_virt means allow virt_img where virt fog exists
        I_image =  virt_imgs * acc_map_virt 
        # (1-acc_map_real) means dont allow I_image (bkgd or virt fog) where real fog exists
        I_image =  I_image * (1-acc_map_real) + real_imgs * acc_map_real 

        # set blank spots: order - virtual first, then real
        I_bgkd = bkgd_img * (1-acc_map_virt) 
        I_bgkd = I_bgkd * (1-acc_map_real) 
        I_render = I_bgkd + I_image

        # plt.imshow(I_render[0]); plt.axis("off")
        # plt.savefig(f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/imgs/to_be_del_a_img.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)
        # import ipdb; ipdb.set_trace() 
        
        #blended_img = (img1*alpha + img2*beta) + gamma
        return I_render

    half_size = rgbs.shape[0]//2

    if args.switch_cam:
        #alpha, beta, gamma = 1.0, 1.0, 0.0

        # perform operation in 0,1 space
        rgbs = rgbs/255.0
        accs = accs/255.0
        #skeletons = skeletons/255.0

        rgbs = blend(rgbs[:half_size], accs[:half_size], rgbs[half_size:], accs[half_size:])
        #skeletons = blend(skeletons[:half_size], accs[:half_size], skeletons[half_size:], accs[half_size:])
        
        accs = np.clip(accs[:half_size] +  accs[half_size:], a_min=0, a_max=1)
        

        # # move back to 0,255 space for saving image
        # rgbs = (rgbs * 255.0).astype(np.uint8)
        # accs = (accs * 255.0).astype(np.uint8)
        # skeletons = (skeletons * 255.0).astype(np.uint8)

        # plt.imshow(skeletons[0])
        # plt.axis("off")
        # plt.savefig(f"/scratch/dajisafe/smpl/A_temp_folder/A-NeRF/checkers/imgs/to_be_del_a_img.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)


        #import ipdb; ipdb.set_trace() 
        #H, W = render_data['hwf'][0], render_data['hwf'][1]
        #valid_idxs = torch.stack(valid_idxs)

        # if not args.white_bkgd:
        #     # compositing both predictions on background image
        #     bkgd_img = render_data['bg_imgs'].reshape(-1,3)
        #     # bkgd_img = np.tile(bkgd_img, [half_size,1,1])

        #     valid_idxs = valid_idxs
        #     valid_idxs_real = valid_idxs[:half_size]
        #     valid_idxs_virt = valid_idxs[half_size:]

        #     def mini_container(accs, preds_on_img, bg_img, v_idxs_real, v_idxs_virt):
        #         n_size = preds_on_img.shape[0]

        #         #bg = (1. - accs.reshape(n_size,-1,1)) * bg_img[v_idxs_real + v_idxs_virt]
        #         bg_img = np.tile(bg_img.reshape(1,H,W,3), [n_size,1,1,1])
        #         bg = (1. - accs) * bg_img
        #         #ipdb.set_trace()

        #         # # set both persons spot to 0 
        #         # bg_img[v_idxs_real.cpu().numpy(), :] = 0
        #         # bg_img[v_idxs_virt.cpu().numpy(), :] = 0

                
        #         # bg_img = np.tile(bg_img.reshape(H,W,3), [n_size,1,1,1])
        #         img_compose = preds_on_img + bg

        #         #ipdb.set_trace()

        #         plt.imshow(bg_img[0])
        #         plt.axis("off")
        #         plt.savefig(f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/imgs/to_be_del_a_img.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)
                
        #         plt.imshow(accs[0])
        #         plt.axis("off")
        #         plt.savefig(f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/imgs/to_be_del_b_img.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)


        #         plt.imshow(preds_on_img[0])
        #         plt.axis("off")
        #         plt.savefig(f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/imgs/to_be_del_c_img.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)
            

        #         plt.imshow(img_compose[0])
        #         plt.axis("off") 
        #         plt.savefig(f"/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/checkers/imgs/to_be_del_d_img.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)
            
        #         ipdb.set_trace()
        #         return img_compose


        #     get background image with black person in parallel
        #     rgbs_compose = list(map(lambda v_idxs_real, v_idxs_virt: mini_container(accs, rgbs, bkgd_img.copy(), v_idxs_real, v_idxs_virt), valid_idxs_real, valid_idxs_virt))
        #     skeletons_compose = list(map(lambda v_idxs_real, v_idxs_virt: mini_container(accs, skeletons, bkgd_img.copy(), v_idxs_real, v_idxs_virt), valid_idxs_real, valid_idxs_virt))

            
        #     #ipdb.set_trace()
        #     # stack images
        #     rgbs = np.concatenate(rgbs_compose)
        #     skeletons = np.concatenate(skeletons_compose)

        # plt.imshow(skeletons[0])
        # plt.axis("off")
        # plt.savefig(f"/scratch/dajisafe/smpl/A_temp_folder/A-NeRF/checkers/imgs/to_be_del_skel.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)


    if not args.white_bkgd:    
        '''converts from [0,1] float64bits (2^64) to [0,255] unsigned 8bits (2^8) - losses information
        due to quantization'''
        # rgbs = (rgbs * 255).astype(np.uint8)
        # accs = (accs * 255).astype(np.uint8)
        #skeletons = (skeletons * 255).astype(np.uint8)

    else: 
        '''if white background''' 
        # I convert to np.uint8 but the red skeleton becomes white. imageio converts internally, red skeleton looks okay.
        pass

        # rgbs = (rgbs * 255)
        # accs = (accs * 255)
        #skeletons = (skeletons * 255)


    #ipdb.set_trace()
    # overlay on body reconstruction
    if args.switch_cam:
        # overlay on body reconstruction
        skel_stage1 = draw_skeletons_3d(rgbs, render_data['kp'],
                                    render_data['render_poses'][:half_size],
                                    *render_data['hwf'])

        skeletons = draw_skeletons_3d(skel_stage1, render_data['kp'],
                                    render_data['render_poses'][half_size:],
                                    *render_data['hwf'])
    
    else:
        skeletons = draw_skeletons_3d(rgbs, render_data['kp'],
                                    render_data['render_poses'],
                                    *render_data['hwf'])

    #ipdb.set_trace()
    real_ids = args.selected_idxs
    for i, (rgb, acc, skel) in enumerate(zip(rgbs, accs, skeletons)):
        #rel_idx = i
        rel_idx = real_ids[i]
        #print(f"i {i}")
        '''temp addition to the filename here'''


        # # plot overlay on original plain image 
        # rel_idx = selected_idxs[i]
        # plain_img = dataset['imgs'][rel_idx].reshape(1, 1080, 1920, 3)

        #ipdb.set_trace()
        # skel_plain = draw_skeletons_3d(imgs=plain_img, skels=render_data['kp'][i:i+1],
        #                             c2ws=render_data['render_poses'][i:i+1],
        #                             H=render_data['hwf'][0], W=render_data['hwf'][1], 
        #                             focals=render_data['hwf'][2])
        #ipdb.set_trace()

        #-----------------------------------------------
        # rel_idx = i
        # plot overlay on volumetric reconstruction 
        #print(f"rgb {rgb.shape}")
        imageio.imwrite(os.path.join(basedir, time, f'image_{view}', f'{rel_idx:05d}.png'), rgb)
        imageio.imwrite(os.path.join(basedir, time, f'acc_{view}', f'{rel_idx:05d}.png'), acc)
        imageio.imwrite(os.path.join(basedir, time, f'skel_{view}', f'{rel_idx:05d}.png'), skel)
    
    #import ipdb; ipdb.set_trace()
    np.save(os.path.join(basedir, time, 'bboxes.npy'), bboxes, allow_pickle=True)
    #imageio.mimwrite(os.path.join(basedir, time, "render_rgb.mp4"), rgbs, fps=args.fps)

    # make the block size dynamic
    kargs = { 'macro_block_size': None }
    print("saving images only.....")
    #imageio.mimwrite(os.path.join(basedir, time, "render_rgb_no_skel.mp4"), rgbs, fps=args.fps,**kargs)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run_render()

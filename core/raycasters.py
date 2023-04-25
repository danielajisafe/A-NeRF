import time
import ipdb
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import grad

import numpy as np
from copy import deepcopy

from .networks import *
from .encoders import *
from .cutoff_embedder import get_embedder
from .utils.ray_utils import *
from .utils.run_nerf_helpers import *
from .utils.skeleton_utils import SMPLSkeleton, rot6d_to_axisang, axisang_to_rot, bones_to_rot, line_intersect, create_plane_updated, normalize_batch_normal


import sys; sys.path.append("/scratch/dajisafe/smpl/mirror_project_dir")
from .utils.extras import save2pickle, load_pickle

def create_raycaster(args, data_attrs, device=None, c_factor=None):
    """Instantiate NeRF's MLP model.
    """
    skel_type = data_attrs["skel_type"]
    near, far = data_attrs["near"], data_attrs["far"]
    n_framecodes = data_attrs["n_views"] if args.n_framecodes is None else args.n_framecodes

    pts_tr_fn = get_pts_tr_fn(args)
    kp_input_fn, input_dims, cutoff_dims = get_kp_input_fn(args, skel_type)
    bone_input_fn, bone_dims = get_bone_input_fn(args, skel_type)
    view_input_fn, view_dims = get_view_input_fn(args, skel_type)
    print(f'KPE: {kp_input_fn.encoder_name}, BPE: {bone_input_fn.encoder_name}, VPE: {view_input_fn.encoder_name}')

    cutoff_kwargs = {
        "cutoff": args.use_cutoff,
        "normalize_cutoff": args.normalize_cutoff,
        "cutoff_dist": args.cutoff_mm * args.ext_scale,
        "cutoff_inputs": args.cutoff_inputs,
        "opt_cutoff": args.opt_cutoff,
        "cutoff_dim": cutoff_dims,
        "dist_inputs":  not(input_dims == cutoff_dims),
        "freq_schedule": args.freq_schedule,
        "init_alpha": args.init_freq,
    }

    #import ipdb; ipdb.set_trace()
    # function to encode input (RGB/view to PE)
    embedpts_fn, input_ch_pts = None, None # only for mnerf
    dist_cutoff_kwargs = deepcopy(cutoff_kwargs)
    dist_cutoff_kwargs['cut_to_cutoff'] = args.cut_to_dist
    dist_cutoff_kwargs['shift_inputs'] = args.cutoff_shift
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed,
                                      input_dims=input_dims,
                                      skel_type=skel_type,
                                      cutoff_kwargs=dist_cutoff_kwargs)

    input_ch_bones = bone_dims
    embedbones_fn = None
    if bone_dims > 0:
        if args.cutoff_bones:
            bone_cutoff_kwargs = deepcopy(cutoff_kwargs)
            bone_cutoff_kwargs["dist_inputs"] = True
        else:
            bone_cutoff_kwargs = {"cutoff": False}

        embedbones_fn, input_ch_bones = get_embedder(args.multires_bones, args.i_embed,
                                                    input_dims=bone_dims,
                                                    skel_type=skel_type,
                                                    cutoff_kwargs=bone_cutoff_kwargs)

    input_ch_views = 0
    embeddirs_fn = None
    # no cutoff for view dir
    if args.use_viewdirs:
        if args.cutoff_viewdir:
            view_cutoff_kwargs = deepcopy(cutoff_kwargs)
            view_cutoff_kwargs["dist_inputs"] = True
        else:
            view_cutoff_kwargs = {"cutoff": False}
        view_cutoff_kwargs["cutoff_dim"] = len(skel_type.joint_trees)
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed,
                                                    input_dims=view_dims,
                                                    skel_type=skel_type,
                                                    cutoff_kwargs=view_cutoff_kwargs)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    nerf_kwargs = {'D': args.netdepth, 'W': args.netwidth,
                   'input_ch': input_ch,
                   'input_ch_bones': input_ch_bones,
                   'input_ch_views': input_ch_views,
                   'output_ch': output_ch, 'skips': skips,
                   'use_viewdirs': args.use_viewdirs,
                   'use_framecode': args.opt_framecode,
                   'framecode_ch': args.framecode_size,
                   'n_framecodes': n_framecodes,
                   'skel_type': skel_type,
                   'density_scale': args.density_scale}

    model = NeRF(**nerf_kwargs)

    # Model learned from fine-grained sampling
    model_fine = None
    if args.N_importance > 0:
        if not args.single_net:
            model_fine = NeRF(**nerf_kwargs)
        else:
            model_fine = model
    
    # from run_nerf.py
    r_color_factor = c_factor[0] if c_factor is not None else None
    v_color_factor = c_factor[1] if c_factor is not None else None

    # create ray caster
    ray_caster = RayCaster(model, embed_fn, embedbones_fn, embeddirs_fn, network_fine=model_fine,
                           joint_coords=torch.tensor(data_attrs['joint_coords']),
                           r_color_factor=r_color_factor, 
                           v_color_factor=v_color_factor,
                           single_net=args.single_net)
    print(ray_caster)
    ray_caster.state_dict()
    # add all learnable grad vars
    grad_vars = get_grad_vars(args, ray_caster)
    # ipdb.set_trace()

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        import os
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f and 'pose' not in f]

    print('Found ckpts', ckpts)
    loaded_ckpt = None

    #ipdb.set_trace()
    #if len(ckpts) > 0 and not args.no_reload:
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        start, ray_caster, optimizer, loaded_ckpt = load_ckpt_from_path(ray_caster,
                                                                        optimizer,
                                                                        ckpt_path,
                                                                        args.finetune)
        if args.finetune:
            start = 0
            print(f"set global step to {start}")

    ##########################
    preproc_kwargs = {
        'pts_tr_fn': pts_tr_fn,
        'kp_input_fn': kp_input_fn,
        'view_input_fn': view_input_fn,
        'bone_input_fn': bone_input_fn,
        'density_scale': args.density_scale,
        'density_fn': get_density_fn(args),
    }
    # copy preproc kwargs and disable peturbation for test-time rendering
    preproc_kwargs_test = {k: preproc_kwargs[k] for k in preproc_kwargs}

    render_kwargs_train = {
        'ray_caster': nn.DataParallel(ray_caster) if not args.debug else nn.DataParallel(ray_caster, device_ids=[0]),
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'ray_noise_std': args.ray_noise_std,
        'ext_scale': args.ext_scale,
        'preproc_kwargs': preproc_kwargs,
        'lindisp': args.lindisp,
        'nerf_type': args.nerf_type,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    # set properties that should be turned off during test here
    # TODO: we have to reuse some inputs during test time (e.g., kp has batch_size == 1)
    # which doesn't work for DataParallel. Any smart way to fix this?
    render_kwargs_test['ray_caster'] = ray_caster
    render_kwargs_test['preproc_kwargs'] = preproc_kwargs_test
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['ray_noise_std'] = 0.
    print(f"#parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # reset gradient
    optimizer.zero_grad()

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, loaded_ckpt

def get_grad_vars(args, ray_caster):

    network, network_fine = ray_caster.get_networks()
    embed_fn, embedbones_fn, embeddirs_fn = ray_caster.get_embed_fns()

    grad_vars = []

    def freeze_weights(pts_linears, layer):
        '''
        only works for standard NeRF now
        '''
        for i, l in enumerate(pts_linears):
            if i >= layer:
                break
            for p in l.parameters():
                p.requires_grad = False

    def get_vars(x, add=True):
        if not add or x is None:
            return []

        trainable_vars = []
        for p in x.parameters():
            if p.requires_grad:
                trainable_vars.append(p)
        return trainable_vars

    get_vars_old = lambda x, add=True: list(x.parameters()) if (x is not None) and add else []

    if args.finetune and args.fix_layer > 0:
        freeze_weights(network.pts_linears, args.fix_layer)
        freeze_weights(network_fine.pts_linears, args.fix_layer)

    if args.weight_decay is not None:
        pass
    else:
        grad_vars += get_vars(network)
        grad_vars += get_vars(network_fine, not args.single_net)
        grad_vars += get_vars(embed_fn)
        grad_vars += get_vars(embedbones_fn)
        grad_vars += get_vars(embeddirs_fn)

    if args.opt_r_color:
        grad_vars += ray_caster.r_color_factor
        if not ray_caster.r_color_factor.requires_grad:
            ipdb.set_trace()

    if args.opt_v_color:
        grad_vars += ray_caster.v_color_factor
        if not ray_caster.v_color_factor.requires_grad:
            ipdb.set_trace()

    return grad_vars

def get_density_fn(args):
    if args.density_type == 'relu':
        return F.relu
    elif args.density_type == 'softplus':
        shift = args.softplus_shift
        softplus = lambda x: F.softplus(x - shift, beta=1) #(1 + (x - shift).exp()).log()
        return softplus
    else:
        raise NotImplementedError(f'density activation {args.density_type} is undefined')

    pass

def get_pts_tr_fn(args):

    if args.pts_tr_type == 'local':
        tr_fn = WorldToLocalEncoder()
    else:
        raise NotImplementedError(f'Point transformation {args.pts_tr_type} is undefined.')
    return tr_fn


def get_kp_input_fn(args, skel_type):

    kp_input_fn = None
    cutoff_dims = len(skel_type.joint_names)
    N_joints = len(skel_type.joint_names)

    if args.kp_dist_type == 'reldist':
        kp_input_fn = RelDistEncoder(N_joints)
    elif args.kp_dist_type == 'cat':
        kp_input_fn = KPCatEncoder(N_joints)
    elif args.kp_dist_type == 'relpos':
        kp_input_fn = RelPosEncoder(N_joints)
    elif args.kp_dist_type == 'querypts':
        kp_input_fn = IdentityEncoder(1, 3)
        cutoff_dims = 3
    else:
        raise NotImplementedError(f'{args.kp_dist_type} is not implemented.')
    input_dims = kp_input_fn.dims

    return kp_input_fn, input_dims, cutoff_dims

def get_view_input_fn(args, skel_type):

    N_joints = len(skel_type.joint_names)

    if args.view_type == "relray":
        view_input_fn = VecNormEncoder(N_joints)
    elif args.view_type == "rayangle":
        view_input_fn = RayAngEncoder(N_joints)
    elif args.view_type == "world":
        view_input_fn = IdentityExpandEncoder(N_joints, 3)
    else:
        raise NotImplementedError(f'{args.view_type} is not implemented.')
    view_dims = view_input_fn.dims

    return view_input_fn, view_dims

def get_bone_input_fn(args, skel_type):

    bone_input_fn = None
    N_joints = len(skel_type.joint_names)

    if args.bone_type.startswith('proj') and args.pts_tr_type == 'bone':
        print(f'bone_type=={args.bone_type} is unnecessary when pts_tr_type==bone.' + \
              f'Use bone_type==dir instead')

    if args.bone_type == 'reldir':
        bone_input_fn = VecNormEncoder(N_joints)
    elif args.bone_type == 'axisang':
        bone_input_fn = IdentityExpandEncoder(N_joints, 3)
    else:
        raise NotImplementedError(f'{args.bone_type} bone function is not implemented')
    bone_dims = 0 if args.bone_type == 'Nope' else bone_input_fn.dims

    return bone_input_fn, bone_dims


class ContrastiveHead(nn.Module):

    def __init__(self, W=256):
        super().__init__()
        W = W * 2
        self.net = nn.Sequential(
                       nn.Linear(W//2, W//4),
                       #nn.BatchNorm1d(W//2),
                       nn.ReLU(),
                       nn.Linear(W//4, W//4),
                       #nn.BatchNorm1d(W//2),
                       nn.ReLU(),
                       nn.Linear(W//4, W//2),
                   )

    def forward(self, h):
        return self.net(h)

class RayCaster(nn.Module):

    def __init__(self, network, embed_fn, embedbones_fn, embeddirs_fn,
                 network_fine=None, joint_coords=None, single_net=False, 
                 r_color_factor=None, v_color_factor=None):
        super().__init__()

        self.network = network
        self.network_fine = network_fine
        self.embed_fn = embed_fn
        self.embedbones_fn = embedbones_fn
        self.embeddirs_fn = embeddirs_fn

        if joint_coords is not None:
            N_J = joint_coords.shape[-3]
            joint_coords = joint_coords.reshape(-1, N_J, 3, 3)
            self.register_buffer('joint_coords', joint_coords)
        
        # becomes an attribute, but one that is differentiable
        if r_color_factor is not None:
            # self.r_color_factor = r_color_factor
            # self.r_color_factor.requires_grad = True
            self.register_parameter('r_color_factor', torch.nn.Parameter(r_color_factor, requires_grad = True))

        if v_color_factor is not None:
            # self.v_color_factor = v_color_factor
            # self.v_color_factor.requires_grad = True
            self.register_parameter('v_color_factor', torch.nn.Parameter(v_color_factor, requires_grad = True))

        self.single_net = single_net

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        return self.render_rays(*args, **kwargs)


    def forward(self, *args, fwd_type='', **kwargs):
        if fwd_type == 'density':
            return self.render_pts_density(*args, **kwargs)
        elif fwd_type == 'density_color':
            return self.render_pts_density(*args, **kwargs, color=True)
        elif fwd_type == 'mesh':
            return self.render_mesh_density(*args, **kwargs)

        if not self.training:
            return self.forward_eval(*args, **kwargs)

        #import ipdb; ipdb.set_trace()
        return self.render_rays(*args, **kwargs)

    def render_rays(self,
                    ray_batch=None,
                    ray_batch_v=None,
                    idx_repeat=None,
                    pixel_loc_repeat=None,
                    pixel_loc_v_repeat=None,
                    N_samples=None,
                    kp_batch=None,
                    kp_batch_v=None,
                    skts=None,
                    cyls=None,
                    cyls_v=None,
                    A_dash=None,
                    m_normal=None,
                    avg_D=None,
                    bones=None,
                    cams=None,
                    subject_idxs=None,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    network_fine=None,
                    raw_noise_std=0.,
                    ray_noise_std=0.,
                    verbose=False,
                    ext_scale=0.001,
                    pytest=False,
                    use_mirr=None,
                    preproc_kwargs={},
                    nerf_type="nerf"):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          kp_batch: array of keypoints for calculating input points.
          kp_input_fn: function that calculate the input using keypoints for the network
          cyls: cylinder parameters for dynamic near/far plane sampling
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          network_fine: "fine" network with same spec as network_fn.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        # Step 1: prep ray data
        # Note: last dimension for ray direction needs to be normalized
        
        #N_samples=32


        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each            
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        def plot_current_pixel(dataset, image_idx, pixel_loc, type="real"):
            '''
            get current pixel data (in np.uint8)
            '''

            img_shape = dataset['img_shape'][:]
            HW = img_shape[1:3]

            color = "og" if type == "real" else "or"
            raw_img = dataset['imgs'][image_idx].reshape(*HW,3).astype(np.float32) / 255.
            plt.imshow(raw_img); plt.axis("off")
            plt.plot(*pixel_loc, color, markersize=2)
            #return plt
            plt.savefig(f"/scratch/dajisafe/smpl/A_temp_folder/A-NeRF/checkers/imgs/pixel_loc.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)
        
        del_folder = "/scratch/dajisafe/smpl/mirror_project_dir/authors_eval_data/temp_dir/to_be_deleted_pickles"
        #h5_path = './data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13/mirror_train_h5py.h5'

        if use_mirr:
            N_rays = ray_batch.shape[0]
            #print(f" *** ray_batch: {ray_batch.shape} | ray_batch_v: {ray_batch_v.shape} **** ")

            #div_factor = 2
            rays_o_v, rays_d_v = ray_batch_v[:,0:3], ray_batch_v[:,3:6] # [N_rays, 3] each
            viewdirs_v = ray_batch_v[:,-3:] if ray_batch_v.shape[-1] > 8 else None
            bounds_v = torch.reshape(ray_batch_v[...,6:8], [-1,1,2])

            '''rough initialization: virt looking at real and real looking at virt should have similar near and far'''
            near_v, far_v = bounds_v[...,0], bounds_v[...,1] # [-1,1]

            init_near, init_far = near, far
            init_near_v, init_far_v = near_v, far_v

            # Step 2: Sample 'coarse' sample from the ray segment within the bounding cylinder
            near, far =  get_near_far_in_cylinder(rays_o=rays_o, rays_d=rays_d, cyl=cyls, near=near, far=far)
            pts, z_vals = self.sample_pts(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_rays=N_rays,
                                         N_samples=N_samples, perturb=perturb, lindisp=lindisp, pytest=pytest, 
                                         ray_noise_std=ray_noise_std)
            
            # Get line of intersection 
            n_m = normalize_batch_normal(m_normal) # same
            pt_on_ray_v = near_v*rays_d_v

            '''for visualization only'''
            intersect_pts=line_intersect(n_m[0].repeat(N_rays).view(N_rays,1,-1), avg_D[0:1].repeat(N_rays,1),\
                        rays_d_v.view(N_rays,-1,1), pt_on_ray_v.view(N_rays,-1))
            #intersect_pts = torch.Tensor(intersect_pts)
        
            '''reflect the virtual points to real space - drop trans part in A matrix 
            (aligns with shiyang comment on transforming the mirr matrix coodinate system to nerf space 
            provided it contains the trans part but no it does not)'''
            rays_ref = (A_dash[0][:3,:3] @ rays_d_v.permute(1,0)).permute(1,0)

            # rays_d_v_homo = torch.cat((rays_d_v, rays_d_v.new_ones(1).expand(*rays_d_v.shape[:-1], 1)), 1)
            # rays_ref = (rays_d_v_homo @  A_dash[0])[:,:3]

            A_mirr = A_dash
            v_cam_pos = A_mirr[:,:3,3:4].reshape(-1,3) # no negative addition (already inhibit)

            #import ipdb; ipdb.set_trace()
            # get near and far based on reflected points to real kp/cyls but from virtual cam position
            near_ref, far_ref = get_near_far_in_cylinder(rays_o=v_cam_pos, rays_d=rays_ref, cyl=cyls, near=init_near_v,
                                                far=init_far_v)

            pts_ref, z_vals_ref = self.sample_pts(rays_o=v_cam_pos, rays_d=rays_ref, near=near_ref, 
                                            far=far_ref, N_rays=N_rays, N_samples=N_samples,
                                            perturb=perturb, lindisp=lindisp, pytest=pytest, ray_noise_std=ray_noise_std)

            # ref_z_vals = sample_from_lineseg(near=near, far=far, N_lines=N_rays, N_samples=64)
            # mirr2real_z_vals = sample_from_lineseg(near=torch.zeros_like(near), far=far, N_lines=n_rays, N_samples=64)

            # h5_path = './data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13/mirror_train_h5py.h5'
            # dataset = h5py.File(h5_path, 'r')
            # #dataset_v = h5py.File(h5_path_v, 'r')
            
            # '''we have pre-seleced 0 from the random pixels, see dataset.py'''
            # plot_current_pixel(dataset, idx_repeat[0].detach().cpu(), pixel_loc_repeat[0].detach().cpu(), type="real")
            # plot_current_pixel(dataset, idx_repeat[0].detach().cpu(), pixel_loc_v_repeat[0].detach().cpu(), type="virt")
  
            filename = f"{del_folder}/raycaster_use_mirr_Sep22_2022.pickle"
            to_pickle = [("pts",pts), ("z_vals", z_vals), 
                        # ("pts_v",pts_v), ("z_vals_v", z_vals_v), 
                        ("pts_ref", pts_ref), 
                        ("z_vals_ref", z_vals_ref),
                        ("intersect_pts", intersect_pts),

                        ("rays_ref", rays_ref),
                        ("rays_o", rays_o), 
                        ("rays_o_v", rays_o_v), 
                        ("rays_d", rays_d), 
                        ("rays_d_v", rays_d_v), 
                        ("skts", skts),
                        
                        ("kp_batch", kp_batch), 
                        #("kp_batch_v", kp_batch_v), 
                        ("cyls", cyls), 
                        ("cyls_v", cyls_v),
                        ("init_near", init_near),
                        ("init_far", init_far),

                        ("init_near_v", init_near_v),
                        ("init_far_v", init_far_v),
                        ("n_m", n_m[0]),
                        ("avg_D", avg_D),
                        ("A_dash", A_dash),
                        ("cyls", cyls), 
                        ("cyls_v", cyls_v),
                        ]

            #save2pickle(filename, to_pickle)
            del intersect_pts
            #import ipdb; ipdb.set_trace()

            # '''z_vals gives amt of change/distance'''
            # if not use_z_direct:
            #     # calculate reflected z_vals and virtual ray direction in the real space
            #     z_vals_ref = torch.norm(ref_pts,dim=-1)
            # else:
            #     z_vals_ref=z_vals_v

            # '''ray direction of reflected points q'''
            # ref_dir = ref_pts[:, -1] - ref_pts[:, 0] # norm of diff vec
            # ref_dir_norm =  torch.norm(ref_dir,dim=-1, keepdim=True)
            # rays_ref = ref_dir/ref_dir_norm   

            #import ipdb; ipdb.set_trace()
            # prepare local coordinate system
            joint_coords = self.get_subject_joint_coords(subject_idxs, pts.device) # is None

            '''Diff directions but same kp'''
            # Step 3: encode
            # we map the sample pts q to local space
            # dan: we maintain and also map the real (view direction) to local space
            encoded = self.encode_inputs(pts, [rays_o[:, None, :], rays_d[:, None, :]], kp_batch,
                                        skts, bones, cam_idxs=cams, subject_idxs=subject_idxs,
                                        joint_coords=joint_coords, network=self.network,
                                        **preproc_kwargs)

            # though rays_o_ref not used
            encoded_ref = self.encode_inputs(pts_ref, [v_cam_pos[:, None, :], rays_ref[:, None, :]], kp_batch,
                                        skts, bones, cam_idxs=cams, subject_idxs=subject_idxs,
                                        joint_coords=joint_coords, network=self.network,
                                        **preproc_kwargs)

            # Step 4: forwarding in NeRF and get coarse outputs
            # dan: pass in encoded real information
            raw = self.run_network(encoded, self.network)
            raw_ref = self.run_network(encoded_ref, self.network)
            #import ipdb; ipdb.set_trace()
            
            '''check density scale - 1.0, raw (1536, 4)'''
            ret_dict = self.network.raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest,
                                            encoded=encoded, B=preproc_kwargs['density_scale'],
                                            act_fn=preproc_kwargs['density_fn'])

            ret_dict_ref = self.network.raw2outputs(raw_ref, z_vals_ref, rays_ref, raw_noise_std, pytest=pytest,
                                            encoded=encoded_ref, B=preproc_kwargs['density_scale'],
                                            act_fn=preproc_kwargs['density_fn'])

            # Step 6: generate fine outputs
            ret_dict0 = None
            ret_dict0_ref =  None

            if N_importance > 0:
                # preserve coarse output
                ret_dict0 = ret_dict
                ret_dict0_ref = ret_dict_ref

                # get the importance samples (z_samples), as well as sorted_idx
                pts_is, z_vals, z_samples, sorted_idxs = self.sample_pts_is(rays_o, rays_d, z_vals, ret_dict0['weights'],
                                                                            N_importance, det=(perturb==0.), pytest=pytest,
                                                                            is_only=self.single_net, ray_noise_std=ray_noise_std)
                
                # get the importance samples (z_samples), as well as sorted_idx
                pts_is_ref, z_vals_ref, z_samples_ref, sorted_idxs_ref = self.sample_pts_is(v_cam_pos, rays_ref, z_vals_ref, ret_dict0_ref['weights'],
                                                                            N_importance, det=(perturb==0.), pytest=pytest,
                                                                            is_only=self.single_net, ray_noise_std=ray_noise_std)

                #import ipdb; ipdb.set_trace()

                # only processed the newly sampled pts (to save some computes)
                encoded_is = self.encode_inputs(pts_is, [rays_o[:, None, :], rays_d[:, None, :]], kp_batch,
                                                skts, bones, cam_idxs=cams, subject_idxs=subject_idxs,
                                                joint_coords=joint_coords, network=self.network_fine,
                                                **preproc_kwargs)

                encoded_is_ref = self.encode_inputs(pts_is_ref, [v_cam_pos[:, None, :], rays_ref[:, None, :]], kp_batch,
                                                skts, bones, cam_idxs=cams, subject_idxs=subject_idxs,
                                                joint_coords=joint_coords, network=self.network_fine,
                                                **preproc_kwargs)

                if not self.single_net:
                    # take the union of both samples
                    N_total_samples = N_importance + N_samples
                    # for real pts
                    encoded_is = self._merge_encodings(encoded, encoded_is, sorted_idxs,
                                                    N_rays, N_total_samples)
                    raw = self.run_network(encoded_is, self.network_fine)

                    # for reflected pts
                    encoded_is_ref = self._merge_encodings(encoded_ref, encoded_is_ref, sorted_idxs_ref,
                                                    N_rays, N_total_samples)
                    raw_ref = self.run_network(encoded_is_ref, self.network_fine)
                
                else: # used?
                    import ipdb; ipdb.set_trace()
                    raw_is = self.run_network(encoded_is, self.network_fine)
                    raw_is_ref = self.run_network(encoded_is_ref, self.network_fine)

                    N_total_samples = N_importance + N_samples
                    encoded_is = self._merge_encodings(encoded, encoded_is, sorted_idxs,
                                                    N_rays, N_total_samples)
                    encoded_is_ref = self._merge_encodings(encoded_ref, encoded_is_ref, sorted_idxs_ref,
                                                    N_rays, N_total_samples)

                    raw  = self._merge_encodings({'raw': raw}, {'raw': raw_is}, sorted_idxs,
                                                N_rays, N_total_samples)['raw']
                    raw_ref  = self._merge_encodings({'raw': raw_ref}, {'raw': raw_is_ref}, sorted_idxs_ref,
                                                N_rays, N_total_samples)['raw']

                    
                
                ret_dict = self.network_fine.raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest,
                                                        encoded=encoded_is, B=preproc_kwargs['density_scale'],
                                                        act_fn=preproc_kwargs['density_fn'])
                
                ret_dict_ref = self.network_fine.raw2outputs(raw_ref, z_vals_ref, rays_ref, raw_noise_std, pytest=pytest,
                                                        encoded=encoded_is_ref, B=preproc_kwargs['density_scale'],
                                                        act_fn=preproc_kwargs['density_fn'])
                                                        

            # import sys; sys.path.append("/scratch/dajisafe/smpl/mirror_project_dir")
            # from util_loading import save2pickle

            # filename = f"{del_folder}/raycaster_paramsC.pickle"
            # to_pickle = [("pts_is",pts_is), ("z_vals", z_vals), 
            # ("pts_is_ref", pts_is_ref), ("z_vals_ref", z_vals_ref),
            
            # ("intersect_pts", intersect_pts),
            # ("rays_ref", rays_ref),

            # ("rays_o", rays_o), 
            # ("rays_o_v", rays_o_v), 
            # ("rays_d", rays_d), 
            # ("rays_d_v", rays_d_v), 
            
            # ("kp_batch", kp_batch), 
            # ("kp_batch_v", kp_batch_v), 
            # ("cyls", cyls), 
            # ("cyls_v", cyls_v),
            # ("cyls_v", cyls_v),
            # ("n_m", n_m[0]),
            #  ("avg_D", avg_D),
            # ]

            # save2pickle(filename, to_pickle)
            # import ipdb; ipdb.set_trace()
        
            #import ipdb; ipdb.set_trace()
            return self._collect_outputs(ret_dict, ret_dict0, ret_dict_ref, ret_dict0_ref, use_mirr=use_mirr)
        
        else:
            '''Rendering time without mirrors'''
            #print("No mirror")
            N_rays = ray_batch.shape[0]
            #print(f"N_rays {N_rays}")

            # Step 2: Sample 'coarse' sample from the ray segment within the bounding cylinder
            near, far =  get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far)
            pts, z_vals = self.sample_pts(rays_o, rays_d, near, far, N_rays, N_samples,
                                        perturb, lindisp, pytest=pytest, ray_noise_std=ray_noise_std)

            n_m = normalize_batch_normal(m_normal)[0] if m_normal is not None else None #readable one-line
            
            
            #dataset = h5py.File(h5_path, 'r')
            #dataset_v = h5py.File(h5_path_v, 'r')
            
            #'''we have pre-seleced 0 from the random pixels, see dataset.py'''
            #plot_current_pixel(dataset, idx_repeat[0].detach().cpu(), pixel_loc_repeat[0].detach().cpu(), type="real")
            #plot_current_pixel(dataset, idx_repeat[0].detach().cpu(), pixel_loc_v_repeat[0].detach().cpu(), type="virt")
  
            filename = f"{del_folder}/raycaster_params_no_mirr_Sep22_2022.pickle"
            to_pickle = [("pts",pts), ("z_vals", z_vals), 
                        # ("rays_ref", rays_ref),
                        ("rays_o", rays_o), 
                        # ("rays_o_v", rays_o_v), 
                        ("rays_d", rays_d), 
                        # ("rays_d_v", rays_d_v), 
                        
                        ("kp_batch", kp_batch), 
                        # ("kp_batch_v", kp_batch_v), 
                        ("cyls", cyls), 
                        # ("cyls_v", cyls_v),
                        ("near", near),
                        ("far", far),
                        ("n_m", n_m),
                        ("avg_D", avg_D),

                        ("skts", skts),
                        ("bones", bones),
                        ("N_rays", N_rays),
                        ("N_samples", N_samples),
                        ("perturb", perturb),
                        ("lindisp", lindisp),
                        ("pytest", pytest),
                        ("ray_noise_std", ray_noise_std),
                        ]

            #save2pickle(filename, to_pickle)
            #import ipdb; ipdb.set_trace()

            # prepare local coordinate system
            joint_coords = self.get_subject_joint_coords(subject_idxs, pts.device)

            # Step 3: encode
            encoded = self.encode_inputs(pts, [rays_o[:, None, :], rays_d[:, None, :]], kp_batch,
                                        skts, bones, cam_idxs=cams, subject_idxs=subject_idxs,
                                        joint_coords=joint_coords, network=self.network,
                                        **preproc_kwargs)

            
            # Step 4: forwarding in NeRF and get coarse outputs
            raw = self.run_network(encoded, self.network)
            ret_dict = self.network.raw2outputs(raw, z_vals=z_vals, rays_d=rays_d, raw_noise_std=raw_noise_std, 
                                                pytest=pytest, use_mirr=use_mirr,
                                                encoded=encoded, B=preproc_kwargs['density_scale'],
                                                act_fn=preproc_kwargs['density_fn'])


            # Step 6: generate fine outputs
            ret_dict0 = None
            if N_importance > 0:
                # preserve coarse output
                ret_dict0 = ret_dict

                # get the importance samples (z_samples), as well as sorted_idx
                pts_is, z_vals, z_samples, sorted_idxs = self.sample_pts_is(rays_o, rays_d, z_vals, ret_dict0['weights'],
                                                                            N_importance, det=(perturb==0.), pytest=pytest,
                                                                            is_only=self.single_net, ray_noise_std=ray_noise_std)

                # only processed the newly sampled pts (to save some computes)
                encoded_is = self.encode_inputs(pts_is, [rays_o[:, None, :], rays_d[:, None, :]], kp_batch,
                                                skts, bones, cam_idxs=cams, subject_idxs=subject_idxs,
                                                joint_coords=joint_coords, network=self.network_fine,
                                                **preproc_kwargs)

                #import ipdb; ipdb.set_trace()
                if not self.single_net:
                    # take the union of both samples:
                    N_total_samples = N_importance + N_samples
                    encoded_is = self._merge_encodings(encoded=encoded, encoded_is=encoded_is, sorted_idxs=sorted_idxs,
                                                    N_rays=N_rays, N_total_samples=N_total_samples, use_mirr=use_mirr)
                    raw = self.run_network(encoded_is, self.network_fine)
                else:
                    import ipdb; ipdb.set_trace()
                    raw_is = self.run_network(encoded_is, self.network_fine)

                    N_total_samples = N_importance + N_samples
                    encoded_is = self._merge_encodings(encoded=encoded, encoded_is=encoded_is, sorted_idxs=sorted_idxs,
                                                    N_rays=N_rays, N_total_samples=N_total_samples, use_mirr=use_mirr)
                    raw  = self._merge_encodings({'raw': raw}, {'raw': raw_is}, sorted_idxs=sorted_idxs,
                                                N_rays=N_rays, N_total_samples=N_total_samples, use_mirr=use_mirr)['raw']
                ret_dict = self.network_fine.raw2outputs(raw, z_vals=z_vals, rays_d=rays_d, raw_noise_std=raw_noise_std, 
                                                        pytest=pytest, use_mirr=use_mirr,
                                                        encoded=encoded_is, B=preproc_kwargs['density_scale'],
                                                        act_fn=preproc_kwargs['density_fn'])

            #import ipdb; ipdb.set_trace()
            return self._collect_outputs(ret_dict, ret_dict0)

    def encode_inputs(self, pts, rays, kps, skts, bones,
                      cam_idxs=None, subject_idxs=None,
                      joint_coords=None, pts_tr_fn=None,
                      kp_input_fn=None, view_input_fn=None,
                      bone_input_fn=None, network=None,
                      density_scale=1.0, **kwargs):
        """
        Args:
          pts: query points in 3D world coordinates
          rays: tuple of (rays_o, rays_d)
          kps: 3D human keypoints
          skts: skeleton-transformation for transforming pts to the local coordinates
          bones: joint rotations
          cam_idxs: camera index of the pts.
          subject_idxs: subject indexs of the pts
          joint_coords: per-joint coordinate system, 3x3 array for each joint (N_axes==3, 3)
          pts_tr_fn: transformation function to apply on 3D world points
          kp_input_fn: encoding function for keypoint location
          view_input_fn: encoding function for view direction
          bone_input_fn: encoding function for joint rotation
          network: NeRF network

        Returns:
          encoded: v (distance), r (rotation), and d(ray direction)
        """


        if pts.shape[0] > kps.shape[0]:
            # expand kps to match the number of rays
            assert kps.shape[0] == 1
            import ipdb; ipdb.set_trace()
            kps = kps.expand(pts.shape[0], *kp_batch.shape[1:])

        # tranform points/rays to local coordinate (if needed)
        if skts is not None:
            pts_t = pts_tr_fn(pts, skts, bones=bones, coords=joint_coords, kps=kps)
            rays_t = transform_batch_rays(rays[0], rays[1], skts)
        else:
            pts_t = pts
            rays_t = rays[1] # only needs ray direction

        # Step 1: keypoint encoding
        v = kp_input_fn(pts, pts_t, kps)
        #if abs(v.mean()) > 1.5:
        #    import ipdb; ipdb.set_trace()

        # Step 2: bone encoding
        # TODO: could be None
        r = bone_input_fn(pts_t, bones=bones, coords=joint_coords)

        # Step 3: view encoding
        d = view_input_fn(rays_t, bones=bones, refs=pts_t, pts_t=pts_t,
                          coords=joint_coords)

        # Positional encoding here?
        # check kp_input_fn type
        # if kp_input_fn != dist -> calculate distance
        if 'Dist' in kp_input_fn.encoder_name:
            j_dists = v
        else:
            j_dists = torch.norm(pts[:, :, None] - kps[:, None], dim=-1, p=2)

        # cw: cutoff weights
        v, cw = self.embed_fn(v, dists=j_dists)
        r, _ = self.embedbones_fn(r, dists=j_dists)
        d, _ = self.embeddirs_fn(d, dists=j_dists)

        if cam_idxs is not None:
            # append cam_idxs after view
            cam_idxs = cam_idxs.view(-1, 1, 1).expand(*d.shape[:2], -1)
            d = torch.cat([d, cam_idxs], dim=-1)

        if subject_idxs is not None:
            N_idxs = subject_idxs.shape[0]
            subject_idxs = subject_idxs.view(N_idxs, 1, -1).expand(*d.shape[:2], -1)
            d = torch.cat([d, subject_idxs], dim=-1)

        encoded = {
            'v': v,
            'r': r,
            'd': d,
        }
        return encoded

    def run_network(self, encoded, network, netchunk=1024*64):


        cat_lists = [encoded['v']]

        # has bone encoding
        if encoded['r'] is not None:
            cat_lists.append(encoded['r'])
        # has view encoding
        if encoded['d'] is not None:
            cat_lists.append(encoded['d'])

        encoded = torch.cat(cat_lists, dim=-1)
        shape = encoded.shape

        # flatten and forward
        outputs_flat = network.forward_batchify(encoded.reshape(-1, shape[-1]), chunk=netchunk)
        reshape = list(shape[:-1]) + [outputs_flat.shape[-1]]
        outputs = torch.reshape(outputs_flat, reshape)

        return outputs

    @torch.no_grad()
    def render_mesh_density(self, kps, skts, bones, subject_idxs=None, radius=1.0, res=64,
                            render_kwargs=None, netchunk=1024*64, v=None):

        # generate 3d cubes (voxels)
        t = np.linspace(-radius, radius, res+1)
        grid_pts = np.stack(np.meshgrid(t, t, t), axis=-1).astype(np.float32)
        sh = grid_pts.shape
        grid_pts = torch.tensor(grid_pts.reshape(-1, 3)) + kps[0, 0]
        grid_pts = grid_pts.reshape(-1, 1, 3)

        # create density forward function (for batchify computation)
        raw_density = self.render_pts_density(grid_pts, kps, skts, bones,
                                              render_kwargs, subject_idxs, netchunk,
                                              v=v)[..., :1]

        return raw_density.reshape(*sh[:-1]).transpose(1, 0)

    def render_pts_density(self, pts, kps, skts, bones, render_kwargs=None,
                           subject_idxs=None, netchunk=1024*64, network=None,
                           color=False, v=None):

        # create density forward function (for batchify computation)
        joint_coords = self.get_subject_joint_coords(subject_idxs, device=pts.device)
        fwd = self._get_density_fwd_fn(kps, skts, bones, joint_coords, network, render_kwargs,
                                       color=color)
        kwargs = {}
        if v is not None:
            kwargs['v'] = v
        raw_density = batchify(fwd, netchunk)(pts, **kwargs)
        return raw_density


    def _get_density_fwd_fn(self, kps, skts, bones,
                            joint_coords, network, render_kwargs, color=False):

        if network is None:
            if self.network_fine is not None:
                network = self.network_fine
            else:
                network = self.network
        kp_input_fn = render_kwargs["kp_input_fn"]
        bone_input_fn = render_kwargs["bone_input_fn"]

        if color:
            assert hasattr(network, 'texture_linears'), 'need to have texture layer!'

        def fwd_fn(pts, v=None):
            # prepare inputs for density network
            pts_t = transform_batch_pts(pts, skts)
            if v is None:
                v = kp_input_fn(pts, pts_t, kps)
            r = bone_input_fn(pts_t, bones=bones, coords=joint_coords)

            if 'Dist' in kp_input_fn.encoder_name:
                j_dists = v
            else:
                j_dists = torch.norm(pts[:, :, None] - kps[:, None], dim=-1, p=2)

            # embedding!
            v, _ = self.embed_fn(v, dists=j_dists)
            r, _ = self.embedbones_fn(r, dists=j_dists)
            embedded = torch.cat([v, r], dim=-1)

            h = network.forward_density(embedded)
            raw_density = network.alpha_linear(h)

            return raw_density

        return fwd_fn

    def sample_pts(self, rays_o, rays_d, near, far, N_rays, N_samples,
                   perturb, lindisp, pytest=False, ray_noise_std=0.):

        z_vals = sample_from_lineseg(near, far, N_rays, N_samples,
                                     perturb, lindisp, pytest=pytest)

        # range of points should be bounded within 2pi.
        # so the lowest frequency component (sin(2^0 * p)) don't wrap around within the bound
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        if ray_noise_std > 0.:
            pts = pts + torch.randn_like(pts) * ray_noise_std

        return pts, z_vals

    def sample_pts_is(self, rays_o, rays_d, z_vals, weights, N_importance,
                      det=True, pytest=False, is_only=False, ray_noise_std=0.):

        z_vals, z_samples, sorted_idxs = isample_from_lineseg(z_vals, weights, N_importance, det=det,
                                                              pytest=pytest, is_only=is_only)

        pts_is = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None] # [N_rays, N_samples + N_importance, 3]

        if ray_noise_std> 0.:
            pts_is = pts_is + torch.randn_like(pts_is) * ray_noise_std


        return pts_is, z_vals, z_samples, sorted_idxs

    def _merge_encodings(self, encoded, encoded_is, sorted_idxs,
                         N_rays, N_total_samples, use_mirr=False,
                        sorted_idxs_v=None, inplace=True):
        """
        merge coarse and fine encodings.
        encoded: dictionary of coarse encodings
        encoded_is: dictionary of fine encodings
        sorted_idxs: define how the [encoded, encoded_is] are sorted
        """

        #import ipdb; ipdb.set_trace()

        if use_mirr:
            print(" we decided to solve this seperately instead")
            import ipdb; ipdb.set_trace() 

            n_sub_samples=N_total_samples//2
            gather_idxs = torch.arange(N_rays * (n_sub_samples)).view(N_rays, -1)

            gather_idxs = torch.gather(gather_idxs, 1, sorted_idxs)
            gather_idxs_v = torch.gather(gather_idxs, 1, sorted_idxs_v)

        else:
            gather_idxs = torch.arange(N_rays * (N_total_samples)).view(N_rays, -1)
            gather_idxs = torch.gather(gather_idxs, 1, sorted_idxs)

        if not inplace:
            merged = {}
        else:
            merged = encoded

        #import ipdb; ipdb.set_trace()
        for k in encoded.keys():
            if k != 'pts':
            
                if use_mirr:
                    import ipdb; ipdb.set_trace() 

                    n_sample_pts, n_is_sample_pts = encoded[k].shape[1]//2, encoded_is[k].shape[1]//2
                    '''v, r and d (merge for each each half - 1st is real, 2nd is virt)'''
                    k_real = merge_samples(encoded[k][:,:n_sample_pts,:], encoded_is[k][:,:n_is_sample_pts,:], gather_idxs, n_sub_samples)
                    k_virt = merge_samples(encoded[k][:,n_sample_pts:,:], encoded_is[k][:,n_is_sample_pts:,:], gather_idxs_v, n_sub_samples)

                    merged[k] = torch.cat([k_real,k_virt],dim=1)

                else:
                    #import ipdb; ipdb.set_trace() 
                    merged[k] = merge_samples(encoded[k], encoded_is[k], gather_idxs, N_total_samples)

        # need special treatment here to preserve the computation graph.
        # (otherwise we can just re-encode everything again, but that takes extra computes)
        if 'pts' in encoded and encoded['pts'] is not None:
            import ipdb; ipdb.set_trace() 
            
            if not inplace:
                merged['pts'] = encoded['pts']

            merged['pts_is'] = encoded_is['pts']
            merged['gather_idxs'] = gather_idxs

            merged['pts_sorted'] = merge_samples(encoded['pts'], encoded_is['pts'],
                                                 gather_idxs, N_total_samples)
        #import ipdb; ipdb.set_trace()
        return merged

    def _collect_outputs(self, ret=None, ret0=None, ret_ref=None, ret0_ref=None, use_mirr=False):
        ''' collect outputs into a dictionary for loss computation/rendering
        ret: outputs from fine networki (or coarse network if we don't have a fine one).
        ret0: outputs from coarse network.
        '''

        # base
        collected = {'rgb_map': ret['rgb_map'], 'disp_map': ret['disp_map'],
                    'acc_map': ret['acc_map'], 'alpha': ret['alpha']}

        if ret0 is not None:
            collected['rgb0'] = ret0['rgb_map']
            collected['disp0'] = ret0['disp_map']
            collected['acc0'] = ret0['acc_map']
            collected['alpha0'] = ret0['alpha']

        if ret_ref is not None and use_mirr:
            # use mirror (base)
            collected['rgb_map_ref'] = ret_ref['rgb_map']
            collected['disp_map_ref'] = ret_ref['disp_map']
            collected['acc_map_ref'] = ret_ref['acc_map']
            collected['alpha_ref'] = ret_ref['alpha']
            
            if ret0_ref is not None and use_mirr:
                collected['rgb0_ref'] = ret0_ref['rgb_map']
                collected['disp0_ref'] = ret0_ref['disp_map']
                collected['acc0_ref'] = ret0_ref['acc_map']
                collected['alpha0_ref'] = ret0_ref['alpha']

        #import ipdb; ipdb.set_trace()
        return collected

    def get_subject_joint_coords(self, subject_idxs=None, device=None):
        joint_coords = self.joint_coords.to(device)
        joint_coords = joint_coords[subject_idxs]
        return joint_coords

    def update_embed_fns(self, global_step, args):

        cutoff_step = args.cutoff_step
        cutoff_rate = args.cutoff_rate

        freq_schedule_step = args.freq_schedule_step
        freq_target = args.multires-1

        self.embed_fn.update_threshold(global_step, cutoff_step, cutoff_rate,
                                       freq_schedule_step, freq_target)


        self.embeddirs_fn.update_threshold(global_step, cutoff_step, cutoff_rate,
                                           freq_schedule_step, freq_target)

        if self.embedbones_fn is not None:
            self.embedbones_fn.update_threshold(global_step, cutoff_step, cutoff_rate,
                                                freq_schedule_step, freq_target)


    # TODO: check if we really need this state_dict/load_state_dict ..
    def state_dict(self):
        state_dict = {}
        modules = self.__dict__['_modules']
        for k in modules:
            m = modules[k]
            # rules for backward compatibility ...
            if k.endswith("_fine"):
                state_dict[f"{k}_state_dict"] = m.state_dict()
            elif k.endswith("_fn"):
                state_dict[f"{k.split('_fn')[0]}_state_dict"] = m.state_dict()
            elif k == "network":
                state_dict["network_fn_state_dict"] = m.state_dict()
            else:
                state_dict[f"{k}_state_dict"] = m.state_dict()
        return state_dict

    def load_state_dict(self, ckpt, strict=True):

        modules = self.__dict__['_modules']
        for k in modules:
            if k.endswith("_fine"):
                target_key = f"{k}_state_dict"
            elif k.endswith("_fn"):
                target_key = f"{k.split('_fn')[0]}_state_dict"
            elif k == "network":
                target_key = "network_fn_state_dict"
            else:
                target_key = f"{k}_state_dict"
            try:
                modules[k].load_state_dict(ckpt[target_key], strict=strict)
            except (KeyError, RuntimeError):
                if k.startswith('network'):
                    print(f'Error occur when loading state dict for network. Try loading with strict=False now')
                    filtered_sd = filter_state_dict(modules[k].state_dict(), ckpt[target_key])
                    modules[k].load_state_dict(filtered_sd, strict=False)
                else:
                    print(f'Error occurr when loading state dict for {target_key}. The entity is not in the state dict?')

    def get_embed_fns(self):
        return self.embed_fn, self.embedbones_fn, self.embeddirs_fn

    def get_networks(self):
        return self.network, self.network_fine

def merge_samples(x, x_is, gather_idxs, N_total_samples):
    """
    merge coarse and fine samples.
    x: coarse samples of shape (N_rays, N_coarse, -1)
    x_is: importance samples of shape (N_rays, N_fine, -1)
    gather_idx: define how the [x, x_is] are sorted
    """
    if x is None:
        return None
    N_rays = x.shape[0]
    x_is = torch.cat([x, x_is], dim=1)
    sh = x_is.shape
    feat_size = np.prod(sh[2:])
    x_is = x_is.view(-1, feat_size)[gather_idxs, :]
    x_is = x_is.view(N_rays, N_total_samples, *sh[2:])

    return x_is

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs, **kwargs):
        # be careful about the concatenation dim
        return torch.cat([fn(inputs[i:i+chunk], **{k: kwargs[k][i:i+chunk] for k in kwargs})
                          for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


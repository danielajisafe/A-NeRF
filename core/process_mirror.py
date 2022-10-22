#standard imports
import os
import torch
import h5py
import datetime
import cv2, json
from glob import glob
from tqdm import tqdm, trange
import numpy as np

# custom imports
import util_skel as skel
import skeletons as skeleton
from util_loading import alpha_to_hip_1st, hip_1st_to_alpha
from util_loading import load_pickle, save2pickle, sort_B_via_A
from core.utils.skeleton_utils import get_kp_bounding_cylinder, get_skeleton_type, plot_cameras, plot_bounding_cylinder, coord_to_homogeneous, line_intersect, create_plane_updated, normalize_batch_normal

class create_data():

    def __init__(self, comb):

        self.project_dir = "/scratch/dajisafe/smpl/mirror_project_dir/"
        #comb = "f8480b9b-b1e2-4bf6-a7cf-a34c19244afd_cam_3" 
        id = comb.split("_cam_")[0]
        self.cam = view = comb.split("_cam_")[1]

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        #self.timestamp = '2022-05-06-18'
        print("timestamp", self.timestamp)

        # if gt_2d and gt_focal:
        #     files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_gt2d_gtfocal/{view}/*{id}.pickle"))       
        # elif not gt_2d and not gt_focal:
        files = sorted(glob(self.project_dir + f"authors_eval_data/new_recon_results_no_gt2d_no_gtfocal_May_11/{view}/*{id}.pickle")) 
        # elif not gt_2d and gt_focal:
        #     files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_no_gt2d/{view}/*{id}.pickle"))       
        # else:
        #     print("do we have this option?")
        #     import ipdb; ipdb.set_trace()

        #files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_no_gt2d_no_gtfocal/{view}/*{id}.pickle")) 
        #files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_no_gt2d/{view}/*{id}.pickle"))       
        #files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_gt2d_gtfocal/{view}/*{id}.pickle"))       

        # TODO: use keys 
        kp3d, kp3d_h, proj2d_real, proj2d_virt, self.img_urls, v_view_h, r_view_h = [], [], [], [], [], [], []
        est_2d_real, est_2d_virt, rotation, theta, bf_build = [], [], [], [], []
        bone_orient_store, k_optim, bone_length = [], [], []
        p_dash_2d, p_2d, N_2d, normal_end_2d = [],[],[],[]
        ground_end_2d, refine_ground_end_2d, otho_end_2d = [],[], []
        l2ws, bf_positive, m_normal, g_normal = [], [], [], []
        chosen_frames, flipped_frames = [], []
        N3d, p3d, p_dash3d, otho = [],[],[], []
        real_feet_mask, virt_feet_mask = [], []
        A, A_dash, A_dash_tuple = [], [], []
        avg_D, plane_d = [], []


        for i in range(len(files)):
            filename = files[0].split(f"{view}_0")[0] + f"{view}_{i}" + files[0].split(f"{view}_0")[1]
            #filename = f"/scratch/dajisafe/smpl/mirror_project_dir/authors_eval_data/new_recon_results_no_gt2d/{view}/result_{view}_{i}_"+"{'A_dash': False, 'A_dash_fewer': False, 'K_op': False, 'bf_op': True, 'bf_build': True, 'bone_sym_2Dloss': False, 'n_m_single': True, 'n_m_single': True}_"+f"{id}.pickle"
            #filename = f"/scratch/dajisafe/smpl/mirror_project_dir/authors_eval_data/new_recon_results_gt2d_gtfocal/{view}/result_{view}_{i}_"+"{'A_dash': False, 'A_dash_fewer': False, 'K_op': False, 'bf_op': True, 'bf_build': True, 'bone_sym_2Dloss': False}_"+f"{id}.pickle"
            from_pickle = load_pickle(filename)
            
            # TODO: reduce lines of code with key indexing
            kp3d.extend(from_pickle["kp3d"]) 
            proj2d_real.extend(from_pickle["proj2d_real"])
            proj2d_virt.extend(from_pickle["proj2d_virt"])
            est_2d_real.extend(from_pickle["est_2d_real"])
            est_2d_virt.extend(from_pickle["est_2d_virt"])
            self.img_urls.extend(from_pickle["img_urls"])
            kp3d_h.extend(from_pickle["kp3d_h"])
            v_view_h.extend(from_pickle["v_view_h"])
            r_view_h.extend(from_pickle["r_view_h"])
            rotation.extend(from_pickle["optim_rotation3x3"])
            theta.extend(from_pickle["optim_theta"]) 
            bf_build.extend(from_pickle["bf_build"])
            bone_orient_store.extend(from_pickle["b_orientation"])
            k_optim.extend(from_pickle["K_optim"])
            bf_positive.extend(from_pickle["bf_positive"])
            #bone_length.extend(from_pickle["bone_length"])
            m_normal.extend(from_pickle["n_m"])
            g_normal.extend(from_pickle["n_g_mini"])
            otho.extend(from_pickle["otho"])
            N3d.extend(from_pickle["N3d"])
            p3d.extend(from_pickle["p3d"])
            p_dash3d.extend(from_pickle["p_dash3d"])
            flipped_frames.extend(from_pickle["flipped_frames"])
            A.extend(from_pickle["final_A"])
            A_dash.extend(from_pickle["final_A_dash"])
            avg_D.extend(from_pickle["avg_D"])
            plane_d.extend(from_pickle["plane_d"])

            p_dash_2d.extend(from_pickle["p_dash_2d"]); p_2d.extend(from_pickle["p_2d"]); N_2d.extend(from_pickle["N_2d"]);
            normal_end_2d.extend(from_pickle["normal_end_2d"]); ground_end_2d.extend(from_pickle["ground_end_2d"])
            refine_ground_end_2d.extend(from_pickle["refine_ground_end_2d"]); otho_end_2d.extend(from_pickle["otho_end_2d"])
            l2ws.extend(from_pickle["l2ws"])


        self.kp3d = torch.stack(kp3d)#[:, skel.alphapose_to_mirror_25, ...]
        self.kp3d_h = torch.stack(kp3d_h)#[:, skel.alphapose_to_mirror_25, ...]
        self.v_view_h = torch.stack(v_view_h)#[:, skel.alphapose_to_mirror_25, ...]
        self.r_view_h = torch.stack(r_view_h)#[:, skel.alphapose_to_mirror_25, ...]
        self.rotation_optim = torch.stack(rotation)#[:, skel.hip_first_to_mirror_25, ...]
        self.theta_optim = torch.stack(theta)#[:, skel.hip_first_to_mirror_25, ...]
        self.bf_build = torch.stack(bf_build)
        self.bone_orient_store = torch.stack(bone_orient_store)#[:, skel.hip_first_to_mirror_25, ...]
        self.k_optim = torch.stack(k_optim).view(-1,3,3)
        self.bf_positive = torch.stack(bf_positive)#.view(-1,3,3)
        self.A = torch.stack(A)#.view(-1,3,3)
        self.A_dash = torch.stack(A_dash)#.view(-1,3,3)

        self.chosen_frames = from_pickle["chosen_frames"]
        self.initial_pose3d = from_pickle["initial_pose3d"]
        

        # real_feet_mask = torch.stack(real_feet_mask)
        # virt_feet_mask = torch.stack(virt_feet_mask)

        self.p_dash_2d, p_2d = torch.stack(p_dash_2d), torch.stack(p_2d)
        self.N_2d, normal_end_2d = torch.stack(N_2d), torch.stack(normal_end_2d)
        self.ground_end_2d = torch.stack(ground_end_2d)
        self.refine_ground_end_2d = torch.stack(refine_ground_end_2d)
        self.otho_end_2d = torch.stack(otho_end_2d)
        self.l2ws = torch.stack(l2ws)
        self.m_normal = torch.stack(m_normal)
        self.g_normal = torch.stack(g_normal)
        self.N3d = torch.stack(N3d)
        self.p3d = torch.stack(p3d)
        self.otho = torch.stack(otho)
        self.p_dash3d = torch.stack(p_dash3d)

        self.proj2d_real = torch.stack(proj2d_real)
        self.proj2d_virt = torch.stack(proj2d_virt)
        self.est_2d_real = torch.stack(est_2d_real)
        self.est_2d_virt = torch.stack(est_2d_virt)

        # Pose
        self.kp3d_h_hipfirst = alpha_to_hip_1st(kp3d_h).detach().numpy()
        self.v_view_h_hipfirst = alpha_to_hip_1st(v_view_h).detach().numpy()
        self.r_view_h_hipfirst = alpha_to_hip_1st(r_view_h).detach().numpy()

        self.B = self.kp3d_h_hipfirst.shape[0]

    
    def save_init_virt_data(self):

        # keypoints¶
        load_filename = self.project_dir+f"/authors_eval_data/temp_dir/seg_files/{self.cam}/{self.comb}/{self.timestamp}/v_view_h_hipfirst.pickle"
        if not os.path.isfile(load_filename):
            tuples  = [("v_view_h_hipfirst", self.v_view_h_hipfirst)]
            save2pickle(load_filename, tuples)

    def scale_rest_pose(self): 
        rest_p = rest_pose.numpy()
        I = np.eye(3)
        tmp_bones = rot_to_axisang(torch.tensor(I)).view(1,3).repeat(26,1).numpy()[None]
        l2ws_up_restp = np.array([get_smpl_l2ws_kc(bone, rest_p, 1.0, bf_positive) for bone in tmp_bones])
        new_rest_p = l2ws_up_restp[:,:, :3, -1]
        new_rest_p[:,25]

    def get_cyls(self,**kwargs):

        v_cylinder_params = get_kp_bounding_cylinder(self.v_view_h_hipfirst,
                                               skel_type=skeleton.CMUSkeleton, extend_mm=250,
                                               head='-y')

        load_filename = self.project_dir+f"/authors_eval_data/temp_dir/seg_files/{self.cam}/{self.comb}/{self.timestamp}/v_cylinder_params_from_kps.pickle"
        if not os.path.isfile(load_filename):
            tuples  = [("v_cylinder_params", v_cylinder_params)]
            save2pickle(load_filename, tuples)

        kwargs["v_cylinder_params"] = v_cylinder_params
        
        return kwargs


    def create_v_data(self, split="train", **kwargs):
        data_file = f'/scratch/dajisafe/smpl/A_temp_folder/A-NeRF/data/mirror/{self.cam}/{self.comb}/{self.timestamp}/v_mirror_{split}_h5py.h5'

        v_final_dict = {
              "kp3d":  self.v_view_h_hipfirst[: self.B].astype(np.float32),
              "cyls": kwargs["v_cylinder_params"][:self.B].astype(np.float32),
              "A_dash": self.A_dash[:self.B].astype(np.float32),
              "m_normal": self.m_normal[:self.B].astype(np.float32),
              "avg_D": self.avg_D[: self.B].astype(np.float32),
                }

        # Write
        with h5py.File(data_file, 'w') as hf:
            for k,val in v_final_dict.items():
                hf.create_dataset(k, data=val)


    def sort_masks(self):

        file = self.project_dir + f"/authors_eval_data/temp_dir/{self.cam}_pha.mp4"
        folder = self.project_dir + f"/authors_eval_data/temp_dir/seg_masks/{self.cam}"

        imgs_fog = glob(folder+"/*.png")
        imgs_fog_ids = list(map(lambda x:int(x.split("name")[-1].split(".")[0]), imgs_fog))
        self.sorted_imgs_fogs = list(map(lambda x: x[1], sorted(zip(imgs_fog_ids, imgs_fog))))

        load_filename = self.project_dir+f"/authors_eval_data/temp_dir/seg_files/{self.cam}/{self.comb}/{self.timestamp}/sorted_imgs_fogs.pickle"
        if not os.path.isfile(load_filename):
            tuples  = [("sorted_imgs_fogs", self.sorted_imgs_fogs)]
            save2pickle(load_filename, tuples)

    
    def save_unit_masks(self):
        for i in trange(self.B):
            load_filename = self.project_dir+f"/authors_eval_data/temp_dir/seg_files/{self.cam}/{self.comb}/{self.timestamp}/virt_masks_{i}.pickle"
            
            # access data
            virt_p = self.get_clean_fog_mask(self.img_urls[i], self.sorted_imgs_fogs[i],self.proj2d_virt[i], p_type="virt")
            
            if not os.path.isfile(load_filename):
                    tuples  = [(f"virt_mask_{i}", virt_p)]
                    save2pickle(load_filename, tuples)

    
    def save_unit_smasks(self,B):
        for i in trange(B):
            load_filename = self.project_dir+f"/authors_eval_data/temp_dir/seg_files/{self.cam}/{self.comb}/{self.timestamp}/v_samp_masks_{i}.pickle"

            # access data
            virt_sm = self.get_sampling_mask(self.img_urls[i], self.sorted_imgs_fogs[i],self.proj2d_virt[i].detach().cpu())
            if not os.path.isfile(load_filename):
                    tuples  = [(f"v_samp_mask_{i}",virt_sm)]
                    save2pickle(load_filename, tuples)


                            # close the file



    def get_sampling_mask(self,frame_img_n, fog_img_n, proj2d_real, head_margin = 40, 
                      foot_margin = 20, side_margin = 15, p_type="real"):
        '''alpha mirror common - 25'''
        image = cv2.imread(frame_img_n)[:,:,::-1]
        f_image = cv2.imread(fog_img_n)[:,:,::-1]

        xr, yr = proj2d_real[:,0].reshape(-1), proj2d_real[:,1].reshape(-1)
        real_min_xr, real_max_xr = xr.min()-side_margin, xr.max()+side_margin
        real_min_yr, real_max_yr = yr.min()-head_margin, yr.max()+foot_margin

        r_not_ys = set(range(f_image.shape[0])) - set(list(range(int(real_min_yr), int(real_max_yr)))) 
        r_not_xs = set(range(f_image.shape[1])) - set(list(range(int(real_min_xr), int(real_max_xr)))) 
        r_image = f_image.copy()

        '''ys first'''
        r_image[list(r_not_ys),:,:] = 0
        r_image[:, list(r_not_xs),:] = 0
        # binary rgb
        r_image_mask = r_image / 255.0

        # sampling mask
        dil_mask = self.sampling_mask(r_image_mask, extend_iter=1)
        return dil_mask[:,:,0]

    def sampling_mask(self,mask,extend_iter=3, erode_border=False):
        '''
        ref: https://github.com/LemonATsu/A-NeRF/blob/d5f583330182a83214f5948d1497a820c70c7817/core/load_zju.py#L31
        Following NeuralBody repo
        https://github.com/zju3dv/neuralbody/blob/master/lib/datasets/light_stage/can_smpl.py#L46    '''
        d_size, e_size = 5, 10
        d_kernel = np.ones((d_size, d_size))
        e_kernel = np.ones((e_size, e_size))

        sampling_mask = cv2.dilate(mask.copy(), d_kernel, iterations=extend_iter)
        if erode_border:
            dilated = cv2.dilate(mask.copy(), d_kernel) 
            eroded = cv2.erode(mask.copy(), e_kernel) 
            sampling_mask[(dilated - eroded) == 1] = 0

        return sampling_mask

    def get_clean_fog_mask(self,frame_img_n, fog_img_n, proj2d_real, head_margin = 40, foot_margin = 20,\
                        side_margin = 15, p_type="real"):
        '''alpha mirror common - 25'''
        
        if p_type == "virt":
            head_margin, foot_margin, side_margin = 40,20,15
            
        image = cv2.imread(frame_img_n)[:,:,::-1]
        f_image = cv2.imread(fog_img_n)[:,:,::-1]

        xr, yr = proj2d_real[:,0].reshape(-1), proj2d_real[:,1].reshape(-1)
        real_min_xr, real_max_xr = xr.min()-side_margin, xr.max()+side_margin
        real_min_yr, real_max_yr = yr.min()-head_margin, yr.max()+foot_margin

        r_not_ys = set(range(f_image.shape[0])) - set(list(range(int(real_min_yr), int(real_max_yr)))) 
        r_not_xs = set(range(f_image.shape[1])) - set(list(range(int(real_min_xr), int(real_max_xr)))) 
        r_image = f_image.copy()

        '''ys first'''
        r_image[list(r_not_ys),:,:] = 0
        r_image[:, list(r_not_xs),:] = 0
        # binary rgb
        r_image_mask = r_image / 255.0
        
        return r_image_mask[:,:,0]


    def get_target_crop_img(self,frame_img_n, fog_img_n, proj2d_real, head_margin = 40, foot_margin = 20, side_margin = 15):
        '''alpha mirror common - 25'''
        image = cv2.imread(frame_img_n)[:,:,::-1]
        f_image = cv2.imread(fog_img_n)[:,:,::-1]

        xr, yr = proj2d_real[:,0].reshape(-1), proj2d_real[:,1].reshape(-1)
        real_min_xr, real_max_xr = xr.min()-side_margin, xr.max()+side_margin
        real_min_yr, real_max_yr = yr.min()-head_margin, yr.max()+foot_margin

        r_not_ys = set(range(f_image.shape[0])) - set(list(range(int(real_min_yr), int(real_max_yr)))) 
        r_not_xs = set(range(f_image.shape[1])) - set(list(range(int(real_min_xr), int(real_max_xr)))) 
        r_image = f_image.copy()

        '''ys first'''
        r_image[list(r_not_ys),:,:] = 0
        r_image[:, list(r_not_xs),:] = 0

        # binary rgb
        r_image_mask = r_image / 255.0
        real_obs = (r_image_mask * image).clip(0, 255).astype(np.uint8)/255.0

        # store observed pixels
        return real_obs





def main_fn():

    kwargs = {}
    comb= "31d71f29-ccfe-4509-849f-a811f72ba08f_cam_3"
    
    # create real data
    cam_ds = create_data(comb=comb)
    cam_ds.save_init_virt_data()
    kwargs = cam_ds.get_cyls(kwargs)
    cam_ds.sort_masks()
    cam_ds.save_unit_masks()

    
    # create virtual data
    cam_ds = create_data(comb=comb)
    cam_ds.save_init_virt_data()
    kwargs = cam_ds.get_cyls(kwargs)
    cam_ds.sort_masks()
    cam_ds.save_unit_masks()


    ## done
    cam_ds.create_v_data(split="train", **kwargs)

if __name__=='__main__':
    main_fn()
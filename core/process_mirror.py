#standard imports
import torch

# custom imports
from util_loading import load_pickle, save2pickle, sort_B_via_A
from glob import glob
import util_skel as skel


def collect_mirror_data(comb, gt_2d=False,gt_focal=True):
    
    data_dict = {}
    
    project_dir = "/scratch/dajisafe/smpl/mirror_project_dir/"
    comb = "f8480b9b-b1e2-4bf6-a7cf-a34c19244afd_cam_3" 
    id = comb.split("_cam_")[0]
    view = cam = comb.split("_cam_")[1]

    if gt_2d and gt_focal:
        files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_gt2d_gtfocal/{view}/*{id}.pickle"))       
    elif not gt_2d and not gt_focal:
        files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_no_gt2d_no_gtfocal/{view}/*{id}.pickle")) 
    elif not gt_2d and gt_focal:
        files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_no_gt2d/{view}/*{id}.pickle"))       
    else:
        print("do we have this option?")
        import ipdb; ipdb.set_trace()

    #files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_no_gt2d_no_gtfocal/{view}/*{id}.pickle")) 
    #files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_no_gt2d/{view}/*{id}.pickle"))       
    #files = sorted(glob(project_dir + f"authors_eval_data/new_recon_results_gt2d_gtfocal/{view}/*{id}.pickle"))       

    # TODO: use keys 
    kp3d, kp3d_h, proj2d_real, proj2d_virt, img_urls, v_view_h, r_view_h = [], [], [], [], [], [], []
    est_2d_real, est_2d_virt, rotation, theta, bf_build = [], [], [], [], []
    bone_orient_store, k_optim, bone_length = [], [], []
    p_dash_2d, p_2d, N_2d, normal_end_2d = [],[],[],[]
    ground_end_2d, refine_ground_end_2d, otho_end_2d = [],[], []
    l2ws, bf_positive, m_normal, g_normal = [], [], [], []
    chosen_frames, flipped_frames = [], []
    N3d, p3d, p_dash3d, otho = [],[],[], []
    real_feet_mask, virt_feet_mask = [], []
    A, A_dash, A_dash_tuple = [], [], []


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
        img_urls.extend(from_pickle["img_urls"])
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

    #     real_feet_mask.extend(from_pickle["real_feet_mask"])
    #     virt_feet_mask.extend(from_pickle["virt_feet_mask"])
        
        p_dash_2d.extend(from_pickle["p_dash_2d"]); p_2d.extend(from_pickle["p_2d"]); N_2d.extend(from_pickle["N_2d"]);
        normal_end_2d.extend(from_pickle["normal_end_2d"]); ground_end_2d.extend(from_pickle["ground_end_2d"])
        refine_ground_end_2d.extend(from_pickle["refine_ground_end_2d"]); otho_end_2d.extend(from_pickle["otho_end_2d"])
        l2ws.extend(from_pickle["l2ws"])

    kp3d = torch.stack(kp3d)#[:, skel.alphapose_to_mirror_25, ...]
    kp3d_h = torch.stack(kp3d_h)#[:, skel.alphapose_to_mirror_25, ...]
    v_view_h = torch.stack(v_view_h)#[:, skel.alphapose_to_mirror_25, ...]
    r_view_h = torch.stack(r_view_h)#[:, skel.alphapose_to_mirror_25, ...]
    rotation_optim = torch.stack(rotation)#[:, skel.hip_first_to_mirror_25, ...]
    theta_optim = torch.stack(theta)#[:, skel.hip_first_to_mirror_25, ...]
    bf_build = torch.stack(bf_build)
    bone_orient_store = torch.stack(bone_orient_store)#[:, skel.hip_first_to_mirror_25, ...]
    k_optim = torch.stack(k_optim).view(-1,3,3)
    bf_positive = torch.stack(bf_positive)#.view(-1,3,3)
    A = torch.stack(A)#.view(-1,3,3)
    A_dash = torch.stack(A_dash)#.view(-1,3,3)

    chosen_frames = from_pickle["chosen_frames"]
    initial_pose3d = from_pickle["initial_pose3d"]

    # real_feet_mask = torch.stack(real_feet_mask)
    # virt_feet_mask = torch.stack(virt_feet_mask)

    p_dash_2d, p_2d = torch.stack(p_dash_2d), torch.stack(p_2d)
    N_2d, normal_end_2d = torch.stack(N_2d), torch.stack(normal_end_2d)
    ground_end_2d = torch.stack(ground_end_2d)
    refine_ground_end_2d = torch.stack(refine_ground_end_2d)
    otho_end_2d = torch.stack(otho_end_2d)
    l2ws = torch.stack(l2ws)
    m_normal = torch.stack(m_normal)
    g_normal = torch.stack(g_normal)
    N3d = torch.stack(N3d)
    p3d = torch.stack(p3d)
    otho = torch.stack(otho)
    p_dash3d = torch.stack(p_dash3d)

    proj2d_real = torch.stack(proj2d_real)
    proj2d_virt = torch.stack(proj2d_virt)
    est_2d_real = torch.stack(est_2d_real)
    est_2d_virt = torch.stack(est_2d_virt)

    return data_dict

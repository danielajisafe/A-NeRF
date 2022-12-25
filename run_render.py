#!/bin/bash

# --------------------------------------------------------------
# NOTICE: Please make all modifications on local Naye Machine
# --------------------------------------------------------------

#PBS -l walltime=12:00:00,select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -N mirror
#PBS -A st-rhodin-1-gpu
#PBS -o /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/terminal/

# load the thing you want to use here
# module load software_you_want

cd /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF
eval "$(conda shell.bash hook)" # solving conda init problem
# Note: need to set up conda init properly for compute node
#conda init bash
conda activate anerf

# ---------------------------------------------------------------------------------------------------------------
#                                       NOTICE
# ---------------------------------------------------------------------------------------------------------------
# Please make sure all jobs in 1 branch is already running, before submitting jobs in another branch 
# accepted requests comes to pick whichever branch they meet
# ---------------------------------------------------------------------------------------------------------------


# Run your code
# *** things to consid-er before running**
# branch?
# noreload?
# use_mirr flag?
# gpu time and no of gpu?
# datapath and new/old date? 
# modelpath?
# pose_stop for body model?
# did pending models use the same script? then try 1hr first, and 3 days next.



# ----------------------------------------------------------------
# CVPR 23

# MESH RENDERING
#python run_render.py --nerf_args logs/tim_model/finetune/args.txt --ckptpath logs/tim_model/finetune/138000.tar --dataset mirror --entry easy --render_type mesh --runname mirror_mesh --selected_framecode 0 --render_refined --data_path /scratch/dajisafe/smpl/A_temp_folder/A-NeRF/data/mirror/0/f9f4444a-e75e-4196-bc97-e3558f504263_cam_0/2022-08-01-11 --selected_idxs 700   

# OVERLAP EVALUATION
# with occlusion model (M1)
# c6
#python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-23-15-27-31/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-23-15-27-31/175000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --train_len 1620 --eval_true_overlap
# c7
#python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-28-11-56-11/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-28-11-56-11/175000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --n_bullet 1 --train_len 1563 --eval_true_overlap

# without occlusion model (M3)
# c6
#python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-18-21-13-03/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-18-21-13-03/175000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --train_len 1620 --eval_true_overlap
# c7
#python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-28-11-57-26/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-28-11-57-26/175000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --n_bullet 1 --train_len 1563 --eval_true_overlap

#np.round(np.linalg.norm(p_aligned - p_ref[eval_cnt], axis=0, ord=2)*1000, 2)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/debug_folder --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 0 --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/debug_folder --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 0 --use_mirr --switch_cam --overlap_rays --layered_bkgd



# KPS EVALUATION

# Mirror anerf 2022-10-23-15-27-31
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-23-15-27-31/finetune/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-23-15-27-31/finetune/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --evaluate_pose --train_len 80 #1620 

# EVALUATE STEP 2 (Initial 3D) FROM HERE - remove render_refined
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-29-04-21-18-c2/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-29-04-21-18-c2/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --n_bullet 1 --train_len 1414 --evaluate_pose  
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-11-03-12-00-32-c5/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-11-03-12-00-32-c5/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --n_bullet 1 --train_len 1240 --evaluate_pose  
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-11-03-12-01-03-c7/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-11-03-12-01-03-c7/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --n_bullet 1 --train_len 1563 --evaluate_pose  

# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-29-01-49-55-c3/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-29-01-49-55-c3/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 1 --train_len 1620 --evaluate_pose  
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-29-01-49-42-c6/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-29-01-49-42-c6/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --train_len 1620 --evaluate_pose  



# Vanilla anerf 2022-07-31-17-19-21
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-11-09-16-09-16-c2/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-11-09-16-09-16-c2/175000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-11-09-06 --n_bullet 1 --train_len 1620 --evaluate_pose
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/175000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 1 --train_len 1620 --evaluate_pose
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-10-28-23-36-59-c5/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-10-28-23-36-59-c5/175000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --n_bullet 1 --train_len 1618 --evaluate_pose
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/175000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --train_len 1620 --evaluate_pose
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-10-28-23-40-42-c7/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-10-28-23-40-42-c7/175000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --n_bullet 1 --train_len 1620 --evaluate_pose
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/finetune/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/finetune/145000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13/ --n_bullet 1 --train_len 1620 --evaluate_pose

# Potential PSNR Calculation
### Vanilla vs Mirror A-NeRF

# vanilla anerf
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/finetune/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/finetune/145000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13/ --n_bullet 1 --selected_idxs 0 #--train_len 1620 --evaluate_pose

# mirror anerf (with occlusion)
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-23-15-27-31/finetune/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-23-15-27-31/finetune/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --white_bkgd --psnr_images --train_len 1620 #--evaluate_pose #1620 
# mirror anerf (no occlusion)
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-18-21-13-03/finetune/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-18-21-13-03/finetune/145000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --white_bkgd --psnr_images --train_len 1620 #--evaluate_pose #1620 
 


# # video creation
# ffmpeg -framerate 7 -pattern_type glob -i '/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/render_output/mirror_bullet/2022-11-11-11/image_6/*.png' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -y -c:v libx264 -r 30 -pix_fmt yuv420p '/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/render_output/mirror_bullet/2022-11-11-11/mirr_anerf_render.mp4'


# RENDERING

# quick for cam 3
#python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-29-01-23-04-c3/finetune/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-29-01-23-04-c3/finetune/155000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 10 --selected_idxs 0 --bullet_ang 90 --white_bkgd #--evaluate_pose --train_len 1620 
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-10-29-01-49-55-c3/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-10-29-01-49-55-c3/175000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 2 --selected_idxs 0 --bullet_ang 30 #--evaluate_pose --train_len 1620 


## Tim Mirror A-NeRF
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname tim_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/0/e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0/2022-11-13-01 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --use_mirr --switch_cam --no_reload --eval_metrics
# debug
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname tim_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/0/e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0/2022-11-13-01 --N_rand 500 --i_testset 10 --i_pose_weights 10 --i_weights 10 --use_mirr --switch_cam --no_reload --eval_metrics --num_workers 1



# Mirror-optimized folder mirror/2022-05-14-13

# fresh start

# 6 (most done)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays # normal background
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload

# 3 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays # normal background
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload

# 2
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays # normal background
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload

# 5
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays # normal background
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload

# 7
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam --overlap_rays # normal background
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload --use_mirr --switch_cam 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 25000 --i_pose_weights 25000 --i_weights 25000 --no_reload


# debug
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 1500 --i_testset 1 --i_pose_weights 1 --i_weights 1 --no_reload --num_workers 1
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 1500 --i_testset 1 --i_pose_weights 1 --i_weights 1 --no_reload --num_workers 1
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 1500 --i_testset 1 --i_pose_weights 1 --i_weights 1 --no_reload --num_workers 1
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 1500 --i_testset 1 --i_pose_weights 1 --i_weights 1 --no_reload --num_workers 1
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 1500 --i_testset 1 --i_pose_weights 1 --i_weights 1 --no_reload --num_workers 1

# debug no of overlap cases
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/debug_folder --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 0 --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/debug_folder --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 0 --use_mirr --switch_cam --overlap_rays --layered_bkgd
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/debug_folder --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 0 --use_mirr --switch_cam --overlap_rays --layered_bkgd

# reload models (200k + 200k finetune)

# 2
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-no-overlap --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-no-overlap --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-01-23-04-c2 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam
# with c-view added to folder?
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-04-21-18-c2 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 

# 3
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-no-overlap  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-no-overlap  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-01-23-04-c3 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam
# with c-view added to folder?
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-01-49-55-c3 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 

# 5
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-no-overlap  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-no-overlap  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-01-49-40-c5 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam
# with c-view added to folder?
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-11-03-12-00-32-c5 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 


# 6
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-23-15-27-31 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-23-15-23-54 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-18-21-13-03 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam
# with c-view added to folder?
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-01-49-42-c6 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 

# 7
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-28-11-56-11  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --layered_bkgd
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-28-11-56-55 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-28-11-57-26 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam
# with c-view added to folder?
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-11-03-12-01-03-c7 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 

# FINE-TUNING 

#c3
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-01-23-04-c3 --ft_path logs/mirror/pose_opt_model/-2022-10-29-01-23-04-c3/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image

# c6
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-23-15-27-31 --ft_path logs/mirror/pose_opt_model/-2022-10-23-15-27-31/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --layered_bkgd --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-23-15-23-54 --ft_path logs/mirror/pose_opt_model/-2022-10-23-15-23-54/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-18-21-13-03 --ft_path logs/mirror/pose_opt_model/-2022-10-18-21-13-03/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-xxxxxxxxxxxxxxxxxxx --ft_path logs/mirror/pose_opt_model/-xxxxxxxxxxxxxxxxxxx/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image
# ----------------------------------------------------------------


# CONTINUE FINE-TUNING (please update) 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-23-15-23-54/finetune --ft_path logs/mirror/pose_opt_model/-2022-10-23-15-23-54/finetune/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image


####render
#python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-22-20-54-07/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-22-20-54-07/200000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-20-22 --n_bullet 5 #--bullet_ang 60

####validation on sockeye
#python run_render.py --nerf_args logs/mirror/pose_opt_model/--2022-05-22-20-54-07/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-22-20-54-07/200000.tar --dataset mirror --entry val --render_type val  --runname mirror_val --selected_framecode 0 --white_bkgd --selected_idxs 0 --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --test_len 180 --eval

#MAIN
# daniel_mirr_anerf branch  **** update model and data path*****
####### reload
# (no mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-22-20-32-47 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-20-22
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-22-20-54-07 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-20-22
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-22-21-25-21 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-20-22
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-23-01-11-14 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-20-22
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-23-01-26-35 --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-20-22


# START FRESH
####### no reload
# (no mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-20-22 --no_reload --opt_pose_stop 200000  
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-20-22 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-20-22 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-20-22 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-20-22 --no_reload --opt_pose_stop 200000


# Vanilla-SMPL folder 


# RE-RUN REBUTTAL - 2022-12-21-23-48-54
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-12-21-01 --no_reload --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --eval_metrics
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-12-21-01 --no_reload --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --eval_metrics

# RE-RUN REBUTTAL DEBUG
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-12-21-01 --no_reload --i_testset 20 --i_pose_weights 20 --i_weights 20 --i_print 20 --eval_metrics --num_workers 1 --N_rand 500
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-12-21-01 --no_reload --i_testset 20 --i_pose_weights 20 --i_weights 20 --i_print 20 --eval_metrics --num_workers 1  --N_rand 500


# RE-REBUTTAL RELOAD#
# 12 hours
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-12-21-23-48-54-c5 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-12-21-01  --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --eval_metrics 
# 24 hours
python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-12-23-07-16-19-c6 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-12-21-01 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --eval_metrics 
# 48 hours (c7)
# 48 hours (c3)
# 48 hours (c2)

# RE-REBUTTAL RELOAD DEBUG
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-12-21-23-48-54-c5 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-12-21-01  --i_testset 20 --i_pose_weights 20 --i_weights 20 --i_print 20 --eval_metrics --N_rand 500

# TENSORBOARD
#tensorboard --logdir=/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/logs/vanilla/pose_opt_model/-2022-12-23-07-16-19-c6 --bind_all
# ssh -L 6005:se061.ib.sockeye:6006 dajisafe@sockeye.arc.ubc.ca

# REBUTTAL FRESH - vanilla/2022-05-14-13

#CUDA_VISIBLE_DEVICES=0,2,3 
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-11-09-06 --no_reload --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --eval_metrics
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13/ --no_reload --opt_pose_stop 200000 
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13/ --no_reload --opt_pose_stop 200000 --i_pose_weights 1000 --i_print 500 
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --no_reload --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --no_reload --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500

# TODO: checker checker image first for 5,7, and 2 before running, then 24 hours run.

# REBUTTAL FRESH DEBUG
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-11-09-06 --no_reload --i_testset 20 --i_pose_weights 20 --i_weights 20 --i_print 20 --eval_metrics --num_workers 1 --N_rand 500
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --no_reload --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 1 --N_rand 500

# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --no_reload --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 1 --N_rand 500
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --no_reload --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 1 --N_rand 500
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --no_reload --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 1 --N_rand 500 
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --no_reload --i_testset 1 --i_pose_weights 1 --i_weights 1 --num_workers 1 --N_rand 500 

# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-07-31-17-19-21 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13/ --no_reload --i_testset 1000 --i_pose_weights 1000 --i_weights 1000


# REBUTTAL RELOAD 

#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-10-28-23-36-59-c5 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-10-28-23-40-42-c7 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500

# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-07-31-17-19-22 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13/ --i_pose_weights 1000 --i_weights 1000
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-07-31-17-19-21 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13/ --i_pose_weights 1000 --i_weights 1000
# with c-view added to folder?

# REBUTTAL RELOAD DEBUG
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-11-09-16-09-16-c2 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-11-09-06/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --eval_metrics
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-10-28-23-36-59-c5 --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500

# models
# 2022-07-31-17-19-21  - 233k (cam6)
# 2022-07-31-17-19-22 - 219k (cam3)

# REBUTTAL FINETUNE
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-07-31-17-19-21 --ft_path logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/200000.tar  --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-07-31-17-19-22 --ft_path logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/200000.tar  --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image

#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-10-28-23-36-59-c5 --ft_path logs/vanilla/pose_opt_model/-2022-10-28-23-36-59-c5/200000.tar  --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-10-28-23-40-42-c7 --ft_path logs/vanilla/pose_opt_model/-2022-10-28-23-40-42-c7/200000.tar  --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image

# REBUTTAL CONTINUE FINETUNE
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-07-31-17-19-22/finetune --ft_path logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/finetune/195000.tar --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13/ --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image

# finetune debug
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-10-28-23-36-59-c5 --ft_path logs/vanilla/pose_opt_model/-2022-10-28-23-36-59-c5/200000.tar  --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13/ --i_testset 1 --i_pose_weights 1 --i_weights 1 --i_print 1 --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image --num_workers 1
# python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model/-2022-10-28-23-40-42-c7 --ft_path logs/vanilla/pose_opt_model/-2022-10-28-23-40-42-c7/200000.tar  --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13/ --i_testset 1 --i_pose_weights 1 --i_weights 1 --i_print 1 --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image --num_workers 1

## REBUTTAL bullet
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/070000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --white_bkgd --selected_idxs 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 15 
# python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/057000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --white_bkgd --selected_idxs 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 5 --bullet_ang 60

# REBUTTAL evaluation on trainset
#python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-22/070000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --white_bkgd --selected_idxs 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 1 --train_len 1620
#python run_render.py --nerf_args logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/args.txt --ckptpath logs/vanilla/pose_opt_model/-2022-07-31-17-19-21/070000.tar --dataset vanilla --entry easy --render_type bullet --runname vanilla_bullet --selected_framecode 0 --white_bkgd --selected_idxs 0 --render_refined --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 1 --train_len 1620


# SSH into interactive node through a connected terminal, and watch GPU usage
# ssh se061 (check the name in interactive terminal)

# debugging
#python run_nerf.py --config configs/vanilla/vanilla.txt --basedir logs/vanilla --expname pose_opt_model --data_path /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/data/vanilla/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13/ --no_reload --num_workers 1
#---------


# MOVE refined poses to CC
#scp /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/logs/mirror/pose_opt_model/-2022-05-22-20-32-47/200000.tar dajisafe@cedar.computecanada.ca:/home/dajisafe/scratch/anerf_mirr/A-NeRF/data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/refined_pose
#scp /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/logs/mirror/pose_opt_model/-2022-05-22-20-54-07/200000.tar dajisafe@cedar.computecanada.ca:/home/dajisafe/scratch/anerf_mirr/A-NeRF/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/refined_pose
#scp /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/logs/mirror/pose_opt_model/-2022-05-22-21-25-21/200000.tar dajisafe@cedar.computecanada.ca:/home/dajisafe/scratch/anerf_mirr/A-NeRF/data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/refined_pose
#scp /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/logs/mirror/pose_opt_model/-2022-05-23-01-11-14/200000.tar dajisafe@cedar.computecanada.ca:/home/dajisafe/scratch/anerf_mirr/A-NeRF/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/refined_pose
#scp /scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF/logs/mirror/pose_opt_model/-2022-05-23-01-26-35/200000.tar dajisafe@cedar.computecanada.ca:/home/dajisafe/scratch/anerf_mirr/A-NeRF/data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/refined_pose




#ABLATION
# daniel_mirr_anerf branch  **** update model and data path*****
####### reload
# (no mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-xxxfolderxxxx --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/a77b39fa-64de-418a-9d93-22fb5b29f0d9_cam_3/2022-05-23-05
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-xxxfolderxxxx --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/aa5355f2-1ce4-45ca-a38f-2e0e316d3b2f_cam_3/2022-05-23-05
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-xxxfolderxxxx --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/83d417be-693d-4cd6-8d4a-471ddfe63d7d_cam_3/2022-05-23-05
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-xxxfolderxxxx --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/0dc04e49-25ec-4225-8fb6-769c7fe83c74_cam_3/2022-05-23-05
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-xxxfolderxxxx --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/dd08d0b8-dfd9-4fbc-a15b-1f93c9d98863_cam_3/2022-05-23-05

#ABLATION
####### no reload
# (no mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/a77b39fa-64de-418a-9d93-22fb5b29f0d9_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/aa5355f2-1ce4-45ca-a38f-2e0e316d3b2f_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/83d417be-693d-4cd6-8d4a-471ddfe63d7d_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/0dc04e49-25ec-4225-8fb6-769c7fe83c74_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/dd08d0b8-dfd9-4fbc-a15b-1f93c9d98863_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000




# testing line
#--i_print 2 --i_weights 2 --i_pose_weights 2 --i_testset 2 --n_iters 7 


# /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-20-22





#!/bin/bash

# --------------------------------------------------------------
# NOTICE: Please make all modfications on local Naye Machine
# --------------------------------------------------------------

#SBATCH --time=0-72:00:00
#SBATCH --account=rrg-rhodin 

#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/dajisafe/anerf_mirr/terminal/result_%t_%j.out

### properly create virtual environment on every node (4 gpus)
source /scratch/dajisafe/Anerf-dev/A-NeRF/anerf/bin/activate
module load python/3.8

# job description and submission script
scontrol show job $SLURM_JOB_ID
scontrol write batch_script $SLURM_JOB_ID -

# fix missing linkage
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib

# ---------------------------------------------------------------------------------------------------------------
# Please make sure all jobs in 1 branch is already running, before submitting jobs in another branch 
# accepted requests comes to pick whichever branch they meet
# May 19 main submission version is in (HEAD detached at 54efe1f)
# ---------------------------------------------------------------------------------------------------------------


# *** things to consider before running**
# did you add top margin for vanilla data? visualize initial runs to adjust value
# added eval_metrics? ie no eval metrics
# branch?
# no_reload?
# use_mirr flag?
# gpu time and no of gpu?
# datapath and new/old date? 
# modelpath?
# pose_stop for body model?
# did pending models use the same script? then try 1hr first on interative node, and 3 days next on submission.

# ----------------------------------------------------------------

## ICCV23
# Fresh mirr anerf runs

# Daniel - 02 -  with eroded mask + (good restpose is used) | refine all data size 2000
# basic with mirror input
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname da_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/2f4b7613-7d51-4a1d-81b0-4ba0a21129df_cam_0/2023-05-09-13_cropping --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 2000 --data_size 2000 --n_framecodes 2000 --eval_metrics --no_reload # --num_workers 1
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname da_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/2f4b7613-7d51-4a1d-81b0-4ba0a21129df_cam_0/2023-05-09-13_cropping --N_rand 3072 --i_testset 2 --i_pose_weights 2 --i_weights 2 --i_print 2 --train_size 2000 --data_size 2000 --n_framecodes 2000 --eval_metrics --no_reload --num_workers 0
# reload - CANCEL 24HR RUN ONCE 3-DAYS RUN BEGIN
python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname da_model/-2023-05-10-15-46-27-c0 --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/2f4b7613-7d51-4a1d-81b0-4ba0a21129df_cam_0/2023-05-09-13_cropping --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 2000 --data_size 2000 --n_framecodes 2000 --eval_metrics # --num_workers 1


# Chunjin - 01 - CANCELLED: restpose unscaled
# basic with mirror input (using a similar but not GT background, lets see if this works)
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname cj_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/c3abfea6-0634-4bb1-a254-3d5ab0bc652c_cam_0/2023-05-04-12_cropping --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --no_reload # --num_workers 1
# reload basic (trial: using a similar but not GT background, it works?)
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname cj_model/-2023-05-06-16-47-31-c0 --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/c3abfea6-0634-4bb1-a254-3d5ab0bc652c_cam_0/2023-05-04-12_cropping --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics #--num_workers 1

# Daniel - 02 - CANCELLED: restpose unscaled
# basic with mirror input
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname da_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/2f4b7613-7d51-4a1d-81b0-4ba0a21129df_cam_0/2023-05-04-12_cropping --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --no_reload --num_workers 1
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname da_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/2f4b7613-7d51-4a1d-81b0-4ba0a21129df_cam_0/2023-05-04-12_cropping --N_rand 3072 --i_testset 2 --i_pose_weights 2 --i_weights 2 --i_print 2 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --no_reload --num_workers 1
# reload basic
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname da_model/-2023-05-05-17-00-43-c0 --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/2f4b7613-7d51-4a1d-81b0-4ba0a21129df_cam_0/2023-05-04-12_cropping --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics #--num_workers 1


# Abi - subj 2
# two-view training
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ab_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --use_mirr --switch_cam --train_size 1800 --data_size 2000 --n_framecodes 1800 #--eval_metrics #--num_workers 1 
# basic with mirror input
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ab_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 1000 --i_pose_weights 1000 --i_weights 1000 --i_print 1000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --num_workers 1 --no_reload
# reload basic
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ab_model/-2023-03-25-11-57-35-c0 --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics #--num_workers 1 #--no_reload
#debug
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ab_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --num_workers 1 --no_reload



# Frank - subj 1
# two-view training
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --use_mirr --switch_cam --train_size 1800 --data_size 2000 --n_framecodes 1800 #--eval_metrics #--num_workers 1 
# basic with mirror input
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --no_reload --num_workers 1
# reload basic
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model/-2023-03-25-12-17-09-c0 --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics #--no_reload --num_workers 0 
#debug
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --no_reload --num_workers 1 

# Tim - subj 3 -> we already have Tim (two-view) from sockeye
# basic with mirror input
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ti_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1178 --data_size 1308 --n_framecodes 1178 --eval_metrics --no_reload --num_workers 1 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ti_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --use_mirr --switch_cam --train_size 1178 --data_size 1308 --n_framecodes 1178 #--eval_metrics #--num_workers 1 

# reload basic
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ti_model/-2023-03-25-12-40-01-c0 --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1178 --data_size 1308 --n_framecodes 1178 --eval_metrics #--no_reload #--num_workers 1 
## debug
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ab_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --use_mirr --switch_cam --train_size 1800 --data_size 2000 --n_framecodes 1800 --num_workers 0 # --eval_metrics
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname chunjin_model --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/dbcd9ff7-202a-484a-bfc4-6faea8c14c27_cam_0/2022-11-01-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --eval_metrics --no_reload

# test kp_map
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname test_kp_map_on_ab_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics  --no_reload --num_workers 1

### TODO: new segmentation for Daniel (done), c6, c3 and internet. Build data. No head cutting. Run refinement 72h.
# # c6
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1620 --data_size 1799 --n_framecodes 1620 --eval_metrics --num_workers 0 --no_reload
# # reload basic
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/xxxxx --data_path ./data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1620 --data_size 1799 --n_framecodes 1620 --eval_metrics --num_workers 1 #--no_reload

# # Internet
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname net_4l5h9ajj2wI_model --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/00af0626-d8fa-4c5c-a8c6-fd5486e496c1_cam_0/2022-11-01-13 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --num_workers 0 #--no_reload
# # reload basic
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname net_4l5h9ajj2wI_model/xxxxx --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/00af0626-d8fa-4c5c-a8c6-fd5486e496c1_cam_0/2022-11-01-13 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics #--num_workers 1 #--no_reload
# #debug

# # c3
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1620 --data_size 1800 --n_framecodes 1620 --eval_metrics --num_workers 0 --no_reload
# # reload basic
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/xxxxx --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --N_rand 3072 --i_testset 5 --i_pose_weights 5 --i_weights 5 --i_print 5 --train_size 1620 --data_size 1800 --n_framecodes 1620 --eval_metrics --num_workers 1 #--no_reload


### A-NeRF finetune (okay to ignore 'idx_map_in_ckpt' key in A-NeRF body modelling)

# # Abi - subj 2
# # two-view training
# # python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ab_model/-2023-03-25-11-57-35-c0 --ft_path logs/mirror/ab_model/-2023-03-25-11-57-35-c0/200000.tar --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --use_mirr --switch_cam --finetune --opt_pose_lrate 0.0 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image # --num_workers 1 
# reload

# # Frank - subj 1
# # two-view training
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model/-2023-03-25-12-17-09-c0 --ft_path logs/mirror/fr_model/-2023-03-25-12-17-09-c0/200000.tar --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --use_mirr --switch_cam --finetune --sub_fine_folder --opt_pose_lrate 0.0 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image # --num_workers 1 
# reload

# # softplus
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model/-2023-03-25-12-17-09-c0 --ft_path logs/mirror/fr_model/-2023-03-25-12-17-09-c0/200000.tar --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --use_mirr --switch_cam --finetune --sub_fine_folder --opt_pose_lrate 0.0 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image --density_type softplus # --num_workers 1 
# reload

# # color factor
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model/-2023-03-25-12-17-09-c0 --ft_path logs/mirror/fr_model/-2023-03-25-12-17-09-c0/200000.tar --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --use_mirr --switch_cam --finetune --sub_fine_folder --opt_pose_lrate 0.0 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image --opt_r_color --opt_v_color # --num_workers 1 
# reload

# # Tim - subj 3 -> from tim basic
# # two-view training
# # python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ti_model/-2023-03-25-12-40-01-c0 --ft_path logs/mirror/ti_model/-2023-03-25-12-40-01-c0/200000.tar --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0/2023-03-02-19 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1178 --data_size 1308 --n_framecodes 1178 --eval_metrics --use_mirr --switch_cam --finetune --opt_pose_lrate 0.0 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image # --num_workers 1
# reload


# test
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname fr_model/-2023-03-25-12-17-09-c0 --ft_path logs/mirror/fr_model/-2023-03-25-12-17-09-c0/200000.tar --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/55fa5330-0708-4ae0-9619-57058150bd4e_cam_0/2023-03-02-19 --N_rand 100 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 5000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --use_mirr --switch_cam --finetune --opt_pose_lrate 0.0 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image # --num_workers 1 --N_sample_images 2 --opt_r_color --opt_v_color

# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname ab_model --data_path /scratch/dajisafe/anerf_mirr/DANBO-pytorch/data/mirror/0/83bed111-cf7e-48cb-b75a-3fd92694280a_cam_0/2023-03-02-19 --N_rand 200 --i_testset 1000 --i_pose_weights 1000 --i_weights 1000 --i_print 1000 --train_size 1800 --data_size 2000 --n_framecodes 1800 --eval_metrics --use_mirr --switch_cam --num_workers 1 --no_reload --N_sample_images 2 --opt_r_color --opt_v_color

#c3 template
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-29-01-23-04-c3 --ft_path logs/mirror/pose_opt_model/-2022-10-29-01-23-04-c3/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image
# CONTINUE FINE-TUNING (please update) 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-10-23-15-23-54/finetune --ft_path logs/mirror/pose_opt_model/-2022-10-23-15-23-54/finetune/200000.tar  --data_path /project/st-rhodin-1/users/dajisafe/mirror_anerf/data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --overlap_rays --finetune --opt_pose_lrate 0.0 --opt_pose_stop 1 --lrate_decay 200000 --lrate_decay_rate 0.25 --mask_image





## CVPR 23

# RENDER
# internet
#python run_render.py --nerf_args logs/mirror/net_4l5h9ajj2wI_model/-2022-11-05-03-31-16-c0/args.txt --ckptpath logs/mirror/net_4l5h9ajj2wI_model/-2022-11-05-03-31-16-c0/080000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/00af0626-d8fa-4c5c-a8c6-fd5486e496c1_cam_0/2022-11-01-13 --selected_idxs 144 --n_bullet 4 --bullet_ang 360
# chunjin
#python run_render.py --nerf_args logs/mirror/chunjin_model/-2022-11-04-22-03-24-c0/args.txt --ckptpath logs/mirror/chunjin_model/-2022-11-04-22-03-24-c0/070000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/dbcd9ff7-202a-484a-bfc4-6faea8c14c27_cam_0/2022-11-01-13 --selected_idxs 144 --n_bullet 4 --bullet_ang 360

# fresh start

# indoor - Tim
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname chunjin_model --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/dbcd9ff7-202a-484a-bfc4-6faea8c14c27_cam_0/2022-11-01-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --eval_metrics --no_reload


# outdoor - chunjin
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname chunjin_model --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/dbcd9ff7-202a-484a-bfc4-6faea8c14c27_cam_0/2022-11-01-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --eval_metrics --no_reload

# internet
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname net_4l5h9ajj2wI_model --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/00af0626-d8fa-4c5c-a8c6-fd5486e496c1_cam_0/2022-11-01-13 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --eval_metrics --no_reload 

# debug fresh
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname net_4l5h9ajj2wI_model --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/00af0626-d8fa-4c5c-a8c6-fd5486e496c1_cam_0/2022-11-01-13 --N_rand 1500 --i_testset 1 --i_pose_weights 1 --i_weights 1 --no_reload --i_print 1 --use_mirr --switch_cam --eval_metrics --num_workers 1


# reload models
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname chunjin_model/-2022-11-04-22-03-24-c0 --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/dbcd9ff7-202a-484a-bfc4-6faea8c14c27_cam_0/2022-11-01-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --eval_metrics
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname net_4l5h9ajj2wI_model/-2022-11-05-03-31-16-c0 --data_path /scratch/dajisafe/anerf_mirr/A-NeRF/data/mirror/0/00af0626-d8fa-4c5c-a8c6-fd5486e496c1_cam_0/2022-11-01-13 --N_rand 3072 --i_testset 5000 --i_pose_weights 5000 --i_weights 5000 --i_print 2500 --use_mirr --switch_cam --eval_metrics


# ----------------------------------------------------------------
# daniel_mirr_anerf branch  **** update model and data path*****
####### reload
# (use mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-13-13-34-31 --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --use_mirr
# old mirr-in-anerf: python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-13-13-34-31 --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-06-18 


#ABLATION
####### reload
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-24-16-25-54 --data_path ./data/mirror/3/a77b39fa-64de-418a-9d93-22fb5b29f0d9_cam_3/2022-05-23-05 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-24-16-38-27 --data_path ./data/mirror/3/aa5355f2-1ce4-45ca-a38f-2e0e316d3b2f_cam_3/2022-05-23-05 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-24-16-46-27 --data_path ./data/mirror/3/83d417be-693d-4cd6-8d4a-471ddfe63d7d_cam_3/2022-05-23-05 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-24-17-07-06 --data_path ./data/mirror/4/0dc04e49-25ec-4225-8fb6-769c7fe83c74_cam_3/2022-05-23-05 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-24-18-36-00 --data_path ./data/mirror/3/dd08d0b8-dfd9-4fbc-a15b-1f93c9d98863_cam_3/2022-05-23-05 

# ABLATION
####### no reload
# (no mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/a77b39fa-64de-418a-9d93-22fb5b29f0d9_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/aa5355f2-1ce4-45ca-a38f-2e0e316d3b2f_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/83d417be-693d-4cd6-8d4a-471ddfe63d7d_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/0dc04e49-25ec-4225-8fb6-769c7fe83c74_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/dd08d0b8-dfd9-4fbc-a15b-1f93c9d98863_cam_3/2022-05-23-05 --no_reload --opt_pose_stop 200000


# FRESH START
####### no reload
# (no mirror)
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --no_reload --opt_pose_stop 200000


# CHECK WHERE POSE_STOP STARTS/RELOADS FROM
# (no mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-16-02-10-36 --data_path ./data/mirror/2//2022-05-14-13 --opt_pose_stop 0
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-16-02-01-28 --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --opt_pose_stop 0
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-16-14-42-42 --data_path ./data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --opt_pose_stop 0
# python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-16-02-23-39 --data_path ./data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --opt_pose_stop 0
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-2022-05-16-14-12-18 --data_path ./data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --opt_pose_stop 0


# 2022-05-16-02-10-36 (cam2) - 407k
# 2022-05-16-02-01-28 (cam3) - 451k
# 2022-05-16-14-42-42 (cam5) - 367k
# 2022-05-16-02-23-39 (cam6) - 372k
# 2022-05-16-14-12-18 (cam7) - 365k


# FORMER ABLATION
# -2022-05-24-16-25-54 /32022-05-23-05 
# -2022-05-24-16-38-27 /32022-05-23-05 
# -2022-05-24-16-46-27 /32022-05-23-05 
# -2022-05-24-17-07-06 /42022-05-23-05 
# -2022-05-24-18-36-00 /32022-05-23-05 



# run_render (works now)
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-10-36/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-10-36/407000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --n_bullet 4 --bullet_ang 360
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 4 --bullet_ang 360
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-42-42/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-42-42/367000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --n_bullet 4 --bullet_ang 360
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-23-39/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-23-39/372000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 4 --bullet_ang 360
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-12-18/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-12-18/365000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --n_bullet 4 --bullet_ang 360

# salloc --time=02:00:00 --gres=gpu:v100l:1 --ntasks=8 --nodes=1 --mem=64G --account=def-rhodin

# ####validation
# #python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry val --render_type val --runname mirror_val --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --test_len 180 --eval

# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-10-36/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-10-36/407000.tar --dataset mirror --entry val --render_type val --runname mirror_val --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --test_len 157 --eval
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry val --render_type val --runname mirror_val --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --test_len 180 --eval
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-42-42/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-42-42/367000.tar --dataset mirror --entry val --render_type val --runname mirror_val --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --test_len 137 --eval
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-23-39/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-23-39/372000.tar --dataset mirror --entry val --render_type val --runname mirror_val --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --test_len 179 --eval
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-12-18/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-12-18/365000.tar --dataset mirror --entry val --render_type val --runname mirror_val --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --test_len 173 --eval


# # images
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry val --render_type val --runname mirror_val --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --test_len 180 


####bullet
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-10-36/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-10-36/407000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --n_bullet 5 --bullet_ang 60
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bullet 30 --bullet_ang 360
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-42-42/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-42-42/367000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --n_bullet 5 --bullet_ang 60
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-23-39/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-23-39/372000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bullet 5 --bullet_ang 60
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-12-18/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-12-18/365000.tar --dataset mirror --entry easy --render_type bullet --runname mirror_bullet --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --n_bullet 5 --bullet_ang 60

####bubble


# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-10-36/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-10-36/407000.tar --dataset mirror --entry easy --render_type bubble --runname mirror_bubble --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/2/fc4f46b9-1f80-4498-8949-ca1b52864d0c_cam_2/2022-05-14-13 --n_bubble 4
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry easy --render_type bubble --runname mirror_bubble --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --n_bubble 15
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-42-42/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-42-42/367000.tar --dataset mirror --entry easy --render_type bubble --runname mirror_bubble --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --n_bubble 4
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-23-39/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-23-39/372000.tar --dataset mirror --entry easy --render_type bubble --runname mirror_bubble --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/6/ea8ddac0-6837-4434-b03a-09316277a4aa_cam_6/2022-05-14-13 --n_bubble 4

# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-14-12-18/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-14-12-18/365000.tar --dataset mirror --entry easy --render_type bubble --runname mirror_bubble --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --n_bubble 4

# ####interpolation
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry easy --render_type interpolate --runname mirror_interp --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13

# ####animate
# python run_render.py --nerf_args logs/mirror/pose_opt_model/-2022-05-16-02-01-28/args.txt --ckptpath logs/mirror/pose_opt_model/-2022-05-16-02-01-28/451000.tar --dataset mirror --entry easy --render_type animate --runname mirror_anim --selected_framecode 0 --white_bkgd --render_refined --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13


####### no reload
# (use mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --no_reload --use_mirr #--opt_pose_stop 200000 
# (no mirror)
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/5/c28e8104-b416-474c-914c-c911baa8540b_cam_5/2022-05-14-13 --no_reload #--opt_pose_stop 200000



##### testing line
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-14-13 --no_reload --opt_pose_stop 200000 --use_mirr --num_workers 1 --i_print 3 --i_weights 3 --i_pose_weights 3 --i_testset 3 --n_iters 7 

#--i_print 3 --i_weights 3 --i_pose_weights 3 --i_testset 3 --n_iters 7 


# mesh
#python run_render.py --nerf_args logs/mirror/pose_opt_model/-000_date/args.txt --ckptpath logs/mirror/pose_opt_model/-000_date/230000.tar --dataset mirror --entry easy --render_type mesh --runname mirror_mesh --data_path ./data/mirror/3/23df3bb4-272d-4fba-b7a6-514119ca8d21_cam_3/2022-05-06-18
#python render_mesh.py --expname mirror_mesh 




# run_nerf (with reload)
# Base (w/o mirror) - daniel branch  **** update model and data path*****
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model/-000_date --data_path ./data/mirror/3/f8480b9b-b1e2-4bf6-a7cf-a34c19244afd_cam_3/2022-05-03-22 
#python run_nerf.py --config configs/mirror/mirror.txt --basedir logs/mirror --expname pose_opt_model --data_path ./data/mirror/7/261970f0-e705-4546-a957-b719526cbc4a_cam_7/2022-05-14-13 --no_reload --opt_pose_stop 200000
#--num_workers 1 --i_print 3 --i_weights 3 --i_pose_weights 3 --i_testset 3 --n_iters 7 


# #!ffmpeg -y -f rawvideo -vcodec rawvideo -s 1920x1080 -pix_fmt rgb24 -r 14.00 -i - -an -vcodec libx264 -pix_fmt yuv420p -crf 25 -v warning /scratch/dajisafe/anerf_mirr/A-NeRF/render_output/mirror_bullet/render_rgb_no_skel.mp4
# --------------------------------------------------------------------------------------------------------------------------------------


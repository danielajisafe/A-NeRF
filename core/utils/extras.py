import os,sys
from re import L
import torch
import numpy as np
import torch.nn.functional as F

# sys.path.append("../")
# sys.path.append("./")
# sys.path.append("/scratch/dajisafe/smpl/mirror_project")

'''This script is a local copy from the initial mirror project
Lesson: If you have a function that does the same, just use
instead of re-implementing'''

import torch
import torch.nn as nn
import numpy as np
#from mirror_project_dir.transforms import flip_h36m
#from utils import rot6d_to_rotmat, axisang_to_rot
from .skeletons import CMUSkeleton
from .dan_skeleton_utils import get_parent_idx, verify_get_parent_idx

#import torch.nn.functional as f 
#from utils import axisang_to_rot6d

# custom imports
#from transforms import rotate_initial_pose



def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)

    w = norm_quat[:, 0]
    x = norm_quat[:, 1]
    y = norm_quat[:, 2]
    z = norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

#     rotMat = torch.stack([
#         w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
#         w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
#         w2 - x2 - y2 + z2
#     ], dim=1).view(batch_size, 3, 3)
    
    
    # NB: similar to Rodrigues' Rotation Formula and Zhou et al. CVPR 2019 Paper
    
    rotMat = torch.stack([
        w2 + x2 - y2 - z2, #checked 1
        2 * xy - 2 * wz, #checked 2
        2 * wy + 2 * xz, #checked 3
        2 * wz + 2 * xy, #checked 4
        w2 - x2 + y2 - z2, #checked 5
        2 * yz - 2 * wx, #checked 6
        2 * xz - 2 * wy, #checked 7
        2 * wx + 2 * yz, #checked 8
        w2 - x2 - y2 + z2 #checked 9
    ], dim=1).view(batch_size, 3, 3)

    return rotMat

def axisang_to_rot(axisang):
    """
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py  # I like this :) 
    https://github.com/nkolot/SPIN/blob/5c796852ca7ca7373e104e8489aa5864323fbf84/utils/geometry.py#L9
    Args:
        The axis/rotation angle same as theta: size = [B, 3] in degree
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]    
    """

    #converts angle from degree to radian first
    angle = torch.norm(axisang + 1e-8, p=2, dim=-1)[..., None]
 
    axisang_norm = axisang / angle
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)

    quat = torch.cat([v_cos, v_sin * axisang_norm], dim=-1)
    rot = quat2mat(quat)
    
    return rot


def pad_mat_to_homogeneous(mat):
    """expects (3,4)"""
    last_row = torch.tensor([[0., 0., 0., 1.]]).to(mat.device)
    if mat.dim() == 3:
        last_row = last_row.expand(mat.size(0), 1, 4)
    return torch.cat([mat, last_row], dim=-2)

def mat_to_homo(mat):
    """expects (3,4)"""
    last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
    return np.concatenate([mat, last_row], axis=0)

# TODO: Finish numpy version: Legacy code
class KinematicChain_Numpy(nn.Module):

    def __init__(self, rest_pose, skeleton_type=CMUSkeleton, use_rot6d=False, theta_shape=None, where=None):
        """
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        use_rot6d: bool, to use 6d rotation instead of 3d axis-angle representation.
                   (see: https://arxiv.org/abs/1812.07035) - continous 6D rotation
                   
        Paper notes: We use Adam optimization with batch size 64 and learning rate 10e−5 for 
        the first 10^4 iterations and 10e−6 for the remaining iterations.
        """
        super().__init__()

        assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"
        self.rest_pose = rest_pose
        self.skeleton_type = skeleton_type
        self.use_rot6d = use_rot6d
        
    def forward(self, theta, bone_factor=None, rest_pose=None, skeleton_type=None, **k_pipe_kwargs):
        """
        theta: float32, (B, N_joints, 3) or (B, N_joints, 6). SMPL pose parameters in axis-angle
               or 6d representation.
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        """

        if rest_pose is None:
            rest_pose = self.rest_pose
            B, N_J, _ = theta.shape # (B, N_J, _) 
            
            # match theta's dimention
            rest_pose = rest_pose[None].repeat(B, 0)
            #rest_pose = rest_pose[None].expand(B, N_J, 3)
            
        if skeleton_type is None:
            skeleton_type = self.skeleton_type
        else:
            assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"

        idx = get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        verify_get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        joint_trees = np.array(idx)
        
        root_id = skeleton_type.root_id
        B, N_J, _ = theta.shape
        #import ipdb; ipdb.set_trace()

        # turn rotation parameters (joint angles) into the proper 3x3 rotation matrices
        if self.use_rot6d:
            #theta6d = axisang_to_rot6d(theta) #converts axis-angle to proper 3x3 (same as 9), then to 6D rep
            rots = rot6d_to_rotmat(theta.reshape(-1, 6)).reshape(B, N_J, 3, 3) #converts from 6D rep to proper 3x3 again
        else:
            '''I assume theta is by defaut in axis-angle representation'''
            rots = axisang_to_rot(theta.reshape(-1, 3)).reshape(B, N_J, 3, 3) #coverts axis-angle to proper 3x3 

        
        #print("is the det of our Rotation matrices +ve?", torch.det(rots) > 0)   
        #import ipdb; ipdb.set_trace()

        # l2w: local-to-world transformation.
        # concatenate the rotation and translation of the root joint
        # to get a 3x4 matrix |R T|
        #                     |0 1|

        root_l2w = torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1)
        import ipdb; ipdb.set_trace()
        root_l2w = np.concatenate([rots[:, root_id], rest_pose[:, root_id, :, None]], axis=2)
        # pad it to 4x4
        root_l2w = mat_to_homo(root_l2w)
        #root_l2w = pad_mat_to_homogeneous(root_l2w) 

        # assume root_id == 0,
        # this is the per-joint-local to-world matrices
        l2ws = [root_l2w]

        # collect all rotations/translation except for the root one
        # B x (N_J - 1) x 3 x 3
        children_rots = torch.cat([rots[:, :root_id], rots[:, root_id+1:]], dim=1)
        # B x (N_J - 1) x 3 x 1
        children_trans = torch.cat([rest_pose[:, :root_id], rest_pose[:, root_id+1:]], dim=1)[..., None]
        parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
        # B x (N_J - 1) x 3 x 1
        parent_trans = rest_pose[:, parent_ids, :, None]
        # B x (N_J - 1) x 3 x 4, matrix |R T|
        bv = children_trans - parent_trans # (N, 15, 3, 1)

        if bone_factor is not None:
            # use optimized bonelength to scale normalized bone vectors
            '''normalize is diff to norm, normalize would normalize and keep the vector form to unit length but norm e.g torch norm gives the length/magnitude of each vector'''
            #bv = f.normalize(bv, p=2, dim=-2, eps=1e-12) * bone_factor 
            
            '''Initialize with true bv and only add free parameter as an offset (and not a scale)'''
            #bv = bv + bone_factor 
            eps = 1e-36
            bone_factor = torch.sqrt(bone_factor**2 + eps)
            if torch.all(bone_factor>0).item() != True:
                import ipdb; ipdb.set_trace()
            assert torch.all(bone_factor>0).item() == True, "bone factor should be positive"
            bv = bv * bone_factor  #bv * bone_factor # fixes the head issue
            
        # concatenate the rotation and translation of other joints
        joint_rel_transforms = torch.cat([children_rots, bv], dim=-1) # --> [Rotation | BONE vectors]

        '''optimized rotation + bone vectors'''
        # pad to 4 x 4: |R T|
        #               |0 1|
        joint_rel_transforms = pad_mat_to_homogeneous(joint_rel_transforms.reshape(-1, 3, 4))
        joint_rel_transforms = joint_rel_transforms.reshape(B, N_J-1, 4, 4) # (N, 15, 4, 4)

        
        '''Run kinematic chain here successively, starting with the root rotation 
        at [zero pose|identity matrix| x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) + 3d translation| point]
        against other joints at [zero pose + bone vector]
        '''
        for i, parent in enumerate(parent_ids): #(15)
            l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
        l2ws = torch.stack(l2ws, dim=-3) # (N, 16, 4, 4)
  
        # the 3d keypoints are the translation part of the final
        # per-joint local-to-world matrices
        kp3d = l2ws[..., :3, -1] # (N, 16, 3)

        # Dan: the orientation of the joints are the final
        # rotational part of the per-joint local-to-world matrices
        orient = l2ws[..., :3, :3]

        return kp3d, orient, l2ws, bone_factor


''' I added a copy of Script to A-NeRF code'''

class KinematicChain(nn.Module):

    def __init__(self, rest_pose, skeleton_type=CMUSkeleton, use_rot6d=False, theta_shape=None, where=None):
        """
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        use_rot6d: bool, to use 6d rotation instead of 3d axis-angle representation.
                   (see: https://arxiv.org/abs/1812.07035) - continous 6D rotation
                   
        Paper notes: We use Adam optimization with batch size 64 and learning rate 10e−5 for 
        the first 10^4 iterations and 10e−6 for the remaining iterations.
        """
        super().__init__()

        assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"
        self.rest_pose = rest_pose
        self.skeleton_type = skeleton_type
        self.use_rot6d = use_rot6d
        #self.on_device = on_device
        
    def forward(self, theta, bone_factor=None, rest_pose=None, skeleton_type=None, **k_pipe_kwargs):
        """
        theta: float32, (B, N_joints, 3) or (B, N_joints, 6). SMPL pose parameters in axis-angle
               or 6d representation.
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        """

        if rest_pose is None:
            rest_pose = self.rest_pose
            B, N_J, _ = theta.shape # (B, N_J, _) 
            # match theta's dimention
            rest_pose = rest_pose[None].expand(B, N_J, 3)
            
        if skeleton_type is None:
            skeleton_type = self.skeleton_type
        else:
            assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"

        idx = get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        verify_get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        joint_trees = np.array(idx)
        
        root_id = skeleton_type.root_id
        B, N_J, _ = theta.shape
        #import ipdb; ipdb.set_trace()

        # turn rotation parameters (joint angles) into the proper 3x3 rotation matrices
        if self.use_rot6d:
            #theta6d = axisang_to_rot6d(theta) #converts axis-angle to proper 3x3 (same as 9), then to 6D rep
            rots = rot6d_to_rotmat(theta.view(-1, 6)).view(B, N_J, 3, 3) #converts from 6D rep to proper 3x3 again
        else:
            '''I assume theta is by defaut in axis-angle representation'''
            rots = axisang_to_rot(theta.view(-1, 3)).view(B, N_J, 3, 3) #coverts axis-angle to proper 3x3 

        #print("is the det of our Rotation matrices +ve?", torch.det(rots) > 0)   
        #import ipdb; ipdb.set_trace()

        # l2w: local-to-world transformation.
        # concatenate the rotation and translation of the root joint
        # to get a 3x4 matrix |R T|
        #                     |0 1|
        root_l2w = torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1)
        # pad it to 4x4
        root_l2w = pad_mat_to_homogeneous(root_l2w)
        #import ipdb; ipdb.set_trace()

        # assume root_id == 0,
        # this is the per-joint-local to-world matrices
        l2ws = [root_l2w]

        # collect all rotations/translation except for the root one
        # B x (N_J - 1) x 3 x 3
        children_rots = torch.cat([rots[:, :root_id], rots[:, root_id+1:]], dim=1)
        # B x (N_J - 1) x 3 x 1
        children_trans = torch.cat([rest_pose[:, :root_id], rest_pose[:, root_id+1:]], dim=1)[..., None]
        parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
        # B x (N_J - 1) x 3 x 1
        parent_trans = rest_pose[:, parent_ids, :, None]
        # B x (N_J - 1) x 3 x 4, matrix |R T|
        bv = children_trans - parent_trans # (N, 15, 3, 1)

        if bone_factor is not None:
            # use optimized bonelength to scale normalized bone vectors
            '''normalize is diff to norm, normalize would normalize and keep the vector form to unit length but norm e.g torch norm gives the length/magnitude of each vector'''
            #bv = f.normalize(bv, p=2, dim=-2, eps=1e-12) * bone_factor 
            
            '''Initialize with true bv and only add free parameter as an offset (and not a scale)'''
            #bv = bv + bone_factor 
            eps = 1e-36
            bone_factor = torch.sqrt(bone_factor**2 + eps)
            if torch.all(bone_factor>0).item() != True:
                import ipdb; ipdb.set_trace()
            assert torch.all(bone_factor>0).item() == True, "bone factor should be positive"
            bv = bv * bone_factor  #bv * bone_factor # fixes the head issue
            
        # concatenate the rotation and translation of other joints
        joint_rel_transforms = torch.cat([children_rots, bv], dim=-1) # --> [Rotation | BONE vectors]

        '''optimized rotation + bone vectors'''
        # pad to 4 x 4: |R T|
        #               |0 1|
        joint_rel_transforms = pad_mat_to_homogeneous(joint_rel_transforms.view(-1, 3, 4))
        joint_rel_transforms = joint_rel_transforms.view(B, N_J-1, 4, 4) # (N, 15, 4, 4)

        
        '''Run kinematic chain here successively, starting with the root rotation 
        at [zero pose|identity matrix| x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) + 3d translation| point]
        against other joints at [zero pose + bone vector]
        '''
        for i, parent in enumerate(parent_ids): #(15)
            l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
        l2ws = torch.stack(l2ws, dim=-3) # (N, 16, 4, 4)
  

        # the 3d keypoints are the translation part of the final
        # per-joint local-to-world matrices
        kp3d = l2ws[..., :3, -1] # (N, 16, 3)

        # Dan: the orientation of the joints are the final
        # rotational part of the per-joint local-to-world matrices
        orient = l2ws[..., :3, :3]

        return kp3d, orient, l2ws, bone_factor

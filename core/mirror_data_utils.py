# ref: https://stackoverflow.com/a/67334192/12761745

import os
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm, trange

from util_loading import load_pickle, save2pickle


project_dir = "/scratch/dajisafe/smpl/mirror_project_dir/"

def mask_chunking(h5_key,data_path,p_type,chunk_factor=24,split="train", args=None):
    Not_avail=False
    
    if not os.path.exists(project_dir+f"/authors_eval_data/temp_dir/seg_files/{args.cam}/{args.comb}/{args.timestamp}/{p_type}_masks_0.pickle"): #exists
        # find other combination under the same camera
        found = glob(project_dir+f"/authors_eval_data/temp_dir/seg_files/{args.cam}/*/*/{p_type}_masks_0.pickle")
        print(f"Found: {found[0] if len(found)>0 else found}")
        if len(found) > 0:
            _ = input("re-using data from other comb but same camera")
            Not_avail=True
            found_list = found[0].split("/")
            new_comb, new_timestamp = found_list[-3], found_list[-2]
        #import ipdb; ipdb.set_trace()
    
    catg = "" if p_type == "real" else "v_"
    data_file = data_path + f'{args.cam}/{args.comb}/{args.timestamp}/'+catg+f'mirror_{split}_h5py.h5'
    #import ipdb; ipdb.set_trace()
    
    f = h5py.File(data_file, 'a')
    if h5_key in f.keys():
        del f[h5_key]  
    
    for i in trange(B):
        load_filename = project_dir+f"/authors_eval_data/temp_dir/seg_files/{args.cam}/{args.comb}/{args.timestamp}/{p_type}_masks_{i}.pickle"
        if Not_avail:
            load_filename = load_filename.replace(args.comb, new_comb)
            load_filename = load_filename.replace(args.timestamp, new_timestamp)
            if i ==0:
                print(f"Now using: {load_filename}") 
                #import ipdb; ipdb.set_trace()
                
        from_pickle = load_pickle(load_filename)
        real_mask = from_pickle[f"{p_type}_mask_{i}"].astype(np.float32)
        #plt.imshow(real_mask)

        if len(real_mask.shape)>2:
            H,W,C = real_mask.shape
        else:
            H,W = real_mask.shape
            C=1
        
        chunk_size = (1, int(H/chunk_factor) * int(W/chunk_factor)) + (C,)
        flatten_shape = (1,H*W,C)
        #import ipdb; ipdb.set_trace()

        # Data to be appended
        new_data = real_mask.reshape(flatten_shape)

        if i == 0:
            # Create the dataset at first
            f.create_dataset(h5_key, data=new_data, chunks=chunk_size,compression='gzip', maxshape=(None,1080*1920,1))

        else:
            # Append new data to it
            f[h5_key].resize((f[h5_key].shape[0] + new_data.shape[0]), axis=0)
            f[h5_key][-new_data.shape[0]:] = new_data
        tqdm.write("I am on iteration {} and 'masks' chunk has shape:{}".format(i,f[h5_key].shape))
    f.close()



# ref: https://stackoverflow.com/a/67334192/12761745

import numpy as np
import h5py


def samp_mask_chunking(h5_key,data_path,p_type,chunk_factor=24,split="train",args=None):
    name = "r" if p_type == "real" else "v"
    
    Not_avail=False
    if not os.path.exists(project_dir+f"/authors_eval_data/temp_dir/seg_files/{args.cam}/{args.comb}/{args.timestamp}/{name}_samp_masks_0.pickle"): #exists
        # find other combination under the same camera
        found = glob(project_dir+f"/authors_eval_data/temp_dir/seg_files/{args.cam}/*/*/{name}_samp_masks_0.pickle")
        print(f"Found: {found[0] if len(found)>0 else found}")
        if len(found) > 0:
            _ = input("re-using data from other comb but same camera")
            Not_avail=True
            found_list = found[0].split("/")
            new_comb, new_timestamp = found_list[-3], found_list[-2]
    
    #import ipdb; ipdb.set_trace()
    catg = "" if p_type == "real" else "v_"
    data_file = data_path + f'{args.cam}/{args.comb}/{args.timestamp}/'+catg+f'mirror_{split}_h5py.h5'
    #import ipdb; ipdb.set_trace()
    
    f = h5py.File(data_file, 'a')
    if h5_key in f.keys():
        del f[h5_key] 
    for i in trange(B):
        load_filename = project_dir+f"/authors_eval_data/temp_dir/seg_files/{args.cam}/{args.comb}/{args.timestamp}/{name}_samp_masks_{i}.pickle"
        if Not_avail:
            load_filename = load_filename.replace(args.comb, new_comb)
            load_filename = load_filename.replace(args.timestamp, new_timestamp)
            if i ==0:
                print(f"Now using: {load_filename}") 
                #import ipdb; ipdb.set_trace()
                
        from_pickle = load_pickle(load_filename)
        r_samp_mask = from_pickle[f"{name}_samp_mask_{i}"].astype(np.float32)

        if len(r_samp_mask.shape)>2:
            H,W,C = r_samp_mask.shape
        else:
            H,W = r_samp_mask.shape
            C=1

        chunk_size = (1, int(H/chunk_factor) * int(W/chunk_factor)) + (C,)
        flatten_shape = (1,H*W,C)
        #import ipdb; ipdb.set_trace()

        # Data to be appended
        new_data = r_samp_mask.reshape(1,1080*1920,1) 
        #print(torch.Tensor(r_samp_mask.reshape(-1)).unique())

        if i==0:
            # Create the dataset at first
            f.create_dataset(h5_key, data=new_data, chunks=chunk_size,compression='gzip', maxshape=(None,1080*1920,1))

        else:
            # Append new data to it
            f[h5_key].resize((f[h5_key].shape[0] + new_data.shape[0]), axis=0)
            f[h5_key][-new_data.shape[0]:] = new_data
        tqdm.write("I am on iteration {} and 'sampling_masks' chunk has shape:{}".format(i,f['sampling_masks'].shape))
    f.close()



def save_img_shape_n_bkgd(split="train", data_file=None,args=None,):
    "Run seperately"
    #ref: https://stackoverflow.com/a/22925117/12761745
    
    data_file = data_file + f'{args.cam}/{args.comb}/{args.timestamp}/mirror_{split}_h5py.h5'
    f1 = h5py.File(data_file, 'r+')     # open the file

    copy = f1['imgs'][:].copy()
    if "img_shape" in f1.keys():
        del f1['img_shape']
    f1['img_shape'] = copy.reshape(1800,1080,1920,3).shape
    print(f1['img_shape'][:])

    if "bkgds" in f1.keys():
        del f1['bkgds']
    bkgd = np.median(copy, axis=0).reshape(1,-1,3).astype(np.uint8)
    f1['bkgds'] = bkgd

    # chk_img = bkgd.reshape(1080,1920,3)
    # plt.axis("off")
    # plt.imshow(chk_img)
    f1.close()  




def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    '''Store true in args is true only when its called (not default)'''

    parser.add_argument("--comb", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--timestamp", type=str, default='./data/llff/fern',
                        help='input data directory')

                    
def train():

    parser = config_parser()
    args = parser.parse_args()

    split="train"
    data_path = '/scratch/dajisafe/smpl/A_temp_folder/A-NeRF/data/mirror/'

    mask_chunking(h5_key='masks',data_path=data_path,p_type="virt",chunk_factor=24, split="train",args=args)
    samp_mask_chunking(h5_key='sampling_masks',data_path=data_path,p_type="virt",chunk_factor=24, split="train",args=args)

    # this takes time
    save_img_shape_n_bkgd(split=split,data_file=data_path,args=args)

    print("is 3600 still your chunk size? for youTube videos, No.")

if __name__=='__main__':
    train()
import cv2
import ipdb
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from core.utils.skeleton_utils import draw_skeleton2d, SMPLSkeleton


def cca_image(kps, BGR_img=None, acc_img=None, img_url=None, plot=False, x_margin=200, y_margin=100, chk_folder=None):
    # ref: https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
    # others: https://stackoverflow.com/a/46442154/12761745 | https://stackoverflow.com/a/51524749/12761745 | https://stackoverflow.com/a/35854198/12761745

    if img_url is not None:
        BGR_img = cv2.imread(img_url)
#         plt.imshow(BGR_img[..., [2,1,0]])
    # else:
    #     # to BGR
    #     BGR_img = image.copy()
    #     # ipdb.set_trace # whats the pixel range?
    
    H,W,C = BGR_img.shape
    
    kps = kps.astype(int)
    # add allowable margin
    min_y, max_y = kps[:,1].min(), kps[:,1].max()
    min_x, max_x = kps[:,0].min(), kps[:,0].max()
    
    min_y = max(0, min_y-y_margin)
    min_x = max(0, min_x-x_margin)
    max_y = min(H, max_y+y_margin)
    max_x = min(W, max_x+x_margin)
    
    # ipdb.set_trace()
    # crop image
    cropped_img = BGR_img[min_y:max_y, min_x:max_x]
    if plot:
        plt.imshow(cropped_img); 
        plt.savefig(f"{chk_folder}/cropped_img.png")
    
    gray_img = cv2.cvtColor(cropped_img , cv2.COLOR_BGR2GRAY)
    
    # 7x7 Gaussian Blur, threshold and component analysis function
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # pre-compute kp mask
    kps_mask = np.ones((H,W), dtype=np.uint8)
    for kp in kps:
        kps_mask[kp[1], kp[0]] = 1
    
#     ipdb.set_trace()
    # plug crop back into full image size
    label_ids_full = np.zeros((H,W), dtype=np.uint8) # start with 0s | usually stand for background
    label_ids_full[min_y:max_y, min_x:max_x] = label_ids 
    
    # get closest component
    composite = label_ids_full*kps_mask
    vals = composite[np.nonzero(composite)]
    component_label, freq = stats.mode(vals)
    # found_locs = np.transpose(np.nonzero(composite))
    
    # extract component mask
    mask = np.zeros((H,W), dtype=np.uint8)
    mask[label_ids_full == component_label] = 1
    
    if plot:
        plt.scatter(kps[:,1], kps[:,0], linewidth=0.5, color="r")
        plt.imshow(mask); plt.show()#; mask.min(); mask.max()
        plt.savefig(f"{chk_folder}/mask.png")

    # filter image with component mask
    # RGB_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB) # faster if cv2 already imported
    filtered_rgb = BGR_img*mask[:,:,None]
    filtered_acc = acc_img*mask[:,:,None]
    
    
    # white background
    rgb_white_bkgd = np.zeros_like(filtered_rgb)
    skel_white_bkgd = np.zeros_like(filtered_rgb)
    rgb_white_bkgd[label_ids_full != component_label] = 255
    # skel_white_bkgd[skel_img == 0] = 255
    # skel_white_bkgd[skel_img != 0] = 0

    rgb = rgb_white_bkgd + filtered_rgb
    skel_img = draw_skeleton2d(rgb, kps, skel_type=SMPLSkeleton, width=3, flip=False)
    # skel = skel_white_bkgd + skel_img
    return rgb, filtered_acc, skel_img
    
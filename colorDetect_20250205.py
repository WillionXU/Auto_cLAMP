import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import pandas as pd
import mmd_numpy_sklearn as mmd

import numpy as np


import numpy as np

def mmd_periodic_linear_binning(X, Y, period=360.0, B=360):

    X = X.flatten()
    Y = Y.flatten()
    n_sample1 = len(X)
    n_sample2 = len(Y)


    hist_x, _ = np.histogram(X, bins=B, range=(0, period))
    hist_y, _ = np.histogram(Y, bins=B, range=(0, period))


    bin_centers = np.linspace(0.5*(period/B), period - 0.5*(period/B), B)

    diff = np.abs(bin_centers[:, None] - bin_centers[None, :])
    diff = np.minimum(diff, period - diff)

    K_bins = np.cos(np.pi * diff / period)  # shape (B, B)


    term_xx = np.sum(hist_x[None, :] * hist_x[:, None] * K_bins) / (n_sample1**2)
    term_xy = np.sum(hist_x[None, :] * hist_y[:, None] * K_bins) / (n_sample1 * n_sample2)
    term_yy = np.sum(hist_y[None, :] * hist_y[:, None] * K_bins) / (n_sample2**2)

    return term_xx - 2*term_xy + term_yy



def mmd_periodic_linear(X, Y, period=360.0):
    """
    MMD using a 'periodic linear' (cosine-based) kernel to handle hue's circular nature.

    Arguments:
        X : np.ndarray of shape (n_sample1,) or (n_sample1, d)
            - If it's multi-dim, you'd need to adapt the kernel calculation.
            - Here we assume at least the first dimension (or the only dimension) is Hue in [0, period).
        Y : np.ndarray of shape (n_sample2,) or (n_sample2, d)
        period : float
            - The period for the hue dimension (default: 360.0).

    Returns:
        MMD_value : float
            - The estimated MMD^2 between distributions P and Q under the periodic linear kernel.
    """


    X = X.reshape(-1)  
    Y = Y.reshape(-1)

    n_sample1 = X.shape[0]
    n_sample2 = Y.shape[0]

 
    dist_xx = np.abs(X[:, None] - X[None, :])              # |H_i - H_j|
    dist_xx = np.minimum(dist_xx, period - dist_xx)        # min(d, period - d)
  
    Kxx = np.cos(np.pi * dist_xx / period)


    dist_xy = np.abs(X[:, None] - Y[None, :])
    dist_xy = np.minimum(dist_xy, period - dist_xy)
    Kxy = np.cos(np.pi * dist_xy / period)


    dist_yy = np.abs(Y[:, None] - Y[None, :])
    dist_yy = np.minimum(dist_yy, period - dist_yy)
    Kyy = np.cos(np.pi * dist_yy / period)


    term_xx = np.sum(Kxx) / (n_sample1 ** 2)
    term_xy = np.sum(Kxy) / (n_sample1 * n_sample2)
    term_yy = np.sum(Kyy) / (n_sample2 ** 2)

    mmd_value = term_xx - 2.0 * term_xy + term_yy

    return mmd_value


def my_BGR2HSV(img):

    # if (len(img.shape) != 3) | (img.shape[2] != 3):
    #     print('Dimention Error!!')
    #     return []


    # img_norm = img.astype('double') / 255

    # Vmax = img_norm.max(axis=2)
    # Vmin = img_norm.min(axis=2)
    # B = img_norm[:,:,0]
    # G = img_norm[:,:,1]
    # R = img_norm[:,:,2]
    # S = 1000 * np.ones_like(Vmax)

    # idx = np.where(Vmax==0)
    # S[idx] = 0

    # idx = np.where(Vmax!=0)
    # S[idx] = (Vmax[idx] - Vmin[idx]) / Vmax[idx]

    # H = 1000 * np.ones_like(Vmax)
    # ch_Vmax = img_norm.argmax(axis=2)

    # idx = np.where(np.abs(Vmax - Vmin)<0.00001)
    # H[idx] = 0

    # idx = np.where((ch_Vmax == 2) & (np.abs(Vmax - Vmin)>0.00001))
    # H[idx] = 60 * (G[idx] - B[idx]) / (Vmax[idx] - Vmin[idx])

    # idx = np.where((ch_Vmax == 1) & (np.abs(Vmax - Vmin)>0.00001))
    # H[idx] = 120 + 60 * (B[idx] - R[idx]) / (Vmax[idx] - Vmin[idx])

    # idx = np.where((ch_Vmax == 0) & (np.abs(Vmax - Vmin)>0.00001))
    # H[idx] = 240 + 60 * (R[idx] - G[idx]) / (Vmax[idx] - Vmin[idx])

    # img_hsv = np.stack((H,S*255,Vmax*255),axis=2)

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    return img_hsv



def clean_tubeIMG(img):

    img_input = img.copy()

    if (len(img_input.shape) != 3) | (img_input.shape[2] != 4):  # Input must be BGRA!!!
       return 'Incorrect Format', img_input

    con_num, mask_con, stats, centroids= cv.connectedComponentsWithStats(image=img_input[:,:,3], connectivity=8)
    img_input[np.isin(mask_con, stats[:, cv.CC_STAT_AREA].argsort()[:-3]), 3] = 0


    while np.sum(img_input[:,:,3] == 255) > 3500:

    #   print(f'Image {img_name} Tube {tube_num+1}: recurrent')
        img_hsv = my_BGR2HSV(img_input[:,:,:3])
        idx = np.where(img_input[:,:,3]==255)
        km = KMeans(n_clusters=2, n_init='auto')
        pixels = img_hsv[idx][:,1]
        km.fit(pixels.reshape(-1,1))
        if np.median(pixels[km.labels_ == 0]) > np.median(pixels[km.labels_ == 1]):
            idx_selected = (idx[0][km.labels_ == 0], idx[1][km.labels_ == 0], 3*np.ones_like(idx[0][km.labels_ == 0]))
        else:
            idx_selected = (idx[0][km.labels_ == 1], idx[1][km.labels_ == 1], 3*np.ones_like(idx[0][km.labels_ == 1]))

        img_input[:,:,3] = 0
        img_input[idx_selected] = 255


    if np.sum(img_input[:,:,3] == 255) < 1200:
    #   print(f'Image {img_name} Tube {tube_num+1}: No tube detected')
        return 'No Tube', img_input

    else:
        img_hsv = my_BGR2HSV(img_input[:,:,:3])
        idx = np.where(img_input[:,:,3]==255)

        IF = IsolationForest(n_estimators=100, contamination=0.05, n_jobs=1)

        inlier_mask = IF.fit_predict(img_hsv[idx][:,0].reshape(-1,1))
        img_input[(idx[0][inlier_mask == -1], idx[1][inlier_mask == -1], 3*np.ones_like(idx[0][inlier_mask == -1]))] = 0
      
    #   print(f'Image {img_name} Tube {tube_num+1}: {np.sum(img_input[:,:,3] == 255)}')
        return 'Tube Confirmed', img_input
    


def batchclean_tubeIMG(img_list):

    img_out_list = []
    status_list = []

    for img in img_list:
        status, img_out = clean_tubeIMG(img)
        img_out_list.append(img_out)
        status_list.append(status)
    
    return status_list, img_out_list


def clean_tubeIMG_highres(img):


    #2000*500!!!
    img_input = img.copy()

    if (len(img_input.shape) != 3) | (img_input.shape[2] != 4):  # Input must be BGRA!!!
       return 'Incorrect Format', img_input

    con_num, mask_con, stats, centroids= cv.connectedComponentsWithStats(image=img_input[:,:,3], connectivity=8)
    img_input[np.isin(mask_con, stats[:, cv.CC_STAT_AREA].argsort()[:-3]), 3] = 0


    while np.sum(img_input[:,:,3] == 255) > 3500*4:

    #   print(f'Image {img_name} Tube {tube_num+1}: recurrent')
        img_hsv = my_BGR2HSV(img_input[:,:,:3])
        idx = np.where(img_input[:,:,3]==255)
        km = KMeans(n_clusters=2, n_init='auto')
        pixels = img_hsv[idx][:,1]
        km.fit(pixels.reshape(-1,1))
        if np.median(pixels[km.labels_ == 0]) > np.median(pixels[km.labels_ == 1]):
            idx_selected = (idx[0][km.labels_ == 0], idx[1][km.labels_ == 0], 3*np.ones_like(idx[0][km.labels_ == 0]))
        else:
            idx_selected = (idx[0][km.labels_ == 1], idx[1][km.labels_ == 1], 3*np.ones_like(idx[0][km.labels_ == 1]))

        img_input[:,:,3] = 0
        img_input[idx_selected] = 255


    if np.sum(img_input[:,:,3] == 255) < 1200: 
    #   print(f'Image {img_name} Tube {tube_num+1}: No tube detected')
        return 'No Tube', img_input

    else:
        img_hsv = my_BGR2HSV(img_input[:,:,:3])
        idx = np.where(img_input[:,:,3]==255)

        IF = IsolationForest(n_estimators=100, contamination=0.20, n_jobs=1)

        inlier_mask = IF.fit_predict(img_hsv[idx][:,0].reshape(-1,1))
        img_input[(idx[0][inlier_mask == -1], idx[1][inlier_mask == -1], 3*np.ones_like(idx[0][inlier_mask == -1]))] = 0
      
    #   print(f'Image {img_name} Tube {tube_num+1}: {np.sum(img_input[:,:,3] == 255)}')
        return 'Tube Confirmed', img_input
    


def batchclean_tubeIMG_highres(img_list):

    img_out_list = []
    status_list = []

    for img in img_list:
        status, img_out = clean_tubeIMG_highres(img)
        img_out_list.append(img_out)
        status_list.append(status)
    
    return status_list, img_out_list


def img2df(img_name, tube_num, img):

    Pixel_num = img.shape
    sub_df_BGR = pd.DataFrame(np.zeros([Pixel_num[0]*Pixel_num[1], 7]), columns=['Image', 'Tube','Pixel_idx','B', 'G', 'R', 'Mask'])
    sub_df_BGR['Image'] = img_name
    sub_df_BGR['Tube'] = str(tube_num + 1)
    sub_df_BGR['Pixel_idx'] = np.arange(Pixel_num[0]*Pixel_num[1])
    sub_df_BGR.loc[:, 'B':'Mask'] = img.reshape([-1,4])
    sub_df_BGR['Mask'] = (sub_df_BGR.loc[:,'Mask'] == 255)

    sub_df_HSV = pd.DataFrame(np.zeros([Pixel_num[0]*Pixel_num[1], 7]), columns=['Image', 'Tube','Pixel_idx','H', 'S', 'V', 'Mask'])
    sub_df_HSV['Image'] = img_name
    sub_df_HSV['Tube'] = str(tube_num + 1)
    sub_df_HSV['Pixel_idx'] = np.arange(Pixel_num[0]*Pixel_num[1])
    sub_df_HSV.loc[:, 'H':'V'] = my_BGR2HSV(img[:,:,:3]).reshape([-1,3])
    sub_df_HSV['Mask'] = sub_df_BGR.loc[:,'Mask']

    sub_df_YUV = pd.DataFrame(np.zeros([Pixel_num[0]*Pixel_num[1], 7]), columns=['Image', 'Tube','Pixel_idx','Y', 'U', 'V', 'Mask'])
    sub_df_YUV['Image'] = img_name
    sub_df_YUV['Tube'] = str(tube_num + 1)
    sub_df_YUV['Pixel_idx'] = np.arange(Pixel_num[0]*Pixel_num[1])
    sub_df_YUV.loc[:, 'Y':'V'] = cv.cvtColor(img[:,:,:3], cv.COLOR_BGR2YUV).reshape([-1,3])
    sub_df_YUV['Mask'] = sub_df_BGR.loc[:,'Mask']


    return sub_df_BGR, sub_df_HSV, sub_df_YUV


def batch_img2df(img_nameList, img_list):

    clean_df_BGR = []
    clean_df_HSV = []
    clean_df_YUV = []

    for img_num, img_name in enumerate(img_nameList):
        for tube_num in range(8):

            img = img_list[img_num*8 + tube_num]
            sub_df_BGR, sub_df_HSV, sub_df_YUV = img2df(img_name, tube_num, img)
            clean_df_BGR.append(sub_df_BGR)
            clean_df_HSV.append(sub_df_HSV)
            clean_df_YUV.append(sub_df_YUV)
  
    clean_df_BGR = pd.concat(clean_df_BGR, ignore_index=True)
    clean_df_HSV = pd.concat(clean_df_HSV, ignore_index=True)
    clean_df_YUV = pd.concat(clean_df_YUV, ignore_index=True)


    return clean_df_BGR, clean_df_HSV, clean_df_YUV



def detectIMGColor(img_name, tubes_IMGList, thres):

    imgVec_List = []
    colorResult_df = pd.DataFrame(columns=['Image', 'Tube 1', 'Tube 2', 'Tube 3', 'Tube 4', 'Tube 5', 'Tube 6', 'Tube 7', 'Tube 8'])
    colorResult_df.loc[0, 'Image'] = img_name
    colorResult_df.loc[0, 'Tube 1':'Tube 8'] = 'X'

    for tube_IMG in tubes_IMGList:
        tube_IMG_hsv = my_BGR2HSV(tube_IMG[:,:,:3])
        imgVec_List.append(tube_IMG_hsv[np.where(tube_IMG[:,:,3] == 255)][:,0].astype(float).reshape(-1,1))

    # Check empty tubes, Threshold 500!!! / 2000 for 2000*500 resize!!!
    status = 0
    for tube_num in range(8):
        if imgVec_List[tube_num].shape[0] < 2000:
            colorResult_df.iloc[0, tube_num+1] = 'EPT'
            status = 1

    if status == 1:
        return colorResult_df
        


    # Check reference tubes
    # d78 = mmd.mmd_linear(imgVec_List[6], imgVec_List[7])
    # d17 = mmd.mmd_linear(imgVec_List[0], imgVec_List[6])
    # d18 = mmd.mmd_linear(imgVec_List[0], imgVec_List[7])

    # d78 = mmd_periodic_linear(imgVec_List[6], imgVec_List[7])
    # d17 = mmd_periodic_linear(imgVec_List[0], imgVec_List[6])
    # d18 = mmd_periodic_linear(imgVec_List[0], imgVec_List[7])

    d78 = mmd_periodic_linear_binning(imgVec_List[6], imgVec_List[7])
    d17 = mmd_periodic_linear_binning(imgVec_List[0], imgVec_List[6])
    d18 = mmd_periodic_linear_binning(imgVec_List[0], imgVec_List[7])

    if (d78 > thres*d17) | (d78 > thres*d18):
        colorResult_df.loc[0, ('Tube 1','Tube 7','Tube 8') ] = 'IC'
        return colorResult_df


    #reference check passed
    colorResult_df.loc[0, 'Tube 1' ] = 'P'
    colorResult_df.loc[0, ('Tube 7','Tube 8') ] = 'Y'

    # d1_78 = mmd.mmd_linear(imgVec_List[0], np.concatenate([imgVec_List[6], imgVec_List[7]]))
    # d1_78 = mmd_periodic_linear(imgVec_List[0], np.concatenate([imgVec_List[6], imgVec_List[7]]))
    d1_78 = mmd_periodic_linear_binning(imgVec_List[0], np.concatenate([imgVec_List[6], imgVec_List[7]]))

    DIST = np.zeros([5,2])

    # calculate pink decision distances
    for tube_num in range(5):
        # d = mmd.mmd_linear(imgVec_List[0], imgVec_List[tube_num+1])
        # d = mmd_periodic_linear(imgVec_List[0], imgVec_List[tube_num+1])
        d = mmd_periodic_linear_binning(imgVec_List[0], imgVec_List[tube_num+1])
        DIST[tube_num, 0] = d

    # calculate yellow decision distances
    for tube_num in range(5):
        # d = mmd.mmd_linear( np.concatenate([imgVec_List[6], imgVec_List[7]]), imgVec_List[tube_num+1] )
        # d = mmd_periodic_linear( np.concatenate([imgVec_List[6], imgVec_List[7]]), imgVec_List[tube_num+1] )
        d = mmd_periodic_linear_binning( np.concatenate([imgVec_List[6], imgVec_List[7]]), imgVec_List[tube_num+1] )
        DIST[tube_num, 1] = d

    TH_value = d1_78 * thres
    # TH_value = d18 * thres
    Decision = np.array([DIST[:,0] < TH_value, DIST[:,1] < TH_value]).T

    for tube_num in range(5):
        if (Decision[tube_num, 0] == True) & (Decision[tube_num, 1] == False):
            colorResult_df.iloc[0, tube_num+2] = 'P'
        elif (Decision[tube_num, 0] == False) & (Decision[tube_num, 1] == True):
            colorResult_df.iloc[0, tube_num+2] = 'Y'
        else:
            colorResult_df.iloc[0, tube_num+2] = 'U'

    return colorResult_df



def batch_detectIMGColor(img_nameList, img_cleaned, thres):

    colorResult_df = []

    for img_num, img_name in enumerate(img_nameList):
        colorResult_df.append(detectIMGColor(img_name, img_cleaned[img_num*8:(img_num+1)*8], thres))

    return pd.concat(colorResult_df, ignore_index=True)



def batchclean_tubeIMG(img_list):

    img_out_list = []
    status_list = []

    for img in img_list:
        status, img_out = clean_tubeIMG(img)
        img_out_list.append(img_out)
        status_list.append(status)
    
    return status_list, img_out_list


def img2df(img_name, tube_num, img):

    Pixel_num = img.shape
    sub_df_BGR = pd.DataFrame(np.zeros([Pixel_num[0]*Pixel_num[1], 7]), columns=['Image', 'Tube','Pixel_idx','B', 'G', 'R', 'Mask'])
    sub_df_BGR['Image'] = img_name
    sub_df_BGR['Tube'] = str(tube_num + 1)
    sub_df_BGR['Pixel_idx'] = np.arange(Pixel_num[0]*Pixel_num[1])
    sub_df_BGR.loc[:, 'B':'Mask'] = img.reshape([-1,4])
    sub_df_BGR['Mask'] = (sub_df_BGR.loc[:,'Mask'] == 255)

    sub_df_HSV = pd.DataFrame(np.zeros([Pixel_num[0]*Pixel_num[1], 7]), columns=['Image', 'Tube','Pixel_idx','H', 'S', 'V', 'Mask'])
    sub_df_HSV['Image'] = img_name
    sub_df_HSV['Tube'] = str(tube_num + 1)
    sub_df_HSV['Pixel_idx'] = np.arange(Pixel_num[0]*Pixel_num[1])
    sub_df_HSV.loc[:, 'H':'V'] = my_BGR2HSV(img[:,:,:3]).reshape([-1,3])
    sub_df_HSV['Mask'] = sub_df_BGR.loc[:,'Mask']

    sub_df_YUV = pd.DataFrame(np.zeros([Pixel_num[0]*Pixel_num[1], 7]), columns=['Image', 'Tube','Pixel_idx','Y', 'U', 'V', 'Mask'])
    sub_df_YUV['Image'] = img_name
    sub_df_YUV['Tube'] = str(tube_num + 1)
    sub_df_YUV['Pixel_idx'] = np.arange(Pixel_num[0]*Pixel_num[1])
    sub_df_YUV.loc[:, 'Y':'V'] = cv.cvtColor(img[:,:,:3], cv.COLOR_BGR2YUV).reshape([-1,3])
    sub_df_YUV['Mask'] = sub_df_BGR.loc[:,'Mask']


    return sub_df_BGR, sub_df_HSV, sub_df_YUV


def batch_img2df(img_nameList, img_list):

    clean_df_BGR = []
    clean_df_HSV = []
    clean_df_YUV = []

    for img_num, img_name in enumerate(img_nameList):
        for tube_num in range(8):

            img = img_list[img_num*8 + tube_num]
            sub_df_BGR, sub_df_HSV, sub_df_YUV = img2df(img_name, tube_num, img)
            clean_df_BGR.append(sub_df_BGR)
            clean_df_HSV.append(sub_df_HSV)
            clean_df_YUV.append(sub_df_YUV)
  
    clean_df_BGR = pd.concat(clean_df_BGR, ignore_index=True)
    clean_df_HSV = pd.concat(clean_df_HSV, ignore_index=True)
    clean_df_YUV = pd.concat(clean_df_YUV, ignore_index=True)


    return clean_df_BGR, clean_df_HSV, clean_df_YUV



def detectIMGColor_4tube(img_name, tubes_IMGList, thres):

    imgVec_List = []
    colorResult_df = pd.DataFrame(columns=['Image', 'Tube 1', 'Tube 2', 'Tube 3', 'Tube 4'])
    colorResult_df.loc[0, 'Image'] = img_name
    colorResult_df.loc[0, 'Tube 1':'Tube 4'] = 'X'

    for tube_IMG in tubes_IMGList:
        tube_IMG_hsv = my_BGR2HSV(tube_IMG[:,:,:3])
        imgVec_List.append(tube_IMG_hsv[np.where(tube_IMG[:,:,3] == 255)][:,0].astype(float).reshape(-1,1))

    # Check empty tubes, Threshold 500!!! / 2000 for 2000*500 resize!!!
    status = 0
    for tube_num in range(4):
        if imgVec_List[tube_num].shape[0] < 2000:
            colorResult_df.iloc[0, tube_num+1] = 'EPT'
            status = 1

    if status == 1:
        return colorResult_df
        


    # Check reference tubes
    # d23 = mmd.mmd_linear(imgVec_List[1], imgVec_List[2])
    # d12 = mmd.mmd_linear(imgVec_List[0], imgVec_List[1])
    # d13 = mmd.mmd_linear(imgVec_List[0], imgVec_List[2])

    # d23 = mmd_periodic_linear(imgVec_List[1], imgVec_List[2])
    # d12 = mmd_periodic_linear(imgVec_List[0], imgVec_List[1])
    # d13 = mmd_periodic_linear(imgVec_List[0], imgVec_List[2])

    d23 = mmd_periodic_linear_binning(imgVec_List[1], imgVec_List[2])
    d12 = mmd_periodic_linear_binning(imgVec_List[0], imgVec_List[1])
    d13 = mmd_periodic_linear_binning(imgVec_List[0], imgVec_List[2])

    if (d23 > thres*d12) | (d23 > thres*d13):
        colorResult_df.loc[0, ('Tube 1','Tube 2','Tube 3') ] = 'IC'
        return colorResult_df


    #reference check passed
    colorResult_df.loc[0, 'Tube 1' ] = 'P'
    colorResult_df.loc[0, ('Tube 2','Tube 3') ] = 'Y'

    # d1_23 = mmd.mmd_linear(imgVec_List[0], np.concatenate([imgVec_List[1], imgVec_List[2]]))
    # d1_23 = mmd_periodic_linear(imgVec_List[0], np.concatenate([imgVec_List[1], imgVec_List[2]]))
    d1_23 = mmd_periodic_linear_binning(imgVec_List[0], np.concatenate([imgVec_List[1], imgVec_List[2]]))

    DIST = np.zeros([1,2])

    # calculate pink decision distances
    # d = mmd.mmd_linear(imgVec_List[0], imgVec_List[3])
    # d = mmd_periodic_linear(imgVec_List[0], imgVec_List[3])
    d = mmd_periodic_linear_binning(imgVec_List[0], imgVec_List[3])
    DIST[0, 0] = d

    # calculate yellow decision distances
    # d = mmd.mmd_linear( np.concatenate([imgVec_List[1], imgVec_List[2]]), imgVec_List[3] )
    # d = mmd_periodic_linear( np.concatenate([imgVec_List[1], imgVec_List[2]]), imgVec_List[3] )
    d = mmd_periodic_linear_binning( np.concatenate([imgVec_List[1], imgVec_List[2]]), imgVec_List[3] )
    DIST[0, 1] = d

    TH_value = d1_23 * thres
    # TH_value = d18 * thres
    Decision = np.array([DIST[:,0] < TH_value, DIST[:,1] < TH_value]).T

    if (Decision[0, 0] == True) & (Decision[0, 1] == False):
        colorResult_df.loc[0, 'Tube 4'] = 'P'
    elif (Decision[0, 0] == False) & (Decision[0, 1] == True):
        colorResult_df.loc[0, 'Tube 4'] = 'Y'
    else:
        colorResult_df.loc[0, 'Tube 4'] = 'U'

    return colorResult_df


def batch_detectIMGColor_4tube(img_nameList, img_cleaned, thres):

    colorResult_df = []

    for img_num, img_name in enumerate(img_nameList):
        colorResult_df.append(detectIMGColor_4tube(img_name, img_cleaned[img_num*8:(img_num+1)*8], thres))

    return pd.concat(colorResult_df, ignore_index=True)
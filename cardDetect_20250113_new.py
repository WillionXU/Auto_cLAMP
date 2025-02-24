import cv2 as cv
from pupil_apriltags import Detector
import numpy as np
import utility


# returns a similarity transform to normalize a set of points
def get_normalization_transform(pts):
    centroid = np.mean(pts, axis=0)
    test = np.linalg.norm(pts - centroid, axis=1)
    mean_dist = np.mean(np.linalg.norm(pts - centroid, axis=1))
    translation = np.array([[1, 0, centroid[0] * -1], [0, 1, centroid[1] * -1], [0, 0, 1]])
    scale = np.array([[1 / mean_dist, 0, 0], [0, 1 / mean_dist, 0], [0, 0, 1]]) * np.sqrt(2)
    transform = np.matmul(scale, translation)
    return transform

# obsolete
# uses the given homography to estimate an ROI for a test tube
def tube_square(num, H):
    
    # location of the first tube (mm from top left of card)
    TUBE1 = np.array([28.4, 34])
    # distance between tubes (mm)
    TUBE_OFFSET = 8.8

    center = TUBE1 + np.array([TUBE_OFFSET * (num - 1), 0])
    width = 2.5
    tl = center + np.array([width * -1, width * -1])
    tr = center + np.array([width, width * -1])
    br = center + np.array([width, width])
    bl = center + np.array([width * -1, width])
    img_tl = np.matmul(H, np.append(tl, 1))[0:2].astype(int)
    img_tr = np.matmul(H, np.append(tr, 1))[0:2].astype(int)
    img_br = np.matmul(H, np.append(br, 1))[0:2].astype(int)
    img_bl = np.matmul(H, np.append(bl, 1))[0:2].astype(int)

    return np.array([img_tl, img_tr, img_br, img_bl])

# as above but for the new cropping method
# the magic numbers in this function are in mm from the top left of the cropped card image,
# which is 80 x 27.5 mm
def cropped_tube_square(num, mm_to_px):
    tube1 = np.array([6, 21.5])
    tube_offset = 9.7
    center = tube1 + np.array([tube_offset * (num - 1), 0])
    width = 2.5
    tl = center + np.array([width * -1, width * -1])
    tr = center + np.array([width, width * -1])
    br = center + np.array([width, width])
    bl = center + np.array([width * -1, width])
    square = (np.array([tl, tr, br, bl]) * mm_to_px).astype(int)
    return square

def cropped_tube_circle(num, mm_to_px):
    tube1 = np.array([6, 21.5])
    tube_offset = 9.7
    center = ((tube1 + np.array([tube_offset * (num - 1), 0])) * mm_to_px).astype(int)
    radius = int(3 * mm_to_px)
    return (center, radius)

# uses the given homography to estimate an ROI to sample the card background color
def color_sample_box(H):
    # 40-50h, 26-94w
    tl = np.matmul(H, np.array([26, 40, 1]))[0:2].astype(int)
    tr = np.matmul(H, np.array([94, 40, 1]))[0:2].astype(int)
    br = np.matmul(H, np.array([94, 50, 1]))[0:2].astype(int)
    bl = np.matmul(H, np.array([26, 50, 1]))[0:2].astype(int)

    return np.array([tl, tr, br, bl])

def draw_quad(img, quad, color, thickness):
    cv.line(img, quad[0,:].astype(int), quad[1,:].astype(int), color, thickness)
    cv.line(img, quad[1,:].astype(int), quad[2,:].astype(int), color, thickness)
    cv.line(img, quad[2,:].astype(int), quad[3,:].astype(int), color, thickness)
    cv.line(img, quad[3,:].astype(int), quad[0,:].astype(int), color, thickness)

# produces an image which only includes the card
def crop_to_card(img, H):
    tl = np.matmul(H, np.array([0, 0, 1]))[0:2].astype(np.float32)
    tr = np.matmul(H, np.array([120, 0, 1]))[0:2].astype(np.float32)
    br = np.matmul(H, np.array([120, 55, 1]))[0:2].astype(np.float32)
    bl = np.matmul(H, np.array([0, 55, 1]))[0:2].astype(np.float32)

    test = np.matmul(H, np.array([0, 0, 1]))
    test1 = np.matmul(H, np.array([120, 0, 1]))
    test2 = np.matmul(H, np.array([120, 55, 1]))
    test3 = np.matmul(H, np.array([0, 55, 1]))

    widthTop = np.linalg.norm(tr - tl)
    widthBottom = np.linalg.norm(br - bl)
    heightLeft = np.linalg.norm(bl - tl)
    heightRight = np.linalg.norm(br - tr)
    widthMax = np.maximum(widthTop, widthBottom)
    heightMax = np.maximum(heightLeft, heightRight)

    dst = np.array([[0, 0], [widthMax - 1, 0], [widthMax - 1, heightMax - 1], [0, heightMax - 1]]).astype(np.float32)

    test = np.array([tl, tr, br, bl])

    warp = cv.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    warped = cv.warpPerspective(img, warp, (int(widthMax), int(heightMax)))

    return warped

# finds a homography transformation between the plane of the test card in mm
# and the plane of the image in pixels
def find_card_homography(img):
    # constants used for locating AprilTags
    # card-coordinate positions of each corner of each tag in mm from the top left
    TL_TL = np.array([3.5, 3.5])
    TL_TR = np.array([11.5, 3.5])
    TL_BR = np.array([11.5, 11.5])
    TL_BL = np.array([3.5, 11.5])

    TR_TL = np.array([108.5, 3.5])
    TR_TR = np.array([116.5, 3.5])
    TR_BR = np.array([116.5, 11.5])
    TR_BL = np.array([108.5, 11.5])

    BR_TL = np.array([108.5, 43.5])
    BR_TR = np.array([116.5, 43.5])
    BR_BR = np.array([116.5, 51.5])
    BR_BL = np.array([108.5, 51.5])

    BL_TL = np.array([3.5, 43.5])
    BL_TR = np.array([11.5, 43.5])
    BL_BR = np.array([11.5, 51.5])
    BL_BL = np.array([3.5, 51.5])


    # detect AprilTags in the image
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    detector = Detector(families="tag16h5")
    detections = detector.detect(gray)

    # filter out poor quality matches which are likely to be spurious
    tags = list(filter(lambda x: x.hamming == 0 and x.decision_margin > 30, detections))

    # create lists of corresponding points in the card plane and the image plane
    img_pts = np.array([]).reshape(0,2)
    card_pts = np.array([]).reshape(0,2)

    for tag in tags:
        #draw_quad(img, tag.corners, (0, 0, 255), 5)f
        if tag.tag_id == 0:
            card_pts = np.vstack([card_pts, np.array([TL_BL, TL_BR, TL_TR, TL_TL])])
            img_pts = np.vstack([img_pts, tag.corners])
        elif tag.tag_id == 1:
            card_pts = np.vstack([card_pts, np.array([TR_BL, TR_BR, TR_TR, TR_TL])])
            img_pts = np.vstack([img_pts, tag.corners])
        elif tag.tag_id == 3:
            card_pts = np.vstack([card_pts, np.array([BR_BL, BR_BR, BR_TR, BR_TL])])
            img_pts = np.vstack([img_pts, tag.corners])
        elif tag.tag_id == 2:
            card_pts = np.vstack([card_pts, np.array([BL_BL, BL_BR, BL_TR, BL_TL])])
            img_pts = np.vstack([img_pts, tag.corners])
        else:
            print("unknown tag ID detected")

    # convert the corresponding points from cartesian to homogeneous coordinates
    # this is necessary to apply translations and projective transforms such as homographies
    ones = np.ones((card_pts.shape[0], 1))
    card_pts_homogeneous = np.hstack([card_pts, ones])
    img_pts_homogeneous = np.hstack([img_pts, ones])

    # get transformations to normalize the coordinate spaces for the
    # image points and the card points
    # this step reduces the effect of noise on our homography
    T_card = get_normalization_transform(card_pts)
    T_img = get_normalization_transform(img_pts)

    # apply the normalizations
    # the [:,0:2] slice converts from homogeneous back to cartesian coordinates
    card_pts_t = np.transpose(np.matmul(T_card, np.transpose(card_pts_homogeneous)))[:,0:2]
    img_pts_t = np.transpose(np.matmul(T_img, np.transpose(img_pts_homogeneous)))[:,0:2]
    test = np.matmul(T_img, np.transpose(img_pts_homogeneous))

    # produce a homography for the normalized data
    H_t, _ = cv.findHomography(card_pts_t, img_pts_t, method=cv.RANSAC)

    # transform the homography to work on real data
    T_img_inv = np.linalg.inv(T_img)
    H = np.matmul(T_img_inv, np.matmul(H_t, T_card))

    return H

def draw_results(img, filename):
    #img_TL = np.matmul(H, np.array([0, 0, 1]))[0:2].astype(int)
    #img_TR = np.matmul(H, np.array([120, 0, 1]))[0:2].astype(int)
    #img_BR = np.matmul(H, np.array([120, 55, 1]))[0:2].astype(int)
    #img_BL = np.matmul(H, np.array([0, 55, 1]))[0:2].astype(int)

    #cv.line(img, img_TL, img_TR, (0, 0, 255), 5)
    #cv.line(img, img_TR, img_BR, (0, 0, 255), 5)
    #cv.line(img, img_BR, img_BL, (0, 0, 255), 5)
    #cv.line(img, img_BL, img_TL, (0, 0, 255), 5)

    for i in range(1, 9):
        #square = cropped_tube_square(i, img.shape[1] / 80)
        #draw_quad(img, square, (0, 255, 0), 3)
        (center, radius) = cropped_tube_circle(i, img.shape[1] / 80)
        center = tuple(center)
        cv.circle(img, center, radius, (0, 255, 0), 3)

    res = cv.imwrite(filename, img)
    if not res:
        print("Failed to write test results image")



# scales an image to 1000px in its largest dimension
# this is done for two reasons:
#  1. to improve performance on very large images by downsampling them
#  2. to ensure that training images aren't more heavily weighted just for being larger
def scale_image(img):
    # if img.shape[0] > img.shape[1]:
    #     height = 1000
    #     scale = 1000 / img.shape[0]
    #     width = int(img.shape[1] * scale)
    #     resized = cv.resize(img, (width, height))
    #     return resized
    # else:
    #     width = 1000
    #     scale = 1000 / img.shape[1]
    #     height = int(img.shape[0] * scale)
    #     resized = cv.resize(img, (width, height))
    #     return resized

    # height = 250
    # width = 1000
    height = 500
    width = 2000
    resized = cv.resize(img, (width, height))
    return resized

def detect_split(img):

    try:
        H = find_card_homography(img)
        warped = crop_to_card(img, H)
    except:
        return [], []

    # Correct image
    # top = int(warped.shape[0] * 0.5)
    # bottom = warped.shape[0] - int(warped.shape[0] * 0.2)
    top = int(warped.shape[0] * 0.4)
    bottom = warped.shape[0] - int(warped.shape[0] * 0.2)
    left = int(warped.shape[1] * 0.2)
    right = warped.shape[1] - left
    
    # Dimensions
    width = abs(right - left + 1)
    height = abs(top - bottom + 1)
    img_cropped = warped[top:bottom, left:right]

    img_scaled = scale_image(img_cropped)
    height_scaled = img_scaled.shape[0]
    width_scaled = img_scaled.shape[1]


    img_ref = warped[bottom:bottom + height - 1, left + round(7 * width / 16):left + round(9 * width / 16)]
    img_ref_scaled = cv.resize(img_ref, (round(width_scaled / 8), height_scaled))

    # Slice image and subtract
    img_split = [''] * 9
    img_split[0:7] = np.hsplit(img_scaled, 8) #- img_ref_scaled
    img_split[8] = img_ref_scaled

    return img_scaled, img_split

def main():

    path = 'Images'
    img_nameList = utility.find_img_name(f'{path}\Raw')
    print(f'{len(img_nameList)} images found!\n')
    
    if len(img_nameList) > 450:
        print('Cannot detect for more than 450 pictures due to paskage limitation!!!')
        return

    for img_name in img_nameList:

        print(f'Detecting {img_name}...')
        img = cv.imread(f"{path}\Raw\{img_name}.jpg")
        
        img_scaled, img_split = detect_split(img)

        if img_split == []:
            print(f'Apriltag not found in {img_name}!!')
            continue

        cv.imwrite(f"{path}/Cropped/{img_name}.jpg", img_scaled)

        for num in range(0,9):
            cv.imwrite(f"{path}/Split/{img_name}_Tube{num+1}.jpg", img_split[num])


if __name__ == "__main__":
    main()

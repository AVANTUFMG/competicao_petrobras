# coding: utf-8

import numpy as np
import cv2
import cv_bridge as bridge

# TODO: renomear função
def get_frame(msg, encoding="bgr8"):
    image = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)

    return image

def get_external_contours(img, thresh_params, kernel=None, iterations=1):
    if kernel is None:
        kernel = np.ones((5, 5), dtype=np.uint8)

    image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    image_threshold = cv2.inRange(
        image_HSV,
        (
            thresh_params["low_H"],
            thresh_params["low_S"],
            thresh_params["low_V"],
        ),
        (
            thresh_params["high_H"],
            thresh_params["high_S"],
            thresh_params["high_V"],
        ),
    )

    image_threshold = cv2.dilate(image_threshold, kernel, iterations=iterations)

    result = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]

    return contours

def get_centroid(contour):
    moments = cv2.moments(contour)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy

def get_position_from_centroid(drone_pos, fx, fy, height, width, centroid, hdg, H):
    hfov = 2 * np.arctan2(width, 2 * fx)
    vfov = 2 * np.arctan2(height, 2 * fy)

    dx = -1 * (centroid[0] - (width / 2))
    dy = -1 * (centroid[1] - (height / 2))
    beta = np.arctan2(dx * np.tan(hfov / 2), (width / 2))
    alpha = np.arctan2(dy * np.tan(vfov / 2), (height / 2))

    x_vec = np.cos(hdg)
    y_vec = np.sin(hdg)

    pos_x = (
        drone_pos.x
        + np.tan(alpha) * (drone_pos.z - H) * x_vec
        - np.tan(beta) * (drone_pos.z - H) * y_vec
    )
    pos_y = (
        drone_pos.y
        + np.tan(alpha) * (drone_pos.z - H) * y_vec
        + np.tan(beta) * (drone_pos.z - H) * x_vec
    )

    pos = [pos_x, pos_y, H]
    return pos

def mask_out_color(img, lower, upper):
    LOWER_RANGE = np.array(lower)
    UPPER_RANGE = np.array(upper)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return cv2.inRange(hsv, LOWER_RANGE, UPPER_RANGE)

def flood_fill(img):
    image = img.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)

    cv2.floodFill(image, mask, (0, 0), 255) #sup esq
    cv2.floodFill(image, mask, (width - 1, 0), 255) #sup dir
    cv2.floodFill(image, mask, (0, height - 1), 255) #inf esq
    cv2.floodFill(image, mask, (width - 1, height - 1), 255) #inf dir

    return image

def is_land_base(contour, img):
        is_land_base = False
        is_moving_base = False
        image = np.copy(img)

        h,w = image.shape[:2]
        roi = np.zeros([h, w, 1], dtype=np.uint8)
        roi.fill(0)

        cv2.fillPoly(roi, [contour], color=(255, 255, 255))
        roiColored = cv2.bitwise_and(image, image, mask=roi)

        yellowMask = mask_out_color(roiColored, [0,40,0], [100,255,255])

        invertedFloodFilled = flood_fill(yellowMask)
        roi2 = invertedFloodFilled | yellowMask
        roi2Colored = cv2.bitwise_and(roiColored,roiColored,mask=roi2)

        blueMask = mask_out_color(roi2Colored, [110,1,0], [179,255,255])

        conts, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        isRectangle = is_quadrilateral(conts[0])

        if np.mean(blueMask) != 0 and isRectangle:
            is_land_base = True

        return is_land_base

def is_fixed_base(contour, img, pad=40):
    image = img.copy()
    is_moving_base = False

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    x, y, h, w = cv2.boundingRect(mask)

    x1 = np.min(np.abs([x, x - pad]))
    y1 = np.min(np.abs([y, y - pad]))
    x2 = np.min([x + w + pad, image.shape[1]])
    y2 = np.min([y + h + pad, image.shape[0]])
    
    input = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if len(np.where(input[y1 : y2, x1 : x2, 2] < 70)[0]) <= 2000:
        is_moving_base = True

    return is_moving_base

def is_quadrilateral(contour, err=0.15):
    peri = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, err * peri, True)
    num_sides = len(vertices)

    if (num_sides != 4):
        return False

    return True

def is_panel(img):
    image = np.copy(img)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white = np.where(image > 200)
    if len(white[0]) < 0.06 * (image.shape[0] * image.shape[1]):
        return False
    return True

# TODO: refatorar essa função
def get_panel(img, upscaling_fact=5):
    image = np.copy(img)

    contours = get_external_contours(image, panel_params)

    masks = list()
    for c in contours:
        if is_quadrilateral(c) == False:
            continue

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masks.append(mask)

    # finalmente encontrar o painel
    rois = list()
    for m in masks:
        # aplicando a máscara para recortar as regiões de interesse
        roi = np.zeros(image.shape, dtype=np.uint8)
        roi = cv2.bitwise_and(image, image, mask=m)

        # recortar a roi da imagem completa (que teria um monte de pixel preto)
        x, y, h, w = cv2.boundingRect(m)
        roi = roi[y : y + h, x : x + w]

        rois.append(roi)

    return rois

def get_panel_numbers(img):
    result = {
        'top-left': None,
        'top-right': None,
        'bottom-left': None,
        'bottom-right': None
    }

    kernel = np.ones((5, 5))
    cv_image = cv2.GaussianBlur(img, (5, 5), 0)
    img_HSV = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)

    img_thresh = cv2.inRange(img_HSV, (0, 0, 0), (179, 255, 30))
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    img_thresh = cv2.dilate(img_thresh, kernel, iterations=0)

    cnts, hier = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(img_gray.shape, dtype="uint8")

    # seleciona maior contorno
    cnt = max(cnts, key=cv2.contourArea)

    rotrect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)

    (x, y, w, h) = cv2.boundingRect(cnt)
    rect = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    cv2.fillPoly(mask, [rect], 255)

    imageROI = img[y : y + h, x : x + w]
    maskROI = mask[y : y + h, x : x + w]

    imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
    imageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)

    imageROI = cv2.GaussianBlur(imageROI, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(imageROI, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -3)
    th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel2)

    (h, w) = th3.shape

    newth = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)

    for i in range(5):
        new_cnts, hierar = cv2.findContours(th3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        circle_cnts = list()
        for i in range(len(new_cnts)):
            (x, y), radius = cv2.minEnclosingCircle(new_cnts[i])
            area = cv2.contourArea(new_cnts[i])
            if area > 0:
                if (np.abs((radius * radius * np.pi) - area) / area < 0.45) and (
                    hierar[0][i][2] == -1
                ):
                    circle_cnts.append(new_cnts[i])
            else:
                pass

        avg_x = 0.0
        avg_y = 0.0
        for i in range(len(circle_cnts)):
            M = cv2.moments(circle_cnts[i])
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            avg_x += cx
            avg_y += cy
        avg_x = avg_x / 4
        avg_y = avg_y / 4
    
    x = int(0.08 * newth.shape[0])
    y = int(0)
    h = int(newth.shape[0] - 1)
    w = int(0.84 * newth.shape[1])

    cropped = imageROI[y : y + h, x : x + w]

    (h, w) = cropped.shape
    ret, cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    sqr1 = cropped[0 : np.int0(h / 2), 0 : np.int0(w / 3)]
    sqr2 = cropped[0 : np.int0(h / 2), (np.int0(w / 3) + 1) : np.int0(2 * w / 3)]
    sqr4 = cropped[np.int0(h / 2) : h, 0 : np.int0(w / 3)]
    sqr5 = cropped[np.int0(h / 2) : h, (np.int0(w / 3) + 1) : (np.int0(2 * w / 3))]

    ret3, sqr1 = cv2.threshold(sqr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, sqr2 = cv2.threshold(sqr2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, sqr4 = cv2.threshold(sqr4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, sqr5 = cv2.threshold(sqr5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result["bottom-left"] = sqr1
    result["top-left"] = sqr2
    result["bottom-right"] = sqr4
    result["top-right"] = sqr5

    return result

def rotate_img(self, img, angle, scale=1):
    height, width = img.shape[:2]
    center = (int(width/2), int(height/2))
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
    return rotated
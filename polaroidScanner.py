import cv2
import numpy as np
from PIL import Image, ImageEnhance

def order_points(pts):
    # wie gehabt
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # wie gehabt
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def orient_polaroid(warped_img):
    # Jetzt sicher: immer mit der dicksten weißen Kante UNTEN!
    best_img = warped_img
    max_white = -1
    for i in range(4):
        h = warped_img.shape[0]
        bottom_strip = warped_img[int(h*0.80):, :]
        # Farb-zu-Grau, falls Bild bunt ist
        if len(bottom_strip.shape) == 3:
            strip_gray = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
        else:
            strip_gray = bottom_strip
        white_score = np.mean(strip_gray)
        if white_score > max_white:
            max_white = white_score
            best_img = warped_img.copy()
        warped_img = np.rot90(warped_img)
    return best_img

def enhance_image(pil_img, contrast=1.3, color=1.3, sharpness=1.2):
    # Kontrast erhöhen
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)
    # Sättigung erhöhen
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(color)
    # Schärfe erhöhen
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(sharpness)
    return pil_img

image_path = "N://EPSCAN//001//EPSON021.JPG"
output_base = "N://EPSCAN//001//" + image_path

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
_, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

polaroid_num = 1
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 10000:
        continue
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    warped = four_point_transform(img, box)
    warped_oriented = orient_polaroid(warped)
    polaroid_pil = Image.fromarray(cv2.cvtColor(warped_oriented, cv2.COLOR_BGR2RGB))
    
    # >>> HIER BILD ENHANCEN <<<
    polaroid_pil = enhance_image(polaroid_pil,
                                 contrast=1.3,   # 1.0 = unverändert, >1 erhöht
                                 color=1.3,      # 1.0 = unverändert, >1 erhöht Sättigung
                                 sharpness=1.2)  # 1.0 = unverändert, >1 erhöht

    polaroid_pil.save(f"{output_base}_{polaroid_num}.jpg")
    polaroid_num += 1

print(f"{polaroid_num-1} Polaroids erkannt, gerichtet, verbessert und gespeichert!")

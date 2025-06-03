import cv2
import numpy as np
from PIL import Image, ImageEnhance
import subprocess
import os

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
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
    best_img = warped_img
    max_white = -1
    for i in range(4):
        h = warped_img.shape[0]
        bottom_strip = warped_img[int(h*0.80):, :]
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
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(color)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(sharpness)
    return pil_img

# --- Einstellungen ---
image_path = "N://EPSCAN//001//EPSON021.JPG"
output_base = "N://EPSCAN//001//"
tmp_folder = "N://EPSCAN//001//tmp_polaroids"
output_folder = "N://EPSCAN//001//output_polaroids"
realesrgan_path = r"C://tools//realesrgan-ncnn-vulkan//realcugan-ncnn-vulkan.exe"

os.makedirs(tmp_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

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
    polaroid_pil = enhance_image(polaroid_pil,
                                 contrast=1,
                                 color=1,
                                 sharpness=1)
    tmp_jpg = os.path.join(tmp_folder, f"{output_base}_{polaroid_num}.jpg")
    upscale_jpg = os.path.join(output_folder, f"{output_base}_{polaroid_num}.jpg")
    polaroid_pil.save(tmp_jpg, quality=95)
    # --- Real-ESRGAN upscaling als EXTERNER PROZESS ---
    subprocess.run([
        realesrgan_path,
        "-i", tmp_jpg,
        "-o", upscale_jpg,
        "-n", "-1", # Denoise
        "-s", "2",  # 2-fach Upscaling
        "-g", "0" # Using GPU 0
    ], check=True)
    print(f"Polaroid {polaroid_num} gespeichert und hochskaliert!")
    polaroid_num += 1

print(f"Fertig! Alle Polaroids erkannt, upgescaled und im Ordner {output_folder} gespeichert.")

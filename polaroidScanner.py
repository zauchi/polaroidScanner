import cv2
import numpy as np
from PIL import Image, ImageEnhance
import subprocess
import os
import glob

def order_points(pts):
    """
    Order the points in the contour to: top-left, top-right, bottom-right, bottom-left.
    This is needed for proper perspective transform.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    Apply a perspective transformation to extract the rectangle defined by pts.
    Returns a top-down, "flattened" view of the detected polaroid.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Coordinates of the new rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Calculate the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def orient_polaroid(warped_img):
    """
    Rotates the cropped polaroid image so that the thick white border is always at the bottom.
    Checks all four 90-degree rotations, measures the mean brightness of the bottom edge, 
    and picks the orientation with the brightest bottom (assumed to be the thick border).
    """
    best_img = warped_img
    max_white = -1
    for i in range(4):
        h = warped_img.shape[0]

        # Get bottom 20% strip of the image
        bottom_strip = warped_img[int(h*0.80):, :]

        # Convert to grayscale for analysis
        if len(bottom_strip.shape) == 3:
            strip_gray = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
        else:
            strip_gray = bottom_strip
        white_score = np.mean(strip_gray)
        if white_score > max_white:
            max_white = white_score
            best_img = warped_img.copy()

        # Rotate 90 degrees for next test
        warped_img = np.rot90(warped_img)
    return best_img

def enhance_image(pil_img, contrast=1.3, color=1.3, sharpness=1.2):
    """
    Enhances the input image:
      - Adjusts contrast, color (saturation), and sharpness.
    Returns the enhanced image.
    """
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(color)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(sharpness)
    return pil_img

# --- Folders and tool paths ---
input_folder = r"N:\EPSCAN\001"
tmp_folder = os.path.join(input_folder, "tmp_polaroids")
output_folder = os.path.join(r"D:\Google schreinersgarage@gmail.com\Fotos\Polaroids")
realesrgan_path = r"C:\tools\realesrgan-ncnn-vulkan\realcugan-ncnn-vulkan.exe"

# Make sure folders exist (create if needed)
os.makedirs(tmp_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Gather all input image files (JPG, JPEG, PNG) in the input folder
input_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
              glob.glob(os.path.join(input_folder, "*.jpeg")) + \
              glob.glob(os.path.join(input_folder, "*.png"))

# --- Main processing loop ---
for image_path in input_files:
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Starting processing of {image_path} ...")
    img = cv2.imread(image_path)

    # Preprocess: convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Thresholding: make background white, polaroids black
    _, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (external only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polaroid_num = 1

    # Process each detected contour that is big enough (likely a polaroid)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000: # Filter out noise and tiny contours
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Crop and straighten the detected polaroid
        warped = four_point_transform(img, box)

        # Rotate so that the thick border is always at the bottom
        warped_oriented = orient_polaroid(warped)

        # Convert to PIL Image for further enhancements
        polaroid_pil = Image.fromarray(cv2.cvtColor(warped_oriented, cv2.COLOR_BGR2RGB))

        # Enhance contrast, color, and sharpness (adjust values as needed)
        polaroid_pil = enhance_image(polaroid_pil,
                                     contrast=1.1, # Adjust as needed
                                     color=1.3,
                                     sharpness=1.0)
        
        # Temporary filename for Real-ESRGAN/RealCUGAN input
        tmp_jpg = os.path.join(tmp_folder, f"{basename}_{polaroid_num}.jpg")

        # Final output filename (upscaled)
        upscale_jpg = os.path.join(output_folder, f"{basename}_{polaroid_num}.jpg")

        # Save the (still small) polaroid image temporarily
        polaroid_pil.save(tmp_jpg, quality=95)

        # --- Run Real-ESRGAN/RealCUGAN as external process for upscaling ---
        subprocess.run([
            realesrgan_path,
            "-i", tmp_jpg,
            "-o", upscale_jpg,
            "-n", "-1",      # denoise parameter (change if needed)
            "-s", "2",       # 2x upscaling (adjust if using a different model)
            "-g", "0"        # Use GPU 0 (main NVIDIA GPU)
        ], check=True)
        print(f"{basename}_{polaroid_num}.jpg saved and upscaled!")
        polaroid_num += 1

print(f"Done! All scans and polaroids processed and upscaled!")
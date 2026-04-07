import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(img):
    h,w = img.shape[:2]

    polygon = np.array([
        [
            (int(0.02*w), h),
            (int(0.40*w), int(0.55*h)),
            (int(0.60*w), int(0.55*h)),
            (int(0.98*w), h)
        ]
    ])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines):
    line_img = np.zeros_like(img)

    if lines is None:
        return img

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return cv2.addWeighted(img, 0.8, line_img, 1, 1)

def detect_lanes(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny edges
    edges = cv2.Canny(blur, 50, 150)

    # 4. ROI mask
    cropped = region_of_interest(edges)

    # 5. Hough Transform
    lines = cv2.HoughLinesP(
        cropped,
        rho=2,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # 6. Draw lines
    lane_img = draw_lines(img_rgb, lines)

    # Plot result
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1) 
    plt.title("Original") 
    plt.imshow(img_rgb) 
    plt.axis("off")

    plt.subplot(1, 3, 2) 
    plt.title("Edges (ROI)") 
    plt.imshow(cropped, cmap="gray") 
    plt.axis("off") 
    
    plt.subplot(1, 3, 3) 
    plt.title("Detected Lanes") 
    plt.imshow(lane_img) 
    plt.axis("off") 
    
    plt.tight_layout() 
    save_path = "lane_result.png" 
    plt.savefig(save_path, dpi=200) 
    print(f"Saved result to {save_path}") 
    plt.close()  

# PUT IMAGE PATH  
detect_lanes("/home/faidbogi/ads/bdd100k_images_10k/10k/val/7d97d173-09388af3.jpg") 
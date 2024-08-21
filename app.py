import cv2
import numpy as np

# Load the background image and logo
background_img = cv2.imread('after.jpg')
logo_img = cv2.imread('LOGO.png')
logo_img = cv2.resize(logo_img, (200, 150))

# Initialize corner points for the logo image
h, w = logo_img.shape[:2]
points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

# Clone points for manipulation
points_copy = points.copy()

# Initialize dragging state
dragging_point = -1

# Function to update the image with the distorted logo
def update_image():
    global points_copy
    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(points, points_copy)
    # Apply the perspective transformation to the logo image
    distorted_logo = cv2.warpPerspective(logo_img, M, (w, h))
    
    # Overlay the transformed logo on the background image
    temp_img = background_img.copy()
    x_offset = 100
    y_offset = 100
    rows, cols, _ = distorted_logo.shape
    roi = temp_img[y_offset:y_offset+rows, x_offset:x_offset+cols]
    
    # Create a mask of the logo and its inverse mask
    logo_gray = cv2.cvtColor(distorted_logo, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(logo_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Black-out the area of the logo in ROI
    background_img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # Take only the region of the logo from the logo image
    logo_fg = cv2.bitwise_and(distorted_logo, distorted_logo, mask=mask)
    
    # Put the logo in the ROI and modify the main image
    dst = cv2.add(background_img_bg, logo_fg)
    temp_img[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst
    
    # Draw the corners
    for point in points_copy:
        x, y = int(point[0] + x_offset), int(point[1] + y_offset)
        cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
    
    # Display the result
    cv2.imshow('Distorted Logo', temp_img)

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global points_copy, dragging_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # Find the nearest corner
        min_dist = float('inf')
        nearest_idx = -1
        for i, point in enumerate(points_copy):
            px, py = int(point[0] + 100), int(point[1] + 100)
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        if min_dist < 10:  # Threshold to detect click near the corner
            dragging_point = nearest_idx

    elif event == cv2.EVENT_MOUSEMOVE and dragging_point != -1:
        points_copy[dragging_point] = [x - 100, y - 100]
        update_image()
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = -1

# Set up the window and mouse callback
cv2.namedWindow('Distorted Logo')
cv2.setMouseCallback('Distorted Logo', mouse_callback)

# Initial display
update_image()

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
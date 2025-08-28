# =============================================================================
#
#   Vision Command: Real-time Video Analysis Suite
#   Author: Utkarsh Tripathi
#
#   Description:
#   An advanced computer vision toolkit built with Python and OpenCV. This
#   application captures video from a webcam and applies a wide range of
#   real-time filters, enhancements, and analyses. It features an interactive
#   GUI to control various visual effects, from edge detection and low-light
#   enhancement to motion tracking and optical flow.
#
# =============================================================================

import cv2
import numpy as np
import time

# Attempt to import scikit-image for LBP feature, with a friendly warning if it's missing.
try:
    from skimage import feature
except ImportError:
    print("="*80)
    print("⚠  Warning: scikit-image is not installed. LBP mode will not work.")
    print("   Please install it by running: pip install scikit-image")
    print("="*80)
    feature = None

# --- 1. CONFIGURATION & TUNING PARAMETERS ---

# --- Low-Light and Enhancement ---
BRIGHTNESS_THRESHOLD = 80       # Below this average brightness, auto low-light enhancement kicks in.
CLAHE_CLIP_LIMIT = 2.0          # Controls the contrast limit for CLAHE algorithm.
CLAHE_GRID_SIZE = (8, 8)        # The grid size for histogram equalization.

# --- Motion Detection ---
MOTION_MIN_AREA = 700           # Ignores any detected motion contours smaller than this area.
STILLNESS_THRESHOLD = 5         # Seconds of no motion before the scene is declared "Clear".

# --- Overlays & Special Effects ---
LIGHT_SOURCE_THRESHOLD = 250    # Brightness value (0-255) to be considered a light source.
SHADOW_LIFT_THRESHOLD = 60      # Pixels darker than this will be brightened.
SHADOW_LIFT_AMOUNT = 1.8        # How much to brighten the shadows (gamma correction factor).

# --- 2. GLOBAL STATE DICTIONARY ---
# A simple way to manage the state from the GUI trackbars.
filter_state = {
    'mode': 0,
    'brightness': 50,
    'contrast': 10,
    'light_detect': 0
}

# --- 3. FILTER & IMAGE PROCESSING FUNCTIONS ---
# Each function takes a frame and returns a processed frame.

def apply_canny(frame):
    """Applies the Canny edge detection algorithm."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_sobel(frame):
    """Applies the Sobel operator to highlight edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_laplacian(frame):
    """Applies the Laplacian operator for edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.cvtColor(cv2.convertScaleAbs(edges), cv2.COLOR_GRAY2BGR)

def apply_morph_gradient(frame):
    """Highlights the outline of objects using morphological gradient."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

def apply_tophat(frame):
    """Highlights small, bright details in the image (Top-hat transform)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((15, 15), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)

def apply_lbp(frame):
    """Visualizes Local Binary Patterns for texture analysis."""
    if feature is None:
        error_frame = np.zeros_like(frame)
        cv2.putText(error_frame, "pip install scikit-image", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return error_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, P=24, R=8, method="uniform")
    return cv2.cvtColor(lbp.astype("uint8"), cv2.COLOR_GRAY2BGR)

def apply_thermal_view(frame):
    """Simulates a thermal camera look by applying a colormap."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

def apply_difference_view(current_frame, prev_frame):
    """Highlights differences between the current and previous frame."""
    if prev_frame is None:
        return current_frame
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    return cv2.applyColorMap(thresh, cv2.COLORMAP_INFERNO)

def apply_green_night_vision(frame):
    """Simulates a classic green night vision goggle (NVG) look."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zeros = np.zeros(gray.shape, dtype="uint8")
    return cv2.merge([zeros, gray, zeros])

def lift_shadows(frame):
    """Brightens only the dark areas (shadows) of the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Create a mask for dark areas
    shadow_mask = cv2.inRange(v, 0, SHADOW_LIFT_THRESHOLD)
    # Apply gamma correction to brighten the masked areas
    inv_gamma = 1.0 / SHADOW_LIFT_AMOUNT
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    v_lifted = np.where(shadow_mask > 0, cv2.LUT(v, table), v)
    final_hsv = cv2.merge([h, s, v_lifted])
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def draw_motion_vectors(current_frame, prev_gray, processed_frame):
    """Calculates and draws optical flow vectors to show motion direction."""
    if prev_gray is None:
        return
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Find good points to track in the previous frame
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if p0 is not None:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # Calculate the optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
        # Select and draw the good points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(processed_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(processed_frame, (int(a), int(b)), 5, (0, 255, 0), -1)

# --- 4. BASE ENHANCEMENT & OVERLAY FUNCTIONS ---

def enhance_low_light(frame):
    """Improves contrast in low-light frames using CLAHE."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced_l = clahe.apply(l)
    enhanced_frame = cv2.merge((enhanced_l, a, b))
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)

def apply_brightness_contrast(frame, brightness, contrast):
    """Manually adjust brightness and contrast based on trackbar values."""
    # The formula is: new_image = alpha * original_image + beta
    alpha = float(contrast) / 10.0  # Contrast control (1.0-3.0)
    beta = int(brightness) - 50     # Brightness control (-50 to 50)
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def detect_light_sources(frame, processed_frame):
    """Finds and circles bright spots in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv) # Use the Value channel for brightness
    mask = cv2.threshold(v, LIGHT_SOURCE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    # Clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(processed_frame, (center_x, center_y), 15, (0, 255, 255), 2)
        cv2.putText(processed_frame, "Light Source", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# --- 5. UI CALLBACK FUNCTIONS ---
# These functions are called by the trackbars to update the global state.

def set_filter_mode(val): filter_state['mode'] = val
def set_brightness(val): filter_state['brightness'] = val
def set_contrast(val): filter_state['contrast'] = val
def set_light_detect(val): filter_state['light_detect'] = val

# --- 6. MAIN APPLICATION LOGIC ---

def main():
    """The main function to run the Vision Command application."""
    # --- Initialization ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    window_name = "Vision Command by Utkarsh Tripathi (Original vs. Processed)"
    cv2.namedWindow(window_name)

    # Create the interactive trackbars for the GUI
    cv2.createTrackbar("Filter Mode", window_name, 0, 11, set_filter_mode)
    cv2.createTrackbar("Light Detect", window_name, 0, 1, set_light_detect)
    cv2.createTrackbar("Brightness", window_name, 50, 100, set_brightness)
    cv2.createTrackbar("Contrast", window_name, 10, 30, set_contrast)

    # Setup for motion detection (Background Subtraction)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    # Initialize variables for stateful operations (like difference view and motion flow)
    prev_frame = None
    prev_gray_for_flow = None
    last_motion_time = 0 # Initialize to avoid NameError on first run

    print("✅ Webcam started. Press 'q' to quit.")

    # Dictionary to map filter modes to their functions and names
    modes = {
        1: (apply_canny, "Canny"), 2: (apply_sobel, "Sobel"),
        3: (apply_laplacian, "Laplacian"), 4: (apply_morph_gradient, "Morph Grad"),
        5: (apply_tophat, "Top-hat"), 6: (apply_lbp, "LBP Texture"),
        7: (apply_thermal_view, "Thermal"), 9: (apply_green_night_vision, "Green NVG"),
        10: (lift_shadows, "Shadow Lift")
    }

    # --- Main Application Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        # --- Pipeline Step 1: Base Visual Enhancement ---
        # Create a copy to work on, preserving the original
        base_enhanced_frame = frame.copy()
        # Automatically enhance if the scene is dark
        if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < BRIGHTNESS_THRESHOLD:
            base_enhanced_frame = enhance_low_light(base_enhanced_frame)
        # Apply manual brightness/contrast from the trackbars
        base_enhanced_frame = apply_brightness_contrast(
            base_enhanced_frame, filter_state['brightness'], filter_state['contrast']
        )

        # --- Pipeline Step 2: Apply Selected Filter Mode ---
        mode = filter_state['mode']
        processed_frame = base_enhanced_frame.copy() # Start with the enhanced frame

        if mode in modes:
            filter_function, filter_name = modes[mode]
            processed_frame = filter_function(base_enhanced_frame)
        elif mode == 8: # Difference View
            processed_frame = apply_difference_view(frame, prev_frame)
        elif mode == 11: # Motion Flow
            draw_motion_vectors(frame, prev_gray_for_flow, processed_frame)
        
        # --- Pipeline Step 3: Motion Detection ---
        gray_motion_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_motion_frame = cv2.GaussianBlur(gray_motion_frame, (21, 21), 0)
        fg_mask = backSub.apply(gray_motion_frame)
        
        # Clean up the mask to reduce noise
        thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_motions = 0
        for contour in contours:
            # If a contour is too small, ignore it (likely noise)
            if cv2.contourArea(contour) < MOTION_MIN_AREA:
                continue
            detected_motions += 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # --- Pipeline Step 4: Apply Overlays and Status ---
        if filter_state['light_detect'] == 1:
            detect_light_sources(frame, processed_frame)

        # Update and display the motion status
        if detected_motions > 0:
            last_motion_time = time.time()
            if detected_motions > 1:
                motion_status = f"ALERT: {detected_motions} Objects in Motion!"
                motion_color = (0, 0, 255) # Red
            else:
                motion_status = "Motion Detected"
                motion_color = (0, 165, 255) # Orange
        else:
            if time.time() - last_motion_time > STILLNESS_THRESHOLD:
                motion_status = "Scene Clear"
                motion_color = (0, 255, 0) # Green
            else:
                motion_status = "PENDING"
                motion_color = (0, 255, 255) # Yellow

        # Draw the status bar at the top of the processed frame
        cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(processed_frame, f"STATUS: {motion_status}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, motion_color, 2)

        # --- Pipeline Step 5: Final Display ---
        # Update state for the next frame's calculations
        prev_frame = frame.copy()
        prev_gray_for_flow = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Stack the original and processed frames side-by-side
        combined_frame = np.hstack((frame, processed_frame))
        cv2.imshow(window_name, combined_frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Script closed successfully.")

# --- 7. SCRIPT ENTRY POINT ---

if __name__ == "__main__":
    main()

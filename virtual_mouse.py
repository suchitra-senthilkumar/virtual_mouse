"""
AI Virtual Mouse using Hand Tracking in Python.
Allows full control of system mouse (movement, left click, right click, double click, scroll, drag & drop)
via hand gestures captured through a webcam.
"""

import cv2
import numpy as np
import pyautogui
import time
import math
from hand_tracking_module import HandDetector

# Safety & Latency Configuration for PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001


def run_virtual_mouse():
    # Camera and Screen Dimensions
    w_cam, h_cam = 640, 480
    frame_margin = 100  # Boundary margin in camera view for screen mapping
    smoothing_factor = 5  # Motion smoothing factor (higher = smoother, lower = faster response)

    w_scr, h_scr = pyautogui.size()
    print(f"[INFO] Screen Resolution: {w_scr} x {h_scr}")
    print(f"[INFO] Camera Target Resolution: {w_cam} x {h_cam}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_cam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_cam)

    detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.7)

    # State variables for motion smoothing & debouncing
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0
    
    last_click_time = 0
    click_cooldown = 0.3  # Seconds between clicks to prevent unintended spamming
    is_dragging = False
    
    prev_scroll_y = 0
    prev_time = 0

    print("\n" + "="*50)
    print("      AI VIRTUAL MOUSE CONTROLLER ACTIVE      ")
    print("="*50)
    print("  GESTURES GUIDE:")
    print("   * Move Cursor : Index Finger UP")
    print("   * Left Click  : Index + Middle Finger Pinch (< 35px)")
    print("   * Right Click : Thumb + Middle Finger Pinch (< 35px)")
    print("   * Scroll Mode : Index + Middle Finger UP & Moved Vertically")
    print("   * Drag & Drop : Thumb + Index Pinch (Hold & Move)")
    print("   * Exit App    : Press 'q' key in video window")
    print("="*50 + "\n")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("[WARNING] Could not read frame from webcam.")
            time.sleep(0.1)
            continue

        # 1. Mirror image for intuitive movement mapping
        img = cv2.flip(img, 1)

        # 2. Find hand landmarks
        img = detector.find_hands(img, draw=True)
        lm_list, bbox = detector.find_positions(img, draw=False)

        # Draw Active Control Region Box
        cv2.rectangle(img, (frame_margin, frame_margin), 
                      (w_cam - frame_margin, h_cam - frame_margin), 
                      (255, 0, 255), 2)

        active_mode_text = "READY / IDLE"
        text_color = (255, 255, 255)

        if len(lm_list) != 0:
            # Landmark IDs
            x_index, y_index = lm_list[8][1], lm_list[8][2]    # Index tip
            x_thumb, y_thumb = lm_list[4][1], lm_list[4][2]    # Thumb tip
            x_middle, y_middle = lm_list[12][1], lm_list[12][2] # Middle tip

            # 3. Check extended fingers
            fingers = detector.fingers_up()

            current_time = time.time()

            # GESTURE 1: DRAG & DROP (Thumb + Index Pinch Hold)
            thumb_index_dist, img, line_info = detector.find_distance(4, 8, img, draw=False)
            if thumb_index_dist < 30 and fingers[1] == 1:
                active_mode_text = "DRAG & DROP"
                text_color = (0, 165, 255)

                # Map & smooth position
                x3 = np.interp(x_index, (frame_margin, w_cam - frame_margin), (0, w_scr))
                y3 = np.interp(y_index, (frame_margin, h_cam - frame_margin), (0, h_scr))
                curr_x = prev_x + (x3 - prev_x) / smoothing_factor
                curr_y = prev_y + (y3 - prev_y) / smoothing_factor

                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                cv2.circle(img, (line_info[4], line_info[5]), 12, (0, 165, 255), cv2.FILLED)

            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

                # GESTURE 2: CURSOR NAVIGATION MODE (Only Index Finger Raised)
                if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                    active_mode_text = "CURSOR NAVIGATION"
                    text_color = (0, 255, 0)

                    # Interpolate camera workspace coordinates to screen resolution
                    x3 = np.interp(x_index, (frame_margin, w_cam - frame_margin), (0, w_scr))
                    y3 = np.interp(y_index, (frame_margin, h_cam - frame_margin), (0, h_scr))

                    # Exponential Moving Average Smoothing
                    curr_x = prev_x + (x3 - prev_x) / smoothing_factor
                    curr_y = prev_y + (y3 - prev_y) / smoothing_factor

                    # Move OS Mouse Cursor
                    pyautogui.moveTo(curr_x, curr_y)
                    cv2.circle(img, (x_index, y_index), 10, (0, 255, 0), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

                # GESTURE 3: LEFT CLICK (Index + Middle Pinch)
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                    index_middle_dist, img, line_info = detector.find_distance(8, 12, img, draw=True)

                    if index_middle_dist < 35:
                        cv2.circle(img, (line_info[4], line_info[5]), 12, (0, 255, 0), cv2.FILLED)
                        active_mode_text = "LEFT CLICK"
                        text_color = (0, 255, 0)

                        if current_time - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = current_time

                    else:
                        # GESTURE 4: SCROLLING MODE (Index + Middle Finger Open but not pinched)
                        active_mode_text = "SCROLL MODE"
                        text_color = (255, 255, 0)

                        if prev_scroll_y != 0:
                            diff_y = y_index - prev_scroll_y
                            if abs(diff_y) > 8:  # Sensitivity threshold
                                scroll_units = int(-diff_y * 3)
                                pyautogui.scroll(scroll_units)
                        prev_scroll_y = y_index

                else:
                    prev_scroll_y = 0

                # GESTURE 5: RIGHT CLICK (Thumb + Middle Finger Pinch)
                if fingers[0] == 1 and fingers[2] == 1 and fingers[1] == 0:
                    thumb_middle_dist, img, line_info = detector.find_distance(4, 12, img, draw=True)
                    if thumb_middle_dist < 35:
                        cv2.circle(img, (line_info[4], line_info[5]), 12, (255, 0, 0), cv2.FILLED)
                        active_mode_text = "RIGHT CLICK"
                        text_color = (255, 0, 0)

                        if current_time - last_click_time > click_cooldown:
                            pyautogui.rightClick()
                            last_click_time = current_time

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Draw HUD & Diagnostic Indicators
        cv2.putText(img, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        cv2.putText(img, f"MODE: {active_mode_text}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        cv2.imshow("AI Virtual Mouse - Camera Feed", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting Virtual Mouse Application...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_virtual_mouse()

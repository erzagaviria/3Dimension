import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to plot a 3D cube
def plot_cube(ax, center, size):
    r = [-size/2, size/2]
    X, Y = np.meshgrid(r, r)
    
    ax.clear()
    
    # Z coordinates for the cube's faces, repeated to match the shape of X and Y
    Z1 = np.full_like(X, r[0] + center[2])
    Z2 = np.full_like(X, r[1] + center[2])
    
    # Plot the six faces of the cube
    ax.plot_surface(X + center[0], Y + center[1], Z1, color='r', alpha=0.6)
    ax.plot_surface(X + center[0], Y + center[1], Z2, color='r', alpha=0.6)
    ax.plot_surface(X + center[0], Z1, Y + center[1], color='r', alpha=0.6)
    ax.plot_surface(X + center[0], Z2, Y + center[1], color='r', alpha=0.6)
    ax.plot_surface(Z1, X + center[0], Y + center[1], color='r', alpha=0.6)
    ax.plot_surface(Z2, X + center[0], Y + center[1], color='r', alpha=0.6)
    
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Initialize the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to track hands and get the positions
def track_hands(cap):
    success, frame = cap.read()
    if not success:
        return None, None

    # Flip the frame horizontally for selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_positions = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks for the index finger tip (landmark #8) and wrist (landmark #0)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # Convert the landmarks to pixel coordinates
            height, width, _ = frame.shape
            cx, cy = int(index_tip.x * width), int(index_tip.y * height)
            hand_positions.append((cx, cy, hand_landmarks))
        
        return hand_positions, frame
    return None, frame

# Open the webcam
cap = cv2.VideoCapture(0)

# Set initial cube size and zoom factor
cube_size = 5
zoom_factor = 1.0

# Loop to track hands and control 3D object
while True:
    # Get the hand positions from the webcam
    hand_positions, frame = track_hands(cap)

    if hand_positions:
        # If two hands are detected, calculate zoom based on the distance between them
        if len(hand_positions) == 2:
            dist = calculate_distance(hand_positions[0][:2], hand_positions[1][:2])
            zoom_factor = dist / 100  # Adjust zoom sensitivity
            cube_size = 5 * zoom_factor

        # Normalize the position of the first hand to the range [-10, 10]
        normalized_x = (hand_positions[0][0] / cap.get(3)) * 20 - 10
        normalized_y = (hand_positions[0][1] / cap.get(4)) * 20 - 10

        # Draw the 3D cube with the new position and size
        plot_cube(ax, [normalized_x, -normalized_y, 0], cube_size)
        plt.pause(0.001)

    # Show the camera frame for debugging
    cv2.imshow('Hand Tracking', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

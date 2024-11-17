import PySimpleGUI as sg
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import data
#seperate list for each and every shot
scount = []
bcount= []
ncount = []

mp_pose = mp.solutions.pose
# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils
lst = []

def detectPose(image, pose, display=True):
            '''
            This function performs pose detection on an image.
            Args:
                image: The input image with a prominent person whose pose landmarks needs to be detected.
                pose: The pose setup function required to perform the pose detection.
                display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                         and the pose landmarks in 3D plot and returns nothing.
            Returns:
                output_image: The input image with the detected pose landmarks drawn.
                landmarks: A list of detected landmarks converted into their original scale.
            '''

            # Create a copy of the input image.
            output_image = image.copy()

            # Convert the image from BGR into RGB format.
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform the Pose Detection.
            results = pose.process(imageRGB)

            # Retrieve the height and width of the input image.
            height, width, _ = image.shape

            # Initialize a list to store the detected landmarks.
            landmarks = []

            # Check if any landmarks are detected.
            if results.pose_landmarks:

                # Draw Pose landmarks on the output image.
                mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                          connections=mp_pose.POSE_CONNECTIONS)

                # Iterate over the detected landmarks.
                for landmark in results.pose_landmarks.landmark:
                    # Append the landmark into the list.
                    landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                      (landmark.z * width)))

            # Check if the original input image and the resultant image are specified to be displayed.
            if display:

                # Display the original input image and the resultant image.
                plt.figure(figsize=[22, 22])
                plt.subplot(121);
                plt.imshow(image[:, :, ::-1]);
                plt.title("Original Image");
                plt.axis('off');
                plt.subplot(122);
                plt.imshow(output_image[:, :, ::-1]);
                plt.title("Output Image");
                plt.axis('off');

                # Also Plot the Pose landmarks in 3D.
                mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                for i in range(2):
                    # Display the found landmarks after converting them into their original scale.
                    print(f'{mp_pose.PoseLandmark(i).name}:')
                    print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * width}')
                    print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * height}')
                    print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * width}')
                    print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

            # Otherwise
            else:

                # Return the output image and the found landmarks.
                return output_image, landmarks


# Read another sample image and perform pose detection on it.


# Initialize the VideoCapture object to read from the webcam.
# video = cv2.VideoCapture(0)

# Initialize the VideoCapture object to read from a video stored in the disk.


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
   # print(f'The calculated angle is {angle}')
    # Return the calculated angle.
    return angle

    # Calculate the angle between the three landmarks.


def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''

    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Calculate the required angles.
    # ----------------------------------------------------------------------------------------------------------------

    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points.
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # ----------------------------------------------------------------------------------------------------------------

    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    # ----------------------------------------------------------------------------------------------------------------

        # Check if the other leg is bended at the required angle.
    if left_knee_angle > 70 and left_knee_angle < 90 or right_knee_angle > 70 and right_knee_angle < 90 and left_elbow_angle >0 and left_elbow_angle<15 or right_elbow_angle >0 and right_elbow_angle<15:

            # Specify the label of the pose that is Warrior II pose.
            print(f'Net Drop')
            label = 'Net Drop'

            ncount.append("Net Drop")

    if right_shoulder_angle > 160 and right_shoulder_angle < 200:

            # Specify the label of the pose that is tree pose.
            print(f'Overhead clear')
            label = 'Overhead clear'
            bcount.append("Overhead clear")

    if right_shoulder_angle > 110 and right_shoulder_angle < 160 and left_shoulder_angle > 110 and left_shoulder_angle < 160 :
        # Specify the label of the pose that is tree pose.
        print(f'Smash')
        label = 'Smash'
        scount.append("Smash")
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        print(f'Unknown Pose')
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)

        # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Check if the resultant image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:

        # Return the output image and the classified label.
        return output_image, label


sample_img = cv2.imread('media/sample.jpg')
sg.theme("BlueMono")
layout = [[sg.T("")],[sg.Text('Badmintion Shot Recognizer')], [sg.Button("START")], [sg.Button("Exit")]]

#image is convert in to base 64 to diplay the image in the start menue

layout3 = [ [sg.Button('', image_data=data.pic_base64,
            button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='Exit')]  ]
layout2 = [[sg.VPush()],
              [sg.Push(), sg.Column(layout,element_justification='c'),sg.Column(layout3), sg.Push()],
              [sg.VPush()]]

###Building Window

window = sg.Window('Batmintion Shot Recognizer', layout2, size=(700, 600))

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "START":
        filename = sg.popup_get_file('Upload to desired Video')
        # Initialize the VideoCapture object to read from a video stored in the disk.
        video = cv2.VideoCapture(filename)

        # Initialize a variable to store the time of the previous frame.
        time1 = 0

        # Iterate until the video is accessed successfully.
        while video.isOpened():

            # Read a frame.
            ok, frame = video.read()

            # Check if frame is not read properly.
            if not ok:
                # Break the loop.
                break

                # Flip the frame horizontally for natural (selfie-view) visualization.
            frame = cv2.flip(frame, 1)

            # Get the width and height of the frame
            frame_height, frame_width, _ = frame.shape

            # Resize the frame while keeping the aspect ratio.
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            # Perform Pose landmark detection.
            frame, landmarks = detectPose(frame, pose_video, display=False)

            # Check if the landmarks are detected.
            if landmarks:
                # Perform the Pose Classification.
                frame, _ = classifyPose(landmarks, frame, display=False)

            # Set the time for this frame to the current time.
            time2 = time()

            # Check if the difference between the previous and this frame time &gt; 0 to avoid division by zero.

            # Update the previous frame time to this frame time.
            # As this frame will become previous frame in next iteration.
            time1 = time2

            # Display the frame.
            cv2.imshow('Pose Detection', frame)

            # Wait until a key is pressed.
            # Retreive the ASCII code of the key pressed
            k = cv2.waitKey(1) & 0xFF

            # Check if 'ESC' is pressed.
            if (k == 27):
                # Break the loop.
                break

        # Release the VideoCapture object.
        video.release()

        overhead =0
        for i in range(len(lst)):
            # convert each item to int type
            a = 'Smash'
            b = 'Overhead clear'
            if (lst[i] != a):
                print("Not equal")
            else:
                scount=scount+1

            if (lst[i] != b):
                    print("Not equal")
            else:
                    bcount = bcount + 1

       # lst2=[]
      #  lst2.append("Smash "+ scount)
      #  lst2.append("Overhead "+ bcount)
        smsize = len(scount)
        omsize = len(bcount)
        nmsize = len(ncount)
        report = " Smash " + str(smsize) +"\n Overhead Clear " + str(omsize) + "\n Net Drop  " + str(nmsize)
        sg.PopupScrolled(report, title="Analytical Report")

        # Close the windows.
        cv2.destroyAllWindows()
    elif event == "START":
        window.Close()
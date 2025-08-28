# DramaticNightVisionCamera
Vision Command: Real-time Video Analysis Suite 👁️‍🗨️
<div align="center">

A powerful computer vision toolkit built with Python and OpenCV that transforms a standard webcam into an intelligent, multi-feature analysis and monitoring system.

</div>

This is what Vision Command looks like in action. The left pane shows the original webcam feed, while the right pane displays the processed output with all enhancements and overlays.

[GIF of the Vision Command GUI in action]

## 📖 About The Project
Vision Command is an advanced-intermediate computer vision application designed to perform a wide range of real-time video processing tasks. It provides an interactive GUI to seamlessly switch between different analytical filters, enhancement modes, and object detection overlays.

Whether you're exploring the fundamentals of image processing or building a sophisticated monitoring system, this project serves as a powerful and educational sandbox.

## ✨ Key Features
🖥️ Real-time Video Processing: Analyzes live video from your webcam with minimal latency.

🎛️ Interactive GUI: Easily control filters, brightness, contrast, and other features on-the-fly with intuitive trackbars.

🎨 Multi-Filter Suite: Includes over 10 different vision filters:

Edge Detection (Canny, Sobel, Laplacian)

Texture Analysis (LBP)

Aesthetic Simulators (Thermal, Green Night Vision)

Image Enhancement (Top-hat, Morphological Gradient)

🏃‍♂️ Intelligent Motion Detection: Uses background subtraction to detect and highlight moving objects, with status updates for "Motion Detected" and "Scene Clear."

🌙 Automatic Low-Light Enhancement: Automatically improves visibility in dark environments using the CLAHE algorithm and shadow lifting.

➡️ Advanced Motion Analysis: Visualizes the direction and flow of movement in the scene using Lucas-Kanade optical flow.

## 🛠️ Built With
This project relies on these amazing open-source libraries:

Python

OpenCV

NumPy

scikit-image (Optional, for LBP filter)

## 🚀 Getting Started
To get a local copy up and running, follow these simple steps.

Prerequisites
Make sure you have Python (3.7 or newer) and pip installed on your system.

Installation
Clone the repository

Bash

git clone https://github.com/your_username/vision-command.git
cd vision-command
Install the required packages

Bash

pip install opencv-python numpy scikit-image
Run the application

Bash

python main.py
(Assuming you save the code as main.py)

## 🕹️ How to Use
Once the application is running, a window will appear showing two video feeds (Original vs. Processed). Use the trackbars in the control window to manipulate the processed feed:

Filter Mode: Slide to switch between the 12 different vision modes (0 is 'Off').

Light Detect: Toggle to 1 to highlight bright light sources in the frame.

Brightness: Adjust the overall brightness of the image (50 is default).

Contrast: Adjust the overall contrast of the image (10 is default).

Press the 'q' key on your keyboard while the video window is active to close the application.

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement."

## 📜 License
Distributed under the MIT License. See LICENSE.txt for more information.

## 👨‍💻 Author
Utkarsh Tripathi

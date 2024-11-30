# Face Detection Methods

This repository contains two different approaches to face detection: **Modern Approach using YOLO-Face** and **Traditional Approach using MTCNN and DeepFace**. The purpose of this repository is to demonstrate the speed, ease of deployment, and effectiveness of these methods in detecting faces in images.

---


### Modern Approach: YOLO-Face

- **YOLO-Face** is a modern and fast face detection model based on the YOLO (You Only Look Once) architecture. It can detect faces in real-time with high accuracy and speed.
- This method is implemented in two files:
  - `yolo_face.ipynb`: A Jupyter notebook to demonstrate the use of YOLO-Face for face detection.
  - `yolo_face.py`: The Python script used for face detection in a programmatic way.
  - `results/`: Contains images showing the results of YOLO-Face face detection.

---


#### Features of YOLO-Face:
- **Speed:** YOLO-Face is known for its fast inference time, even in real-time applications.
- **Ease of Deployment:** It’s easy to deploy and use with pre-trained models available for a variety of tasks.
- **Scalability:** This method works efficiently on a range of devices, from personal computers to edge devices like the Jetson.

---


### Traditional Approach: MTCNN and DeepFace

- **MTCNN (Multi-task Cascaded Convolutional Networks)** and **DeepFace** are traditional methods for face detection. Both are widely used for face detection and recognition tasks.
- This approach is implemented in the following files:
  - `mtcnn_deepface.ipynb`: A Jupyter notebook demonstrating the use of both MTCNN and DeepFace for face detection.
  - `results/`: Contains images showing the results of MTCNN and DeepFace face detection.

---


#### Features of MTCNN and DeepFace:
- **MTCNN**: MTCNN is a popular method for face detection that works well with aligned faces in the image. It uses a cascade structure that allows it to detect faces in various sizes and poses.
- **DeepFace**: DeepFace is a high-level face recognition library that supports several pre-trained models for face detection and recognition.

---


### Comparison of YOLO-Face vs MTCNN/DeepFace

| **Feature**       | **YOLO-Face**                 | **MTCNN & DeepFace**          |
|-------------------|-------------------------------|-------------------------------|
| **Speed**         | Fast, real-time performance    | Slower compared to YOLO       |
| **Ease of Deployment** | Easy to deploy and integrate  | Requires additional setup    |
| **Accuracy**      | High accuracy, performs well in real-world scenarios | Good for aligned faces, but less efficient with scale variations |
| **Use Case**      | Real-time detection in various conditions | Best for smaller datasets or controlled environments |
---

### Results: Showing Comparison Images

You can visualize the results from both methods by running the following syntax in your Python environment.

---

#### YOLO-Face Result:
```python
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('path_to_your_image')

# Show the result (you should have added YOLO bounding boxes on the image)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('YOLO-Face Result')
plt.axis('off')
plt.show()
```


---

#### DeepFace Result:
```python

import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('path_to_your_image')

# Show the result (you should have added MTCNN/DeepFace bounding boxes on the image)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('MTCNN/DeepFace Result')
plt.axis('off')
plt.show()
```
---

#### YOLO-Face Image Results

| Original Image                     | Detected Faces (YOLO-Face)        |
|------------------------------------|-----------------------------------|
| ![](/Modern/Data/test2.jpg)        | ![](/Modern/Data/yolo-face.jpg)   |

#### DeepFace Image Results

| Original Image                     | Detected Faces (DeepFace)         |
|------------------------------------|-----------------------------------|
| ![](/Traditional/result/test2.jpg) | ![](/Traditional/result/deepface.png) |


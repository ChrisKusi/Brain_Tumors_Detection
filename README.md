# Brain Tumor Detection Using YOLOv8 #

Brain Tumor Detection Using YOLOv8
This project was developed by Christian Kusi, the Robotics, AI, and IoT club lead for Valley View University. The aim of this project is to detect brain tumors using YOLO v8, an object detection algorithm.

Dataset
The dataset used for this project is the Brain MRI Images for Brain Tumor Detection dataset, which can be found here --> "https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection". The dataset contains MRI images of the brain.

Methodology
The YOLO v8 algorithm was used to detect brain tumors in the MRI images. The algorithm was trained on the dataset using transfer learning. The pre-trained weights of the YOLO v8 model were used as the starting point for training on the dataset. The training was carried out on a GPU, which significantly reduced the training time.

Results
The trained YOLO v8 model was able to accurately detect brain tumors in the test set with a high degree of accuracy. The precision and recall metrics were both above 90%, indicating that the model is highly effective in detecting brain tumors.


How to Use
To use the brain tumor detection system, follow these steps:

Clone the GitHub repository
Download the BTS dataset
Install the necessary dependencies
Train the YOLO v8 model on the dataset
Test the trained model on new MRI images
For detailed instructions on how to use the system, refer to the README.md file in the repository.

Acknowledgments
We would like to thank the creators of the BTS dataset for making it available for research purposes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

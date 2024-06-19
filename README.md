# Sleep Detection
This project uses a trained Convolutional Neural Network (CNN) model and the MTCNN face detector to detect if a person is sleepy or awake in images and videos.
It utlizes MobileNetV2 architecture pre-trained on ImageNet for the classification which was further fine-tuned to improve its accuracy.
The dataset was retrieved from kaggle having a Yolo format, which was then modified to have train, test and validation folders to include 
the classes instead of images and labels using the dataset_format_conversion script. 
The conversion was necessary for the proper training via ImageDataGenerator

## Dependencies
- OpenCV
- MTCNN
- NumPy==1.26.4
- TensorFlow==2.10.1
- Gradio

## How to Run
1. clone this repository
2. pip install `requirements.txt` or install the dependencies manually (Please make sure the libraries having its version
    mentioned explicitly be installed as such to avoid conflicts and errors)
3. open the terminal from your IDE and make sure the directory is set to the path of the repository
4. run `sleep_detection.py`

## Functionality
The script `sleep_detection.py` contains several functions:

- `classify_faces(img)`: This function takes an image as input, detects faces in the image, and classifies each face as 'Awake' or 'Sleepy'. It also draws bounding boxes around the sleepy faces, labels them and counts the number of faces that were detected as sleepy
  
- `process_image(image_path)`: This function reads an image from the specified path, processes it, and passes it to classify_faces(img) for face detection and classification
  
- `process_video(video_path)`: This function reads a video from the specified path, processes each frame, and passes them to classify_faces(img) for the same
  
- `image_interface(image)`: This function provides an interface for processing images which is then consolidated in the tabbed interface of gradio
  
- `video_interface(video_path)`: Same as the previous, it provides an interface for the processing of videos
- 
## Deployment
The deployment of this project was done using `Gradio` and it is hosted on `Hugging Face Spaces`
Link to the website:- https://huggingface.co/spaces/Cosmos48/gradio-sleep_detection

## Author
Joy Biswas

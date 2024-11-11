# Fruit Classification Using CNN

## Table of Contents
- [Project Description](#project-description)
- [Limitations](#limitations)
- [Packages Used](#packages-used)
- [License](#license)

## Project Description
This project uses a dataset containing images of 11 different fruits. Each fruit in the dataset is labeled with one of three conditions: *Good*, *Rotten*, or *Mild*. The images are processed and fed into a Convolutional Neural Network (CNN) model, which is trained to classify both the type of fruit and its condition.

The primary goal of this project is to showcase fruit classification using CNNs, while also experimenting with the model's ability to generalize based on the dataset's characteristics.

## Limitations
- **Generalization**: The model's generalization is limited due to the small size of the dataset and the lack of variability in the images. 
- **Image Uniformity**: The images in the dataset have very little background noise, and the fruits are positioned similarly in each image. This reduces the variety the model encounters during training, making it harder to generalize to new, real-world images.
  
## Packages Used
- `tensorflow`: Deep learning library used to build and train the CNN model.
- `scikit-learn`: For data preprocessing and evaluation metrics.
- `matplotlib`: Used for visualizing the dataset and results.
- `warnings`: For managing warnings in the code.
- `json`: To handle JSON files for saving results and model configurations.
- `numpy`: Used for numerical operations, such as array manipulation.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

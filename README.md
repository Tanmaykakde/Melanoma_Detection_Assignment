# <font style = 'u'>Melanoma_Detection_Assignment</font>
# Skin Cancer Detection using CNN

Skin cancer is a deadly disease that can be life-threatening if not detected early. This project aims to build a CNN-based model that accurately detects melanoma, the deadliest form of skin cancer. The model evaluates images and alerts dermatologists about the presence of melanoma, reducing the manual effort needed for diagnosis.

## Dataset

The Dataset can be downloaded from [Here.](https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view)

The dataset used in this project consists of 2357 images of malignant and benign oncological diseases, sourced from the International Skin Imaging Collaboration (ISIC). The dataset contains nine sub-directories, each representing a specific type of skin cancer:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

The dataset was divided into training and testing sets, with approximately 80% of the images used for training and 20% for validation.

## Technologies Used

- Python
- TensorFlow
- Keras
- Augmentor
- NumPy
- Matplotlib
- OpenCV
- PIL

## Model Architecture

The CNN model used in this project consists of several convolutional and pooling layers, followed by fully connected layers. The model was trained to classify the nine different types of skin cancer present in the dataset. Data augmentation techniques, such as random flipping, rotation, and zoom, were applied to handle class imbalances and improve model performance.

## Results

After training the model for around 50 epochs, the following results were achieved:

- Training Accuracy: 80%
- Validation Accuracy: 75%
- Training Loss: 0.5
- Validation Loss: 0.7

The model showed significant improvement in accuracy and reduced overfitting compared to earlier versions. Class rebalancing using data augmentation played a crucial role in achieving these results.

## Conclusion

The developed CNN model shows promising results in accurately detecting different types of skin cancer. By leveraging deep learning techniques and data augmentation, the model can assist dermatologists in diagnosing melanoma and other skin cancer types. Further improvements can be made by collecting more diverse and balanced datasets and fine-tuning the model architecture.


#### Created by :
- Tanmay Kakde
- Ranjan Lokegaonkar
- Shubhi Mishra


## References

[1] ISIC - The International Skin Imaging Collaboration: https://www.isic-archive.com/<br>
[2] Augmentor Documentation: https://augmentor.readthedocs.io/<br>
[3] TensorFlow Documentation: https://www.tensorflow.org/<br>
[4] Keras Documentation: https://keras.io/<br>
[5] OpenCV Documentation: https://docs.opencv.org/<br>
[6] NumPy Documentation: https://numpy.org/doc/<br>
[7] Matplotlib Documentation: https://matplotlib.org/<br>
[8] PIL Documentation: https://pillow.readthedocs.io/<br>




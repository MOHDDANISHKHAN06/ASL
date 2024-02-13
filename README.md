# ASLHandDetection
## Multiple Classifiers that detect the hand orientation and predict the sign shown, in Americal Sign Language.

The repository consists of 4 different classifiers- 
* Convolutional Neural Networks (CNN)
* Random Forest
* Decision Tree
* SVM

In Random Forest, 3 separate decision trees are being used. 
In SVM, the kernels are set as linear, and rbf, in 2 separate models. 

Running the main function will show the result for all the classifiers. 

The data set being used is a json, which projects details such as the finger postions, hand position, orientation [using quaternions], hand direction (front or back) etc. 
We are extracting the following features - 

1. Angle between each fingers
2. The positions of the fingers
3. Converting quaternions to Euler's notation
4. Hand Direction

The accuracy, after running these classifiers, are as follows- 
* Accuracy- CNN: ~0.85
* Accuracy- Decision tree: 0.98
* Accuracy- SVM(RBF) : 0.86
* Accuracy- SVM(LINEAR) : 0.96
* Accuracy- Random Forest: 0.94


# ASL Gesture Recognition with Leap Motion and Unity

## Project Overview

This innovative project leverages the Leap Motion Controller and Unity to recognize American Sign Language (ASL) gestures. By capturing the detailed hand and finger movements and rendering them into visual data, we aim to accurately predict ASL gestures for alphabets (A-Z) and numerals (0-10).

## Materials and Methods

### Leap Motion Controller

A cutting-edge 3D motion-sensing device that interprets human hand movements into digital signals. It uses infrared cameras and LEDs to track hands and gestures, converting these into 3D coordinates for digital interaction.

#### Ultraleap Hand Tracking Software

Enhances the Leap Motion data, identifying individual fingers and gestures in real-time with high accuracy. Ideal for developers seeking precise hand tracking capabilities.

#### Ultraleap Plugin for Unity

Enables easy integration of Leap Motion's hand tracking into Unity-based applications, facilitating immersive user experiences with hand gestures in VR, AR, and other interactive platforms.

### Unity

A versatile platform for game development and interactive content creation, providing a rich environment for rendering 3D hand models captured by the Leap Motion Controller.

### Data Collection

- **Participants:** 10 volunteers from our university, comprising an equal mix of males and females.
- **Procedure:** Participants performed ASL gestures, which were captured using the Leap Motion Controller and visualized in Unity. Data was stored in JSON format alongside synthetic images for training machine learning models.

### Feature Extraction

Detailed features of hand and finger movements were extracted, including palm data, finger data, and bone data. Engineered features like angles between fingers and distances between fingertips were also calculated to enrich the dataset.

## Objectives and Findings

The goal was to evaluate the effectiveness of machine learning models, including SVM, Random Forests, and DNNs, in predicting ASL gestures. Our models achieved impressive accuracies, demonstrating the potential of this technology in enhancing communication accessibility.

## Contributions

This project offers a comprehensive LMC dataset of ASL gestures, a 3D reconstruction of these gestures using Unity, and insights into the predictive accuracies of various machine learning models. It underscores the Leap Motion Controller's versatility and potential applications in HCI, particularly for the deaf and hard of hearing.

## Future Directions

While focused on static gestures, this groundwork facilitates further exploration into dynamic gesture recognition, broadening the application scope of ASL recognition technologies.

---

For more information, contributions, or inquiries, please contact [Author Name](mailto:e-mail@e-mail.com).


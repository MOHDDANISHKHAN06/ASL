# ASL Gesture Recognition with Leap Motion and Unity
![collage](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/68fbe285-0d28-41da-8532-31e3a6f63d36)

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

![WhatsApp Image 2023-10-24 at 9 46 29 PM](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/a37aa442-553f-4d2f-9a9f-3df8c5f2f776)
![WhatsApp_Image_2023-10-26_at_8 24 16_PM-removebg-preview](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/7b17101d-86e4-4daa-994e-a8e50fd91242)


### Feature Extraction

Detailed features of hand and finger movements were extracted, including palm data, finger data, and bone data. Engineered features like angles between fingers and distances between fingertips were also calculated to enrich the dataset.

## Objectives and Findings

The goal was to evaluate the effectiveness of machine learning models, including SVM, Random Forests, and DNNs, in predicting ASL gestures. Our models achieved impressive accuracies, demonstrating the potential of this technology in enhancing communication accessibility.

# ASL Recognition Results

This section showcases the results of applying different machine learning models, including SVM (Support Vector Machine), Random Forest, and DNN (Deep Neural Network), on the ASL (American Sign Language) recognition task.

## SVM and Random Forest Results

![SVM and Random Forest Result 1](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/b75c2c7c-5322-4242-b28d-d9c520793440)

![SVM and Random Forest Result 2](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/50a9c191-0374-4cd5-aa1b-f4b553fe25d2)

![SVM and Random Forest Combined](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/3422bd1b-1eb2-4d39-b553-11bf0a225be6)

## DNN Results

![DNN Result 1](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/f5b565e4-9a93-4071-b0c0-8cb4d1a343d3)

![DNN Combined Results](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/5fbea9fe-31e8-4885-8a1c-40ecc434b473)



## Future Directions

While focused on static gestures, this groundwork facilitates further exploration into dynamic gesture recognition, broadening the application scope of ASL recognition technologies.

---

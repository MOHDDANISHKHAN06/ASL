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
![output](https://github.com/MOHDDANISHKHAN06/ASL/assets/47732298/c69c157d-766d-4257-afef-a529700a2bcf)

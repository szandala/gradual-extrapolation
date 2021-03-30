# Gradual Extrapolation

## Idea
We propose an enhancement technique for the Gradient-weighted Class Activation Mapping (Grad-CAM) method, which presents visual explanations of decisions from CNN-based models. Our idea, called gradual saliency extrapolation, can supplement Grad-CAM or any other method that generates a heatmap picture. Instead of producing a coarse localization map highlighting the important predictive regions in the image, our method outputs the specific shape that most contributes to the model output. In this way, it improves the accuracy of saliency maps. In validation tests on a chosen set of images, the proposed method significantly improved the localization detection of the neural networks’ attention. Furthermore, the proposed method is applicable to any deep neural network model.


## Sample input images

We have prepared a set of several images to present output of the method.

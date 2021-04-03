# Gradual Extrapolation

## Idea
We propose an enhancement technique for the Gradient-weighted Class Activation Mapping (Grad-CAM) method, which presents visual explanations of decisions from CNN-based models. Our idea, called gradual saliency extrapolation, can supplement Grad-CAM or any other method that generates a heatmap picture. Instead of producing a coarse localization map highlighting the important predictive regions in the image, our method outputs the specific shape that most contributes to the model output. In this way, it improves the accuracy of saliency maps. In validation tests on a chosen set of images, the proposed method significantly improved the localization detection of the neural networks’ attention. Furthermore, the proposed method is applicable to any deep neural network model.

## Usage

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python vision.py
```

First we have to import GradualExtrapolator class. Next we have to load the model, in this case a pretrained `vgg16` has been chosen and then we have to call the first GradualExtrapolator method to register required hooks in the model.
With such a prepared model we can perform the classification and, since the actual output is not needed, we can just ignore it. Model execution is followed by using the second GradualExtrapolator method to calculate smooth maps for the processed batch.

## Sample results

We have prepared a set of several images to present output of the method.

### Grad-CAM

![Grad-CAM](https://github.com/szandala/gradual-extrapolation/blob/master/outputs/output_Grad-CAM.jpg?raw=true)

### Gradual Grad-CAM

![Gradual Grad-CAM](https://github.com/szandala/gradual-extrapolation/blob/master/outputs/output_Gradual-Grad-CAM.jpg?raw=true)

### Contrastive Excitation BP

![Contrastive Excitation BP](https://github.com/szandala/gradual-extrapolation/blob/master/outputs/output_Contr-Excit-BP.jpg?raw=true)

### Gradual Contrastive Excitation BP

![Gradual Contrastive Excitation BP](https://github.com/szandala/gradual-extrapolation/blob/master/outputs/output_Gradual-Contr-Excit-BP.jpg?raw=true)

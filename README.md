# Transforming-How-We-Diagnose-Heart-Disease
Transforming How We Diagnose Heart Disease is a convolutional neural network build to determine if a patient has or not heart diseases.

The neural network was designed to solve the Second Annual Data Science Bowl competition: https://www.kaggle.com/c/second-annual-data-science-bowl

To accomplishment this challenge, the neuroan network determinate the frame fot the diastola (when heart has the maximum amount of blood through it) and the frame for the siastola (when heart has the minimum amount of blood through it ).
All the slices for each frames are summed individually and its result the volume of blood at sistola and diastole. With those results it's calculated the ejection fraction, which determiante if a pacient has problem or not with the heart.

The neural network was written with TensorFlow library in Python version 3.5 and it was trained on Sunnybrook dataset.

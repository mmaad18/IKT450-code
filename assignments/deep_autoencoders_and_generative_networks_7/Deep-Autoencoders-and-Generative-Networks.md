# 7 - Deep Autoencoders and Generative Networks 

In this task you are supposed to implement an autoencoder removing systematic noise. Some cameras add a date stamp to images. 
You should add systematic noise in form of date stamps on images and use autoencoders to remove these time stamps.

- Download the food 11 dataset https://mmspg.epfl.ch/food-image-datasets (same data set as earlier)
- Manually add date stamps (e.g. August 15, 2024) on the images using any Python library you prefer.
- Train an autoencoder and a GAN to avoid the date stamp. As an example, if an image has the text “August 15, 2024” on it, 
  the autoencoder or GAN should be able to accurately remove the text.
- Hint: Start with a subset of the dataset.
- Choose the network architecture with care.
- Train and validate all algorithms.
- Make the necessary assumptions.


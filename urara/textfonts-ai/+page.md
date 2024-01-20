---
title: "Training Generative Models for Typefaces"
tags: 
  - "computer vision"
  - "generative models"
image_caption: Styles across one of the dimensions in latent space
image: /textfonts-ai/images/tensor-fonts.gif
created: 2021-07-31
---

*You can find [this project's repository on Github](https://github.com/nestorSag/textfont-ai), along with pretrained models and an interactive Dash web app.*

## Motivation

At some point last year I was watching one of the episodes of [Abstract: the art of design](https://en.wikipedia.org/wiki/Abstract%3A_The_Art_of_Design) on Netflix, when I realised typeface generation was possibly a low hanging fruit from a machine learning perspective. Deep generative models have been shown to achieve impressive results in a variety of tasks, particularly image generation, and this includes datasets such as faces, landscapes and even painting styles, so it is reasonable to expect they wouldn't also excel at typeface generation. 


This is not a new idea, for example in [this article](https://distill.pub/2017/aia/) they do exactly that, and approach the problem from the broader context of AI-assisted design. There are also a few Github repositories on the subject. I thought it would be fun to give it a shot myself. However, I did not try to produce typeface files per se but only typeface images, i.e. images of typeface characters, as the former would involve an awful lot of work dealing with the internal complexities of `otf` and related file formats.

## Data

The main problem here was to get data. Google makes their fonts publicly available, so thats around 4k examples. There are plenty of websites that offer free fonts online, but they don't have an API of course, so there was no other way than scraping a few of them. In the end I got a bit under 130k fonts, or around 20 GB of images.


Mapping fonts to labeled images ready for model consumption was relatively easy, with an Apache Beam job on Google Cloud extracting 64-by-64 character images, and Tensorflow's `Dataset` abstraction to consume remote files. Being free fonts, there were lots of corrupted files, corrupted characters within otherwise ok files, mislabeled characters, and fonts that looked nothing like characters.


I did train a classifier on the dataset and discarded all misclassified characters (around 10%); discarded images were mostly fonts or characters that were just too extravagant or simply defective, and I found that this improved the quality of generative models downstream. I also restricted the character set to uppercase to save a bit of time.

## Architecture

Once plenty of clean data was available the next problem was deciding on a model architecture. I am not an expert in generative models, but thought the architecture outlined in [Adversarial Autoencoders by Goodfellow et all](https://arxiv.org/abs/1511.05644) looked good for this problem, as it enabled the model to also receive label information. I ended up doing one slight modification to this starting architecture, and the workflow looked like the following diagram (**this might be hard to see with a dark background**):

![architecture](/textfonts-ai/images/architecture.png)

The only difference with the paper I mentioned is that I split the encoding phase in 2: first the image is encoded by an *image encoder*, then a *full encoder* takes the encoded image features *and* the labels (this is, the one-hot-encoded charater labels) to finally produce the embedded style representation. I did this hoping that the labels help not only on the decoding phase but also on the encoding one, say, by underlining the right features given the character label, e.g. if it's an H, curviness is probably more important to the font's style than if its a C, which I hoped would speed up training.



### Character style models

The following image shows one of the model's style components for a randomly sampled font, once the model plateaud to a MSE of around 0.020 (this is the pixelwise MSE using normalised pixels in [0,1]) by training it with minibatches of randomly sampled character images across the dataset:

![chars](/textfonts-ai/images/chars.gif)
*Transition through a straight line in feature space between two randomly generated fonts.*

There was a caveat though: generating all characters for a given style vector does not necessarily produce consistent image styles across the character set. I think this is because the model is only encoding the style of individual characters, as during training there is nothing that indicates any association between characters from the same font, and so, the latent style space ends up encoding styles slightly differently for different characters. To be fair, this was a relatively uncommon occurrence, but it did mean that this model wasn't ideal for font generation. 


### Font style models: a self-supervised approach

In order to address the caveat mentioned above, I started taking font minibatches rather than image minibatches; this restricted the training to around 70k examples of fonts that were complete (i.e., no character was lost due to corrupted data or misclassification). The trick here was to use a bit of self supervised learning to try and make the model learn the fonts' style rather than the character style. 


To do this, I shuffled the images and labels randomly when passing them to the decoder. So for example, the decoder might get the style vector from an 'A', but be required to reconstruct a 'B' instead, which should be possible to do from just the style vector and the one-hot-encoded label for 'B'. This worked, and the styles were now consistent across characters for all style vectors, but the images were more blurry than I expected, even after the model plateaued, with a mean squared error of 0.075:

![fonts](/textfonts-ai/images/fonts.gif)
*Transition through a straight line in feature space between two randomly generated fonts.*

An interesting phenomenon was that this model consistently used just 5 dimensions in the style space even when there were more than that, making the rest useless; I suspect this means that there are (broadly speaking) only as many high-level characteristics that can be generalised from a single character to entire font styles, e.g. tickness, height/width ratio and so on.

### Font style models: fonts as 26-channel images

My second attempt was to take fonts as images with 26 channels where each channel was associated to a character. With this architecture, there was no need for labels anymore, as now channels acted implicitly as labels; since labels were gone, there wasn't any need for splitting the encoding stage in 2 parts, so the whole setup reduced to the usual autoencoder architecture, plus the discriminator network on the side, simplifying things quite a bit. 


This model worked better in general, achieving a lower reconstruction error and having faster training times. Since fonts are passed as multi-channel images, this is less intensive on the GPU's memory as well, because intermediate representations are per-font and not per-image.


![fonts](/textfonts-ai/images/tensor-fonts.gif)
*Transition through a straight line in feature space between two randomly generated fonts.*

I have to admit all results looked worse than expected at first. Then again, training high-quality generative models is not easy. Anyway, I think with a bit more data to generalise better, and with a sequence model to map images to points on the plane, (and with an expert that helps me navigate the technical aspects of font files!) it would even be possible to generate usable font files and not just images. Maybe this would be a nice bit of help for designers, to have a starting point when they set out to create a new font.

This project is available on [Github](https://github.com/nestorSag/textfont-ai), along with some pretrained decoders, and a Dash app in which to visualise style spaces.


## Lessons in MLOps

This project was more than anything an excuse to get my hands dirty with MLOps practices, and I placed a lot of emphasis on this along the project. A few lessons I learned:

* Experiment tracking does make a difference in project organisation, and MLFlow is a great tool for this

* Configuration became the project's center of gravity. Comprehensive YAML configuration schemas is what enabled adding complexity without chaos as a byproduct.



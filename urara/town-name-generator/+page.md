---
title: "A Generative Model For English Town Names"
tags: 
  - "NLP"
  - "generative models"
image_caption: Styles across one of the dimensions in latent space
image: /town-name-generator/images/sign.jpg
created: 2024-01-18
---

## Motivation

It seems [I am not the only one](https://www.instagram.com/reel/Cv5SkdLrVoq/?igsh=MXIzaDgzODBxZDlsZg==) that find some English town names slightly quirky or outright funny. Maybe it is because their spelling is quite different to the rest of English vocabulary we use one our daily lives, or maybe some of these places were named a long time ago when English looked and sounded very different from today.

Whatever the reason, I thought it would be fun to train a small transformer to come up with similar-sounding names, if only to see if we would fall for a deepfake sign on a road. 

## Data

Relevant data for most countries can be easily accessed from `http://download.geonames.org`, and the data contain administrative information too, so it is possible to use data for a single state or even county. In the case of England, available data contains a bit over 16,000 names.

## Architecture


This project was done with Keras NLP. The model is a transformer decoder, identical to the one in [Attention Is All You Need](https://arxiv.org/abs/1706.03762), with the following parameters

* Attention heads: 3
* Transformer layers: 3
* Fully connected layer size: 512
* Embedding size: 32

Training parameters were:

* Batch size: 32
* Epochs: 200
* Optimizer: ADAM with `learning_rate = 0.001`

All of this took around 10 minutes on a CPU.

The tokenizer worked at character level, which for English produced 30 characters, including beginning and end of text tokens

## Results 

Here are some of the generated names I liked the most for England

* Upminster
* Whippleigh
* Kelingbrough
* Millers mill
* Croomfleet
* Chillarton
* Egerton on the hill
* Kilkinster
* Ashton Dingley
* Hegleton

As a non-native speaker I have to admit I wouldn't bat an eye if I saw any of those on a road sign (Perhap's Miller's Mill would raise some suspicion), which somehow makes it more amusing.

This little experiment can be re-run for any other country or subregion with barely any change in command line parameters too, so I also had a chuckle doing this for Mexican towns. Some of the best ones:

* San Juan Guilalapam
* El Malo
* Llano Grande
* Yuchiqui de la Luma
* Quinicuelo


# Code 

The project's repo is [here](https://github.com/nestorSag/towngen).


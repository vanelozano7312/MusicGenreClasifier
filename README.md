# Music classifier by genres using CNN and KNN models

The goal of this project is to build machine learning models that process audio files (in wav format for this particular case) and classify them into the correct music genre.

Different types of machine learning algorithms and solutions are capable of solving this problem, from simple and supervised machine learning algorithms that can be used to solve classification and regression problems such as KNN, to convolutional neural networks (CNN). For this particular project, we focused on building a CNN and KNN that can accurately predict the genre of different audio files.

## Slides and Report 

- [Slides]()

- [Report]()

## structure
In order to use the .ipynb file we need to download the *Data CNN* and *Data* folder and put it in the google collab folder. The *Data* file is big so it'll work better if you upload it in your drive and used from there in the google collab, changing the "DATASET_PATH = "/content/Data/genres_original" rute" for your drive rute and uncomment # save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10) line of the code.  This collab will take a lot of minutes, around 30, but it will give the best results.

- code: 
	- jupyter notebook: `CNN_for_Music_Genre_Classification_Test.ipynb`
  - jupyter notebook: 
- models: pre-trained models

### Code 
In this repository we provide the scripts to fine-tune the pre-trained model and a quick music genre prediction algorithm using our own weights. 

Currently the genres supported are the [GTZAN dataset](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi0w9v6ndv7AhW2RzABHV9IBWUQFnoECBMQAQ&url=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fandradaolteanu%2Fgtzan-dataset-music-genre-classification&usg=AOvVaw3TTtu7nUZPCc-fNioZPjzz) tags:

- blues
- Clasic
- Country
- Disco
- Hip hop
- jazz
- Metal
- Pop
- reggae
- Rock

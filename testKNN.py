import librosa
import math
import os
import re
import random
import numpy as np

import matplotlib.pyplot as plt



def menu():
    print("""----Music genre clasifier KNN---- 
We have some example audio for you to try our genre guesser
1: classical_music.mp3
2: country_music.mp3
3: disco_music.mp3
4: hip-hop_music.mp3
5: jazz_music.mp3
6: metal_music.mp3
7: pop_music.mp3
8: reggae_music.mp3
    """)
    n = int(input("Your choice:"))
    return n
    
 
def extract_audio_features(list_of_audiofiles):
    data = np.zeros(
        (len(list_of_audiofiles), 128, 33), dtype=np.float64
    )

    for i, file in enumerate(list_of_audiofiles):
        y, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, hop_length=512, n_mfcc=13
        )
        spectral_center = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=512
        )
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=512
        )
        
        data[i, :, 0:13] = mfcc.T[0:128, :]
        data[i, :, 13:14] = spectral_center.T[0:128, :]
        data[i, :, 14:26] = chroma.T[0:128, :]
        data[i, :, 26:33] = spectral_contrast.T[0:128, :]

        print(
            "Extracted features audio track %i of %i."
            % (i + 1, len(list_of_audiofiles))
        )

    return data


def load_deserialize_data():

    train_X = np.load("DataFeatures/data_train.npy")
    train_Y = np.load("DataFeatures/data_train_target.npy")

    test_X = np.load("DataFeatures/data_test.npy")
    test_Y = np.load("DataFeatures/data_test_target.npy")
    
    return train_X, train_Y, test_X, test_Y


def get_neighbors(train_data, train_genre, test_data_ins, k):
    distances_mfcc=[]
    distances_sce=[]
    distances_chroma=[]
    distances_sco=[]
    for i in range(len(train_data)):
        train_mfcc=train_data[i][:][0:13]
        train_sce=train_data[i][:][13:14]
        train_chroma=train_data[i][:][14:26]
        train_sco=train_data[i][:][26:33]
        
        test_mfcc=test_data_ins[:][0:13]
        test_sce=test_data_ins[:][13:14]
        test_chroma=test_data_ins[:][14:26]
        test_sco=test_data_ins[:][26:33]
        
        
        dist_mfcc = np.linalg.norm(train_mfcc - test_mfcc)
        dist_sce = np.linalg.norm(train_sce - test_sce)
        dist_chroma = np.linalg.norm(train_chroma - test_chroma)
        dist_sco = np.linalg.norm(train_sco - test_sco)
        
        if len(distances_mfcc)<k:
            distances_mfcc.append([i, dist_mfcc])
        else:
            for j in range(len(distances_mfcc)):
                if dist_mfcc < distances_mfcc[j][1]:
                    distances_mfcc[j]=[i, dist_mfcc]
                    break
        
        if len(distances_sce)<k:
            distances_sce.append([i, dist_sce])
        else:
            for j in range(len(distances_sce)):
                if dist_sce < distances_sce[j][1]:
                    distances_sce[j]=[i, dist_sce]
                    break
        
        if len(distances_chroma)<k:
            distances_chroma.append([i, dist_chroma])
        else:
            for j in range(len(distances_chroma)):
                if dist_chroma < distances_chroma[j][1]:
                    distances_chroma[j]=[i, dist_chroma]
                    break
                
        if len(distances_sco)<k:
            distances_sco.append([i, dist_sco])
        else:
            for j in range(len(distances_sco)):
                if dist_sco < distances_sco[j][1]:
                    distances_sco[j]=[i, dist_sco]
                    break
    return distances_mfcc, distances_sce, distances_chroma, distances_sco
                
                
def list_to_genre(genre_binary):
    genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    count=0
    for i in genre_binary:
        if i == 1:
            break
        else:
            count+=1
    genre = genre_list[count]
    return genre
    
    
def nearest_class(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre):
    genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    guess = []
    for neighbors in [neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco]:
        genre = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for x in range(len(neighbors)):
            neighbor_genre_list = train_genre[neighbors[x][0]]
            count=0
            for i in neighbor_genre_list:
                if i == 1:
                    break
                else:
                    count+=1
            genre[count]=genre[count]+1
        
        for i in range(10):
            if genre[i]==max(genre):
                guess.append(genre_list[i])
    final_guess=[]
    times=0
    for i in genre_list:
        if guess.count(i)>=times:
            times = guess.count(i)
    for i in genre_list:
        if guess.count(i)>=times:
            final_guess.append(i)
            
    return final_guess 


def grafic_neighbors(values):
    N = 10
    ind = np.arange(N) 
    width = 0.2
    
    bar1 = plt.bar(ind, values[0], width, color = 'r')
    
    bar2 = plt.bar(ind+width, values[1], width, color='g')
    
    bar3 = plt.bar(ind+width*2, values[2], width, color = 'b')
    
    bar4 = plt.bar(ind+width*3, values[3], width, color = 'y')
    plt.xticks(fontsize=6)
    
    plt.xlabel("Genres")
    plt.ylabel('Score')
    plt.title("K neighbors distribution")
    
    plt.xticks(ind+width,["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
    plt.legend( (bar1, bar2, bar3, bar4), ('MFCC', 'Spectral center', 'Chroma', 'Spectral contrast') )
    plt.show()

 
def get_values(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre, k):
    values=[]
    for neighbors in [neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco]:
        genre = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for x in range(len(neighbors)):
            neighbor_genre_list = train_genre[neighbors[x][0]]
            count=0
            for i in neighbor_genre_list:
                if i == 1:
                    break
                else:
                    count+=1
            genre[count]=genre[count]+1
        for i in range(10):
            genre[i]=genre[i]/k
        values.append(genre)
        
    return values


def main():
    exa= ["classical_music.wav", "country_music.wav", "disco_music.mp3", "hip-hop_music.mp3", "jazz_music.mp3", "metal_music.mp3", "pop_music.mp3", "reggae_music.mp3"]
    n = menu()
    k=20
    file = ["ExampleAudios/"+exa[n-1]]
    data=extract_audio_features(file)
    train_data, train_genre, test_data, test_genre = load_deserialize_data()
    prediction=[]
    neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco=get_neighbors(train_data, train_genre, data[0], k)
    guess_genre=nearest_class(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre)
    print(guess_genre)
    # grafic_neighbors(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre)
    values=get_values(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre, k)
    grafic_neighbors(values)

if __name__ == "__main__":
	main()

    
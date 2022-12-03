import librosa
import numpy as np

import matplotlib.pyplot as plt
from GenreFeatureData import load_deserialize_data
from GenreFeatureData import get_neighbors
from GenreFeatureData import nearest_class
from GenreFeatureData import list_to_genre


def menu():
    """
    Function that shows the user the example audios to try the model
    
    Returns:
        number of the audio chosen
    """
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
9: 2pac-Hit'em up (HipHop)
10: Chopin - Nocturne op.9 No.2 (Classical)
11: Olivia Rodrigo - drivers license (pop)
12: ACDC - Highway to Hell
13: I will survive-Gloria Gaynor
14: Is this love bob marley
    """)
    n = int(input("Your choice:"))
    return n
    
 
def extract_audio_features(list_of_audiofiles):
    """
    Function that given a list of audio files extracts it features using librosa library
    
    Args:
        list_of_audiofiles: list of paths to audiofiles, list with no length restriction
        
    Returns:
        data: 3-dimensional numpy matrix of size number of audiofiles x 128 x 33, stores the audio features
        
        
    """
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
                
                
def get_values(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre, k):
    """
    Functions that given the list of neighbors for each feature prepares the data to be graphed according to the proportions
    
    Args:
        neighbors_mfcc: list of size k x 2 (int x int), contains the k nearest neighbors according to mfcc and its distance 
        neighbors_sce: list of size k x 2 (int x int), contains the k nearest neighbors according to spectral center and its distance 
        neighbors_chroma: list of size k x 2 (int x int), contains the k nearest neighbors according to chroma and its distance 
        neighbors_sco: list of size k x 2 (int x int), contains the k nearest neighbors according to spectral contrast and its distance 
        train_genre: 2-dimensional numpy matrix of size: number of train audiofiles x 10, that stores the genre of each audio
        k: integer, number of neighbors allowed
    Returns:
        values: list of 4 lists for each of the audio features (mfcc, sce, chroma, sco), each one contains 10 values
                between 0 and 1 representing the proportion of the k neigbors found for that genre using that feature
    """
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


def grafic_neighbors(values):
    """
    Function that graphs the proportion of neighbors for each feature using matplotlib
    
    Args:
        values: list of 4 lists for each of the audio features (mfcc, sce, chroma, sco), each one contains 10 values
                between 0 and 1 
                
        
    """
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

 
def main():
    """
    Function that executes the menu and uses the model to show the user the obtained results
          
        
    """
    k=5
    exa= ["classical_music.wav", "country_music.wav", "disco_music.mp3", "hip-hop_music.mp3", "jazz_music.mp3", "metal_music.mp3", "pop_music.wav", "reggae_music.wav", "2pac-Hit'em up.mp3", "Chopin - Nocturne op.9 No.2.mp3", "Olivia Rodrigo - drivers license.mp3", "ACDC - Highway to Hell.mp3", "I will survive-Gloria Gaynor.mp3", "Is this love bob marley.mp3"]
    n = menu()
    file = ["ExampleAudios/"+exa[n-1]]
    data=extract_audio_features(file)
    train_data, train_genre, test_data, test_genre = load_deserialize_data()
    prediction=[]
    neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco=get_neighbors(train_data, train_genre, data[0], k)
    guess_genre=nearest_class(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre)
    print(guess_genre)
    values=get_values(neighbors_mfcc, neighbors_sce, neighbors_chroma, neighbors_sco, train_genre, k)
    grafic_neighbors(values)

if __name__ == "__main__":
	main()

    
# Raag_Classification
Ravindra Bisram and Joya Debi 

## Dataset 
The dataset for this project was obtained through the use of the Dunya/PyCompMusic API. Dunya comprises the music corpora and related software tools that have been developed as part of the CompMusic project. These corpora have been created with the aim of studying particular music traditions and they include audio recordings plus complementary information that describes the recordings.

The actual data consists of rougly 60+ different ragas, each with a select number of mp3 files corresponding to musical pieces played in said raag. 

## Preprocessing.py
- generate_dataset(): Iterate through target directory and collect the relative path of each recording.

 Parameters:
        - dataset_path: the absolute path to the directory containing all of the data. Each subdirectory inside of this should
        contain all of the recordings for a specific raga, and the name of the subdirectory should be the common name of the raga
        in question. 

    Returns: 
        - df: dataframe containing the paths to each audio file, the name of the raga they correspond to, and the one-hot encoded version 
        of the raga names
        - enc: the OneHotEncoder using to encode the ragas (so we can invert the process later)
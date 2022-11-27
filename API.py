from compmusic import dunya

print("Package imported successfully")
dunya.set_token("63c6dd8c2e1e4d977cd56a8b2bf96b9b97b22c21")
print("Token set successfully\n")

raags = dunya.hindustani.get_raags()

if False:
    print(raags)

    for i in range(len(raags)):
        print(raags[i])

    print("Information has been retrieved!")

    print(dunya.hindustani.get_raag('d9c603fa-875f-4b84-b851-c6a345427898').keys())
    songs = dunya.hindustani.get_raag('d9c603fa-875f-4b84-b851-c6a345427898')['recordings']
    for song in songs:
        print(song)

    dunya.hindustani.download_mp3('41e85340-5071-4cdf-a988-6e0aabed3dd6', './recordings')

    print(dunya.hindustani.get_raag('d9c603fa-875f-4b84-b851-c6a345427898')['aliases'])

    print(dunya.hindustani.get_recording('41e85340-5071-4cdf-a988-6e0aabed3dd6'))



# Use generator to search for a specific raag by common name
def query_raga(common_name = 'Bageshree'):
    '''Uses a ragas common name to display all of it's recordings and their durations in minutes'''
    print("=" * 25, common_name, "=" * 25)
    raga = next(item for item in raags if item["common_name"] == common_name)
    raga_recordings =  dunya.hindustani.get_raag(raga['uuid'])['recordings']
    for i, recording in enumerate(raga_recordings):
        time = dunya.hindustani.get_recording(recording['mbid'])['length']
        print(f"{i}: {round(time/1000/60, 2)} - {recording['title']}")


def create_raag_data(common_name = 'Bageshree', destination = './raga-data/Bageshree'):
    '''Uses a ragas common name to download all the audio files for said raga to destination folder'''
    print("=" * 25, common_name, "=" * 25)
    raga = next(item for item in raags if item["common_name"] == common_name)
    raga_recordings =  dunya.hindustani.get_raag(raga['uuid'])['recordings']

    print(f"Starting download of {len(raga_recordings)} audio files for raga {common_name}")
    
    for i, recording in enumerate(raga_recordings):
        if i < 3:
            continue 
        
        time = dunya.hindustani.get_recording(recording['mbid'])['length']
        print(f"{i}/{len(raga_recordings)}: {round(time/1000/60, 2)} - Downloading - {recording['title']}")
        dunya.hindustani.download_mp3(recording['mbid'], destination)
        print("~ Download complete!")

    print("Files successfully downloaded")

if __name__ == "__main__":
    # query_raga()
    create_raag_data()
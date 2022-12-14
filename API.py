from compmusic import dunya
import os

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


def create_raag_data(common_name = 'Khamaj', destination = './raga-data/Khamaj', starting_num = -1):
    '''Uses a ragas common name to download all the audio files for said raga to destination folder'''
    print("=" * 25, common_name, "=" * 25)
    raga = next(item for item in raags if item["common_name"] == common_name)
    raga_recordings =  dunya.hindustani.get_raag(raga['uuid'])['recordings']

    destination = f'./raga-data/{common_name}'
    CHECK_FOLDER = os.path.isdir(destination)
    if not CHECK_FOLDER:
        os.makedirs(destination)
        print("created folder : ", destination)
    else:
        print("Skipping raga")
        return 0

    print(f"Starting download of {len(raga_recordings)} audio files for raga {common_name}")
    
    for i, recording in enumerate(raga_recordings):
        if i <= starting_num:
            continue 
        
        time = dunya.hindustani.get_recording(recording['mbid'])['length']
        print(f"{i}/{len(raga_recordings)-1}: {round(time/1000/60, 2)} - Downloading - {recording['title']}")
        dunya.hindustani.download_mp3(recording['mbid'], destination)
        print("~ Download complete!")

    print("Files successfully downloaded")

if __name__ == "__main__":
    # query_raga()
    ragas = [{'uuid': '7e9ac165-68bd-4e6d-b1c0-b8d2f18ce3c3', 'common_name': 'Mangal bhairav', 'name': 'Maṅgal bhairav'},
            {'uuid': '54c4214c-05b9-4acc-8f77-6d5786e43a2e', 'common_name': 'Marubihag', 'name': 'Mārūbihāg'},
            {'uuid': '3eb7ba30-4b94-432e-9618-875ee57e01ab', 'common_name': 'Marwa', 'name': 'Mārvā'},
            {'uuid': '0437cc3e-6e02-491f-aa9b-5b4e1fec8993', 'common_name': 'Marwashri marwa', 'name': 'Mārvāśrī mārvā'},
            {'uuid': '86599528-c95a-441d-82dc-eb3ec5657045', 'common_name': 'Meera malhar', 'name': 'Mīrā malhār'},
            {'uuid': 'cd0f0430-a499-4f38-8d50-adf3435cf1e3', 'common_name': 'Megh', 'name': 'Mēgh'},
            {'uuid': '7acd42d8-e5ad-4cff-9a1f-61157eafb10b', 'common_name': 'Megh malhar', 'name': 'Mēgh malahār'},
            {'uuid': 'd205eaa9-079e-4e0f-8ae4-8db8ae231c12', 'common_name': 'Mishra gaara', 'name': 'Miśra gārā'},
            {'uuid': '97af40c5-8537-4128-a18f-fe10d8aebdf0', 'common_name': 'Mishra kalingada', 'name': 'Miśra kaliṅgaḍā'},
            {'uuid': '1ec5f9ec-7320-4beb-9469-18bf69655645', 'common_name': 'Mishra maand', 'name': 'Miśra māṇḍ'},
            {'uuid': 'eedc8258-2270-4da4-b7a6-388c842ae77a', 'common_name': 'Mishra piloo', 'name': 'Miśra pīlū'}
]
    for raga in ragas:
        create_raag_data(common_name = raga['common_name'], starting_num = 0)

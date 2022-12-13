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
    ragas = [{'uuid': '007a5094-e226-41d0-b626-0f938f7e67e0', 'common_name': 'Shudh maaru', 'name': 'Śuddh mārū'},
            {'uuid': 'f95072ad-729d-49b7-8ec0-c5be64337395', 'common_name': 'Shudha basant', 'name': 'Śuddh basant'},
            {'uuid': 'cadee83a-6aaa-404c-8a68-1250dc62d320', 'common_name': 'Shyam kalyan', 'name': 'Śyām kalyāṇ'},
            {'uuid': '7fad8e4e-8ba2-402d-9d01-6586acc5b458', 'common_name': 'Sindhu bhairavi', 'name': 'Sindhu bhairavi'},
            {'uuid': '7e1db214-c0f2-4e68-b748-35ea464c1b39', 'common_name': 'Sindhura', 'name': 'Sindhūra'},
            {'uuid': '3b18e6fe-5ea0-4326-8d3d-c1fbcbe1cc9e', 'common_name': 'Sohini', 'name': 'Sōhinī'},
            {'uuid': '7f8cd8d5-6dba-40f9-8f26-0c9fff6da3e2', 'common_name': 'Sohini bahar', 'name': 'Sōhinī bahār'},
            {'uuid': '76497571-61e8-4f5a-9e26-b52715be9fd1', 'common_name': 'Sohoni bhatiyar', 'name': 'Sōhōnī bhaṭiyār'},
            {'uuid': '72420351-7aa7-4118-9fc1-ed60de170172', 'common_name': 'Sorath', 'name': 'Sōrath'},
            {'uuid': 'f132c2f3-90f9-4dd3-bcb2-b6f95fd234b8', 'common_name': 'Subhalakshmi', 'name': 'Śubhalakṣmi'},
            {'uuid': '3800077c-9fd5-46e8-ab7b-d86955a864f1', 'common_name': 'Sughrai', 'name': 'Sugharaī'},
            {'uuid': 'faa182a5-687f-4c12-8ef7-616c537e10b0', 'common_name': 'Suha', 'name': 'Suhā'},
            {'uuid': '17a7b583-71b9-4d75-a881-7dd5d2627b37', 'common_name': 'Suha sugharai', 'name': 'Sur sugharai'},
            {'uuid': '6bb4b24a-deb0-4a47-b99a-094ca9605d45', 'common_name': 'Surmalhar', 'name': 'Sur malhār'},
            {'uuid': 'd7ebeced-2c40-4188-9556-907392a9df9b', 'common_name': 'Swarashtram', 'name': 'Saurāṣṭram'},
            {'uuid': '214aa9c2-b69b-4431-aaa4-1ad935e54334', 'common_name': 'Tankeshri', 'name': 'Tankēśrī'},
            {'uuid': '45528063-4723-4522-8a8c-e5b1da747112', 'common_name': 'Tilak bihari', 'name': 'Tilak bihārī'}]
    for raga in ragas:
        create_raag_data(common_name = raga['common_name'])

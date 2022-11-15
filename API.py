print("hello Workd")

from compmusic import dunya

print("Package imported successfully")
dunya.set_token("63c6dd8c2e1e4d977cd56a8b2bf96b9b97b22c21")
print("Token set successfully")

raags = dunya.hindustani.get_raags()

#print(raags)

for i in range(len(raags)):
    print(raags[i])

print("Information has been retrieved!")

print(dunya.hindustani.get_raag('d9c603fa-875f-4b84-b851-c6a345427898').keys())
songs = dunya.hindustani.get_raag('d9c603fa-875f-4b84-b851-c6a345427898')['recordings']
for song in songs:
    print(song)

dunya.hindustani.download_mp3('41e85340-5071-4cdf-a988-6e0aabed3dd6', './recordings')
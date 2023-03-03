import pickle
import pandas as pd

#To load from pickle file
with open("intermediate_camera_2_files_output.pkl", 'rb') as fr:
    infos = pickle.load(fr)

print(infos)
df_infos = pd.DataFrame(infos)
print("blub")
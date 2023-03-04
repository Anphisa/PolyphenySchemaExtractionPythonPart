import pickle
import pandas as pd

#To load from pickle file
#with open("intermediate_camera_2_files_output.pkl", 'rb') as fr:
#    infos = pickle.load(fr)

with open("intermediate_results/camera_2/camera_buy.net,www.henrys.com_0_0.8_COMA_OPT_False.pkl", "rb") as fr:
    info = pickle.load(fr)

print(info)
df_infos = pd.DataFrame(infos)
print("blub")
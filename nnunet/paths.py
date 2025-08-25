import os

# Use the environment variable to set the base path for inference weights
Server_Base_Path = os.getenv('INFERENCE_WEIGHTS_DIR', '/app/3D_Slicer/CTPelvic1K/inference_weights')
print(Server_Base_Path)

# Define paths
my_output_identifier = "CTPelvic1K"
default_plans_identifier = "nnUNetPlans"
default_data_identifier = "nnUNet"

try:
    base = os.path.join(Server_Base_Path, 'nnUNet')
    raw_dataset_dir = os.path.join(base, "nnUNet_raw")
    splitted_4d_output_dir = os.path.join(base, "nnUNet_raw_splitted")
    cropped_output_dir = os.path.join(base, "nnUNet_raw_cropped")
    raw_data_real = os.path.join(base, "rawdata", "ipcai2021_ALL_Test")

    os.makedirs(splitted_4d_output_dir, exist_ok=True)
    os.makedirs(raw_dataset_dir, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)
    os.makedirs(raw_data_real, exist_ok=True)
except KeyError:
    print("nnUNet_raw_data_base is not defined. nnU-Net can only be used on data for which preprocessed files are already present on your system. Please read documentation/setting_up_paths.md for information on how to set this up properly.")
    cropped_output_dir = splitted_4d_output_dir = raw_dataset_dir = base = None

try:
    preprocessing_output_dir = os.path.join(Server_Base_Path, 'nnUNet', 'nnUNet_processed')
    os.makedirs(preprocessing_output_dir, exist_ok=True)
except KeyError:
    print("nnUNet_preprocessed is not defined. nnU-Net cannot be used for preprocessing or training. Please read documentation/setting_up_paths.md for information on how to set this up.")
    preprocessing_output_dir = None

try:
    network_training_output_dir = os.path.join(Server_Base_Path, 'nnUNet', 'nnUNet_results_folder', my_output_identifier)
    os.makedirs(network_training_output_dir, exist_ok=True)
except KeyError:
    network_training_output_dir = None
    print("RESULTS_FOLDER was not in your environment variables, network_training_output_dir could not be determined. Please go to nnunet/paths.py and manually set network_training_output_dir. You can ignore this warning if you are using nnunet only as a toolkit and don't intend to run network trainings.")




# """
# Put your personal paths in here. This file will shortly be added to gitignore so that your personal paths will not be tracked
# """
# import os
# #from os.path import join 

# home_dir = '/app'
# #home_dir = os.path.expanduser('~')
# Server_Base_Path = os.path.join(home_dir,'3D_Slicer','CTPelvic1K', 'inference_weights')

# print(Server_Base_Path)


# # You need to set the following folders: base, preprocessing_output_dir and network_training_output_dir. See below for details.

# # do not modify these unless you know what you are doing
# my_output_identifier = "CTPelvic1K"
# default_plans_identifier = "nnUNetPlans"
# default_data_identifier = "nnUNet"

# #server_base_path = os.path.join()
# #base = os.environ["nnUNet_raw_data_base"] #if "nnUNet_raw_data_base" in os.environ.keys() else None 
# #preprocessing_output_dir = os.environ["nnUNet_preprocessed"] #if "nnUNet_preprocessed" in os.environ.keys() else None
# #network_training_output_dir_base = os.path.join(os.environ["RESULTS_FOLDER"]) #if "RESULTS_FOLDER" in os.environ.keys() else None

# try:
#     # base is the folder where the raw data is stored. You just need to set base only, the others will be created
#     # automatically (they are subfolders of base).
#     # Here I use environment variables to set the base folder. Environment variables allow me to use the same code on
#     # different systems (and our compute cluster). You can replace this line with something like:
#     #base = "C:\\Users\\akh\\all_data_2\\nnUNet"
#     #base = os.environ['nnUNet_base']
#     base = f'{Server_Base_Path}\\nnUNet'
#     raw_dataset_dir = os.path.join(base, "nnUNet_raw")
#     splitted_4d_output_dir = os.path.join(base, "nnUNet_raw_splitted")
#     cropped_output_dir = os.path.join(base, "nnUNet_raw_cropped")
#     raw_data_real = os.path.join(base,"rawdata","ipcai2021_ALL_Test")

#     os.makedirs(splitted_4d_output_dir, exist_ok=True)
#     os.makedirs(raw_dataset_dir, exist_ok=True)
#     os.makedirs(cropped_output_dir, exist_ok=True)
#     os.makedirs(raw_data_real, exist_ok=True)
# except KeyError:
#     print("nnUNet_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
#           "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
#           "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
#     cropped_output_dir = splitted_4d_output_dir = raw_dataset_dir = base = None

# # preprocessing_output_dir is where the preprocessed data is stored. If you run a training I very strongly recommend
# # this is a SSD!
# try:
#     # Here I use environment variables to set the folder. Environment variables allow me to use the same code on
#     # different systems (and our compute cluster). You can replace this line with something like:
#     # preprocessing_output_dir = "/path/to/my/folder_with_preprocessed_data"

#     # preprocessing_output_dir = os.environ['nnUNet_preprocessed']
#     preprocessing_output_dir = f'{Server_Base_Path}/nnUNet/nnUNet_processed'
#     os.makedirs(preprocessing_output_dir, exist_ok=True)
# except KeyError:
#     print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
#           "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.")
#     preprocessing_output_dir = None

# # This is where the trained model parameters are stored
# try:
#     # Here I use environment variables to set the folder. Environment variables allow me to use the same code on
#     # different systems (and our compute cluster). You can replace this line with something like:
#     # network_training_output_dir = "/path/to/my/folder_with_results"

#     # network_training_output_dir = os.path.join(os.environ['RESULTS_FOLDER'], my_output_identifier)
#     network_training_output_dir = os.path.join(f'{Server_Base_Path}','nnUNet','nnUNet_results_folder', f'{my_output_identifier}')
#     #network_training_output_dir = os.path.join(network_training_output_dir_base, my_output_identifier)
#     os.makedirs(network_training_output_dir, exist_ok = True)
# except KeyError:
#     network_training_output_dir = None
#     print("RESULTS_FOLDER was not in your environment variables, network_training_output_dir could not be determined. "
#           "Please go to nnunet/paths.py and manually set network_training_output_dir. You can ignore this warning if "
#           "you are using nnunet only as a toolkit and don't intend to run network trainings")

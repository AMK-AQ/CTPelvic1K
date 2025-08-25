import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nnunet.paths import my_output_identifier

# Paths
home_dir = '/mnt/inference_weights'
inference_weights_dir = home_dir
train_dir = os.path.join(inference_weights_dir, 'rawdata', 'Task11_CTPelvic1K')
output_dir = os.path.join(inference_weights_dir, 'nnUNet_raw', 'Task11_CTPelvic1K')
processed_path = os.path.join(inference_weights_dir, 'nnUNet_processed', 'Task11_CTPelvic1K')
check_save_path = os.path.join(inference_weights_dir, 'nnUNet_processed', 'Task11_CTPelvic1K', 'Task11_check')

TASK = 'Task22_ipcai2021'
FOLD = 0
GPU = 0

test_data_path = os.path.join(inference_weights_dir, 'rawdata', 'ipcai2021_ALL_Test')

command = f'python inference/predict_simple.py ' \
          f'-i {test_data_path} ' \
          f'-o {test_data_path}/{TASK}__{my_output_identifier}__fold{FOLD}_2d_pred ' \
          f'-t {TASK} ' \
          f'-tr nnUNetTrainer ' \
          f'-m 2d ' \
          f'-f {FOLD} ' \
          f'--num_threads_preprocessing 8 ' \
          f'--num_threads_nifti_save 4 ' \
          f'--gpu {GPU}'

if __name__ == '__main__':
    print('\n'*2)
    print(command, '\n'*2)
    os.system(command)




# import sys
# import os

# # Add the parent directory of 'nnunet' to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from nnunet.paths import my_output_identifier

# # Paths
# home_dir = '/app'  # Use /app as the base directory in the container
# inference_weights_dir = os.path.join(home_dir, 'inference_weights','nnUNet')
# train_dir = os.path.join(inference_weights_dir, 'rawdata', 'Task11_CTPelvic1K')
# output_dir = os.path.join(inference_weights_dir, 'nnUNet_raw', 'Task11_CTPelvic1K')
# processed_path = os.path.join(inference_weights_dir, 'nnUNet_processed', 'Task11_CTPelvic1K')
# check_save_path = os.path.join(inference_weights_dir, 'nnUNet_processed', 'Task11_CTPelvic1K', 'Task11_check')

# TASK = 'Task22_ipcai2021'
# FOLD = 0
# GPU = 0

# test_data_path = os.path.join(inference_weights_dir, 'rawdata', 'ipcai2021_ALL_Test')

# command = f'python inference/predict_simple.py ' \
#           f'-i {test_data_path} ' \
#           f'-o {test_data_path}/{TASK}__{my_output_identifier}__fold{FOLD}_2d_pred ' \
#           f'-t {TASK} ' \
#           f'-tr nnUNetTrainer ' \
#           f'-m 2d ' \
#           f'-f {FOLD} ' \
#           f'--num_threads_preprocessing 8 ' \
#           f'--num_threads_nifti_save 4 ' \
#           f'--gpu {GPU}'

# if __name__ == '__main__':
#     print('\n'*2)
#     print(command, '\n'*2)
#     os.system(command)

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import argparse
from nnunet.preprocessing.preprocessing import resample_data_or_seg

import nnunet
from nnunet.run.default_configuration import get_default_configuration
from multiprocessing import Pool

from nnunet.training.model_restore import recursive_find_trainer
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer


def resample_and_save(predicted, target_shape, output_file):
    predicted_new_shape = resample_data_or_seg(predicted, target_shape, False, order=1, do_separate_z=False, cval=0)
    seg_new_shape = predicted_new_shape.argmax(0)
    np.savez_compressed(output_file, data=seg_new_shape.astype(np.uint8))


def predict_next_stage(trainer, stage_to_be_predicted_folder, fold):
    output_folder = join(pardir(trainer.output_folder), f"pred_next_stage_{fold}")
    os.makedirs(output_folder, exist_ok=True)

    process_manager = Pool(2)
    results = []

    print(len(trainer.dataset_val.keys()))
    for pat in trainer.dataset_val.keys():

        if os.path.exists(output_folder+'/'+pat+"_segFromPrevStage.npz"):
            print(pat,'has been predicted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!continue!!!!!!!!!')
            continue



        print(pat)
        data_file = trainer.dataset_val[pat]['data_file']
        try:
            data_preprocessed = np.load(data_file)['data'][:-1]
        except Exception as _:
            data_preprocessed = np.load(data_file.replace('.npz','.npy'))[:-1]

        predicted = trainer.predict_preprocessed_data_return_softmax(data_preprocessed, False, # do_mirroring
                                                                     1, False, 1,
                                                                     trainer.data_aug_params['mirror_axes'],
                                                                     True, True, 2, trainer.patch_size, True)
        data_file_nofolder = data_file.split("/")[-1]
        data_file_nextstage = join(stage_to_be_predicted_folder, data_file_nofolder)
        try:
            data_nextstage = np.load(data_file_nextstage)['data']
        except Exception as _:
            data_nextstage = np.load(data_file_nextstage.replace('.npz','.npy'))

        target_shp = data_nextstage.shape[1:]
        output_file = join(output_folder, data_file_nextstage.split("/")[-1][:-4] + "_segFromPrevStage.npz")
        results.append(process_manager.starmap_async(resample_and_save, [(predicted, target_shp, output_file)]))

    _ = [i.get() for i in results]

    ### train
    print(len(trainer.dataset_tr.keys()))
    for pat in trainer.dataset_tr.keys():

        if os.path.exists(output_folder+'/'+pat+"_segFromPrevStage.npz"):
            print(pat,'has been predicted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!continue!!!!!!!!!')
            continue

        print(pat)
        data_file = trainer.dataset_tr[pat]['data_file']
        try:
            data_preprocessed = np.load(data_file)['data'][:-1]
        except Exception as _:
            data_preprocessed = np.load(data_file.replace('.npz','.npy'))[:-1]

        predicted = trainer.predict_preprocessed_data_return_softmax(data_preprocessed, False, # do_mirroring
                                                                     1, False, 1,
                                                                     trainer.data_aug_params['mirror_axes'],
                                                                     True, True, 2, trainer.patch_size, True)
        data_file_nofolder = os.path.basename(data_file)
        data_file_nextstage = join(stage_to_be_predicted_folder, data_file_nofolder)
        try:
            data_nextstage = np.load(data_file_nextstage)['data']
        except Exception as _:
            data_nextstage = np.load(data_file_nextstage.replace('.npz','.npy'))

        target_shp = data_nextstage.shape[1:]
        output_file = join(output_folder, os.path.basename(data_file_nextstage)[:-4] + "_segFromPrevStage.npz")
        results.append(process_manager.starmap_async(resample_and_save, [(predicted, target_shp, output_file)]))

    _ = [i.get() for i in results]


if __name__ == "__main__":
    """
    RUNNING THIS SCRIPT MANUALLY IS USUALLY NOT NECESSARY. USE THE run_training.py FILE!
    
    This script is intended for predicting all the low resolution predictions of 3d_lowres for the next stage of the 
    cascade. It needs to run once for each fold so that the segmentation is only generated for the validation set 
    and not on the data the network was trained on. Run it with
    python predict_next_stage TRAINERCLASS TASK FOLD"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("network_trainer")
    parser.add_argument("task")
    parser.add_argument("fold", type=int)

    args = parser.parse_args()

    trainerclass = args.network_trainer
    task = args.task
    fold = args.fold

    plans_file, folder_with_preprocessed_data, output_folder_name, dataset_directory, batch_dice, stage = \
        get_default_configuration("3d_lowres", task)
    
    trainer_class = recursive_find_trainer([join(nnunet.__path__[0], "training", "network_training")], trainerclass,
                                           "nnunet.training.network_training")
    
    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnunetTrainer"
    
    trainer = trainer_class(plans_file, fold, folder_with_preprocessed_data,output_folder=output_folder_name,
                            dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage)

    trainer.initialize(False)
    trainer.load_dataset()
    trainer.do_split()
    trainer.load_best_checkpoint(train=False)

    stage_to_be_predicted_folder = join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1)
    output_folder = join(pardir(trainer.output_folder), "pred_next_stage")
    os.makedirs(output_folder, exist_ok=True)

    predict_next_stage(trainer, stage_to_be_predicted_folder)


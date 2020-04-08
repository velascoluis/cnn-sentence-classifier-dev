import argparse
import shutil
import logging
import os
#add GPU support
def move_data_pvc(keras_model_path, x_train_path, x_val_path, y_train_path, y_val_path,workdir):
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting move data to PVC step ..')
    logging.info('Input data ..')
    logging.info('keras_model_path:{}'.format(keras_model_path))
    logging.info('x_train_path:{}'.format(x_train_path))
    logging.info('x_val_path:{}'.format(x_val_path))
    logging.info('y_train_path:{}'.format(y_train_path))
    logging.info('y_val_path:{}'.format(y_val_path))
    logging.info('workdir:{}'.format(workdir))

    mount_dir = workdir
    shutil.copyfile(x_train_path, mount_dir + 'x_train.bin')
    shutil.copyfile(y_train_path, mount_dir + 'y_train.bin')
    shutil.copyfile(x_val_path, mount_dir + 'x_val.bin')
    shutil.copyfile(y_val_path, mount_dir + 'y_val.bin')
    shutil.copyfile(keras_model_path, mount_dir + 'model.bin')
    logging.info('STEP: MV DATA PVC (1/1) Data moved.')
    logging.info('Files copied. Content of {} is {}'.format(mount_dir,os.system('ls -lrta '+mount_dir)))
    logging.info('Move data to PVC step finished.')

def main(params):
    move_data_pvc(params.keras_model_path, params.x_train_path, params.x_val_path, params.y_train_path, params.y_val_path, params.workdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='04. Move data to PVC step')
    parser.add_argument('--keras_model_path', type=str, default='None')
    parser.add_argument('--x_train_path', type=str, default='None')
    parser.add_argument('--x_val_path', type=str, default='None')
    parser.add_argument('--y_train_path', type=str, default='None')
    parser.add_argument('--y_val_path', type=str, default='None')
    parser.add_argument('--workdir', type=str, default='None')
    params = parser.parse_args()
    main(params)









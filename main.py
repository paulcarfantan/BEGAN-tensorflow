import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config
from scipy.misc import imread

def main(config):
    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        setattr(config, 'batch_size', 64)
        if config.test_data_path is None:
            data_path = config.data_path
            #print('\n data_path ',data_path,'\n')
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    data_path_real = './img_real/'
    data_path_gen = './img_gen/'

    data_loader = get_loader(
            data_path, config.batch_size, config.input_scale_size,
            config.data_format, config.split)
    data_loader_real = get_loader(                   # change paths !
            data_path_real, config.batch_size, config.input_scale_size,
            config.data_format, config.split)
    data_loader_gen = get_loader(                    # change paths !
            data_path_gen, config.batch_size, config.input_scale_size,
            config.data_format, config.split)
    trainer = Trainer(config, data_loader, data_loader_real, data_loader_gen)
    
    if config.is_train:
        save_config(config)
        trainer.train()
        return None
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()
        img_real = imread(config.img_real)
        img_gen = imread(config.img_gen)
        d_loss = trainer.d_loss_out(img_real,img_gen)
        print("d_loss : ",d_loss)
        return d_loss

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)

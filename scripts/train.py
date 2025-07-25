import numpy as np
import argparse
from omnifold import Multifold
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow.keras.backend as K
import utils
from dataloader import Dataset

# tf.random.set_seed(1234)
# np.random.seed(1234)


if __name__ == "__main__":
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        default="/global/cfs/cdirs/m3246/H1/h5",
        help="Folder containing data and MC files",
    )
    parser.add_argument(
        "--config",
        default="config_general.json",
        help="Basic config file containing general options",
    )
    parser.add_argument(
        "--config_omnifold",
        default="config_omnifold.json",
        help="Basic config file containing general options",
    )

    parser.add_argument(
        "--dataset",
        default="ep",
        help="Choice between ep or em datasets",
    )
    parser.add_argument(
        "--closure",
        action="store_true",
        default=False,
        help="Train omnifold for a closure test using simulation",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        default=False,
        help="Pretrain the model on step 1 rapgap vs djangoh",
    )
    parser.add_argument(
        "--load_pretrain",
        action="store_true",
        default=False,
        help="Load pretrained model instead of starting from scratch",
    )
    parser.add_argument(
        "--nstrap", type=int, default=0, help="Unique id for bootstrapping"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Which omnifold iteration to start with"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Display additional information during training",
    )
    parser.add_argument("--jobNum", type=int, default=-1, help="SBATCH Job Number")

    flags = parser.parse_args()

    if flags.verbose:
        print(80 * "#")
        print(
            "Total hvd size {}, rank: {}, local size: {}, local rank{}".format(
                hvd.size(), hvd.rank(), hvd.local_size(), hvd.local_rank()
            )
        )
        print(80 * "#")

    opt = utils.LoadJson(flags.config)
    version = "{}_{}".format(opt["NAME"], flags.dataset)

    mc_names = opt["MC_NAMES"]
    mc_file_names = [utils.get_sample_name(name, flags.dataset) for name in mc_names]
    data_file_names = [utils.get_sample_name("data", flags.dataset)]

    if flags.closure or flags.pretrain:
        # use djangoh as pseudo data
        data_file_names = [utils.get_sample_name("Djangoh", flags.dataset)]
        version += "_closure"
        # Keep closure data with around the same amount as the true data
        nmax = 350000 if flags.closure else None
    else:
        nmax = None

    if flags.jobNum != -1:
        np.random.seed(1234 * (flags.jobNum + 1))
        if not (flags.nstrap > 0):  # ensure tf seed isn't same for multiple bootstraps
            tf.random.set_seed(1234 * (flags.jobNum + 1))
        version += f"_job{flags.jobNum}"

    data = Dataset(
        data_file_names,
        flags.data_folder,
        is_mc=False,
        rank=hvd.rank(),
        size=hvd.size(),
        nmax=nmax,
    )  # match the normalization from MC files

    mc = Dataset(
        mc_file_names,
        flags.data_folder,
        is_mc=True,
        rank=hvd.rank(),
        size=hvd.size(),
        nmax=None if flags.pretrain else 5_000_000,
        norm=data.nmax,
    )  # Pretrain using all the events, but use smaller dataset during normal training

    if flags.nstrap > 0:
        assert not flags.closure and not flags.pretrain, (
            "ERROR: bootstrapping cannot run with  closure or pretraining!!"
        )
        if flags.verbose and hvd.rank() == 0:
            print(80 * "#")
            print("Running booststrap with ID: {}".format(flags.nstrap))
            np.random.seed(flags.nstrap * (hvd.rank() + 1))
            print(80 * "#")
        data.weight = np.random.poisson(1, data.weight.shape[0])

    K.clear_session()
    mfold = Multifold(
        version=version,
        config_file=flags.config_omnifold,
        nstrap=flags.nstrap,
        start=flags.start,
        pretrain=flags.pretrain,
        load_pretrain=flags.load_pretrain,
        verbose=flags.verbose,
    )

    mfold.mc = mc
    mfold.data = data
    mfold.Preprocessing()
    mfold.Unfold()

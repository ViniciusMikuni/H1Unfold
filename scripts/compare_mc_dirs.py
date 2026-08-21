import argparse
import gc
from dataloader import Dataset
import utils
import horovod.tensorflow as hvd
import tensorflow as tf
from omnifold import Multifold


hvd.init()
utils.SetStyle()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the same MC sample (e.g. Djangoh_Eplus0607_prep.h5) "
            "produced in two different folders/pipelines (e.g. old vs new merge)."
        )
    )

    parser.add_argument(
        "--data_folder1",
        required=True,
        help="First folder containing the MC h5 file(s) (e.g. old pipeline output)",
    )
    parser.add_argument(
        "--data_folder2",
        required=True,
        help="Second folder containing the MC h5 file(s) (e.g. new pipeline output)",
    )
    parser.add_argument(
        "--tag1",
        default="X",
        help="Suffix identifying data_folder1, combined with the MC sample name for the "
        "legend label, e.g. 'old' -> 'Djangoh old'",
    )
    parser.add_argument(
        "--tag2",
        default="Y",
        help="Suffix identifying data_folder2, combined with the MC sample name for the "
        "legend label (also used as the histogram reference), e.g. 'new' -> 'Djangoh new'",
    )
    parser.add_argument(
        "--period", default="Eplus0607", help="Period tag used in the h5 file names"
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["Rapgap", "Djangoh"],
        help="Which MC samples to compare (default: Rapgap and Djangoh). "
        "Each sample gets its own set of plots.",
    )
    parser.add_argument(
        "--weights", default="../weights", help="Folder to store trained weights"
    )
    parser.add_argument(
        "--config",
        default="config_general.json",
        help="Basic config file containing general options",
    )
    parser.add_argument(
        "--plot_folder", default="../plots", help="Folder to store plots"
    )
    parser.add_argument(
        "--reco",
        action="store_true",
        default=False,
        help="Load reco-level events instead of gen/fiducial-level",
    )
    parser.add_argument(
        "--niter", type=int, default=0, help="Omnifold iteration to load"
    )
    parser.add_argument(
        "--nmax", type=int, default=1000000, help="Maximum number of events to load"
    )
    parser.add_argument(
        "--img_fmt", default="pdf", help="Format of the output figures"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Increase print level"
    )

    return parser.parse_args()


def get_sample_names(samples, period="Eplus0607"):
    """Map sample name -> h5 file name. Same file name is used in both folders."""
    return {mc: f"{mc}_{period}_prep.h5" for mc in samples}


def load_sample(flags, file_name, data_folder):
    """Load a single MC sample h5 file from a given folder."""
    if flags.reco:
        dl = Dataset(
            [file_name],
            data_folder,
            is_mc=True,
            rank=hvd.rank(),
            size=hvd.size(),
            nmax=flags.nmax,
            pass_reco=True,
        )
        del dl.gen  # free a bit of memory
        dl.evts = dl.reco
        del dl.reco
    else:
        dl = Dataset(
            [file_name],
            data_folder,
            is_mc=True,
            rank=hvd.rank(),
            size=hvd.size(),
            nmax=flags.nmax,
            pass_fiducial=True,
        )
        del dl.reco  # free a bit of memory
    gc.collect()
    return dl


def gather_data(dataloaders):
    for dl in dataloaders:
        # 1. Store the original mask shape/mask before flattening
        # so we can use it for weight alignment later.
        original_mask = dataloaders[dl].mask  # shape (N_events, N_part)

        # 2. Gather particles
        dataloaders[dl].part = hvd.allgather(tf.constant(
            dataloaders[dl].part.reshape(-1, dataloaders[dl].part.shape[-1])[original_mask.flatten()]
        )).numpy()

        # 3. Gather the mask itself so it matches the gathered particles
        dataloaders[dl].mask = hvd.allgather(tf.constant(original_mask.flatten())).numpy().astype(bool)

        # 4. Gather event-level info
        dataloaders[dl].event = hvd.allgather(tf.constant(dataloaders[dl].event)).numpy()
        dataloaders[dl].weight = hvd.allgather(tf.constant(dataloaders[dl].weight)).numpy()


def plot_particles(flags, dataloaders, mc_name, version, num_part, reference_name):
    """Plot particle-level observables comparing the two folders for a single MC sample."""
    for feature in range(dataloaders[reference_name].part.shape[-1]):
        feed_dict = {key: dataloaders[key].part[:, feature] for key in dataloaders}
        

        weights = {
            key: (
                dataloaders[key].weight
                .reshape(-1, 1, 1)
                .repeat(num_part, axis=1)
                .reshape(-1)[dataloaders[key].mask]
            )
            for key in dataloaders
        }

        fig, ax = utils.HistRoutine(
            feed_dict,
            xlabel=utils.particle_names.get(str(feature), f"Feature {feature}"),
            weights=weights,
            reference_name=reference_name,
            label_loc="upper left",
            uncertainty=None,
            binning=None,
        )

        if hvd.rank() == 0:
            fig.savefig(f"{flags.plot_folder}/{version}_{mc_name}_part_{feature}.{flags.img_fmt}")


def plot_event(flags, dataloaders, mc_name, version, reference_name):
    """Plot event-level observables comparing the two folders for a single MC sample."""
    for feature in range(dataloaders[reference_name].event.shape[-1]):
        feed_dict = {key: dataloaders[key].event[:, feature] for key in dataloaders}
        weights = {key: dataloaders[key].weight for key in dataloaders}

        fig, ax = utils.HistRoutine(
            feed_dict,
            xlabel=utils.event_names.get(str(feature), f"Feature {feature}"),
            weights=weights,
            reference_name=reference_name,
            label_loc="upper left",
            uncertainty=None,
            binning=None,
        )

        if hvd.rank() == 0:
            fig.savefig(f"{flags.plot_folder}/{version}_{mc_name}_event_{feature}.{flags.img_fmt}")


def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)
    mc_files = get_sample_names(flags.samples, flags.period)

    if flags.verbose and hvd.rank() == 0:
        print(
            f"Comparing samples {list(mc_files.keys())} between:\n"
            f"  tag1={flags.tag1}: {flags.data_folder1}\n"
            f"  tag2={flags.tag2}: {flags.data_folder2}"
        )


    for mc_name, file_name in mc_files.items():
        # MC-name-specific labels, e.g. "Djangoh old" / "Djangoh new" — these become
        # both the dict keys and the legend labels in the plots.
        label1 = f"{mc_name} {flags.tag1}"
        label2 = f"{mc_name} {flags.tag2}"

        if flags.verbose and hvd.rank() == 0:
            print(f"Loading {mc_name} ({file_name}) from both folders as '{label1}' / '{label2}'...")

        dl1 = load_sample(flags, file_name, flags.data_folder1)
        dl2 = load_sample(flags, file_name, flags.data_folder2)

        dataloaders = {label1: dl1, label2: dl2}

        utils.undo_standardizing(flags, dataloaders)

        # Padding capacity must be captured BEFORE gather_data flattens/filters part & mask
        num_part = dataloaders[label1].part.shape[1]

        gather_data(dataloaders)

        plot_particles(
            flags, dataloaders, mc_name, opt["NAME"],
            num_part=num_part, reference_name=label1,
        )
        plot_event(
            flags, dataloaders, mc_name, opt["NAME"],
            reference_name=label1,
        )

        del dataloaders, dl1, dl2
        gc.collect()


if __name__ == "__main__":
    main()
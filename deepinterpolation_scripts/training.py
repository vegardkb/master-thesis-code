import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime

STEPS_PER_EPOCH = 5

"""
    Training stuff saved here
"""
JOB_DIR_PARENT = "runs"

"""
    Location of training data
"""
WORKING_DIR = os.path.join("..", "..", "processed")
TRAIN_EXP_NAME = "20211112_18_30_27_GFAP_GCamp6s_F5_c2"
VAL_EXP_NAME = "20211117_17_33_00_GFAP_GCamp6s_F4_PTZ"
CROP_ID = "OpticTectum"
H5_FNAME = "aligned.h5"
H5_CHAN2_FNAME = "aligned_chan2.h5"
GENERATOR_NAME = "OphysGenerator"

""" CHANNELS = {
    "HuC-GCamp6s": H5_FNAME,
    "GFAP-jRGECO": H5_CHAN2_FNAME,
} """
CHANNELS = {
    "GFAP-GCamp6s": H5_FNAME,
}


def set_training_param(run_uid):
    training_param = {}
    training_param["type"] = "trainer"
    training_param["name"] = "core_trainer"
    training_param["run_uid"] = run_uid
    training_param["steps_per_epoch"] = STEPS_PER_EPOCH
    training_param["period_save"] = 25
    # network model is potentially saved
    # during training between a regular nb epochs
    training_param["nb_gpus"] = 0
    training_param["apply_learning_decay"] = 0
    training_param["nb_times_through_data"] = 1
    # if you want to cycle through the entire data.
    # Two many iterations will cause noise overfitting
    training_param["learning_rate"] = 0.0001
    training_param["loss"] = "mean_absolute_error"
    training_param["nb_workers"] = 1
    training_param["caching_validation"] = False

    return training_param


def set_generator_param(h5_fname):
    generator_param = {}

    # Those are parameters used for the main data generator
    generator_param["type"] = "generator"
    generator_param["steps_per_epoch"] = STEPS_PER_EPOCH
    generator_param["name"] = GENERATOR_NAME
    generator_param["pre_post_frame"] = 30
    generator_param["train_path"] = os.path.join(WORKING_DIR, TRAIN_EXP_NAME, CROP_ID, h5_fname)
    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = 6000
    generator_param["pre_post_omission"] = 0

    return generator_param


def set_generator_test_param(h5_fname):
    generator_test_param = {}
    # Those are parameters used for the Validation test generator. Here the
    # test is done on the beginning of the data but
    # this can be a separate file
    generator_test_param["type"] = "generator"  # type of collection
    generator_test_param["name"] = GENERATOR_NAME
    # Name of object in the collection
    generator_test_param[
        "pre_post_frame"
    ] = 30  # Number of frame provided before and after the predicted frame
    generator_test_param["train_path"] = os.path.join(WORKING_DIR, VAL_EXP_NAME, CROP_ID, h5_fname)
    generator_test_param["batch_size"] = 5
    generator_test_param["start_frame"] = 2000
    generator_test_param["end_frame"] = 2099
    generator_test_param[
        "pre_post_omission"
    ] = 1  # Number of frame omitted before and after the predicted frame
    generator_test_param["steps_per_epoch"] = -1
    # No step necessary for testing as epochs are not relevant.
    # -1 deactivate it.

    return generator_test_param


def set_network_param():
    network_param = {}
    # Those are parameters used for the network topology
    network_param["type"] = "network"
    network_param["name"] = "unet_single_1024"
    # Name of network topology in the collection

    return network_param


def set_aggregate_params(
    training_param, generator_param, generator_test_param, network_param, run_uid
):
    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
    training_param["model_string"] = (
        network_param["name"]
        + "_"
        + training_param["loss"]
        + "_"
        + training_param["run_uid"]
    )
    jobdir = os.path.join(
        JOB_DIR_PARENT,
        training_param["model_string"] + "_" + run_uid,
    )
    training_param["output_dir"] = jobdir

    return training_param, generator_param, generator_test_param, network_param, jobdir


def train(indicator, h5_fname):
    # This is used for record-keeping
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")

    # Initialize meta-parameters objects
    training_param, generator_param, generator_test_param, network_param = (
        set_training_param(run_uid),
        set_generator_param(h5_fname),
        set_generator_test_param(h5_fname),
        set_network_param(),
    )

    (
        training_param,
        generator_param,
        generator_test_param,
        network_param,
        jobdir,
    ) = set_aggregate_params(
        training_param, generator_param, generator_test_param, network_param, run_uid
    )

    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")


    with open(os.path.join(jobdir, "metainformation.txt"), mode="w") as f:
        f.write(f"Indicator: {indicator}\n")
        f.write(f"Training set: {TRAIN_EXP_NAME}\n")
        f.write(f"Validation set: {VAL_EXP_NAME}\n")

    # Here we create all json files that are fed to the training.
    # This is used for recording purposes as well as input to the
    # training process
    path_training = os.path.join(jobdir, "training.json")
    json_obj = JsonSaver(training_param)
    json_obj.save_json(path_training)

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_test_generator = os.path.join(jobdir, "test_generator.json")
    json_obj = JsonSaver(generator_test_param)
    json_obj.save_json(path_test_generator)

    path_network = os.path.join(jobdir, "network.json")
    json_obj = JsonSaver(network_param)
    json_obj.save_json(path_network)

    # We find the generator obj in the collection using the json file
    generator_obj = ClassLoader(path_generator)
    generator_test_obj = ClassLoader(path_test_generator)

    # We find the network obj in the collection using the json file
    network_obj = ClassLoader(path_network)

    # We find the training obj in the collection using the json file
    trainer_obj = ClassLoader(path_training)

    # We build the generators object. This will, among other things,
    # calculate normalizing parameters.
    train_generator = generator_obj.find_and_build()(path_generator)
    test_generator = generator_test_obj.find_and_build()(path_test_generator)

    # We build the network object. This will, among other things,
    # calculate normalizing parameters.
    network_callback = network_obj.find_and_build()(path_network)

    # We build the training object.
    training_class = trainer_obj.find_and_build()(
        train_generator, test_generator, network_callback, path_training
    )

    # Start training. This can take very long time.
    training_class.run()

    # Finalize and save output of the training.
    training_class.finalize()


def main():
    for indicator, h5_fname in CHANNELS.items():
        train(indicator, h5_fname)



if __name__ == "__main__":
    main()

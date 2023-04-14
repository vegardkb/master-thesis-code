import os
from deepinterpolation.generic import JsonSaver, ClassLoader

JOB_DIR_PARENT = "runs"

MODELS = {
    "GFAP_GCamp6s_5Hz": {
        "model_dir": "GFAP_GCamp6s_5Hz_2023_02_15",
        "model_name": "2023_02_15_17_26_unet_single_1024_mean_absolute_error_2023_02_15_17_26_model",
    },
    "HuC_GCamp6s_5Hz": {
        "model_dir": "HuC_GCamp6s_5Hz_2022_12_13",
        "model_name": "2022_12_13_20_23_unet_single_1024_mean_absolute_error_2022_12_13_20_23_model",
    },
    "GFAP_jRGECO_5Hz": {
        "model_dir": "GFAP_jRGECO_5Hz_2022_12_13",
        "model_name": "2022_12_13_23_21_unet_single_1024_mean_absolute_error_2022_12_13_23_21_model",
    },
}

MODEL = "GFAP_GCamp6s_5Hz"
MODEL_CHAN2 = "GFAP_jRGECO_5Hz"

WORKING_DIR = os.path.join("..", "..", "processed")
MODEL_PATH = os.path.join(JOB_DIR_PARENT, MODELS[MODEL]["model_dir"], MODELS[MODEL]["model_name"] + ".h5")
MODEL_CHAN2_PATH = os.path.join(JOB_DIR_PARENT, MODELS[MODEL_CHAN2]["model_dir"], MODELS[MODEL_CHAN2]["model_name"] + ".h5")

INPUT_FNAME = "aligned.h5"
INPUT_CHAN2_FNAME = "aligned_chan2.h5"
OUTPUT_DIR = "denoised"
OUTPUT_FNAME = "denoised.h5"
OUTPUT_CHAN2_DIR = "denoised_chan2"
OUTPUT_CHAN2_FNAME = "denoised_chan2.h5"
OUTPUT_DTYPE = 'uint16'

GENERATOR_NAME = "OphysGenerator"
EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    #"20211117_14_17_58_GFAP_GCamp6s_F2_C",
    "20211117_17_33_00_GFAP_GCamp6s_F4_PTZ",
    "20211117_21_31_08_GFAP_GCamp6s_F6_PTZ",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    "20211119_18_15_06_GFAP_GCamp6s_F5_C",
    "20211119_21_52_35_GFAP_GCamp6s_F7_C",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    #"20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    #"20220412_10_43_04_GFAP_GCamp6s_F1_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    #"20220412_16_06_54_GFAP_GCamp6s_F4_PTZ",
]
CROP_IDS = ["OpticTectum"]

WORKING_DIR = os.path.join("..", "..", "processed")

""" CHANNELS = {
    "HuC-GCamp6s": {
        "input_fname": INPUT_FNAME,
        "output_dir": OUTPUT_DIR,
        "output_fname": OUTPUT_FNAME,
        "model_path": MODEL_PATH,
    },
    "GFAP-jRGECO": {
        "input_fname": INPUT_CHAN2_FNAME,
        "output_dir": OUTPUT_CHAN2_DIR,
        "output_fname": OUTPUT_CHAN2_FNAME,
        "model_path": MODEL_CHAN2_PATH,
    }
} """

CHANNELS = {
    "GFAP_GCamp6s": {
        "input_fname": INPUT_FNAME,
        "output_dir": OUTPUT_DIR,
        "output_fname": OUTPUT_FNAME,
        "model_path": MODEL_PATH,
    }
}

def do_inference(exp_name, crop_id, ch_dict):

    exp_dir = os.path.join(WORKING_DIR, exp_name, crop_id)

    generator_param = {}
    inferrence_param = {}

    # We are reusing the data generator for training here.
    generator_param["type"] = "generator"
    generator_param["name"] = GENERATOR_NAME
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 0
    generator_param["steps_per_epoch"] = -1
    # No steps necessary for inference as epochs are not relevant.
    # -1 deactivate it.

    generator_param["train_path"] = os.path.join(exp_dir, ch_dict["input_fname"])

    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = -1  # -1 to go until the end.
    generator_param["randomize"] = 0
    # This is important to keep the order
    # and avoid the randomization used during training

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    inferrence_param["model_path"] = ch_dict["model_path"]

    # Replace this path to where you want to store your output file
    if not os.path.isdir(os.path.join(exp_dir, ch_dict["output_dir"])):
        os.makedirs(os.path.join(exp_dir, ch_dict["output_dir"]))
    inferrence_param["output_file"] = os.path.join(exp_dir, ch_dict["output_dir"], ch_dict["output_fname"])
    inferrence_param["output_datatype"] = OUTPUT_DTYPE

    jobdir = os.path.join(JOB_DIR_PARENT, exp_name, crop_id)
    try:
        os.makedirs(jobdir)
    except Exception:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inferrence_class.run()


def main():
    for exp_name in EXP_NAMES:
            for crop_id in CROP_IDS:
                exp_dir = os.path.join(WORKING_DIR, exp_name, crop_id)
                for _, ch_dict in CHANNELS.items():
                    output_path = os.path.join(exp_dir, ch_dict["output_dir"], ch_dict["output_fname"])
                    if os.path.isfile(output_path):
                        print(f"Skipping: data already denoised: {output_path}")
                        continue
                    
                    do_inference(exp_name, crop_id, ch_dict)

if __name__ == "__main__":
    main()
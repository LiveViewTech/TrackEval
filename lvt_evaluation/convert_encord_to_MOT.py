import math
import os
import glob
import csv
import tqdm
import json
import configparser
import click
import shutil
from pathlib import Path

class EncordMetadataParser:
    """
    class to handle conversion of Encord (json) outputs to TrackEval MOT format
    required for downstream processing and evaluation
    """

    def __init__(self, annotation_dir, mot_gt_dir, dataset_version, dataset_name):
        self.annotation_dir = annotation_dir
        self.mot_gt_dir = mot_gt_dir
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.seq_name = f"{dataset_name}-{dataset_version}-all"
        self.mot_gt_dataset_dir = os.path.join(mot_gt_dir, self.seq_name)
        if os.path.exists(self.mot_gt_dataset_dir):
            shutil.rmtree(self.mot_gt_dataset_dir)

        os.makedirs(self.mot_gt_dataset_dir)



    @staticmethod
    def write_dict_to_ini(dictionary, filename):
        config_object = configparser.ConfigParser()
        for section, options in dictionary.items():
            config_object.add_section(section)
            for key, value in options.items():
                config_object.set(section, key, str(value))

        with open(filename, "w") as file:
            config_object.write(file)


    def run(self):
        seqmaps = []
        annotation_files = list(glob.glob(os.path.join(self.annotation_dir, "*.json")))
        for annotation_file in tqdm.tqdm(annotation_files):
            with open(annotation_file) as json_file:
                dataset_metadata = json.load(json_file)

            for dataset_metadata in dataset_metadata:
                data_info = dataset_metadata["data_units"][dataset_metadata["data_hash"]]
                video_file_name = data_info["data_title"].split("/")[-1]
                video_name = video_file_name.split(".")[0]
                data_title = f"{self.dataset_name}-{video_name}"
                seqmaps.append({"name": data_title})
                data_title_dir = os.path.join(self.mot_gt_dataset_dir, data_title)
                if not os.path.exists(data_title_dir):
                    os.makedirs(data_title_dir)
                data_title_gt_dir = os.path.join(data_title_dir, "gt")
                if not os.path.exists(data_title_gt_dir):
                    os.makedirs(data_title_gt_dir)
                person_data = []

                seqinfo_dict = {
                    "Sequence": {
                        "name": data_title,
                        "imDir": "img1",
                        "frameRate": int(math.ceil(data_info["data_fps"])),
                        # "seqLength": len(data_info["labels"].keys()),
                        "seqLength": max(int(data_info["data_duration"] * data_info["data_fps"]), len(data_info["labels"].keys())),
                        "imWidth": data_info["width"],
                        "imHeight": data_info["height"],
                        "imExt": ".jpg",
                    }
                }
                self.write_dict_to_ini(seqinfo_dict, os.path.join(data_title_dir, "seqinfo.ini"))
                key_map = {}
                for frame_num_str, frame_objects in data_info["labels"].items():
                    frame_num = int(frame_num_str)+1
                    for frame_object in frame_objects["objects"]:
                        if frame_object["value"] == "person":
                            if frame_object["objectHash"] not in key_map.keys():
                                key_map[frame_object["objectHash"]] = len(key_map.keys())+1

                            left = int((frame_object["boundingBox"]["x"])
                                       * data_info["width"])
                            top = int((frame_object["boundingBox"]["y"]) *
                                       data_info["height"])
                            width = int(frame_object["boundingBox"]["w"] * data_info["width"])
                            height = int(frame_object["boundingBox"]["h"] * data_info["height"])

                            person_data.append({"frame_num": frame_num,
                                                "id": key_map[frame_object["objectHash"]],
                                                "left": left,
                                                "top": top,
                                                "width": width,
                                                "height": height,
                                                "x": 1,
                                                "y": 1,
                                                "z": 1})

                with open(os.path.join(data_title_gt_dir, "gt.txt"), "w", newline='') as csvfile:
                    fieldnames = ["frame_num", "id", "left", "top", "width", "height", "x", "y", "z"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerows(person_data)

        seqmaps_filepath = os.path.join(self.mot_gt_dir, "seqmaps", f"{self.seq_name}.txt")
        if os.path.exists(seqmaps_filepath):
            os.remove(seqmaps_filepath)

        with open(seqmaps_filepath, "w", newline='') as csvfile:
            fieldnames = ["name"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(seqmaps)

@click.command()
@click.option("--encord-dir", type=click.Path(path_type=Path), required=True)
@click.option("--mot-gt-dir", type=click.Path(path_type=Path), required=True)
@click.option("--dataset-name", type=str, default="MOTLVT")
@click.option("--dataset-version", type=str, required=True)
def main(encord_dir, mot_gt_dir, dataset_name, dataset_version):
    encord_data_parser = EncordMetadataParser(encord_dir, mot_gt_dir, dataset_version, dataset_name)
    encord_data_parser.run()

if __name__ == '__main__':
    main()
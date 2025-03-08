import click
import csv
import os
import glob
import shutil
from pathlib import Path

class BoundingBox(object):
    def __init__(
        self,
        object_id,
        left,
        top,
        width,
        height,
        confidence,
        obj_label,
        frame_num=None,
        timestamp=None,
    ):
        self.object_id = object_id
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.confidence = confidence
        self.obj_label = obj_label
        self.frame_num = frame_num
        self.timestamp = timestamp
        self.centroid = (
            int((self.left + self.width / 2)),
            int((self.top + self.height / 2)),
        )
        self.area = self.width * self.height

    def get_centroid(self):
        return self.centroid

    def get_XYXY(self):
        return (self.left, self.top, self.left + self.width, self.top + self.height)


    def to_mot_dict(self):
        return {
            "frame_num": self.frame_num,
            "id": self.object_id,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "probability": self.confidence,
            "x": -1,
            "y": -1,
            "z": -1,
        }

class DeepstreamMetadataParser:
    """
    class to handle conversion of Deepstream (csv) outputs to TrackEval MOT format
    required for downstream processing and evaluation
    """

    def __init__(self, ds_benchmark_dir, mot_track_dir, dataset_name, dataset_version,
                 tracker_name, tracker_version):
        self.ds_benchmark_dir = ds_benchmark_dir
        self.mot_track_dir = mot_track_dir
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.tracker_name = tracker_name
        self.tracker_version = tracker_version
        self.seq_name = f"{dataset_name}-{dataset_version}-all"
        self.mot_dataset_dir = os.path.join(mot_track_dir, self.seq_name)
        self.mot_tracker_dir = os.path.join(self.mot_dataset_dir , f"{tracker_name}-{tracker_version}")

        if not os.path.exists(self.mot_dataset_dir):
            os.makedirs(self.mot_dataset_dir)

        if os.path.exists(self.mot_tracker_dir):
            self.mot_tracker_dir += "-duplicate"
        self.mot_tracker_dir = os.path.join(self.mot_tracker_dir, "data")
        os.makedirs(self.mot_tracker_dir)

    @staticmethod
    def load_bbox_data(file_path, vid_width=1920, vid_height=1080):
        bboxes = []
        with open(file_path, "r", encoding="UTF8", newline="") as in_file:
            reader = csv.DictReader(in_file)
            # reader.__next__()
            for line in reader:
                left = max(int(line["bbox_left"]), 0)
                width = int(line["bbox_width"])

                top = max(int(line["bbox_top"]), 0)
                height = int(line["bbox_height"])

                if width == 0 or height == 0:
                    continue
                bboxes += [
                    BoundingBox(
                        object_id=line["track_id"],
                        frame_num=line["frame"],
                        left=left,
                        top=top,
                        width=width,
                        height=height,
                        obj_label=line["type"],
                        confidence=float(line["score"]),
                    )
                ]
        return bboxes

    def run(self):
        pred_files = list(glob.glob(os.path.join(self.ds_benchmark_dir, "*.csv")))
        for pred_file in pred_files:
            pred_file_name = (pred_file.split("/")[-1])
            pred_file_name = pred_file_name.replace(".csv", ".txt")
            pred_file_name = pred_file_name.replace("_pred", "")
            tracks_list = self.load_bbox_data(pred_file)
            tracks_list = list(map(BoundingBox.to_mot_dict, tracks_list))

            if len(tracks_list) > 0:
                csv_filepath = os.path.join(self.mot_tracker_dir, f"{self.dataset_name}-{pred_file_name}")
                with open(csv_filepath, "w", newline='') as csvfile:
                    # fieldnames = ["frame_num", "id", "left", "top", "width", "height", "probability", "x", "y", "z"]
                    fieldnames = tracks_list[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerows(tracks_list)



@click.command()
@click.option("--ds-benchmark-dir", type=click.Path(path_type=Path), required=True, default=[])
@click.option("--mot-track-dir", type=click.Path(path_type=Path), required=True)
@click.option("--dataset-name", type=str, default="MOTLVT")
@click.option("--dataset-version", type=str, required=True)
@click.option("--tracker-name", type=str, required=True)
@click.option("--tracker-version", type=str, required=True)
def main(ds_benchmark_dir, mot_track_dir, dataset_name, dataset_version, tracker_name, tracker_version):
    ds_data_parser = DeepstreamMetadataParser(ds_benchmark_dir, mot_track_dir, dataset_name, dataset_version,
                                                  tracker_name, tracker_version)
    ds_data_parser.run()


if __name__ == "__main__":
    main()

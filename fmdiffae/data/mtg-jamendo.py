import os
import csv
import webdataset as wds
from fmdiffae.data.data_utils import save_webdataset


def load_jamendo_tsv(tsv_path):
    records = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            track_id, artist_id, album_id, path, duration, *tags = row
            records.append(
                {
                    "track_id": track_id,
                    "artist_id": artist_id,
                    "album_id": album_id,
                    "path": path,
                    "duration": float(duration),
                    "tags": tags,
                }
            )

    return records


if __name__ == "__main__":
    raw_base_dir = "/data/hai-res/shared/datasets/mtg-jamendo/raw"
    save_base_dir = "/data/hai-res/shared/datasets/mtg-jamendo/processed"

    tsv_paths = {
        "train": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-train.tsv",
        "valid": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-validation.tsv",
    }

    for split in ["train", "valid"]:
        records = load_jamendo_tsv(tsv_paths[split])
        records_one_instrument = [x for x in records if len(x["tags"]) == 1]

        total_duration = sum([x["duration"] for x in records_one_instrument]) / (3600)
        print(f"Total Length for {split}: {total_duration:.3f} Hours")

        rel_audio_paths = [x["path"] for x in records_one_instrument]
        names = []

        audio_paths = [os.path.join(raw_base_dir, path) for path in rel_audio_paths]
        audio_names = [
            os.path.splitext(path.replace("/", "_"))[0] for path in rel_audio_paths
        ]

        save_webdataset(
            audio_paths=audio_paths,
            audio_names=audio_names,
            save_dir=os.path.join(save_base_dir, split),
            maxcount=8192,
            transform_kwargs={"load_model_on_init": False},
        )

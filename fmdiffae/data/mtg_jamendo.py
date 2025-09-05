import os
import csv
import glob
import argparse
import numpy as np
import webdataset as wds
from tqdm import tqdm
from fmdiffae.data.data_utils import save_webdataset
from fmdiffae.utils.fad import get_embeddings_vggish
from fmdiffae.data.data_utils import get_webdataset


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
    parser = argparse.ArgumentParser(description="Processing the MTG Jamendo Dataset")
    parser.add_argument(
        "save_name",
        help="Folder name for the current processed dataset",
    )
    parser.add_argument(
        "--only_inst_tagged",
        help="Use the instrument-tagging dataset, which includes only songs with instrument tags",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--filter_by_one_inst_tag",
        help="Filter by songs with one instrument tag",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exclude_voice",
        help="Exclude songs tagged with voice",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--chunk_length_samples",
        help="Length of each data chunk in samples",
        type=int,
        default=65536,
    )
    parser.add_argument(
        "--wds_maxcount",
        help="Num. Examples in each .tar file",
        type=int,
        default=8192,
    )
    args = parser.parse_args()

    # Below code orients us within our filesystem - edit this based on where your directories are.
    raw_base_dir = "/data/hai-res/shared/datasets/mtg-jamendo/raw"
    # directory to all processed versions of the dataset
    save_base_dir = "/data/hai-res/ycda/processed-datasets/mtg-jamendo"

    if args.only_inst_tagged:
        tsv_paths = {
            "train": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-train.tsv",
            "valid": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-validation.tsv",
            "test": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-test.tsv",
        }
    else:
        tsv_paths = {
            "train": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging-train.tsv",
            "valid": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging-validation.tsv",
            "test": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-test.tsv",
        }

    # Write Audio
    save_dir = os.path.join(save_base_dir, args.save_name)
    for split in ["train", "valid", "test"]:
        records = load_jamendo_tsv(tsv_paths[split])

        if args.filter_by_one_inst_tag:
            records = [x for x in records if len(x["tags"]) == 1]

        if args.exclude_voice:
            records = [x for x in records if "instrument---voice" not in x["tags"]]

        total_duration = sum([x["duration"] for x in records]) / (3600)
        print(f"Total length for {split}: {total_duration:.3f} Hr. Before Process")

        rel_audio_paths = [x["path"] for x in records]

        audio_paths = [os.path.join(raw_base_dir, path) for path in rel_audio_paths]
        # Replaces slash, removes extension.
        audio_names = [
            os.path.splitext(path.replace("/", "_"))[0] for path in rel_audio_paths
        ]

        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        save_webdataset(
            audio_paths=audio_paths,
            audio_names=audio_names,
            save_dir=split_dir,
            maxcount=args.wds_maxcount,
            transform_kwargs={"load_model_on_init": False},
            chunk_audio_kwargs={"chunk_length_samples": args.chunk_length_samples},
        )

    # Save VGGish Embeddings
    for split in ["valid", "test"]:
        dataset = get_webdataset(
            split=split, base_dir=save_dir, data_type="audio", shuffle_size=None
        )
        vggish_embeddings = get_embeddings_vggish(dataset, pbar=True)

        vggish_embeddings = vggish_embeddings.numpy().reshape(-1, 128)
        np.save(
            os.path.join(save_dir, f"{split}_vggish_embeddings.npy"),
            vggish_embeddings,
        )

        mean = np.mean(vggish_embeddings, axis=0)
        np.save(os.path.join(save_dir, f"{split}_vggish_mean.npy"), mean)

        cov = np.cov(vggish_embeddings, rowvar=False)
        np.save(os.path.join(save_dir, f"{split}_vggish_cov.npy"), cov)

        # Save approx one-chunk-per-song valid/test subset for easy metric computation
        shard_paths = sorted(glob.glob(os.path.join(save_dir, split, "data-*.tar")))
        dataset = (
            wds.WebDataset(shard_paths, resampled=False, shardshuffle=True)
            .shuffle(args.wds_maxcount)
            .decode()
        )

        subset_audio = []
        subset_spec = []

        while len(subset_audio) < 2048:
            seen_song_ids = set()
            for example in tqdm(dataset, desc="Selecting One Chunk Per Song"):
                song_id = example["__key__"].rsplit("_", 1)[0]
                if song_id not in seen_song_ids:
                    print(song_id)
                    subset_audio.append(example["audio.npy"])
                    subset_spec.append(example["spec.npy"])
                    seen_song_ids.add(song_id)

                    if len(subset_audio) >= 2048:
                        break

        subset_audio = np.stack(subset_audio)
        subset_spec = np.stack(subset_spec)

        print(subset_audio.shape)
        print(subset_spec.shape)

        np.save(os.path.join(save_dir, f"{split}_subset_audio.npy"), subset_audio)
        np.save(os.path.join(save_dir, f"{split}_subset_spec.npy"), subset_spec)

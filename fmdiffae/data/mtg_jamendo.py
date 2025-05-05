import os
import csv
import glob
import torch
import webdataset as wds
from tqdm import tqdm
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


def get_jamendo_wds(
    split="train",
    base_dir="/data/hai-res/shared/datasets/mtg-jamendo/processed",
    data_type="spec",
    shuffle_size=2048,
):
    shard_paths = sorted(glob.glob(os.path.join(base_dir, split, f"{data_type}-*.tar")))

    if split == "train":
        dataset = wds.WebDataset(
            shard_paths, resampled=True, shardshuffle=True
        ).shuffle(shuffle_size)
    else:
        dataset = wds.WebDataset(shard_paths, resampled=False)

    dataset = (
        dataset.decode()
        .to_tuple(f"{data_type}.npy")
        .map_tuple(torch.from_numpy)
        .map(lambda x: x[0])
    )

    return dataset


def load_valid_subset(
    path="/data/hai-res/shared/datasets/mtg-jamendo/processed/valid_subset_one_chunk_per_song.npy",
):
    return torch.from_numpy(np.load(path))


if __name__ == "__main__":
    raw_base_dir = "/data/hai-res/shared/datasets/mtg-jamendo/raw"
    save_base_dir = "/data/hai-res/shared/datasets/mtg-jamendo/processed"

    tsv_paths = {
        "train": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-train.tsv",
        "valid": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-validation.tsv",
    }

    # Write Audio
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

    # # Save VGGish Emebddings
    import numpy as np
    from fmdiffae.utils.fad import get_embeddings_vggish

    valid_dataset = get_jamendo_wds(
        base_dir=save_base_dir, split="valid", data_type="audio"
    )
    valid_vggish_embeddings = get_embeddings_vggish(valid_dataset, pbar=True)

    valid_vggish_embeddings = valid_vggish_embeddings.numpy().reshape(-1, 128)
    mean = np.mean(valid_vggish_embeddings, axis=0)
    cov = np.cov(valid_vggish_embeddings, rowvar=False)

    np.save(
        os.path.join(save_base_dir, "valid_vggish_embeddings.npy"),
        valid_vggish_embeddings,
    )
    np.save("valid_vggish_mean.npy", mean)
    np.save("valid_vggish_cov.npy", cov)

    # Save approx one-chunk-per-song valid subset for easy metric computation
    shard_paths = sorted(glob.glob(os.path.join(save_base_dir, "valid", "spec-*.tar")))
    valid_dataset = (
        wds.WebDataset(shard_paths, resampled=False, shardshuffle=True)
        .shuffle(8192)
        .decode()
    )

    valid_subset = []
    while len(valid_subset) < 2048:
        seen_song_ids = set()
        for example in tqdm(valid_dataset, desc="Selecting One Chunk Per Song"):
            song_id = example["__key__"].rsplit("_", 1)[0]
            if song_id not in seen_song_ids:
                print(song_id)
                valid_subset.append(example["spec.npy"])
                seen_song_ids.add(song_id)
                if len(valid_subset) >= 2048:
                    break

    valid_subset = np.stack(valid_subset)
    print(valid_subset.shape)
    np.save(os.path.join(save_base_dir, "valid_subset_spec.npy"), valid_subset)

import os
import csv
import glob
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


if __name__ == "__main__":
    filter_by_one_inst_tag = False

    raw_base_dir = "/data/hai-res/shared/datasets/mtg-jamendo/raw"

    if filter_by_one_inst_tag:
        save_base_dir = "/data/hai-res/ycda/processed-datasets/mtg-jamendo/one-inst-tag"
        tsv_paths = {
            "train": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-train.tsv",
            "valid": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-validation.tsv",
        }

    else:
        save_base_dir = "/data/hai-res/ycda/processed-datasets/mtg-jamendo/full"
        tsv_paths = {
            "train": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging-train.tsv",
            "valid": "/data/hai-res/shared/datasets/mtg-jamendo/mtg-jamendo-dataset/data/splits/split-0/autotagging-validation.tsv",
        }

    # Write Audio
    for split in ["train", "valid"]:
        records = load_jamendo_tsv(tsv_paths[split])

        if filter_by_one_inst_tag:
            records = [x for x in records if len(x["tags"]) == 1]

        total_duration = sum([x["duration"] for x in records]) / (3600)
        print(f"Total Length for {split}: {total_duration:.3f} Hours")

        rel_audio_paths = [x["path"] for x in records]

        audio_paths = [os.path.join(raw_base_dir, path) for path in rel_audio_paths]
        # Replaces slash, removes extension.
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
    from fmdiffae.data.data_utils import get_webdataset

    valid_dataset = get_webdataset(
        split="valid", base_dir=save_base_dir, data_type="audio"
    )
    valid_vggish_embeddings = get_embeddings_vggish(valid_dataset, pbar=True)

    valid_vggish_embeddings = valid_vggish_embeddings.numpy().reshape(-1, 128)
    np.save(
        os.path.join(save_base_dir, "valid_vggish_embeddings.npy"),
        valid_vggish_embeddings,
    )

    mean = np.mean(valid_vggish_embeddings, axis=0)
    np.save(os.path.join(save_base_dir, "valid_vggish_mean.npy"), mean)

    cov = np.cov(valid_vggish_embeddings, rowvar=False)
    np.save(os.path.join(save_base_dir, "valid_vggish_cov.npy"), cov)

    # Save approx one-chunk-per-song valid subset for easy metric computation
    shard_paths = sorted(glob.glob(os.path.join(save_base_dir, "valid", "data-*.tar")))
    valid_dataset = (
        wds.WebDataset(shard_paths, resampled=False, shardshuffle=True)
        .shuffle(8192)
        .decode()
    )

    valid_subset_audio = []
    valid_subset_spec = []

    while len(valid_subset_audio) < 2048:
        seen_song_ids = set()
        for example in tqdm(valid_dataset, desc="Selecting One Chunk Per Song"):
            song_id = example["__key__"].rsplit("_", 1)[0]
            if song_id not in seen_song_ids:
                print(song_id)
                valid_subset_audio.append(example["audio.npy"])
                valid_subset_spec.append(example["spec.npy"])

                seen_song_ids.add(song_id)
                if len(valid_subset_audio) >= 2048:
                    break

    valid_subset_audio = np.stack(valid_subset_audio)
    valid_subset_spec = np.stack(valid_subset_spec)

    print(valid_subset_audio.shape)
    print(valid_subset_spec.shape)

    np.save(os.path.join(save_base_dir, "valid_subset_audio.npy"), valid_subset_audio)
    np.save(os.path.join(save_base_dir, "valid_subset_spec.npy"), valid_subset_spec)

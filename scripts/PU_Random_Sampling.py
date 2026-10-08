"""Build a positive-unlabeled copy of a YOLO detection dataset by dropping boxes.

A PU dataset is one where some real objects are boxed and many are not -- the
normal state of an annotation project. This makes one on purpose, from a fully
labeled dataset, so that PU training can be measured against a known answer.

Three drop modes, because "drop 20%" has more than one honest reading:

    box             20% of every box in the split, sampled from one global pool.
                    The default. Per-image counts vary the way they do in a real
                    project: a 3-box image may lose 0 or 2.
    box-per-image   20% of each image's boxes, rounded. Every annotated image is
                    thinned by the same fraction.
    image           every box in 20% of the annotated images. A different kind of
                    missingness -- whole frames nobody touched, rather than
                    partial review -- and worth testing separately.

Only the splits named by --split are touched. Leave val and test alone unless
you mean to: a PU model's recovered boxes score as false positives against
thinned ground truth, which inverts the result you are trying to measure.
"""

import os
import yaml
import shutil
import random
import argparse
from pathlib import Path

SPLIT_KEYS = ('train', 'val', 'valid', 'test')


def _resolve_root(input_yaml, config, root_override=None):
    """The dataset root the yaml's `path` refers to.

    Ultralytics allows `path` to be relative to its own datasets_dir rather than
    to the yaml, which is how the packaged configs (african-wildlife.yaml) work,
    so a relative path that does not exist beside the yaml is tried there too.
    """
    if root_override:
        return Path(root_override).absolute()

    raw = str(config.get('path', '') or '')
    if raw and os.path.isabs(raw):
        return Path(raw)

    candidates = []
    if raw:
        candidates.append(Path(input_yaml).parent / raw)
        try:
            from ultralytics.utils import SETTINGS
            candidates.append(Path(SETTINGS.get('datasets_dir', '')) / raw)
        except Exception:
            pass
    candidates.append(Path(input_yaml).parent)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.absolute()
    return candidates[0].absolute()


def _label_dir(image_dir):
    """The label directory for an image directory, by Ultralytics' convention.

    Ultralytics' img2label_paths swaps the last /images/ component for /labels/,
    so images/train pairs with labels/train. A dataset laid out as train/images
    instead pairs with a sibling train/labels, which is the fallback.
    """
    parts = list(Path(image_dir).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == 'images':
            parts[i] = 'labels'
            return Path(*parts)
    return Path(image_dir).parent / 'labels'


def _split_image_dir(root, value):
    """A split's image directory, whether the yaml gave it absolute or relative.

    A split may also be a .txt list or a list of directories; those are not
    handled, and the caller reports them rather than guessing.
    """
    if isinstance(value, (list, tuple)):
        return None
    path = Path(str(value))
    path = path if path.is_absolute() else root / str(value)
    return path if path.is_dir() else None


def _read_boxes(path):
    """Non-empty label lines, which is one box each."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line for line in f.read().splitlines() if line.strip()]


def _write_boxes(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("".join(line + "\n" for line in lines))


def _choose_drops(files, boxes, mode, ratio, rng):
    """Which box indices to drop per file: {file: set(indices)}.

    `boxes` maps file -> its list of lines, so every mode samples from the same
    material and the three are directly comparable.
    """
    drops = {f: set() for f in files}

    if mode == 'image':
        annotated = [f for f in files if boxes[f]]
        count = int(round(ratio * len(annotated)))
        for f in rng.sample(annotated, count):
            drops[f] = set(range(len(boxes[f])))
        return drops

    if mode == 'box-per-image':
        for f in files:
            n = len(boxes[f])
            if not n:
                continue
            count = int(round(ratio * n))
            drops[f] = set(rng.sample(range(n), count))
        return drops

    # mode == 'box': one global pool, so the dropped fraction is exact over the
    # whole split rather than only in expectation per image.
    pool = [(f, i) for f in files for i in range(len(boxes[f]))]
    count = int(round(ratio * len(pool)))
    for f, i in rng.sample(pool, count):
        drops[f].add(i)
    return drops


def create_noisy_yolo_dataset(input_yaml, output_folder, splits_to_drop, drop_ratio,
                              mode='box', seed=0, root_override=None):
    try:
        with open(input_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return None

    original_base_path = _resolve_root(input_yaml, config, root_override)
    if not original_base_path.is_dir():
        print(f"[!] Dataset root not found: {original_base_path}")
        return None

    output_base_path = Path(output_folder).absolute()
    rng = random.Random(seed)

    # 1. Copy the entire dataset, so the source stays the clean reference
    print(f"[*] Source dataset: {original_base_path}")
    print(f"[*] Copying dataset to: {output_base_path}")
    if output_base_path.exists():
        shutil.rmtree(output_base_path)
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original_base_path, output_base_path,
                    ignore=shutil.ignore_patterns('*.py', '.git', '__pycache__', '*.cache'))

    stats = {'mode': mode, 'ratio': drop_ratio, 'seed': seed, 'splits': {}}

    # 2. Iterate through each requested split
    for split_key in splits_to_drop:
        actual_key = split_key
        if split_key not in config:
            # val/valid are the same split under two spellings
            alias = {'val': 'valid', 'valid': 'val'}.get(split_key)
            if alias and alias in config:
                actual_key = alias
            else:
                print(f"[!] Warning: Split '{split_key}' not found in YAML. Skipping.")
                continue

        image_dir = _split_image_dir(output_base_path, config[actual_key])
        if image_dir is None:
            print(f"[!] Warning: Split '{split_key}' is not a single directory "
                  f"({config[actual_key]!r}). Skipping.")
            continue

        target_label_dir = _label_dir(image_dir)
        if not target_label_dir.is_dir():
            print(f"[!] Warning: Label directory not found at {target_label_dir}. "
                  f"Skipping '{split_key}'.")
            continue

        # 3. Drop the boxes for this split
        print(f"[*] Processing '{split_key}' labels in: {target_label_dir}")

        files = sorted(f for f in os.listdir(target_label_dir)
                       if f.endswith('.txt') and f != 'classes.txt')
        boxes = {f: _read_boxes(target_label_dir / f) for f in files}
        total_before = sum(len(v) for v in boxes.values())

        drops = _choose_drops(files, boxes, mode, drop_ratio, rng)

        dropped = 0
        files_modified = 0
        emptied = 0
        for f in files:
            drop_idx = drops[f]
            if not drop_idx:
                continue
            kept = [line for i, line in enumerate(boxes[f]) if i not in drop_idx]
            _write_boxes(target_label_dir / f, kept)
            dropped += len(drop_idx)
            files_modified += 1
            if not kept:
                emptied += 1

        annotated = sum(1 for v in boxes.values() if v)
        stats['splits'][split_key] = {
            'label_files': len(files),
            'annotated_files': annotated,
            'boxes_before': total_before,
            'boxes_dropped': dropped,
            'boxes_after': total_before - dropped,
            'files_modified': files_modified,
            'files_emptied': emptied,
        }
        pct = (100.0 * dropped / total_before) if total_before else 0.0
        print(f"    -> Dropped {dropped}/{total_before} boxes ({pct:.1f}%) "
              f"across {files_modified} files; {emptied} files left empty.")

    # 4. Update YAML paths. `path` becomes the new root and the split entries
    #    keep whatever form the source used, so the layout is preserved.
    config['path'] = str(output_base_path).replace("\\", "/")
    for key in SPLIT_KEYS:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, str) and os.path.isabs(value):
            resolved = _split_image_dir(output_base_path, value)
            if resolved is None:
                try:
                    rel = Path(value).relative_to(original_base_path)
                except ValueError:
                    rel = Path(Path(value).name)
                resolved = output_base_path / rel
            config[key] = str(resolved).replace("\\", "/")

    new_yaml_path = output_base_path / f"data_dropped_{int(round(drop_ratio * 100))}.yaml"
    with open(new_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    totals = stats['splits'].values()
    print("\n--- Final Summary ---")
    print(f"Mode / ratio / seed:  {mode} / {drop_ratio} / {seed}")
    print(f"Total Files Modified: {sum(s['files_modified'] for s in totals)}")
    print(f"Total Boxes Dropped:  {sum(s['boxes_dropped'] for s in totals)}")
    print(f"New Config Created:   {new_yaml_path}")

    stats['yaml'] = str(new_yaml_path).replace("\\", "/")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop YOLO boxes from multiple splits.")
    parser.add_argument("--yaml", type=str, required=True, help="Input data.yaml")
    parser.add_argument("--out", type=str, required=True, help="New dataset directory")
    # nargs='+' allows multiple values: --split train val test
    parser.add_argument("--split", type=str, nargs='+', default=["train"],
                        help="Space-separated splits (e.g., train val)")
    parser.add_argument("--ratio", type=float, default=0.5, help="0 to 1 ratio of boxes to drop")
    parser.add_argument("--mode", type=str, default="box",
                        choices=["box", "box-per-image", "image"],
                        help="box: that fraction of all boxes in the split (default). "
                             "box-per-image: that fraction of each image's boxes. "
                             "image: every box in that fraction of annotated images.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, so a drop is reproducible")
    parser.add_argument("--root", type=str, default=None,
                        help="Dataset root, if the yaml's 'path' cannot be resolved")

    args = parser.parse_args()
    create_noisy_yolo_dataset(args.yaml, args.out, args.split, args.ratio,
                              mode=args.mode, seed=args.seed, root_override=args.root)

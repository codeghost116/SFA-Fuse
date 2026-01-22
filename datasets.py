import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebDFDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class FramesDataset(Dataset):
    def __init__(
        self,
        root_split: str,
        transform: Callable,
        label_fn: Callable[[Path], Optional[int]],
        max_frames_per_video: Optional[int] = None,
        video_depth: Optional[int] = None,
        video_label_map: Optional[Dict[str, int]] = None,
        seed: int = 1337,
        show_index_progress: bool = True,   
    ):
        self.root = Path(root_split)
        self.transform = transform

        all_imgs = [p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if len(all_imgs) == 0:
            raise RuntimeError(f"No images found under: {root_split}")

        labeled: List[Tuple[Path, int, str]] = []
        skipped_unknown = 0

        iterator = all_imgs
        if show_index_progress:
            iterator = tqdm(all_imgs, desc=f"Indexing {self.root.name}", leave=False)

        for p in iterator:
            vid = get_video_name(p, self.root, video_depth)

            if video_label_map is not None:
                if vid not in video_label_map:
                    skipped_unknown += 1
                    continue
                y = int(video_label_map[vid])
            else:
                y = label_fn(p)
                if y is None:
                    skipped_unknown += 1
                    continue

            labeled.append((p, y, vid))

        if len(labeled) == 0:
            raise RuntimeError(
                f"Could not infer ANY labels under: {root_split}\n\n"
            )

        if max_frames_per_video is not None:
            from collections import defaultdict
            import random
            rng = random.Random(seed)

            buckets = defaultdict(list)
            for p, y, vid in labeled:
                buckets[(vid, y)].append((p, y, vid))

            capped = []
            for _, items in buckets.items():
                if len(items) > max_frames_per_video:
                    capped.extend(rng.sample(items, max_frames_per_video))
                else:
                    capped.extend(items)
            labeled = capped

        self.samples = [(p, y) for (p, y, _) in labeled]
        self.skipped_unknown = skipped_unknown

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)

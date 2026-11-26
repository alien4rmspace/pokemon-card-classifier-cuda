import asyncio
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import aiohttp
from PIL import Image
from tcgdexsdk import TCGdex, Query

DOWNLOAD_CARDS = True
SPLIT_CARDS = True

SPLIT_SEED = 42

RAW_ROOT = Path("data/pokemon_cards/raw")
SPLIT_ROOT = Path("data/pokemon_cards/split")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Max differing bits (of 256) for two images to count as the same artwork.
# Reprints of one illustration across sets land as high as ~38 because of
# different borders and holo patterns, so err high: over-grouping only makes
# the test set stricter, while under-grouping leaks reprints into it.
NEAR_DUPLICATE_MAX_BITS = 40

MAX_CONCURRENT_CARDS = 30
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
PROGRESS_EVERY = 100


import re

def normalize_pokemon_label(card_name: str) -> str:
    name = card_name.strip().lower()

    # Normalize separators first
    name = name.replace("é", "e")
    name = re.sub(r"[.\']", "", name)
    name = re.sub(r"[-\s]+", "_", name)

    # Remove common card-form suffixes at the end
    suffix_pattern = (
        r"(_("
        r"ex|gx|v|vmax|vstar|lv_x|break|delta|δ"
        r"))+$"
    )
    name = re.sub(suffix_pattern, "", name)

    # Clean any leftover repeated underscores
    name = re.sub(r"_+", "_", name).strip("_")

    return name

def _content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _perceptual_hash(path: Path) -> int:
    """256-bit difference hash: each bit says whether a pixel is brighter than its right neighbour."""
    with Image.open(path) as img:
        pixels = img.convert("L").resize((17, 16), Image.BILINEAR).tobytes()

    bits = 0
    for row in range(16):
        for col in range(16):
            i = row * 17 + col
            bits = (bits << 1) | (pixels[i + 1] > pixels[i])
    return bits


def group_duplicates(image_files: list[Path]) -> list[list[Path]]:
    """Drop exact copies, then group reprints of the same artwork.

    Each group goes to a single split, so an image (or a reprint of it) can
    never be in both train and test.
    """
    unique: dict[str, Path] = {}
    for path in image_files:
        unique.setdefault(_content_hash(path), path)

    files = list(unique.values())
    hashes = [_perceptual_hash(path) for path in files]

    parent = list(range(len(files)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if (hashes[i] ^ hashes[j]).bit_count() <= NEAR_DUPLICATE_MAX_BITS:
                parent[find(i)] = find(j)

    groups: dict[int, list[Path]] = defaultdict(list)
    for i, path in enumerate(files):
        groups[find(i)].append(path)
    return list(groups.values())


def split_dataset(source_root: Path, split_root: Path, seed: int = SPLIT_SEED) -> None:
    rng = random.Random(seed)

    # Start from empty split folders. Copying into folders left by an earlier
    # run is what let the same image end up in both train and test.
    for split_name in ("train", "val", "test"):
        split_dir = split_root / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

    classes = sorted(folder for folder in source_root.iterdir() if folder.is_dir())

    for class_dir in classes:
        class_name = class_dir.name
        # Sort before shuffling so the split depends only on the seed, not on
        # the order the filesystem lists files in.
        image_files = sorted(f for f in class_dir.iterdir() if f.is_file())
        groups = group_duplicates(image_files)
        rng.shuffle(groups)

        total = sum(len(group) for group in groups)
        train_target = int(total * TRAIN_RATIO)
        val_target = train_target + int(total * VAL_RATIO)

        splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        assigned = 0
        for group in groups:
            if assigned < train_target:
                split_name = "train"
            elif assigned < val_target:
                split_name = "val"
            else:
                split_name = "test"
            splits[split_name].extend(group)
            assigned += len(group)

        for split_name, split_files in splits.items():
            split_dir = split_root / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for file_path in split_files:
                shutil.copy2(file_path, split_dir / file_path.name)

        print(
            f"{class_name}: total={total} (dropped {len(image_files) - total} exact copies), "
            f"groups={len(groups)}, train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )


async def fetch_and_download_card(
    sdk: TCGdex,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    card_summary,
) -> tuple[str, str]:
    async with sem:
        try:
            card_id = getattr(card_summary, "id", None)
            card_name = getattr(card_summary, "name", None)
            card_image = getattr(card_summary, "image", None)

            if not card_id:
                return ("skipped", "unknown_id missing id")

            if not card_name or not card_image:
                full_card = await sdk.card.get(card_id)

                if not full_card:
                    return ("skipped", f"{card_id} missing full card data")

                card_name = getattr(full_card, "name", None)
                card_image = getattr(full_card, "image", None)

                if not card_name or not card_image:
                    return ("skipped", f"{card_id} missing metadata")

            pokemon_label = normalize_pokemon_label(card_name)
            folder = RAW_ROOT / pokemon_label
            folder.mkdir(parents=True, exist_ok=True)

            filename = folder / f"{card_id}.png"
            if filename.exists():
                return ("exists", str(filename))

            image_url = f"{card_image}/high.png"

            async with session.get(image_url) as response:
                response.raise_for_status()
                content = await response.read()

            filename.write_bytes(content)
            return ("downloaded", str(filename))

        except Exception as e:
            return ("failed", f"{getattr(card_summary, 'id', 'unknown')}: {e}")


async def download_cards() -> None:
    sdk = TCGdex("en")
    sem = asyncio.Semaphore(MAX_CONCURRENT_CARDS)

    pokemon_cards = await sdk.card.list(Query().equal("category", "Pokemon"))
    print(f"Found {len(pokemon_cards)} Pokémon cards")

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_CARDS)
    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT, connector=connector) as session:
        tasks = [
            fetch_and_download_card(sdk, session, sem, card_summary)
            for card_summary in pokemon_cards
        ]

        downloaded = 0
        skipped = 0
        failed = 0
        exists = 0
        processed = 0

        for coro in asyncio.as_completed(tasks):
            status, message = await coro
            processed += 1

            if status == "downloaded":
                downloaded += 1
            elif status == "exists":
                exists += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"FAILED: {message}")

            if processed % PROGRESS_EVERY == 0:
                print(
                    f"Processed {processed}/{len(tasks)} | "
                    f"downloaded={downloaded}, exists={exists}, skipped={skipped}, failed={failed}"
                )

        print(f"Downloaded: {downloaded}")
        print(f"Already existed: {exists}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")


def main() -> None:
    print("Downloading cards...")
    if DOWNLOAD_CARDS:
        asyncio.run(download_cards())

    print("Splitting dataset...")
    if SPLIT_CARDS:
        split_dataset(RAW_ROOT, SPLIT_ROOT)


if __name__ == "__main__":
    main()
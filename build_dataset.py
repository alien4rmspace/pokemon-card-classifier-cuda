import asyncio
import random
import re
import shutil
from pathlib import Path

import aiohttp
from tcgdexsdk import TCGdex, Query

DOWNLOAD_CARDS = True
SPLIT_CARDS = True

random.seed(42)

RAW_ROOT = Path("data/pokemon_cards/raw")
SPLIT_ROOT = Path("data/pokemon_cards/split")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

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

def split_dataset(source_root: Path, split_root: Path) -> None:
    classes = [folder for folder in source_root.iterdir() if folder.is_dir()]

    for class_dir in classes:
        class_name = class_dir.name
        image_files = [f for f in class_dir.iterdir() if f.is_file()]
        random.shuffle(image_files)

        total = len(image_files)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]

        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            split_dir = split_root / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for file_path in split_files:
                shutil.copy2(file_path, split_dir / file_path.name)

        print(
            f"{class_name}: total={total}, "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
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
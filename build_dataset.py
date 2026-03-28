import os
import random
import shutil

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

import aiohttp
import asyncio
import requests
from tcgdexsdk import TCGdex, Query
from pathlib import Path

random.seed(42)

SOURCE_ROOT = Path("data/pokemon_cards/pokemons")
SPLIT_ROOT = Path("data/pokemon_cards/split")
DOWNLOAD_CARDS = False
SPLIT_CARDS = False
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

POKEMON_NAMES = ["Pikachu", "Charizard", "Mewtwo", "Eevee", "Meowth", "Raichu", "Lucario", "Snorlax"]


def split_dataset(source_root: Path, split_root: Path) -> None:
    classes: list[Path] = [folder for folder in source_root.iterdir() if folder.is_dir()]

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
                shutil.copy(file_path, split_dir / file_path.name)

            print(f"{class_name}: total = {total}, train = {len(train_files)}, val = {len(val_files)}, test = {len(test_files)}")

async def download_card_image(sem: asyncio.Semaphore, session: aiohttp.ClientSession, card, index: int, pokemonName: str) -> None:
    if not card.image:
        print(f"{pokemonName}{index} has no image")
        return

    image_url = f"{card.image}/high.png"
    filename = f"data/pokemon_cards/pokemons/{pokemonName.lower()}/{pokemonName}{index}.png"

    async with sem:
        async with session.get(image_url) as response:
            response.raise_for_status()
            content = await response.read()

    with open(filename, "wb") as f:
        f.write(content)

    print(f"Downloaded: {filename}")

async def download_cards() -> None:
    sem = asyncio.Semaphore(10)
    sdk = TCGdex("en")

    async with aiohttp.ClientSession() as session:
        for pokemonName in POKEMON_NAMES:
            os.makedirs(f"data/pokemon_cards/pokemons/{pokemonName.lower()}", exist_ok=True)

            pokemon_cards = await sdk.card.list(
                Query().equal("name",
                              f"{pokemonName}|"
                              f"{pokemonName} ex|"
                              f"{pokemonName} EX|"
                              f"{pokemonName} Ex|"
                              f"{pokemonName} VSTAR|"
                              f"{pokemonName} VStar|"
                              f"{pokemonName} VMAX|"
                              f"{pokemonName} GX|"
                              f"{pokemonName} gx|"
                              f"{pokemonName} V|"
                              )
            )
            tasks = [
                download_card_image(sem, session, card, i, pokemonName)
                for i, card in enumerate(pokemon_cards)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

def main() -> None:
    print("Downloading Cards...")
    if DOWNLOAD_CARDS:
        asyncio.run(download_cards())

    print("Splitting Dataset...")
    if SPLIT_CARDS:
        split_dataset(SOURCE_ROOT, SPLIT_ROOT)

if __name__ == "__main__":
    main()


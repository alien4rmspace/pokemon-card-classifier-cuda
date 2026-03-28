import os

import aiohttp
import torch
import numpy as np
import asyncio
import requests
from tcgdexsdk import TCGdex, Query

sem = asyncio.Semaphore(10)

async def download_card_image(session, card, index, pokemonName):
    if not card.image:
        print(f"{pokemonName}{index} has no image")
        return

    image_url = f"{card.image}/high.png"
    filename = f"data/pokemon_cards/pokemons/{pokemonName.lower()}/{pokemonName}{[index]}.png"

    async with sem:
        async with session.get(image_url) as response:
            response.raise_for_status()
            content = await response.read()

    with open(filename, "wb") as f:
        f.write(content)

    print(f"Downloaded: {filename}")

async def main():
    ## Fetch pikachu cards
    # sdk = TCGdex("en")
    # pikachu_cards = await sdk.card.list(Query().equal("name", "Pikachu"))
    #
    # print(f"Found {len(pikachu_cards)} Pikachu cards")
    # for i, card in enumerate(pikachu_cards[:100]):
    #     if not card.image:
    #         print(f"pikachu{i} has no image")
    #         continue
    #
    #     image_url = f"{card.image}/high.png"
    #     response = requests.get(image_url)
    #     response.raise_for_status()
    #
    #     with open(f"data/pokemon_cards/pikachus/pikachu{[i]}.jpg", "wb") as f:
    #         f.write(response.content)
    #         print(f"Downloaded: pikachu{[i]}.jpg")
    #         print("Saved as pikachu.jpg")
    pokemonNames = ["Pikachu", "Charizard", "Mewtwo", "Eevee" ,"Meowth", "Raichu", "Lucario", "Snorlax"]
    for pokemonName in pokemonNames:
        os.makedirs(f"data/pokemon_cards/pokemons/{pokemonName.lower()}", exist_ok=True)

        sdk = TCGdex("en")
        pokemon_cards = await sdk.card.list(
            Query().equal("name",
                          f"{pokemonName}|"
                          f"{pokemonName} ex|"
                          f"{pokemonName} EX|"
                          f"{pokemonName} Ex|"
                          f"{pokemonName} VSTAR|"
                          f"{pokemonName} VStar|"
                          f"{pokemonName} VMAX|"
                          f"{pokemonName} VSTAR|"
                          f"{pokemonName} GX|"
                          f"{pokemonName} gx|"
                          f"{pokemonName} V|"
                          )
        )
        async with aiohttp.ClientSession() as session:
            tasks = [
                download_card_image(session, card, i, pokemonName)
                for i, card in enumerate(pokemon_cards)
            ]
            await asyncio.gather(*tasks)

asyncio.run(main())

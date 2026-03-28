import torch
import numpy as np
import asyncio
import requests
from tcgdexsdk import TCGdex

async def main():
    tcgdex = TCGdex("en")
    card = await tcgdex.card.get("swsh3-136")
    image_url = f"{card.image}/high.png"

    response = requests.get(image_url)
    response.raise_for_status()

    with open("furret.jpg", "wb") as f:
        f.write(response.content)

    print("Downloaded: ", image_url)
    print("Saved as furret.jph")

    print(card)
    print(image_url)

asyncio.run(main())

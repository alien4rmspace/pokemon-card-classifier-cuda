# GPU-Accelerated Pokémon Card Classifier

A deep learning image classification project built with PyTorch accelerated with CUDA to identify Pokémon from Pokémon card images. The project includes asynchronous dataset collection with TCGdex and aiohttp, dataset preprocessing and splitting, ResNet-18 training, and profiling with NVIDIA Nsight Systems.

## Features

- Asynchronous Pokémon card image collection using TCGdex and aiohttp
- Automatic dataset organization and train/validation/test splitting
- ResNet-18 fine-tuning in PyTorch
- CUDA-accelerated training and inference
- NVIDIA Nsight Systems profiling with NVTX markers
- Evaluation on held-out test data

## Motivation

This project was built to explore GPU-accelerated deep learning workflows end-to-end, from dataset ingestion and preprocessing to model training, inference, and performance profiling. It also serves as a practical computer vision project focused on real image data and reproducible experimentation.

## Tech Stack

- Python
- PyTorch
- Torchvision
- CUDA
- NVIDIA Nsight Systems
- asyncio
- aiohttp
- TCGdex API / SDK

## Project Structure

```text
data/
  pokemon_cards/
    raw/
    split/
      train/
      val/
      test/

prepare_dataset.py
train.py
test.py
README.md

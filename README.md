# GPU-Accelerated Pokémon Card Classifier

A deep learning image classification project built with PyTorch and accelerated with CUDA to identify Pokémon from Pokémon card images. The project includes asynchronous dataset collection with TCGdex and aiohttp, dataset preprocessing and splitting, ResNet-18 training, and profiling with NVIDIA Nsight Systems.

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
```

## Data Cleaning / Label Normalization

Raw card names from TCGdex often include card-specific mechanics, prefixes, and variants that are not useful for Pokémon-level classification. To keep labels consistent, the dataset pipeline normalizes card names to the base Pokémon identity.

Examples:
- `Tapu Bulu-GX` → `tapu_bulu`
- `Tangrowth LV.X` → `tangrowth`
- `Talonflame BREAK` → `talonflame`
- `ns_darmanitan` → removed or normalized because the prefix is not informative for Pokémon identity

This cleaning step reduces noisy labels, merges variant-specific card names into a shared Pokémon class, and improves dataset consistency for supervised learning.

## Profiling & Performance Analysis

<img width="1021" height="327" alt="Nsight Systems profiling output for the base model before fine-tuning" src="https://github.com/user-attachments/assets/6e2a7220-4507-4b11-9829-45ba8bfb62d7" />

Figure 1. Nsight Systems timeline of the base model before fine-tuning. The profile shows repeated training batch activity on the GPU, with small gaps between batch executions that likely correspond to CPU-side work such as data loading, preprocessing, synchronization, or launch overhead.

---
<img width="1021" height="324" alt="Nsight Systems profiling output after fine tuning base model" src="https://github.com/user-attachments/assets/b50b1054-8966-40e1-922d-b8c1f6567d46" />

Figure 2. Nsight Systems timeline after increasing the DataLoader worker count to 4 for training, validation, and testing. The profile shows that training time decreased substantially, with train:epoch_1 dropping from about 28.2 s to 12.4 s, indicating improved input pipeline throughput. However, the validation and test phases became more prominent, suggesting that the overhead of additional workers is less beneficial for smaller evaluation workloads.

---
<img width="1025" height="320" alt="image" src="https://github.com/user-attachments/assets/2ef28940-c3d0-44e3-a1f9-75684574dc7e" />

Figure 3. Nsight Systems timeline after reducing the DataLoader worker count to 0 for validation and testing while leaving training unchanged. The profile shows that validation and test execution times returned closer to their original levels, indicating that additional workers introduced unnecessary overhead for these smaller evaluation workloads.

---
<img width="1032" height="332" alt="image" src="https://github.com/user-attachments/assets/851ff584-828c-4b73-bd87-1e1a8d953122" />
Figure 4. Nsight Systems timeline after increasing the training DataLoader worker count from 4 to 8. Compared with 4 workers, the additional workers yield no meaningful reduction in training time and appear to introduce extra overhead, suggesting that performance declines beyond a certain threshold depending on the size of data.

---
<img width="1463" height="442" alt="image" src="https://github.com/user-attachments/assets/b4afe159-deed-46b4-a16a-372069fb7d53" />
Figure 5. Nsight Systems timeline for a two-epoch training run with the training DataLoader worker count set to 4 and the validation and test DataLoader worker counts set to 2. Compared with profiling a single epoch, this run better reflects realistic training behavior by showing that the worker startup overhead is concentrated at the beginning of the first epoch, while subsequent epochs benefit from the established data loading pipeline and incur much less initialization cost.
Note: The blue blocked state regions suggest that the validation and test phases still spend time waiting, indicating there may be room to increase worker counts further for evaluation.


## Performance Summary

The profiling results from **Figures 1–4** are summarized below.

Increasing `num_workers` allows PyTorch DataLoaders to prefetch data on the CPU while the GPU is executing training work, which helps hide input pipeline latency and improve throughput.

| Configuration | Observed Effect |
|---------------|-----------------|
| Baseline (`num_workers=0`) | Higher training time and visible gaps between batch executions |
| `num_workers=4` for all DataLoaders | Training time reduced from ~28.2 s to ~12.4 s |
| `num_workers=4` only for training | Best overall balance between training throughput and evaluation overhead |
| `num_workers=8` for training | Training slowed to ~18.9 s, showing diminishing returns from extra workers |
| Two-epoch run (`num_workers=4` for training, `2` for validation/test) | Worker startup overhead is concentrated at the beginning of the first epoch, while later epochs benefit from improved steady-state throughput |

Overall, these results show that moderate DataLoader parallelism improves training throughput by reducing input pipeline stalls and keeping the GPU fed with data more consistently. However, excessive worker counts or applying multiprocessing to smaller evaluation workloads can introduce overhead that outweighs the performance benefit.

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

This project was built to learn GPU-accelerated deep learning in a practical, end-to-end setting, covering dataset collection, preprocessing, model training, inference, and evaluation. It was also designed to build experience with profiling tools and performance analysis, with a focus on identifying bottlenecks and understanding how CPU-side input pipelines and GPU execution impact overall training efficiency.

## Tech Stack

- Python
- PyTorch
- Torchvision
- CUDA
- NVIDIA Nsight Systems
- asyncio
- aiohttp
- TCGdex API / SDK

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

---
<img width="1340" height="437" alt="Screenshot 2026-04-15 173103" src="https://github.com/user-attachments/assets/5ab77633-e184-4a32-beb4-0dd23d02f9fc" />
<img width="1187" height="433" alt="Screenshot 2026-04-15 173552" src="https://github.com/user-attachments/assets/ea91a2d9-0557-490d-9057-8d57a425d430" />
Figs. 6 and 7. Nsight Systems timelines comparing baseline and optimized 5-epoch training runs. Fig. 6 shows the baseline configuration with a total runtime of 199.871 s, while Fig. 7 shows the optimized configuration with a total runtime of 66.714 s. The optimized run achieved an approximately 66.6% runtime reduction (~3.0× speedup) through data pipeline tuning, including adjustment of num_workers, use of persistent_workers, pinned memory with non_blocking=True, and prefetch_factor=4.

---

## Performance Summary

## Performance Summary

The profiling results from **Figures 1–7** are summarized below.

Early profiling focused on how `DataLoader` worker counts affected GPU utilization and end-to-end training time, while later profiling evaluated multi-epoch behavior and a fully tuned data pipeline. Overall, the results show that moderate CPU-side parallelism improves throughput by reducing input pipeline stalls and keeping the GPU more consistently supplied with data, but overly aggressive worker counts or applying multiprocessing to smaller evaluation workloads can introduce overhead that offsets those gains.

| Configuration | Observed Effect |
|---------------|-----------------|
| **Figure 1 — Baseline (`num_workers=0`)** | Highest training time, with visible idle gaps between batch executions that suggest CPU-side data loading and preprocessing overhead. |
| **Figure 2 — `num_workers=4` for all DataLoaders** | Training time dropped substantially, with `train:epoch_1` decreasing from about **28.2 s** to **12.4 s**, showing improved input pipeline throughput; however, validation and test overhead became more noticeable. |
| **Figure 3 — `num_workers=4` for training only** | Restoring validation and test worker counts to 0 reduced unnecessary evaluation overhead and produced a better overall balance between training throughput and end-to-end runtime. |
| **Figure 4 — `num_workers=8` for training** | Increasing workers beyond 4 slowed training to about **18.9 s**, indicating diminishing returns and added multiprocessing overhead. |
| **Figure 5 — Two-epoch run (`num_workers=4` for training, `2` for validation/test)** | Showed that worker startup overhead is concentrated near the beginning of the first epoch, while later epochs benefit from a more stable steady-state pipeline. |
| **Figure 6 — Baseline 5-epoch run** | The untuned 5-epoch configuration required **199.871 s**, providing a longer-horizon baseline for end-to-end comparison. |
| **Figure 7 — Optimized 5-epoch run** | The fully tuned configuration completed in **66.714 s**, a **66.6% runtime reduction** and roughly **3.0× speedup**, achieved through tuning `num_workers`, enabling `persistent_workers`, using pinned memory with `non_blocking=True`, and setting `prefetch_factor=4`. |

Overall, **Figures 1–7** show a clear progression from identifying CPU-side input pipeline stalls to validating a more efficient steady-state training configuration. Moderate `DataLoader` parallelism improved overlap between CPU data preparation and GPU execution, while the final tuned pipeline demonstrated that reducing worker startup overhead and improving transfer behavior can significantly lower end-to-end training time across longer runs.

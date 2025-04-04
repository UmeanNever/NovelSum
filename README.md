<p align="center">
<!--  <img src="diversity.png" style="height: 80px;"> -->
 <h2 align="center">Measuring Data Diversity for Instruction Tuning: <br> A Systematic Analysis and A Reliable Metric </h2>
</p>

<p align="center">
 <a href="https://github.com/UmeanNever/NovelSum/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/UmeanNever/NovelSum"></a>
 <a href="https://arxiv.org/abs/2502.17184"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-red"></a>
</p>

## 📋 Overview

In this research, we tackle the fundamental challenge of accurately measuring dataset diversity for instruction tuning and introduce **NovelSum**, a reliable diversity metric that demonstrates strong correlation with model performance. Furthermore, we leverage NovelSum as an optimization objective to develop a greedy diversity-oriented data selection strategy called **NovelSelect** that outperforms existing approaches, validating both the effectiveness and practical significance of our metric.

- 📖 **Paper**: [Read our paper on arXiv](https://arxiv.org/abs/2502.17184)
- 🛠️ **Code**: All implementation code and resources are available in this repository

## 📑 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
  - [Data Format](#data-format)
  - [Embedding Calculation](#embedding-calculation)
  - [NovelSum Metric](#novelsum-metric)
  - [NovelSelect Strategy](#novelselect-strategy)
- [Extensions](#-extensions)
- [Citation](#-citation)

## 🔧 Installation

```bash
git clone https://github.com/UmeanNever/NovelSum.git
cd NovelSum
pip install -r requirements.txt
```

## 🚀 Quick Start

### Calculate NovelSum for a dataset

To calculate the NovelSum of a dataset, provide a JSON file containing data embeddings and a reference directory used for estimating 'density':

```bash
python novelsum.py --single_dataset_path input_data.json --dense_ref_dir your_dir --output_csv output.csv
```

### Use NovelSelect for data selection

To apply the NovelSelect strategy, specify text directory, embedding directory, and output directory:

```bash
python novelselect.py --text_dir your_first_dir --figure_dir your_embedding_dir --output_dir output
```

## 📚 Usage Guide

### Data Format

Due to the computational demands of processing large-scale embeddings, we recommend splitting your dataset into multiple JSON files, each containing approximately 5,000 data points (resulting in files of ~0.5GB).

Our implementation uses separate directories for storing text data and embeddings, with each JSON file named numerically (e.g., `18.json`). Both text and embedding data are organized as lists within these files.

#### Examples

<details>
  <summary><b>SFT Data Format</b></summary>

```json
{
    "from": "user",
    "value": "Create a detailed and exhaustive HTML guide to assist potential buyers in making a well-informed decision when purchasing a laptop."
}
```
</details>

<details>
  <summary><b>Text Data Format (for embedding)</b></summary>

```
Create a detailed and exhaustive HTML guide to assist potential buyers in making a well-informed decision when purchasing a laptop.
```
</details>

<details>
  <summary><b>Corresponding Embedding Format</b></summary>

```json
[
    -0.06069426238536835,
    -0.07991443574428558,
    ...
    0.04622051864862442
]
```
</details>

### Embedding Calculation

You can generate embeddings for your dataset using various models. Thanks to vLLM, we were able to compute embeddings for 400,000 data points in just 2 hours using 8×H800 GPUs.

<details>
  <summary><b>Embedding Calculation Options</b></summary>

```
Usage: python embedding.py [OPTIONS]

  Calculate embeddings for text data in a specified directory.

Options:
  --input_dir TEXT       Directory containing text data as JSON files (list of strings)
  --model_path TEXT      Path to the embedding model
  --gpu_id INT           Specify which GPU to use (default: 0)
  --max_length INT       Maximum sequence length for embedding calculation (default: 256)
  --output_dir TEXT      Directory to store the embedding data (maintains the same file
                         structure as the input directory)
```
</details>

> **Note**: We truncate all sequences to 256 tokens to mitigate the influence of length variations on embedding distributions, as detailed in our paper.

### NovelSum Metric

<details>
  <summary><b>NovelSum Calculation Options</b></summary>

```
Usage: python novelsum.py [OPTIONS]

  Calculate NovelSum for a single embedding file or a directory of embeddings.

Options:
  --single_dataset_path TEXT    Path to a single embedding file
  --multi_datasets_dir TEXT     Directory containing multiple embedding files
  --dense_ref_dir TEXT          Directory with reference dataset for density calculation
  --output_csv TEXT             Path for the output CSV file with results
  --gpu_id INT                  Specify which GPU to use (default: 0)
  --density_powers LIST         List of density powers to calculate NovelSum 
                                (default: [0, 0.25, 0.5])
  --neighbors INT               Number of nearest neighbors for approximate density calculation
                                (default: [5, 10])
  --proximity_powers LIST       List of distance powers for NovelSum calculation 
                                (default: [0, 1, 2])
```
</details>

> **Note**: Our code utilizes Faiss-GPU for accelerated density computation. If you don't have GPU resources available, you can modify the code to use Faiss-CPU instead.

### NovelSelect Strategy

The code efficiently processes your dataset and returns two outputs: the selected text samples and their corresponding indices. Please structure your input text as a list—this can be either raw text content or formatted training data.

<details>
  <summary><b>NovelSelect Options</b></summary>
 

```
Usage: python novelselect.py [OPTIONS]

  Select a subset with maximized NovelSum.

Options:
  --text_dir TEXT          Directory containing text data
  --figure_dir TEXT        Directory containing embeddings (same order as text)
  --output_dir TEXT        Directory to store selected data and indices
  --k INT                  Target number of samples to select (default: 10000)
  --gpu_id INT             Specify which GPU to use (default: 0)
  --density_power FLOAT    Density power parameter for NovelSum (default: 0.5)
  --neighbors INT          Number of nearest neighbors for density calculation (default: 10)
  --proximity_power INT    Distance power parameter for NovelSum (default: 1)
  --seed INT               Random seed for initial point selection (default: 42)
```
</details>

## 🔍 Extensions

In our research, we utilized base models for embedding calculations, but for practical applications, we recommend employing specialized LLM-based embedding models that have been fine-tuned specifically for embedding tasks to achieve superior results.

Based on our comprehensive ablation studies, we recommend the following hyperparameter configuration for optimal alignment with model performance and high-quality dataset selection:
- `density_power`: 0.5
- `proximity_power`: 1
- `neighbors`: 10

When extending our methodology to specialized domains such as code repositories or mathematical problem sets, we encourage experimenting with different hyperparameter combinations guided by our experimental framework to discover optimal configurations for your specific use case.

## 📝 Citation

If you use NovelSum in your research, please cite our paper:

```bibtex
@article{yang2025measuring,
  title={Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric},
  author={Yang, Yuming and Nan, Yang and Ye, Junjie and Dou, Shihan and Wang, Xiao and Li, Shuo and Lv, Huijie and Gui, Tao and Zhang, Qi and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2502.17184},
  year={2025}
}
```

### Acknowledgements

We build upon the following excellent open-source frameworks:

- **[vLLM](https://github.com/vllm-project/vllm)**: High-throughput and memory-efficient inference for LLMs
  
- **[Faiss](https://github.com/facebookresearch/faiss)**: Library for efficient similarity search and clustering of dense vectors

<!-- <a href="https://www.flaticon.com/free-icons/respect" title="respect icons">Respect icons created by Freepik - Flaticon</a> -->
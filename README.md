<p align="center">
<!--  <img src="diversity.png" style="height: 80px;"> -->
 <h2 align="center">Measuring Data Diversity for Instruction Tuning: <br> A Systematic Analysis and A Reliable Metric </h2>
</p>

<p align="center">
 <a href="https://github.com/UmeanNever/NovelSum/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/UmeanNever/NovelSum"></a>
 <a href="https://arxiv.org/abs/2502.17184"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-red"></a>
</p>

## Overview

In this research, we tackle the fundamental challenge of accurately measuring dataset diversity for instruction tuning and introduce **NovelSum**, a reliable diversity metric that demonstrates strong correlation with model performance. Furthermore, we leverage NovelSum as an optimization objective to develop a greedy diversity-oriented data selection strategy called **NovelSelect** that outperforms existing approaches, validating both the effectiveness and practical significance of our metric.

 - 📖 **Paper**: [Read our paper on arXiv](https://arxiv.org/abs/2502.17184)
 - 🛠️ **Code**: All implementation code and resources are available in this repository

<details open>
  <summary><b>Table of Contents</b></summary>

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Usage Guide](#usage-guide)
    - [Data Format](#data-format)
    - [Embedding Calculation](#embedding-calculation)
    - [NovelSum Metric](#novelsum-metric)
    - [NovelSelect Strategy](#novelselect-strategy)
4. [Limitations](#limitations)
5. [Citation](#citation)

</details>


## Installation

```bash
git clone https://github.com/UmeanNever/NovelSum.git
cd NovelSum
pip install -r requirements.txt
```

## Quick Start

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

## Usage Guide

### Data Format

Due to the computational demands of processing large-scale embeddings, we recommend splitting your dataset into multiple JSON files, each containing approximately 5,000 data points (resulting in files of ~1GB). Our implementation uses separate directories for storing text data and embeddings, with each JSON file named numerically (e.g., `18.json`). Both text and embedding data are organized as lists within these files.

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
                         Note: As mentioned in our paper's appendix, embedding distributions
                         can be heavily influenced by sequence length variations. Setting a
                         maximum length helps control for this factor.
  --output_dir TEXT      Directory to store the embedding data (maintains the same file
                         structure as the input directory)
```
</details>


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
  --density_powers LIST         List of density powers to calculate NovelSum (default: [0, 0.25, 0.5])
  --neighbors INT               Number of nearest neighbors for approximate density calculation
                               (default: [5, 10])
  --distance_powers LIST        List of distance powers for NovelSum calculation (default: [0, 1, 2])
```
</details>

### NovelSelect Strategy

<details>
  <summary><b>NovelSelect Options</b></summary>

```
Usage: python novelselect.py [OPTIONS]

  Select a subset with maximized NovelSum.

Options:
  --text_dir TEXT         Directory containing text data
  --figure_dir TEXT       Directory containing embeddings (same order as text)
  --output_dir TEXT       Directory to store selected data and indices
  --k INT                 Target number of samples to select (default: 10000)
  --gpu_id INT            Specify which GPU to use (default: 0)
  --density_powers FLOAT  Density power parameter for NovelSum (default: 0.5)
  --neighbors INT         Number of nearest neighbors for density calculation (default: 10)
  --distance_powers INT   Distance power parameter for NovelSum (default: 1)
  --seed INT              Random seed for initial point selection (default: 42)
```
</details>

## Limitations

Truncating data to a fixed length is our primary method for mitigating the effect of sequence length variations on embedding distributions. This approach could be optimized in future work. For optimal alignment with model performance and high-quality dataset selection, we recommend the following hyperparameters based on our ablation studies:
- `density_powers`: 0.5
- `distance_powers`: 1
- `neighbors`: 10

## Citation

If you use NovelSum in your research, please cite our paper:

```bibtex
@article{novelsum2025,
  title={Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric},
  author={Yuming Yang, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Mingqi Wu, Tao Gui, Qi Zhang, Xuanjing Huang},
  journal={arXiv preprint arXiv:2502.17184},
  year={2025}
}
```

We also build upon the following works:

```bibtex
@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

<!-- <a href="https://www.flaticon.com/free-icons/respect" title="respect icons">Respect icons created by Freepik - Flaticon</a> -->
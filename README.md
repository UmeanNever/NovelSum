<p align="center">
<!--  <img src="diversity.png" style="height: 80px;"> -->
 <h2 align="center">Measuring Data Diversity for Instruction Tuning: <br> A Systematic Analysis and A Reliable Metric </h2>
</p>

<p align="center">
 <a href="https://github.com/UmeanNever/NovelSum/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/UmeanNever/NovelSum"></a>
 <a href="https://arxiv.org/abs/2502.17184"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-red"></a>
</p>

## 📋 Overview

In this research, we tackle the fundamental challenge of accurately measuring dataset diversity for instruction tuning and introduce **NovelSum**, a reliable diversity metric that jointly accounts for inter-sample distances and information density, and shows a strong correlation with model performance. Furthermore, we leverage NovelSum as an optimization objective to develop a greedy diversity-oriented data selection strategy called **NovelSelect** that outperforms existing approaches, validating both the effectiveness and practical significance of our metric.

- 📖 **Paper**: [Read our paper on arXiv](https://arxiv.org/abs/2502.17184)
- 🛠️ **Code**: All codes and resources are available in this repository.

Our codebase supports the following functionalities:
- **Dataset Diversity Measurement (NovelSum)**:  
  Measures dataset diversity using our NovelSum metric, which exhibits a 0.97 correlation with instruction-tuned model performance. While our experiments focus on general instruction-tuning datasets, NovelSum is broadly applicable to textual datasets across various tasks. See Section 3 and 7 of our paper for more information.
- **Data Selection with Optimized Diversity (NovelSelect)**:  
  Selects a diverse subset from a source dataset under a given data budget using our NovelSelect strategy, which outperforms other diversity-oriented data selection strategy. Notably, NovelSelect can seamlessly integrate with quality-based data selection methods. See Section 6 of our paper for more information.
 

## 📑 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
  * [Data Preparation](#data-preparation)
    + [Examples](#examples)
  * [Embedding Calculation](#embedding-calculation)
  * [NovelSum Metric (Diversity Measurement)](#novelsum-metric-diversity-measurement)
  * [NovelSelect Strategy (Data Selection)](#novelselect-strategy-data-selection)
- [Extensions](#-extensions)
- [Citation](#-citation)

## 🔧 Installation

```bash
git clone https://github.com/UmeanNever/NovelSum.git
cd NovelSum
pip install -r requirements.txt
```

## 🚀 Quick Start

### Compute the NovelSum of a dataset

To compute the NovelSum diversity of a dataset, provide a JSON file (or a directory of JSON files) containing data embeddings, along with the reference (source) dataset directory used for estimating the information density factor:

```bash
python novelsum.py --single_dataset_path input_data.json --dense_ref_dir your_ref_dir --output_csv output.csv
```
or 
```bash
python novelsum.py --multi_datasets_dir your_dir --dense_ref_dir your_ref_dir --output_csv output.csv
```

### Use NovelSelect for data selection

To apply the NovelSelect strategy, specify the paths to the source dataset’s text and embedding directories, as well as the desired output directory.

```bash
python novelselect.py --text_dir your_text_dir --figure_dir your_embedding_dir --output_dir your_output_dir
```

See below Usage Guide for details.

## 📚 Usage Guide

### Data Preparation

The input to **NovelSum** consists of a target dataset, for which diversity is computed, and a source (reference) dataset used to estimate the information density factor. If a reference dataset is not readily available, we suggest using a general large-scale (open-source) dataset that is relevant to your task. For instance, in our instruction-tuning experiments, we use a combined dataset consisting of WizardLM, ShareGPT, and UltraChat (using just one of them is also feasible) as the reference. In practice, the reference dataset can be flexibly chosen based on the task at hand; any domain-specific dataset may be used to compute NovelSum for specialized scenarios.

The input to **NovelSelect** requires only the source dataset (i.e., the dataset from which samples are selected).

Both the target dataset and the source dataset use the same conversation-style data format and must be transformed into **sample embeddings** (one embedding vector for each sample in the dataset) before computing NovelSum or running NovelSelect. Due to the computational demands of processing large-scale embeddings, we recommend organizing the dataset as a directory of multiple JSON files, each containing approximately 5,000 samples (~0.5GB per file). In our implementation, text data and embeddings are stored in separate directories, with JSON files named numerically (e.g., `18.json`) for easy indexing and mapping. Each file contains a list of samples, with each entry storing either text or embedding data.

The examples below illustrate the expected data formats:

#### Examples

<details>
  <summary><b>Raw SFT Data Format</b></summary>

```json
[
 [
  {
     "from": "user",
     "value": "Create a detailed and exhaustive HTML guide to assist potential buyers in making a well-informed decision when purchasing a laptop."
  },
  {
     "from": "assistant",
     "value": "..."
  },
  {
     "from": "user",
     "value": "..."
  },
  ...
 ],
 ...
]
```
</details>

<details>
  <summary><b>Corresponding Text Data Format (used for embedding)</b></summary>

```json
[
 "Create a detailed and exhaustive HTML guide to assist potential buyers in making a well-informed decision when purchasing a laptop.\n ...\n ...",
 ...
]
```
</details>

<details>
  <summary><b>Corresponding Embedding Format</b></summary>

```json
[
 [
    -0.06069426238536835,
    -0.07991443574428558,
    ...
    0.04622051864862442
 ],
 ...
]
```
</details>

### Embedding Calculation

You can generate embeddings for your dataset using various models. In our implementation, we use pretrained base LLMs such as LLaMA-3-8B or Qwen-2.5-7B. For details, see Appendix A.1 in our paper. Thanks to vLLM, we were able to compute embeddings for 400,000 instruction-tuning samples in just two hours using 8×H800 GPUs. 

You may refer to the following commands to embed both the target and source (reference) datasets separately. Note that your data should first be converted into plain text format—by joining conversation turns with `\n`—before generating embeddings, as shown in the example above.

<!---<details>
  <summary><b>Embedding Calculation Options</b></summary>-->

```
Usage: python embedding.py [OPTIONS]

  Calculate embeddings for text data in a specified directory.

Options:
  --input_dir TEXT       Directory containing text data as JSON files (list of strings)
  --model_path TEXT      Path to the embedding model. We use pretrained base LLMs such as LLaMA-3-8B or Qwen-2.5-7B.
  --gpu_id INT           Specify which GPU to use (default: 0)
  --max_length INT       Maximum sequence length for embedding calculation (default: 256)
  --output_dir TEXT      Directory to store the embedding data (maintains the same file
                         structure as the input directory)
```
</details>

> **Note**: We truncate all sequences to 256 tokens to mitigate the influence of length variations on embedding distributions, as detailed in our paper.

### NovelSum Metric (Diversity Measurement)

<!---<details>
  <summary><b>NovelSum Calculation Options</b></summary>-->

`single_dataset_path` (or `multi_datasets_dir`) and `dense_ref_dir` point to the embedding files of your target and source (reference) dataset, respectively. Both should be computed and saved in the same manner as illustrated in the data preparation sections above.

```
Usage: python novelsum.py [OPTIONS]

  Calculate NovelSum for a single embedding file or a directory of embeddings.

Options:
  --single_dataset_path TEXT    Path to a single embedding file
  --multi_datasets_dir TEXT     Directory containing multiple embedding files
  --dense_ref_dir TEXT          Directory with reference dataset for density calculation
  --output_csv TEXT             Path for the output CSV file with results
  --gpu_id INT                  Specify the GPU device to use for Faiss (default: 0)
  --neighbors LIST              List of the numbers of nearest neighbors used for approximating the density factor
                                (default: [5, 10])
  --density_powers LIST         List of power parameters applied to the density factor when computing NovelSum
                                (default: [0, 0.25, 0.5])
  --distance_powers LIST        List of power parameters applied to the proximity weight when computing NovelSum
                                (default: [0, 1, 2])
```
</details>

> **Note**: Our code utilizes Faiss-GPU for accelerated density computation. If you don't have GPU resources available, you can modify the code to use Faiss-CPU instead.

### NovelSelect Strategy (Data Selection)

<!---<details>
  <summary><b>NovelSelect Options</b></summary>-->

Please refer to the sections above for data preparation details. The code efficiently processes your input dataset and returns two outputs: the selected text samples and their corresponding indices. Please structure your input text as a list—this can be either raw text content or formatted training data. 

By default, NovelSelect uses the input dataset itself as the reference dataset to estimate the information density factor. However, you may customize our code to specify a different reference dataset with precomputed embeddings.

Moreover, you can integrate NovelSelect with quality-based data selection methods by making a straightforward modification to our code—incorporating sample-wise quality scores as multipliers when selecting the sample with maximum "novelty" $v(x)$ at each iteration of NovelSelect, as discussed in Section 6 of our paper.

```
Usage: python novelselect.py [OPTIONS]

  Select a subset with maximized NovelSum.

Options:
  --text_dir TEXT          Directory containing text data
  --figure_dir TEXT        Directory containing embeddings (same order as text)
  --output_dir TEXT        Directory to store selected data and indices
  --k INT                  Target number of samples to select (default: 10000)
  --gpu_id INT             Specify the GPU device to use for Faiss (default: 0)
  --neighbors INT          Number of nearest neighbors used for approximating the density factor (default: 10)
  --density_power FLOAT    Power parameter for the density factor in NovelSelect (default: 0.5)
  --distance_power FLOAT   Power parameter for the proximity weight in NovelSelect (default: 1.0)
  --seed INT               Random seed for initial point selection (default: 42)
```
</details>

*For more information and a detailed introduction to NovelSum and NovelSelect, please refer to our paper.*

## 🔍 Extensions

In our implementation, we use pretrained base LLMs such as LLaMA-3-8B for sample embedding calculations, primarily for research purposes. For practical applications, you may choose to employ SOTA LLM-based embedding models that have been fine-tuned specifically for embedding tasks to achieve potentially superior performance.

Based on our comprehensive hyperparameter analysis, we recommend the following configuration for NovelSum and NovelSelect, which demonstrates generalizable alignment with model performance and facilitates the selection of high-quality datasets:
- `density_power`: 0.5
- `proximity_power`: 1
- `neighbors`: 10

When adapting our methodology to specialized domains such as code generation or mathematical problems, the above settings serve as a solid starting point. If needed, you can experiment with different hyperparameter combinations, using the correlation analysis in our paper as a guide, to identify the configuration best suited to your use case.

## 📝 Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@article{yang2025measuring,
  title={Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric},
  author={Yang, Yuming and Nan, Yang and Ye, Junjie and Dou, Shihan and Wang, Xiao and Li, Shuo and Lv, Huijie and Wu, Mingqi and Gui, Tao and Zhang, Qi and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2502.17184},
  year={2025}
}
```

## Acknowledgements

We build upon the following excellent open-source frameworks:

- **[vLLM](https://github.com/vllm-project/vllm)**: High-throughput and memory-efficient inference for LLMs
  
- **[Faiss](https://github.com/facebookresearch/faiss)**: Library for efficient similarity search and clustering of dense vectors

<!-- <a href="https://www.flaticon.com/free-icons/respect" title="respect icons">Respect icons created by Freepik - Flaticon</a> -->

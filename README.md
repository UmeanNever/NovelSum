<p align="center">
<!--  <img src="diversity.png" style="height: 80px;"> -->
 <h2 align="center"> NovelSum & NovelSelect </h2>
 <p align="center"> From the paper <strong>"Measuring Data Diversity for Instruction Tuning: <br> A Systematic Analysis and A Reliable Metric"</strong> </p>
</p>

<p align="center">
 <a href="https://github.com/UmeanNever/NovelSum/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/UmeanNever/NovelSum"></a>
 <a href="https://arxiv.org/abs/2502.17184"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-red"></a>
 <a href="https://huggingface.co/datasets/Sirius518/NovelSum"><img alt="Data" src="https://img.shields.io/badge/📀-Data-blue"></a>
</p>

<p align="center">
  <img src="/assets/novelsum.png" alt="NovelSum" width="450"/>
  <br>
  <em>
    The diversity metric, NovelSum, computes each sample’s novelty as a proximity-weighted sum of its density-aware distances to other samples, where closer points have greater influence and high-density regions produce larger distances. The overall dataset diversity is then obtained by summing the novelty scores of all samples.
  </em>
</p>

## 📋 Overview

In this research, we tackle the fundamental challenge of accurately measuring dataset diversity for instruction tuning and introduce **NovelSum**, a reliable **diversity metric** that jointly accounts for inter-sample distances and information density, and shows a strong correlation with model performance.

Furthermore, we leverage NovelSum as an optimization objective to develop a greedy **diversity-oriented data selection strategy** called **NovelSelect** that outperforms existing approaches, validating both the effectiveness and practical significance of our metric.

- 📖 **Paper**: [Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](https://arxiv.org/abs/2502.17184) . Accepted to **ACL 2025** (Main Conference). 🎉
- 🛠️ **Code**: Available in this repository.
- 📀 **Data**: We release both the instruction-tuning dataset curated by NovelSelect and the full source dataset of 396K samples used in our study; see the [Dataset](#-dataset) section below for details.

**Our codebase supports the following functionalities:**
- **Dataset Diversity Measurement (NovelSum)**:  
  Measures dataset diversity using our NovelSum metric, which exhibits a 0.97 correlation with the performance of instruction-tuned models. While our experiments focus on general instruction-tuning datasets, NovelSum is broadly applicable to textual datasets across various tasks. See Section 3 and 7 of our paper for more information. To quickly get started, follow the instructions [below](#compute-the-novelsum-of-a-dataset).
- **Data Selection with Optimized Diversity (NovelSelect)**:  
  Selects a diverse subset from a source dataset under a given data budget using our NovelSelect strategy, which outperforms other diversity-oriented data selection strategies. Notably, NovelSelect can seamlessly integrate with quality-based data selection methods. See Section 6 of our paper for more details. To quickly get started, follow the instructions [below](#use-novelselect-for-data-selection).

For a detailed introduction to NovelSum and NovelSelect, please refer to our paper.

<p align="center">
  <img src="/assets/Scatter_NovelSum.png" alt="Scatter plot" width="500"/>
  <br>
  <em>
    Figure: Our diversity metric, NovelSum, exhibits strong correlation with model performance across IT datasets. <br> 
    Moreover, our NovelSum-based data selection strategy, NovelSelect, achieves superior instruction-tuning performance.
  </em>
</p>

## 📑 Table of Contents

- [🔧 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
  - [Compute the NovelSum of a dataset](#compute-the-novelsum-of-a-dataset)
  - [Use NovelSelect for data selection](#use-novelselect-for-data-selection)
- [📀 Dataset](#-dataset)
- [📚 Usage Guide](#-usage-guide)
  - [Data Preparation](#data-preparation)
    - [Examples](#examples)
  - [Embedding Calculation](#embedding-calculation)
  - [NovelSum Metric (Diversity Measurement)](#novelsum-metric-diversity-measurement)
  - [NovelSelect Strategy (Data Selection)](#novelselect-strategy-data-selection)
- [🔍 Extensions](#-extensions)
- [📝 Citation](#-citation)
- [Acknowledgements](#acknowledgements)

## 🔧 Installation

```bash
git clone https://github.com/UmeanNever/NovelSum.git
cd NovelSum
pip install -r requirements.txt
```

## 🚀 Quick Start

### Compute the NovelSum of a dataset

To compute the NovelSum diversity of a dataset, you need to provide:
- A JSON file (or a directory of JSON files) containing data embeddings for the **target dataset** to be measured.
- (**Optional**) A directory containing data embeddings for the **reference (source) dataset**, which is used to estimate the information density factor.
  - The reference dataset is optional and can be flexibly chosen based on the use case; see the [Data Preparation](#data-preparation) section of the usage guide below for practical suggestions.
  - We provide our preprocessed source dataset of 396K instruction-tuning samples on Hugging Face, which can be downloaded and used as reference dataset for general IT tasks. See the dataset section below for details.

```bash
python novelsum.py --single_dataset_path input_data.json --dense_ref_dir your_ref_dir --output_csv output.csv
```
or 
```bash
python novelsum.py --multi_datasets_dir your_dir --dense_ref_dir your_ref_dir --output_csv output.csv
```
Detailed instructions are provided in the Usage Guide below.

### Use NovelSelect for data selection

To apply the NovelSelect strategy, specify the paths to the source dataset’s text and embedding directories, as well as the desired output directory.

```bash
python novelselect.py --text_dir your_text_dir --figure_dir your_embedding_dir --output_dir your_output_dir
```
Detailed instructions are provided in the Usage Guide below.

## 📀 Dataset

We release two datasets to support reproducibility and further research:
- **A curated 10k instruction-tuning (IT) dataset**, selected using our NovelSelect strategy to maximize diversity based on NovelSum. This subset enables fine-tuned LLMs to achieve strong performance, as evaluated by AlpacaEval and MT-Bench.
- **A combined source dataset of 396k IT samples**, comprising data from WizardLM, ShareGPT, and UltraChat, which serves as the candidate pool for selection (see the paper for more details). This dataset has been preprocessed and embedded, making it ready for downstream data engineering procedures such as data selection.

Both datasets are available on [Hugging Face](https://huggingface.co/datasets/Sirius518/NovelSum). You may also refer to them for the expected input data format when using our codebase.

## 📚 Usage Guide

### Data Preparation

The input to **NovelSum** consists of a target dataset, for which diversity is computed, and a reference (source) dataset used to estimate the information density factor. In our instruction-tuning experiments, we use a combined dataset of WizardLM, ShareGPT, and UltraChat as the reference dataset (using just one of them is also feasible).

In practice, **the reference (source) dataset for NovelSum can be flexibly chosen** based on your use case; any domain-specific dataset may be used for specialized scenarios.
 - For general instruction-tuning tasks, the reference dataset released in our work (described above) can be directly reused.
 - If you are assessing or enhancing the target dataset relative to an existing dataset that represents a desired sample distribution, the latter dataset can naturally serve as the reference dataset.
   - The target dataset itself may also be used as the reference. However, this treats the target dataset’s current sample distribution as the underlying information distribution, causing existing outliers to be regarded as uninformative samples rather than unique ones. As this may deviate from the "diversity" one aims to measure, we generally recommend directly skipping the computation of information density (discussed below) instead when no reference distribution is to be considered.
 - If no such dataset is readily available, one may instead choose a medium- to large-scale dataset with a relatively natural distribution relevant to the task domain (e.g., one derived from open-source data), or alternatively use the validation set of the target task—provided that it adequately reflects the desired distribution, though this may limit generalization.
 - Moreover, if one wishes to skip the computation of information density or if the reference distribution is unknown, the reference dataset can be omitted entirely (see guidance below). In such cases, NovelSum will be computed solely from inter-sample distances (via the proximity-weighted sum), which still provides valuable insights, as shown in our ablation study.

The input to **NovelSelect** requires only the source dataset (i.e., the dataset from which samples are selected).

Both the target dataset and the source dataset use the same conversation-style data format and must be transformed into **sample embeddings** (one embedding vector for each sample in the dataset) before computing NovelSum or running NovelSelect. Due to the computational demands of processing large-scale embeddings, we recommend organizing the dataset as a directory of multiple JSON files, each containing approximately 5,000 samples (~0.5GB per file). In our implementation, text data and embeddings are stored in separate directories, with JSON files named numerically (e.g., `18.json`) for easy indexing and mapping. Each file contains a list of samples, with each entry storing either text or embedding data.

We provide both the target and source datasets used in our study in the dataset section above. These datasets are already processed in the required format and can serve as references. The examples below further illustrate the expected data formats:

#### Examples

<details>
  <summary><b>Raw SFT Data Format</b></summary>

```
[
 [
  {
     "from": "user",
     "value": "Create a detailed and exhaustive HTML guide to assist potential buyers in making a well-informed decision when purchasing a laptop."
  },
  {
     "from": "assistant",
     "value": "<!DOCTYPE html> <html lang="en"> ..."
  },
  {
     "from": "user",
     "value": "Don't use JavaScript."
  },
  {
     "from": "assistant",
     "value": "..."
  }
 ],
 ...
]
```
</details>

<details>
  <summary><b>Corresponding Plain Text Format (used for embedding)</b></summary>

```
[
 "Create a detailed and exhaustive HTML guide to assist potential buyers in making a well-informed decision when purchasing a laptop." + "\n" +"<!DOCTYPE html> <html lang="en"> ..." + "\n" + "Don't use JavaScript." + "\n" + "...", # multi-turn
 "Write a short story about Japanese folklore while also incorporating the idea of reincarnation. Your story should include elements from at least two different Japanese myths or legends." + "\n" + "Once upon a time, there was ...", # single-turn
 ...
]
```
</details>

<details>
  <summary><b>Corresponding Embedding Format</b></summary>

```
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

You can generate embeddings for your dataset using various models. In our implementation, we use pretrained base (i.e., non–instruction-tuned) LLMs such as LLaMA-3-8B or Qwen-2.5-7B. For further details, see Appendix A.1 of our paper. Thanks to vLLM, we were able to compute embeddings for approximately 400,000 instruction-tuning samples in just two hours using 8×H800 GPUs. 

You may refer to the following commands to embed both the target and source (reference) datasets separately. Note that your data should first be converted into plain text format—by joining conversation parts with `\n`—before generating embeddings, as shown in the plain text example above.

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

`single_dataset_path` (or `multi_datasets_dir`) and `dense_ref_dir` point to the embedding files of your target (input) and source (reference) dataset, respectively. The source dataset is optional and can be flexibly chosen based on the use case. Both datasets should be embedded and saved in the same manner. Please refer to the Data Preparation and Embedding Calculation sections above for detailed guidance.

If you prefer to omit the reference dataset and skip the information-density computation, you can set `density_powers` to `[0]` and either point `dense_ref_dir` to any dataset (it will have no effect anyway; you can simply reuse the target dataset) or manually bypass the relevant code. This way, the computed diversity will then rely solely on inter-sample distances (via the proximity-weighted sum), which still provides valuable insights, as shown in our ablation study.

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

Please refer to the sections above for data preprocessing details. The code efficiently applies NovelSelect to your input dataset and produces two outputs: the selected text samples and their corresponding indices. The input text samples should be provided as a list, where each element can be either raw text content or formatted training data. The selected samples will follow the same format as the input text samples.

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

*For more details on the implementation and methodology of NovelSum and NovelSelect, please refer to our paper.*  

*Regarding the code for fine-tuning LLMs, in general, using popular fine-tuning frameworks such as llama-factory can work well, and the practical differences are typically small. Our own fine-tuning code is adapted from [FastChat](https://github.com/lm-sys/FastChat/tree/main). If you're interested, you can refer to Issue [#8](https://github.com/UmeanNever/NovelSum/issues/8), which includes a concise version of our full code.*


## 🔍 Extensions

In our implementation, we use pretrained base (i.e., non–instruction-tuned) LLMs such as LLaMA-3-8B for sample embedding calculations, primarily for research purposes. For practical applications, you may choose to employ SOTA LLM-based embedding models that have been fine-tuned specifically for embedding tasks to achieve potentially superior performance.

Based on our comprehensive hyperparameter analysis, we recommend the following configuration for NovelSum and NovelSelect, which demonstrates generalizable alignment with model performance and facilitates the selection of high-quality datasets:
- `density_power`: 0.5
- `proximity_power`: 1
- `neighbors`: 10

When adapting our methodology to specialized domains such as code generation or mathematical problems, the above settings serve as a solid starting point. If needed, you can experiment with different hyperparameter combinations, guided by the correlation analysis (Section 2) and hyperparameter analysis (Appendix E.1) in our paper, to identify the configuration best suited to your use case.

## 📝 Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@inproceedings{yang-etal-2025-measuring,
    title = "Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric",
    author = "Yang, Yuming  and
      Nan, Yang  and
      Ye, Junjie  and
      Dou, Shihan  and
      Wang, Xiao  and
      Li, Shuo  and
      Lv, Huijie  and
      Gui, Tao  and
      Zhang, Qi  and
      Huang, Xuanjing",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.908/",
    doi = "10.18653/v1/2025.acl-long.908",
    pages = "18530--18549",
    ISBN = "979-8-89176-251-0"
}
```

## Acknowledgements

We build upon the following excellent open-source frameworks:

- **[vLLM](https://github.com/vllm-project/vllm)**: High-throughput and memory-efficient inference for LLMs
  
- **[Faiss](https://github.com/facebookresearch/faiss)**: Library for efficient similarity search and clustering of dense vectors


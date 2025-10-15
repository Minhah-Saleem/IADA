# IADA [![DOI](https://zenodo.org/badge/738799287.svg)](https://zenodo.org/doi/10.5281/zenodo.10693892)
One Shot Intent Aware Data Augmentation <br />
Official Implementation of the paper titled "Intent Aware Data Augmentation by Leveraging Generative AI for Stress Detection in Social Media Texts"
# IADA  
Intent-Aware Data Augmentation (IADA) for Stress Detection in Social Media Texts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![PeerJ Article](https://img.shields.io/badge/paper-PeerJ-CS%20cs-2156-blue)](https://peerj.com/articles/cs-2156/)  
[![GitHub Issues](https://img.shields.io/github/issues/Minhah-Saleem/IADA)](https://github.com/Minhah-Saleem/IADA/issues)  
[![GitHub Stars](https://img.shields.io/github/stars/Minhah-Saleem/IADA)](https://github.com/Minhah-Saleem/IADA/stargazers)

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Key Contributions](#key-contributions)  
- [Paper & Citation](#paper--citation)  
- [Architecture & Method](#architecture--method)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
  - [Training / Augmentation](#training--augmentation)  
  - [Evaluation](#evaluation)  
- [Dataset & Preprocessing](#dataset--preprocessing)  
- [Results & Discussion](#results--discussion)  
- [Limitations & Future Work](#limitations--future-work)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)

---

## Project Overview

**IADA** (Intent-Aware Data Augmentation) is a framework designed to improve stress detection in social media texts by generating augmented samples guided by the **intent** behind the text.  
Instead of blind augmentation (e.g. random synonym replacement), IADA uses generative models to preserve or steer the *intent* in augmented data, making the downstream classifier more robust and intent-sensitive.

This repository contains code, scripts, and experiments used in the research.

---

## Key Contributions

- A novel **intent-aware augmentation** approach using generative models  
- Integration of intent guidance so that augmented samples are not only diverse, but also semantically consistent with the original intent  
- Application to stress detection in social media text  
- Empirical evaluation showing improved performance and robustness  

---

## Paper & Citation

**Title:** *Intent aware data augmentation by leveraging generative AI for stress detection in social media texts* :contentReference[oaicite:0]{index=0}  
**Authors:** Minhah Saleem, Jihie Kim :contentReference[oaicite:1]{index=1}  
**Published in:** *PeerJ Computer Science* (2024) :contentReference[oaicite:2]{index=2}  
**DOI / Link:** https://peerj.com/articles/cs-2156/ :contentReference[oaicite:3]{index=3}  

You can cite this work using:

> Saleem, M., & Kim, J. (2024). Intent aware data augmentation by leveraging generative AI for stress detection in social media texts. *PeerJ Computer Science*. https://doi.org/10.7717/peerj-cs.2156

---

## Architecture & Method

Original Text + Intent Label
│
▼
Generative Model (guided by intent constraint)
│
▼
Augmented Text (preserving target intent)
│
▼
Combined Dataset → Classifier Training
│
▼
Evaluation on test set (robustness, accuracy)

- **Intent encoding**: Represent the underlying intent or label as a guiding signal  
- **Generative model**: Fine-tuned language model or seq2seq model  
- **Augmentation filter / discriminator**: Ensures augmented examples remain consistent with original intent  
- **Classifier**: Stress detection model (e.g. transformer-based or traditional ML)

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Deep learning framework (e.g. PyTorch or TensorFlow)  
- Transformers / language model libraries (e.g. Hugging Face)  
- Common libraries: `numpy`, `pandas`, `scikit-learn`, `torch` (or tf), etc.  
- (Optional) GPU environment for training

### Installation

```bash
# Clone the repository
git clone https://github.com/Minhah-Saleem/IADA.git
cd IADA

# (Optional) Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
````

Ensure any pretrained model weights or checkpoints are downloaded or placed in the expected directories, as referenced in config files.

---

## Usage

### Training / Augmentation

To perform intent-aware augmentation and train your model:

```bash
python run_augmentation.py --mode augment --config config/augment_config.yaml
python run_train.py --mode train --config config/train_config.yaml
```

* `augment` mode generates augmented examples
* `train` mode trains the downstream classifier using augmented + original data

You may configure hyperparameters (learning rate, batch size, augmentation ratio) in the YAML config files.

### Evaluation

After training, run evaluation:

```bash
python run_eval.py --config config/eval_config.yaml
```

This script will output metrics (accuracy, F1, recall, etc.) and robustness tests (e.g. under perturbations).

---

## Dataset & Preprocessing

* The dataset is derived from social media text annotated with **stress-related labels**, plus **intent labels**.
* Preprocessing steps:

  1. Tokenization, cleaning, normalization
  2. Intent label encoding
  3. Splitting into train / validation / test sets
  4. (Optional) Filtering or balancing classes

You may need to adapt file paths in `data_loader.py` or config files, depending on your local setup.

---

## Results & Discussion

* Tabulate classification results (with vs. without IADA augmentation)
* Show robustness tests: e.g. under noisy inputs or adversarial perturbations
* Provide analysis of sample augmented texts (before / after)
* Discussion of when IADA helps most (e.g. rare classes, low-resource data)

---

## Limitations & Future Work

* **Intent dependence**: The method requires annotated intent signals; may not generalize to unlabeled intent scenarios
* **Generative errors**: Sometimes augmentation may drift away from original meaning
* **Scalability**: Large models or many augmentations might require heavy compute
* **Future enhancements**:

  * Better intent representation / embeddings
  * Joint training of generator + classifier
  * Extension to multilingual or multimodal data

---

## Contributing

Contributions, bug fixes, feature requests, or forks are welcome!

```bash
git checkout -b feature/your-feature
# make changes
git commit -m "Add new feature"
git push origin feature/your-feature
```

Then open a **Pull Request** on GitHub. Please add tests or examples if possible.

---

## License

This project is licensed under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

## Acknowledgments

* The PeerJ article and its reviewers
* Open-source NLP & generative model communities
* Tools and libraries: Hugging Face Transformers, PyTorch/TensorFlow, scikit-learn, etc.
* Mentors, colleagues, and collaborators

---

> 🔍 *“Intent-aware augmentation helps the model see not just more data, but more meaningful data.”*

```

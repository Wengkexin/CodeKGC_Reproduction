
# Reproducing and Extending CodeKGC for Medical Knowledge Graph Completion

## Overview

This project reproduces the methodology of **CodeKGC** while introducing enhancements to address its reliance on proprietary models. **CodeKGC** has shown promising results in **Knowledge Graph Completion (KGC)** tasks, particularly in the medical domain. By replicating its approach and providing insights into generalization performance across diverse datasets and an enhancement on evaluation method.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup](#setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Training and Testing](#model-training-and-testing)
6. [Evaluation](#evaluation)
7. [Using Proprietary Models (Codex/GPT)](#using-proprietary-models-codexgpt)
8. [Results](#results)
9. [License](#license)

---

## Introduction

Knowledge Graphs (KGs) are powerful tools for structured information representation and are increasingly critical in healthcare for tasks like:
- **Drug-Adverse Effect Extraction**
- **Biomedical Entity Recognition**

While proprietary models like Codex perform well in general tasks, they lack reproducibility and domain-specific adaptability. This project explores alternatives like **CodeT5** combined with **schema-aware prompting** to address:
1. **Limitations of proprietary models.**
2. **Generation and evaluation of complex medical relationships.**
3. **Generalization ability across datasets.**

---

## Features

- Implementation of **schema-aware prompting** for extracting drug-adverse effect triples.
- Evaluation of **open-source sequence-to-sequence models** (e.g., CodeT5, BioGPT).
- Comparison against **proprietary models** (e.g., Codex, GPT).
- Fine-tuning using **Low-Rank Adaptation (LoRA)**.
- Testing on multiple datasets, including:
  - **ADE Corpus V2**
  - **BC5CDR**

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Wengkexin/CodeKGC_Reproduction.git
   cd CodeKGC_Reproduction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure GPU support for faster training and inference.

---

## Dataset Preparation

### ADE Corpus V2

- Load the ADE Corpus V2 dataset using the `datasets` library:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
  ```

- Preprocess the dataset to create prompts and labels:
  ```python
  def create_prompt_components(icl_examples):
      # Schema-aware prompt and ICL examples generation
      ...
  ```

### BC5CDR Dataset

- Download and preprocess the **BC5CDR** dataset for generalization testing:
  ```python
  from datasets import load_dataset
  dataset = load_dataset('bigbio/bc5cdr', 'bc5cdr_bigbio_kb')
  ```

- Generate sentence-level samples with chemical-disease relationships:
  ```python
  def process_bc5cdr_sample_per_sentence(sample):
      # Process samples to extract chemical-disease pairs
      ...
  ```

---

## Model Training and Testing

### Baseline Testing (Before Fine-Tuning)

- Evaluate **CodeT5** on ADE Corpus V2 without fine-tuning:
  ```python
  def run_baseline_with_evaluation(processed_test_data, tokenizer, model):
      # Baseline evaluation method
      ...
  ```

### Fine-Tuning with LoRA

- Fine-tune the **CodeT5** model using Low-Rank Adaptation (LoRA):
  ```python
  def finetuning_Lora(train_data, test_data, model, tokenizer, epochs=3):
      # LoRA fine-tuning implementation
      ...
  ```

- Save the fine-tuned model:
  ```python
  model.save_pretrained('fine_tuned_model')
  tokenizer.save_pretrained('fine_tuned_model')
  ```

---

## Evaluation

- Parse and evaluate generated triples:
  ```python
  def parse_generated_code(generated_code):
      # Extract triples (drug, relation, adverse_effect)
      ...
  ```

- Compute accuracy metrics for drug and adverse effect extraction:
  ```python
  def evaluate_entity_accuracy(generated_triples, true_triples):
      # Compare generated and true triples
      ...
  ```

---

## Using Proprietary Models (Codex/GPT)

- Evaluate proprietary models like Codex or GPT using OpenAI's API:
  ```python
  import openai
  openai.api_key = "YOUR_API_KEY"
  
  def run_baseline_with_evaluation_for_codex(processed_test_data):
      # Codex evaluation method
      ...
  ```

- Evaluate with GPT models:
  ```python
  def evaluate_gpt(prompt, model="gpt-3.5-turbo"):
      # Generate text using GPT-3.5 Turbo or GPT-4
      ...
  ```

---

## Results

- **ADE Corpus V2**:
  - Fine-Tuned Drug Accuracy: `0.8002%`
  - Fine-Tuned Effect Accuracy: `0.5598%`

- Test finetuned model on **BC5CDR**:
  - Generalization Drug Accuracy: `0.7401%`
  - Generalization Effect Accuracy: `0.4699%`

---


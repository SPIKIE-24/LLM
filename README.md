# CS781 Capstone — Legal NLP with Decoder-Only Language Models

This project investigates the development and task-specific adaptation of compact decoder-only Transformer language models for legal natural language processing. Four model variants were implemented and evaluated across seven tasks, covering legal-domain text generation, information extraction, classification, summarization, and case retrieval.

The project focuses on how architectural choices, particularly attention mechanisms, normalization and positional encoding designs, and downstream adaptation strategies, affect performance across different legal NLP tasks.

## Project Overview

The work is organized around four language models and seven task phases.

### Model Architectures

| Model | Transformer Design | Configuration |
|---|---|---|
| **M1** | Multi-Head Attention (MHA) | 12 layers, hidden size 768, 12 attention heads; RoPE, pre-RMSNorm, GELU FFN |
| **M2** | Multi-Query Attention (MQA) | 18 layers, hidden size 640, 10 query heads and 1 KV head; ALiBi, post-LayerNorm, GELU FFN |
| **M3** | Grouped-Query Attention (GQA) | 16 layers, hidden size 768, 12 query heads and 4 KV heads; RoPE, sliding-window attention, pre-LayerNorm, SwiGLU FFN |
| **M4** | Multi-head Latent Attention (MLA) | 16 layers, hidden size 768; compressed query and KV representations, RoPE, RMSNorm, SwiGLU FFN |

The models use a custom 32K byte-level BPE tokenizer and tied token embeddings. The individual notebooks contain the complete model configurations and implementation details.

## Tasks and Methodology

The project consists of seven legal NLP tasks. The following table summarizes the selected implementation for each phase.

| Phase | Task | Selected Model and Approach |
|---|---|---|
| **1** | Legal-domain pretraining and text generation | M4 — MLA |
| **2** | Legal Named-Entity Recognition (LNER) | M1 — Character-boundary prediction |
| **3** | Rhetorical-Role Classification (RR) | M1 — Supervised classification |
| **4** | Case/Judgment Prediction (CJPE) | M3 — LoRMA+ |
| **5** | Legal Information Extraction (LSI) | M1 — LoRMA+ |
| **6** | Legal Document Summarization | M2 — LoRA |
| **7** | Prior Case Retrieval (PCR) | M1 — LoRMA+ Bi-Encoder |

### Phase 1 — Legal-Domain Pretraining and Text Generation

This phase focuses on pretraining decoder-only language models on a legal-domain corpus and evaluating their ability to generate legal text.

Four different Transformer architectures were explored, with variations in attention mechanisms, positional encoding, normalization and feed-forward network design.

M4, which uses Multi-head Latent Attention (MLA), was selected for this phase.

**Methodology and implementation:**
- Construction and use of a custom 32K byte-level BPE tokenizer.
- Legal-domain corpus preparation and train-validation splitting.
- Autoregressive language-model pretraining.
- Evaluation using validation loss and perplexity.
- Autoregressive text generation using temperature sampling and nucleus sampling.
- Repetition penalty to reduce repetitive generation.

Generation used temperature 0.7, top-p 0.95 and repetition penalty 1.15.

**Implementation:** `notebooks/M4_full_notebook.ipynb`

The M1 notebook also contains the corresponding pretraining implementation and serves as an architectural comparison.

### Phase 2 — Legal Named-Entity Recognition

This phase addresses the extraction of named entities from legal documents.

The selected M1 implementation formulates entity extraction as a character-boundary prediction problem.

A task-specific prediction head predicts entity start and end positions over character positions. These predictions are subsequently decoded into labeled character spans.

**Methodology and implementation:**
- Character-level entity boundary prediction.
- Mapping between tokenizer offsets and original character positions.
- Processing long documents through overlapping chunks.
- Byte and positional feature representations for character-level prediction.
- Partial fine-tuning of the final Transformer blocks.
- Entity span decoding and threshold-based prediction.

The task covers 12 legal entity categories.

**Implementation:** `notebooks/M1_full_notebook.ipynb`

The M4 notebook also contains an alternative experiment for this phase.

### Phase 3 — Rhetorical-Role Classification

This phase focuses on classifying portions of legal documents into rhetorical roles.

The selected implementation uses M1 as the pretrained backbone with a task-specific classification head.

**Methodology and implementation:**
- Preparation of rhetorical-role classification data.
- Extraction of contextual representations from the pretrained language model.
- Supervised fine-tuning using a classification objective.
- Evaluation using development-set classification metrics.
- Comparison of different model architectures and adaptation strategies.

**Implementation:** `notebooks/M1_full_notebook.ipynb`

The M2, M3 and M4 notebooks contain additional experiments for this task.

### Phase 4 — Case/Judgment Prediction

This phase addresses the prediction of case or judgment outcomes using legal text.

The selected implementation uses M3 with LoRMA+ parameter-efficient adaptation and a binary classification head.

**Methodology and implementation:**
- Preparation of case/judgment prediction data.
- Adaptation of the pretrained M3 backbone using LoRMA+.
- Binary classification using a task-specific prediction head.
- Evaluation using Macro-F1, precision and recall.
- Development-set threshold selection to improve classification performance.

The M3 notebook also contains experiments involving LoRA and LoRMA+.

Additional experiments using M1 full fine-tuning, M2 BitFit and M4 knowledge distillation are retained in their respective notebooks.

**Implementation:** `notebooks/M3_full_notebook.ipynb`

Development-set threshold-tuned metrics and official Codabench scores are different evaluation quantities and should not be treated as interchangeable.

### Phase 5 — Legal Information Extraction

This phase focuses on extracting structured legal information from legal documents through multi-label classification.

The selected implementation uses M1 with LoRMA+ adapters and a multi-label prediction head.

**Methodology and implementation:**
- Legal document preprocessing and chunk construction.
- Construction of multi-label training targets.
- Parameter-efficient adaptation using LoRMA+.
- Multi-label classification using binary cross-entropy loss.
- Threshold-based conversion of predicted probabilities into label sets.
- Development-set evaluation and test prediction generation.

**Implementation:** `notebooks/M1_full_notebook.ipynb`

Other model notebooks contain additional experiments. Some of these experiments were incomplete or did not produce completed validation results.

### Phase 6 — Legal Document Summarization

This phase focuses on generating summaries of long legal documents.

The selected implementation uses M2 with LoRA adapters while keeping the pretrained backbone frozen.

Long source documents are divided into overlapping chunks, which are used to train the model to generate summary text.

**Methodology and implementation:**
- Legal document preprocessing.
- Chunk-based processing of long documents.
- Construction of source-summary training examples.
- LoRA-based parameter-efficient fine-tuning.
- Validation using language-model loss.
- Autoregressive summary generation for test documents.

The recorded M2 run completed one epoch with a validation loss of 0.98878. A subsequent epoch was interrupted.

**Implementation:** `notebooks/M2_full_notebook.ipynb`

The M1 notebook contains an alternative LoRMA+ experiment, while M3 contains a prefix-tuning experiment.

### Phase 7 — Prior Case Retrieval

This phase focuses on retrieving relevant prior cases for a given legal query.

The selected implementation uses M1 as a shared encoder in a bi-encoder retrieval architecture.

Query and candidate-case representations are projected into a lower-dimensional embedding space and L2-normalized.

**Methodology and implementation:**
- Preparation of query and candidate-case data.
- Document chunking and representation extraction.
- Shared-encoder bi-encoder architecture.
- LoRMA+ parameter-efficient adaptation.
- Contrastive training using multiple positive and negative examples.
- Distributed training and development-set ranking evaluation.
- Retrieval of the top seven candidate cases.

The system uses a 256-dimensional projected embedding space and evaluates retrieval using Micro-F1@7.

**Implementation:** `notebooks/M1_full_notebook.ipynb`

## Experimental Results

The following table summarizes the official competition scores recorded for the seven phases.

| Phase | Task | Official Score |
|---|---|---:|
| 1 | Legal-domain pretraining and generation | 0.8311 |
| 2 | Legal Named-Entity Recognition | 0.3228 |
| 3 | Rhetorical-Role Classification | 0.3955 |
| 4 | Case/Judgment Prediction | 0.5783 |
| 5 | Legal Information Extraction | — |
| 6 | Legal Document Summarization | 0.8162 |
| 7 | Prior Case Retrieval | 0.1683 |

A dash indicates that no valid official score was recorded for that phase.

The official competition scores and local development-set metrics represent different evaluation settings. The selected implementations described in this README reflect the project's model-selection decisions and should not automatically be assumed to be the models responsible for the highest official score in each phase.

## Repository Structure

```text
.
├── README.md
└── notebooks/
    ├── M1_full_notebook.ipynb
    ├── M2_full_notebook.ipynb
    ├── M3_full_notebook.ipynb
    └── M4_full_notebook.ipynb
```

The four notebooks contain the implementations and experiments for the different model architectures and task phases.

| Notebook | Model | Main Phase Coverage |
|---|---|---|
| `M1_full_notebook.ipynb` | M1 — MHA | Phases 1–7 |
| `M2_full_notebook.ipynb` | M2 — MQA | Phases 1–6 |
| `M3_full_notebook.ipynb` | M3 — GQA | Phases 1–6 |
| `M4_full_notebook.ipynb` | M4 — MLA | Phases 1–6 |

The notebooks preserve their recorded cell outputs and execution history.

## Experimental Setup

The experiments were developed primarily in GPU-enabled Kaggle and Jupyter environments using Python and PyTorch.

Depending on the phase, the notebooks use:
- PyTorch for model implementation and training.
- Hugging Face Datasets and Hub utilities for dataset access.
- Tokenization libraries for text preprocessing.
- NumPy and pandas for data processing.
- Scikit-learn for evaluation utilities.
- tqdm for training progress monitoring.
- PEFT and bitsandbytes for parameter-efficient and quantized experiments where required.

### Dependencies

A basic environment can be prepared using:

```bash
pip install torch numpy pandas scikit-learn tqdm transformers tokenizers datasets huggingface_hub
```

Additional dependencies may be required depending on the experiment, including PEFT and bitsandbytes.

Exact package versions and GPU availability may affect reproducibility.

## Running the Experiments

1. Open the notebook corresponding to the model and phase of interest.
2. Use a GPU-enabled environment. Some experiments require substantial GPU memory.
3. Make the capstone dataset and required Hugging Face resources available to the runtime.
4. Execute the notebook cells in order, beginning with imports, configuration and data preparation.
5. Update environment-specific paths, such as Kaggle input and working-directory paths, when running outside Kaggle.
6. For downstream tasks, provide the corresponding pretrained model and tokenizer checkpoints before starting fine-tuning or inference.
7. Execute the relevant training, evaluation and test prediction cells.

Some experiments use distributed training and require multiple GPUs. The corresponding distributed launch configurations are included in the notebooks.

The notebooks document the experiments as executed. Certain cells depend on checkpoints or intermediate artifacts produced in earlier cells or stored outside the repository, so a fresh runtime may require those artifacts to be recreated.

## Reproducibility and Limitations

- Notebook outputs and execution counts are retained as records of the original experiments.
- Results may vary with random seeds, library versions, GPU hardware and checkpoint selection.
- Some experiments were interrupted or did not produce complete evaluation results. Their status is described in the corresponding notebook and phase documentation above.
- Incomplete experiments should not be interpreted as completed models or submissions.
- Model weights, checkpoints, datasets and credentials are not included in this repository.
- Dataset access and redistribution must follow the applicable course and dataset terms.

## Acknowledgements

This project was completed as part of the CS781 course capstone. We acknowledge the course instructors and the dataset and tooling providers whose resources supported the implementation and evaluation of the project.

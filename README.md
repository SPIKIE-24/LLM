# CS781 Capstone — Legal NLP with Decoder-Only Language Models

This project investigates the development and task-specific adaptation of compact decoder-only Transformer language models for legal natural language processing. Four model variants were implemented and evaluated across seven tasks, covering legal-domain text generation, information extraction, classification, summarization, and case retrieval.

The project focuses on how architectural choices—particularly attention mechanisms and normalization/position-encoding designs—and downstream adaptation strategies affect performance across different legal NLP tasks.

## Project overview

The work is organized around four language models and seven task phases.

| Model | Transformer design | Configuration |
|---|---|---|
| **M1** | Multi-Head Attention (MHA) | 12 layers, hidden size 768, 12 attention heads; RoPE, pre-RMSNorm, GELU FFN |
| **M2** | Multi-Query Attention (MQA) | 18 layers, hidden size 640, 10 query heads and 1 KV head; ALiBi, post-LayerNorm, GELU FFN |
| **M3** | Grouped-Query Attention (GQA) | 16 layers, hidden size 768, 12 query heads and 4 KV heads; RoPE, sliding-window attention, pre-LayerNorm, SwiGLU FFN |
| **M4** | Multi-head Latent Attention (MLA) | 16 layers, hidden size 768; compressed query and KV representations, RoPE, RMSNorm, SwiGLU FFN |

The models use a custom 32K byte-level BPE tokenizer and tied token embeddings. The model-specific notebook contains the complete configuration and implementation details.

## Tasks and selected approaches

The following table summarizes the selected implementation for each phase. The corresponding notebooks also retain additional model experiments and comparisons.

| Phase | Task | Selected model and approach | Main implementation |
|---|---|---|---|
| **1** | Legal-domain pretraining and text generation | **M4 — MLA** | Autoregressive language-model pretraining followed by sampled text generation |
| **2** | Legal named-entity recognition (LNER) | **M1** | Character-boundary prediction head over contextual representations; partial backbone fine-tuning |
| **3** | Rhetorical-role classification (RR) | **M1** | Supervised classification using a task-specific prediction head |
| **4** | Case/judgment prediction (CJPE) | **M3 — LoRMA+** | Parameter-efficient adaptation with development-set threshold selection |
| **5** | Legal information extraction (LSI) | **M1 — LoRMA+** | Multi-label classification with a binary cross-entropy objective |
| **6** | Legal document summarization | **M2 — LoRA** | Chunk-based abstractive summarization with parameter-efficient fine-tuning |
| **7** | Prior case retrieval (PCR) | **M1 — LoRMA+ bi-encoder** | Contrastive retrieval training with normalized query and case embeddings |

### Phase 1 — Pretraining and generation

The project pretrained compact decoder-only language models on a legal-domain corpus and evaluated their ability to generate legal text. The models differ in their attention mechanisms and supporting architectural choices. M4, using Multi-head Latent Attention, was the selected model for this phase. M1 provides a useful MHA comparison.

Generation used nucleus sampling with temperature and repetition controls. The relevant notebook contains the corpus preparation, tokenizer/model setup, training configuration, generation and prediction code.

### Phase 2 — Legal named-entity recognition

The selected M1 system formulates entity extraction as character-boundary prediction. A task-specific head predicts entity start and end positions, which are decoded into labeled character spans. The implementation includes conversion between tokenizer offsets and character positions, document chunking with overlap, and partial fine-tuning of the backbone.

The task covers 12 legal entity categories. The M4 notebook also contains an alternative experiment for this phase.

### Phase 3 — Rhetorical-role classification

This phase classifies legal text into rhetorical roles using a task-specific classifier on top of the pretrained language model. M1 is the selected implementation. The model notebooks retain experiments with other backbones and adaptation strategies, allowing their development behavior to be compared.

### Phase 4 — Case/judgment prediction

The selected implementation uses M3 with LoRMA+ parameter-efficient adaptation and a binary prediction head. A threshold sweep on the development set was used to select the operating threshold. The notebook also contains comparisons with M1 full fine-tuning, M2 BitFit, and M4 knowledge distillation.

Development-set threshold-tuned metrics and official Codabench scores are distinct measurements; they should not be treated as interchangeable. In the recorded official results, M1 obtained a higher Codabench score than M3, while M3's threshold-tuned development result was higher than M1's development result.

### Phase 5 — Legal information extraction

The selected M1 implementation uses LoRMA+ adapters and a multi-label prediction head. The training objective is binary cross-entropy over the legal information labels, and a prediction threshold converts label probabilities into the final set of labels. The notebook includes preprocessing, chunk/label construction, training, validation and test prediction code.

Other model notebooks contain experiments that were incomplete or did not produce a completed validation result; these are retained as experiment records rather than presented as completed alternatives.

### Phase 6 — Legal document summarization

The selected implementation uses M2 with LoRA adapters while keeping the pretrained backbone frozen. Long documents are divided into overlapping chunks, and the model is trained to generate summary text. The notebook includes chunk preparation, adapter configuration, training, validation and test-time generation.

The recorded M2 run completed one epoch and produced a validation loss of 0.98878; a later epoch was interrupted. M1 LoRMA+ and M3 prefix-tuning experiments are also present in their respective notebooks. The recorded M3 inference attempt did not complete successfully.

### Phase 7 — Prior case retrieval

The selected system uses M1 as a shared encoder in a bi-encoder retrieval architecture. Query and candidate-case representations are projected into a lower-dimensional embedding space and L2-normalized. LoRMA+ adapters are used for parameter-efficient adaptation, and a contrastive objective trains the model to rank relevant prior cases above negatives.

The implementation includes candidate preparation, chunk encoding, distributed training, validation ranking and test prediction generation. Retrieval is evaluated using Micro-F1@7.

## Experimental results

The official competition scores recorded for the project are summarized below.

| Phase | Task | Official score | Selected implementation |
|---|---|---:|---|
| 1 | Pretraining and generation | 0.8311 | M4 — MLA |
| 2 | Legal NER | 0.3228 | M1 |
| 3 | Rhetorical-role classification | 0.3955 | M1 |
| 4 | Case/judgment prediction | 0.5783 | M3 — LoRMA+ |
| 5 | Legal information extraction | — | M1 — LoRMA+ |
| 6 | Summarization | 0.8162 | M2 — LoRA |
| 7 | Prior case retrieval | 0.1683 | M1 — LoRMA+ bi-encoder |

A dash indicates that no valid official score was recorded for that phase. These are the competition scores recorded in the project report; local development metrics are reported separately in the notebooks and should not be substituted for official scores.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── notebooks/
│   ├── M1_full_notebook.ipynb
│   ├── M2_full_notebook.ipynb
│   ├── M3_full_notebook.ipynb
│   └── M4_full_notebook.ipynb
└── phases/
    ├── phase1_pretraining/
    ├── phase2_legal_ner/
    ├── phase3_rhetorical_roles/
    ├── phase4_judgment_prediction/
    ├── phase5_legal_information_extraction/
    ├── phase6_summarization/
    └── phase7_prior_case_retrieval/
```

The notebooks are the primary implementation artifacts. They preserve the recorded cells, outputs and execution history. The phase directories provide short entry points to the selected implementation for each task.

## Implementation and experimental environment

The experiments were developed primarily in GPU-enabled Kaggle/Jupyter environments using Python and PyTorch. Depending on the phase, the notebooks also use Hugging Face Datasets/Hub utilities, tokenization libraries, NumPy, pandas, scikit-learn and tqdm. Adapter and quantization experiments may require additional packages such as PEFT and bitsandbytes.

A basic environment can be prepared with:

```bash
pip install torch numpy pandas scikit-learn tqdm transformers tokenizers datasets huggingface_hub
```

Install any additional dependencies imported by the notebook being run. Exact package versions and hardware availability may affect reproducibility.

## Running the experiments

1. Open the notebook corresponding to the model and phase of interest.
2. Use a GPU-enabled environment. Some training cells require substantial GPU memory; distributed training experiments require multiple GPUs and use the distributed launch configuration in the notebook.
3. Make the capstone dataset and any required Hugging Face resources available to the runtime.
4. Run the notebook's setup, data preparation, model initialization and training/evaluation cells in order.
5. Update environment-specific paths (for example, Kaggle input and working-directory paths) when running outside Kaggle.
6. For downstream phases, provide the corresponding pretrained model and tokenizer checkpoints before starting fine-tuning or inference.

These notebooks document the experiments as executed. Some cells depend on checkpoints or intermediate artifacts produced in earlier cells or stored outside the repository, so a fresh runtime may require those artifacts to be recreated. Model weights, checkpoints, datasets and credentials are intentionally not included.

## Notes on reproducibility

- Notebook outputs and execution counts are retained as records of the original runs.
- Results can vary with random seeds, library versions, GPU hardware and checkpoint selection.
- Some experiments were interrupted or did not produce complete evaluation results. Their status is described in the relevant phase section and notebook.
- Do not interpret an incomplete experiment as a completed model or submission.
- Dataset access and redistribution must follow the applicable course and dataset terms.

# Learning Page Order in Shuffled WOO Releases

> **Note (5 November 2026):** We found an error in how the results were evaluated. A corrected version is on the way.

[![arXiv](https://img.shields.io/badge/arXiv-2602.11040-b31b1b.svg)](https://arxiv.org/abs/2602.11040)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue.svg)](https://huggingface.co/datasets/E-8814/woo-page-ordering)

> Can neural networks learn to reorder shuffled pages using only embeddings?

We study this on **5,461 Dutch WOO (freedom of information) documents** — messy, real-world PDFs containing emails, legal texts, and spreadsheets — compiled into single PDFs where pages have been shuffled and semantic ordering cues are unreliable. Using only page-level embeddings from OpenAI's `text-embedding-3-large` (3072-d), we compare five approaches: heuristic baselines (random, greedy nearest neighbour, TSP), a BiLSTM position classifier, pointer networks (LSTM and MLP variants), seq2seq transformers, and pairwise ranking transformers with length-specialized training.

**Paper:** [arXiv:2602.11040](https://arxiv.org/abs/2602.11040)
**Authors:** Efe Kahraman, Giulio Tosato

## Findings

| | |
|---|---|
| **Specialization works** | Training separate models per length range boosts long-document τ by 2.17× |
| **Seq2seq transformers collapse on long documents** | Kendall's τ plummets from 0.918 (2–5 pages) to 0.014 (21–25 pages) |
| **Curriculum learning backfires** | Direct training beats curriculum by 39% on long documents |
| **Short ≠ long** | Short documents rely on local pairwise cues; long documents need global structural attention |
| **Reliable reordering up to 15 pages** | Our best model achieves Kendall's τ of 0.95 (2–5 pages) and 0.72 (11–15 pages), making it practical for the majority of real-world documents |

## Repo structure

```
models/                         # 12 self-contained model scripts (train + evaluate)
├── random_baseline.py          # Random ordering
├── greedy_nearest_neighbour.py # Greedy nearest neighbour
├── tsp_similarity.py           # TSP / spectral / centroid
├── bilstm_position.py          # BiLSTM position classifier
├── pointer_lstm.py             # Pointer network (LSTM)
├── pointer_mlp.py              # Pointer network (MLP)
├── seq2seq_transformer.py      # Seq2seq, learned positional encodings
├── seq2seq_transformer_sinusoidal.py
├── seq2seq_transformer_no_pos.py
├── transformer_pairwise_ranking.py
├── specialized_pairwise_ranking_direct.py      
└── specialized_pairwise_ranking_curriculum.py

analysis/                       # Experiments & figures
├── generate_paper_figures.py   # Reproduce all 9 paper figures
├── evaluate_decoding_diagnostics.py
├── analyze_attention_patterns.py
├── transfer_short_to_long.py
├── compare_positional_encodings.py
└── ablation_positional_vs_none.py

preprocessing/                  # Data pipeline
├── pdf_to_text.py              # PDF → text (PyMuPDF + OCR fallback)
└── generate_embeddings.py      # Text → OpenAI embeddings

results/                        # Pre-computed result reports
figures/                        # Pre-generated paper figures (PDF)
data/                           # Page embeddings (download separately)
```

Every model script is **self-contained**: it defines the architecture, loads data, trains, and evaluates. Just run it.

## Dataset

5,461 Dutch WOO documents, 2–25 pages each. Each page is embedded with OpenAI `text-embedding-3-large` (3072-d).

Download the pre-computed embeddings from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download E-8814/woo-page-ordering --repo-type dataset --local-dir data/
```

This will populate the `data/` directory with one JSON file per document.

Each JSON file in `data/` has this structure:

```json
{
  "doc_id": "uuid",
  "page_numbers": [1, 2, 3, 4, 5],
  "embeddings": [[3072 floats], ...],
  "embedding_dim": 3072,
  "num_pages": 5,
  "correct_order": [1, 2, 3, 4, 5]
}
```

Raw PDFs and OCR-extracted text are available on request — contact the authors.

## Reproducing embeddings from scratch

```bash
# Extract text from PDFs
python preprocessing/pdf_to_text.py

# Generate embeddings (requires Azure OpenAI credentials)
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_KEY="your-key"
python preprocessing/generate_embeddings.py --data_dir text_pages/ --output_dir data/
```

## Quick start

```bash
git clone https://github.com/<your-repo>/embedding-sorting-clean.git
cd embedding-sorting-clean
pip install -r requirements.txt

# Download embeddings
pip install huggingface_hub
huggingface-cli download E-8814/woo-page-ordering --repo-type dataset --local-dir data/

# Train the best model
python models/specialized_pairwise_ranking_direct.py --data_dir data/
```

## Citation

```bibtex
@article{kahraman2026learning,
  title={Learning Page Order in Shuffled WOO Releases},
  author={Kahraman, Efe and Tosato, Giulio},
  journal={arXiv preprint arXiv:2602.11040},
  year={2026}
}
```

A CUDA-capable GPU is required for training.

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

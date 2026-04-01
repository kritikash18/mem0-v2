# Supporting Multimodality in Long-Term Memory Mechanisms

**Author:** Kritika Sharma
**Supervisor:** Dr. Dattatraya Parle
**Institution:** Liverpool John Moores University
**Programme:** Masters in Artificial Intelligence
**Year:** 2025

---

## Overview

This repository contains the implementation and evaluation code for the thesis *"Supporting Multimodality in Long-Term Memory Mechanisms"*, submitted in partial fulfilment of the requirements for the degree of Masters in Artificial Intelligence at Liverpool John Moores University.

The work extends the [Mem0](https://github.com/mem0ai/mem0) memory framework to support multimodal inputs — specifically audio — enabling long-term memory mechanisms to operate beyond text-only modalities. It investigates how speech can be transcribed, encoded, and stored as structured memories, and evaluates retrieval quality across a range of ASR models, LLMs, and configuration parameters.

### Key Contributions

- An audio-to-memory pipeline integrating ASR models (Whisper, Wav2Vec2, AssemblyAI, Google STT) with the Mem0 memory layer
- A systematic evaluation framework comparing ASR models, LLMs, embedding models, and retrieval parameters
- Ablation studies and full-system evaluations measuring BLEU, F1, and LLM-judge scores
- Comparison against Cognee, an alternative multimodal memory system

---

## Repository Structure

```
.
├── evaluation/           # Evaluation pipelines and results
│   ├── audio_eval/       # Audio-to-memory evaluation framework
│   ├── cognee_eval/      # Cognee comparison experiments
│   ├── local_asr/        # Local ASR model setup
│   └── results/          # All experimental results
├── mem0/                 # Core Mem0 library (upstream, extended)
├── docs/                 # Documentation
└── README.md
```

---

## License

This repository contains two components under separate licenses:

### Thesis Research Contributions
Copyright (c) 2025 Kritika Sharma

The research contributions in this repository — including the audio evaluation pipeline, experimental configurations, results, and all code written as part of the thesis — are licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to share and adapt this material for any purpose, provided appropriate credit is given, a link to the license is provided, and any changes are indicated.

Full license text: https://creativecommons.org/licenses/by/4.0/

### Upstream Mem0 Framework
The underlying Mem0 library and associated code (originally developed by mem0ai) is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full terms.

Copyright (c) 2023 Taranjeet Singh and the mem0ai contributors.

---

## Citations

If you use or build upon this work, please cite both the thesis and the original Mem0 paper.

### This Thesis

```bibtex
@mastersthesis{sharma2025multimodalmemory,
  title     = {Supporting Multimodality in Long-Term Memory Mechanisms},
  author    = {Sharma, Kritika},
  school    = {Liverpool John Moores University},
  year      = {2025},
  type      = {Masters Dissertation},
  note      = {Supervisor: Dr. Dattatraya Parle}
}
```

### Original Mem0 Framework

```bibtex
@article{mem0,
  title   = {Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author  = {Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal = {arXiv preprint arXiv:2504.19413},
  year    = {2025}
}
```

---

## Acknowledgements

This work builds on the open-source [Mem0](https://github.com/mem0ai/mem0) framework developed by mem0ai. Sincere thanks to Dr. Dattatraya Parle for supervision and guidance throughout this project.

Healthcare AI Analyzer

Healthcare AI Analyzer is a Python application for advanced analysis of unstructured clinical text using transformer-based medical NLP models. It supports concept extraction, summarization, semantic similarity search, urgency classification, and specialty routing through an interactive terminal interface.

The tool is intended for clinicians, healthcare data scientists, and NLP researchers who require a ready-to-use pipeline for clinical documentation analysis using models such as BioBERT, BART, and MNLI.

Key Features
Capability	Description
Medical concept extraction	Identifies relevant biomedical terminology using BioBERT embeddings.
Clinical note summarization	Generates concise summaries using BART (abstractive summarization).
Semantic case similarity	Compares notes using cosine similarity of BioBERT embeddings.
Urgency classification	Categorizes cases as emergency, urgent, routine, or follow-up using zero-shot classification.
Specialty routing	Suggests the most appropriate clinical specialty (e.g., cardiology, neurology, pulmonology).
Batch analysis	Processes all notes in a dataset automatically.
Optional GPU acceleration	Uses CUDA if available for faster model inference.
Dependencies

The script automatically detects missing dependencies and offers to install them.

Required Python packages:

transformers
torch
scikit-learn
numpy


Manual installation example:

pip install transformers torch scikit-learn numpy


GPU acceleration requires a CUDA-enabled installation of PyTorch and a compatible NVIDIA GPU.

Input Format

The application expects a plain text file containing one or more clinical notes. Notes may be separated with blank lines. For example:

Patient presents with chest pain radiating to the left arm.
EKG shows ST elevations and diaphoresis is present.

Patient with COPD exacerbation requiring BiPAP.
Increased sputum production and diminished breath sounds bilaterally.


Each paragraph is treated as a separate clinical note.

Usage

Run the script:

python clinical_BERT_analyzer.py


On launch, the program will request:

Path to the clinical text file

CPU or GPU selection

The main menu then becomes available:

1. Extract Medical Concepts
2. Summarize Clinical Note
3. Find Similar Cases
4. Classify Urgency
5. Route to Specialty
6. Full Analysis Report
7. Analyze All Notes in File
8. Load Different File
9. Help
0. Exit


Most options allow the user to select a specific note or perform analysis on the entire document.

Under-the-Hood Design

BioBERT is used for semantic embeddings and concept extraction.

BART (facebook/bart-large-cnn) is used for abstractive summarization.

Zero-shot MNLI classification is used for:

Urgency scoring

Specialty routing

Cosine similarity from scikit-learn enables fast semantic case comparisons.

Cached embeddings prevent redundant computation and improve performance.

A fallback rules-based system provides limited functionality if transformer models cannot be loaded.

Typical Workflow

Load a text file containing multiple clinical notes.

Summarize each note for high-level interpretation.

Determine urgency classification to assist with case prioritization.

Assign likely specialty routing.

Search for similar clinical cases within the dataset.

Disclaimer

This project is designed for research, experimentation, and software development purposes only.
It is not intended for direct clinical decision making or patient care.
Medical text models may produce inaccurate or misleading interpretations; all output must be verified manually.

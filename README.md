🏥 Healthcare AI Analyzer

Healthcare AI Analyzer is a Python application that performs advanced analysis on unstructured clinical text using transformer-based medical NLP models. It supports concept extraction, summarization, semantic similarity search, urgency classification, and specialty routing — all through an interactive terminal menu.

This tool is intended for clinicians, medical data scientists, and healthcare NLP researchers who want a fast way to explore clinical documentation using models such as BioBERT, BART, and MNLI without building the pipeline from scratch.

🚀 Features
Feature	Description
🧠 Medical Concept Extraction	Uses BioBERT embeddings to identify key biomedical terms.
📝 Clinical Note Summarization	Generates concise summaries using BART (abstractive).
🔍 Similar Case Search	Finds top semantically similar clinical notes using cosine similarity.
🚨 Urgency Classification	Zero-shot classification into emergency / urgent / routine / follow-up levels.
🏥 Specialty Routing	Recommends appropriate medical specialty (cardiology, neurology, pulmonology, etc.).
📊 Batch Analysis	Applies classification and routing to all notes inside the dataset.
🖥️ GPU Support	Uses CUDA automatically if available for faster inference.
📦 Requirements

The script automatically detects missing dependencies and offers to install them.

Core packages:

transformers
torch
scikit-learn
numpy


You may install them manually if preferred:

pip install transformers torch scikit-learn numpy


GPU acceleration requires a CUDA build of PyTorch and an NVIDIA GPU.

📂 Input Format

The program expects a plain text file containing one or more clinical notes.
Notes may be separated with blank lines, for example:

Patient presents with chest pain radiating to left arm...
EKG shows ST elevations...

Patient with COPD exacerbation requiring BiPAP...
Increasing sputum production...


Each block is treated as a separate clinical note.

▶️ Usage

Run the script:

python clinical_BERT_analyzer.py


You will be prompted to enter:

Path to the medical text file

CPU or GPU selection

You will then see the menu:

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


Most options allow you to:

Select a note by index

Or run the model on the entire document

⚙️ Architecture Highlights

BioBERT used for semantic embeddings and concept extraction

BART (facebook/bart-large-cnn) used for summarization

Zero-shot MNLI model used for:

Urgency classification

Specialty routing

Cosine similarity search over cached BioBERT embeddings for fast comparisons

Graceful fallback logic when transformer models are not available

🧪 Example Workflow

Load file containing raw clinical notes

Summarize each note to extract high-level insight

Determine urgency classification to triage cases

Route cases to specialty subspecialties

Find similar cases for differential and population insight

🛡️ Notes / Disclaimer

This project is for research and educational purposes only.

It is not a clinical decision support tool and should not guide real-world patient care.

Transformer-based NLP may generate errors; always validate outputs manually.

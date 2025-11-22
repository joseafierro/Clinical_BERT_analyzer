"""
Healthcare AI Analyzer
Uses BioBERT/ClinicalBERT for advanced medical text analysis
Requires: transformers, torch, scikit-learn, numpy
"""

import os
import re
import json
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np

def auto_install_dependencies(): # Installs any missing dependencies
    """Auto-install required packages if missing."""
    import subprocess
    import sys
    
    required_packages = {
        'transformers': 'transformers',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    # Check which packages are missing
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("⚠️  Missing required packages detected!")
        print(f"Missing: {', '.join(missing_packages)}\n")
        
        response = input("Would you like to auto-install them now? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\n🔄 Installing packages (this may take several minutes)...\n")
            try:
                for package in missing_packages:
                    print(f"Installing {package}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package, "--quiet"
                    ])
                print("\n✅ All packages installed successfully!")
                print("⚠️  Please restart the program for changes to take effect.\n")
                return False
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Installation failed: {e}")
                print("Please install manually: pip install transformers torch scikit-learn numpy\n")
                return False
        else:
            print("\n⚠️  Continuing with limited functionality...")
            print("To install manually: pip install transformers torch scikit-learn numpy\n")
            return False
    
    return True

# Check and auto-install dependencies
dependencies_ok = auto_install_dependencies()

# Check for required libraries
try:
    from transformers import (
        AutoTokenizer, 
        AutoModel, 
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    if not dependencies_ok:
        print("⚠️  Some features will be limited until dependencies are installed.\n")


class HealthcareAIAnalyzer:
    """Advanced medical text analyzer using transformer models."""
    
    def __init__(self, filepath, device=-1):
        """Initialize with medical AI models.
        
        Args:
            filepath: Path to medical text file
            device: -1 for CPU, 0 for GPU (CUDA)
        """
        self.filepath = filepath
        self.data = None
        self.clinical_notes = []
        self.embeddings_cache = {}
        self.device = device
        self.device_name = "GPU (CUDA)" if device == 0 else "CPU"
        
        # Initialize models
        self.init_models()
        
        # Load file
        self.load_file()
    
    def init_models(self):
        """Initialize transformer models for medical analysis."""
        print(f"🔄 Initializing AI models on {self.device_name} (this may take a moment)...\n")
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  AI features disabled - install required packages\n")
            self.models_loaded = False
            return
        
        try:
            # BioBERT for embeddings and concept extraction
            print("Loading BioBERT for medical concept extraction...")
            self.biobert_tokenizer = AutoTokenizer.from_pretrained(
                "dmis-lab/biobert-v1.1"
            )
            self.biobert_model = AutoModel.from_pretrained(
                "dmis-lab/biobert-v1.1"
            )
            
            # Move BioBERT to selected device
            if self.device == 0:
                self.biobert_model.to('cuda')
            else:
                self.biobert_model.to('cpu')
            
            # Summarization pipeline
            print("Loading summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=self.device
            )
            
            # Classification (we'll use zero-shot for flexibility)
            print("Loading classification model...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            self.models_loaded = True
            print(f"✅ All models loaded successfully on {self.device_name}!\n")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Continuing with limited functionality...\n")
            self.models_loaded = False
    
    def load_file(self):
        """Load and parse the medical text file."""
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"File not found: {self.filepath}")
            
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.data = f.read()
            
            # Split into clinical notes (assuming double newline separation)
            self.clinical_notes = [
                note.strip() 
                for note in self.data.split('\n\n') 
                if note.strip()
            ]
            
            print(f"✅ File loaded: {len(self.data)} chars, {len(self.clinical_notes)} note(s)\n")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def extract_medical_concepts(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract key medical concepts using BioBERT embeddings."""
        if not self.models_loaded:
            return self._fallback_concept_extraction(text, top_n)
        
        try:
            # Extract potential medical terms (simple heuristic)
            medical_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[a-z]+(?:itis|osis|emia|pathy|algia)\b'
            candidates = list(set(re.findall(medical_pattern, text)))
            
            if not candidates:
                return []
            
            # Get embeddings for each candidate
            concept_scores = []
            for candidate in candidates:
                inputs = self.biobert_tokenizer(
                    candidate, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                )
                
                # Move inputs to same device as model
                if self.device == 0:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.biobert_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                
                # Score based on medical relevance (simplified)
                score = float(embedding.norm())
                concept_scores.append((candidate, score))
            
            # Sort by score and return top N
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            return concept_scores[:top_n]
        
        except Exception as e:
            print(f"Error in concept extraction: {e}")
            return self._fallback_concept_extraction(text, top_n)
    
    def _fallback_concept_extraction(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Fallback method using keyword patterns."""
        medical_terms = re.findall(
            r'\b(?:diabetes|hypertension|cardiac|pulmonary|renal|hepatic|'
            r'pneumonia|sepsis|stroke|myocardial|infarction|hemorrhage|'
            r'fracture|trauma|cancer|tumor|malignancy|metastasis)\b',
            text,
            re.IGNORECASE
        )
        
        term_counts = Counter(medical_terms)
        return term_counts.most_common(top_n)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get BioBERT embedding for text."""
        if not self.models_loaded:
            return np.random.rand(768)  # Dummy embedding
        
        # Check cache
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            inputs = self.biobert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to same device as model
            if self.device == 0:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.biobert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            self.embeddings_cache[text] = embedding
            return embedding
        
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.random.rand(1, 768)
    
    def summarize_note(self, text: str, max_length: int = 130) -> str:
        """Generate summary of clinical note."""
        if not self.models_loaded:
            return self._fallback_summary(text, max_length)
        
        try:
            # Ensure text is long enough to summarize
            if len(text.split()) < 50:
                return text
            
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            return summary[0]['summary_text']
        
        except Exception as e:
            print(f"Error in summarization: {e}")
            return self._fallback_summary(text, max_length)
    
    def _fallback_summary(self, text: str, max_length: int) -> str:
        """Simple extractive summary."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:max_length]
        
        # Take first few sentences
        summary = ". ".join(sentences[:3]) + "."
        return summary if len(summary) <= max_length * 5 else summary[:max_length * 5]
    
    def find_similar_cases(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Find similar clinical notes using semantic similarity."""
        if not self.clinical_notes:
            return []
        
        query_emb = self.get_embedding(query)
        
        similarities = []
        for idx, note in enumerate(self.clinical_notes):
            note_emb = self.get_embedding(note)
            sim = cosine_similarity(query_emb, note_emb)[0][0]
            similarities.append((idx, sim, note))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def classify_urgency(self, text: str) -> Dict[str, float]:
        """Classify urgency level of clinical note."""
        if not self.models_loaded:
            return self._fallback_urgency(text)
        
        try:
            candidate_labels = ["emergency", "urgent", "routine", "follow-up"]
            result = self.classifier(text, candidate_labels)
            
            # Convert to dict
            urgency_scores = {
                label: score 
                for label, score in zip(result['labels'], result['scores'])
            }
            return urgency_scores
        
        except Exception as e:
            print(f"Error in urgency classification: {e}")
            return self._fallback_urgency(text)
    
    def _fallback_urgency(self, text: str) -> Dict[str, float]:
        """Rule-based urgency classification."""
        text_lower = text.lower()
        
        emergency_terms = ['emergency', 'critical', 'acute', 'severe', 'stat', 'code']
        urgent_terms = ['urgent', 'immediate', 'asap', 'priority']
        routine_terms = ['routine', 'stable', 'chronic', 'management']
        
        scores = {
            'emergency': sum(term in text_lower for term in emergency_terms) * 0.3,
            'urgent': sum(term in text_lower for term in urgent_terms) * 0.25,
            'routine': sum(term in text_lower for term in routine_terms) * 0.2,
            'follow-up': 0.1
        }
        
        # Normalize
        total = sum(scores.values()) or 1
        return {k: v/total for k, v in scores.items()}
    
    def route_to_specialty(self, text: str) -> Dict[str, float]:
        """Route case to appropriate medical specialty."""
        if not self.models_loaded:
            return self._fallback_routing(text)
        
        try:
            specialties = [
                "cardiology",
                "neurology",
                "pulmonology",
                "gastroenterology",
                "orthopedics",
                "emergency medicine",
                "internal medicine",
                "pediatrics"
            ]
            
            result = self.classifier(text, specialties)
            
            routing_scores = {
                label: score 
                for label, score in zip(result['labels'], result['scores'])
            }
            return routing_scores
        
        except Exception as e:
            print(f"Error in specialty routing: {e}")
            return self._fallback_routing(text)
    
    def _fallback_routing(self, text: str) -> Dict[str, float]:
        """Rule-based specialty routing."""
        text_lower = text.lower()
        
        specialty_keywords = {
            'cardiology': ['cardiac', 'heart', 'myocardial', 'ecg', 'arrhythmia'],
            'neurology': ['neurological', 'stroke', 'seizure', 'brain', 'cva'],
            'pulmonology': ['pulmonary', 'respiratory', 'lung', 'pneumonia', 'copd'],
            'gastroenterology': ['gastrointestinal', 'abdominal', 'liver', 'hepatic'],
            'orthopedics': ['fracture', 'bone', 'joint', 'musculoskeletal'],
            'emergency medicine': ['emergency', 'trauma', 'acute', 'critical'],
        }
        
        scores = {}
        for specialty, keywords in specialty_keywords.items():
            score = sum(keyword in text_lower for keyword in keywords)
            if score > 0:
                scores[specialty] = score
        
        # Normalize
        if scores:
            total = sum(scores.values())
            return {k: v/total for k, v in scores.items()}
        else:
            return {'internal medicine': 1.0}


def print_menu(device_name: str):
    """Display the main menu with current device info."""
    print("\n" + "="*70)
    print(f"HEALTHCARE AI ANALYZER - MENU [{device_name}]")
    print("="*70)
    print("1. 🧠 Extract Medical Concepts (BioBERT)")
    print("2. 📝 Summarize Clinical Note")
    print("3. 🔍 Find Similar Cases (Semantic Search)")
    print("4. 🚨 Classify Urgency Level")
    print("5. 🏥 Route to Specialty")
    print("6. 📊 Full Analysis Report")
    print("7. 💾 Analyze All Notes in File")
    print("8. 📁 Load Different File")
    print("9. ❓ Help")
    print("0. 🚪 Exit")
    print("="*70)


def print_help():
    """Display help information."""
    print("\n" + "="*70)
    print("HELP - AI Features")
    print("="*70)
    print("1. Medical Concepts: Extracts diseases, symptoms, treatments using BioBERT")
    print("2. Summarization: Creates concise summary of clinical notes")
    print("3. Similar Cases: Finds semantically similar cases using embeddings")
    print("4. Urgency: Classifies as emergency/urgent/routine/follow-up")
    print("5. Specialty Routing: Recommends appropriate medical specialty")
    print("6. Full Report: Combines all analyses into comprehensive report")
    print("7. Batch Analysis: Analyzes all notes in the file")
    print("\nModels Used:")
    print("  • BioBERT: Medical concept extraction and embeddings")
    print("  • BART: Clinical note summarization")
    print("  • Zero-shot Classification: Urgency and specialty routing")
    print("="*70)


def select_device():
    """Prompt user to select CPU or GPU and validate availability.
    
    Returns:
        int: -1 for CPU, 0 for GPU
    """
    print("\n" + "="*70)
    print("DEVICE SELECTION")
    print("="*70)
    
    # Check CUDA availability
    cuda_available = False
    if TRANSFORMERS_AVAILABLE:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
        else:
            print("⚠️  No GPU detected (CUDA not available)")
    else:
        print("⚠️  PyTorch not loaded - GPU status unknown")
    
    print("\nOptions:")
    print("  1. CPU (slower, works on all systems)")
    if cuda_available:
        print("  2. GPU (faster, requires CUDA-compatible GPU)")
    else:
        print("  2. GPU (⚠️  NOT AVAILABLE - CUDA not detected)")
    print("="*70)
    
    while True:
        choice = input("\nSelect device (1=CPU, 2=GPU, default=1): ").strip()
        
        # Default to CPU if empty
        if not choice:
            choice = "1"
        
        if choice == "1":
            print("✅ Using CPU")
            return -1
        
        elif choice == "2":
            if cuda_available:
                print("✅ Using GPU (CUDA)")
                return 0
            else:
                print("❌ GPU not available - CUDA not detected")
                print("⚠️  Falling back to CPU\n")
                return -1
        
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def main():
    """Main program loop."""
    print("="*70)
    print("HEALTHCARE AI ANALYZER")
    print("Powered by BioBERT, BART, and Transformer Models")
    print("="*70)
    
    # Get file path
    filepath = input("\nEnter path to medical text file: ").strip()
    
    if not filepath:
        print("❌ No file path provided. Exiting.")
        return
    
    # Device selection
    device = select_device()
    
    try:
        analyzer = HealthcareAIAnalyzer(filepath, device=device)
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    # Main loop
    while True:
        print_menu(analyzer.device_name)
        choice = input("\nSelect option (0-9): ").strip()
        
        if choice == "0":
            print("\n✅ Thank you for using Healthcare AI Analyzer!")
            break
        
        elif choice == "1":
            note_idx = input("Enter note number (or 'all' for full text, 1-based): ").strip()
            if note_idx.lower() == 'all':
                text = analyzer.data
            else:
                try:
                    idx = int(note_idx) - 1
                    text = analyzer.clinical_notes[idx] if 0 <= idx < len(analyzer.clinical_notes) else analyzer.data
                except:
                    text = analyzer.data
            
            print("\n🧠 Extracting medical concepts...\n")
            concepts = analyzer.extract_medical_concepts(text)
            print("Top Medical Concepts:")
            for i, (concept, score) in enumerate(concepts, 1):
                print(f"  {i}. {concept} (score: {score:.2f})")
        
        elif choice == "2":
            note_idx = input("Enter note number to summarize (1-based): ").strip()
            try:
                idx = int(note_idx) - 1
                text = analyzer.clinical_notes[idx] if 0 <= idx < len(analyzer.clinical_notes) else analyzer.data
            except:
                text = analyzer.data
            
            print("\n📝 Generating summary...\n")
            summary = analyzer.summarize_note(text)
            print("Summary:")
            print(f"  {summary}")
        
        elif choice == "3":
            query = input("Enter query text (or note number): ").strip()
            try:
                idx = int(query) - 1
                query_text = analyzer.clinical_notes[idx]
            except:
                query_text = query
            
            similar = analyzer.find_similar_cases(query_text, top_k=3)
            print("\n🔍 Most Similar Cases:\n")
            for rank, (idx, score, note) in enumerate(similar, 1):
                print(f"{rank}. Note #{idx+1} (similarity: {score:.3f})")
                print(f"   Preview: {note[:150]}...")
                print()
        
        elif choice == "4":
            note_idx = input("Enter note number (1-based): ").strip()
            try:
                idx = int(note_idx) - 1
                text = analyzer.clinical_notes[idx] if 0 <= idx < len(analyzer.clinical_notes) else analyzer.data
            except:
                text = analyzer.data
            
            print("\n🚨 Classifying urgency...\n")
            urgency = analyzer.classify_urgency(text)
            print("Urgency Classification:")
            for level, confidence in urgency.items():
                bar = "█" * int(confidence * 20)
                print(f"  {level:12s}: {bar} {confidence:.1%}")
        
        elif choice == "5":
            note_idx = input("Enter note number (1-based): ").strip()
            try:
                idx = int(note_idx) - 1
                text = analyzer.clinical_notes[idx] if 0 <= idx < len(analyzer.clinical_notes) else analyzer.data
            except:
                text = analyzer.data
            
            print("\n🏥 Routing to specialty...\n")
            routing = analyzer.route_to_specialty(text)
            print("Recommended Specialties:")
            for specialty, confidence in routing.items():
                bar = "█" * int(confidence * 20)
                print(f"  {specialty:20s}: {bar} {confidence:.1%}")
        
        elif choice == "6":
            note_idx = input("Enter note number for full analysis (1-based): ").strip()
            try:
                idx = int(note_idx) - 1
                text = analyzer.clinical_notes[idx] if 0 <= idx < len(analyzer.clinical_notes) else analyzer.data
            except:
                text = analyzer.data
            
            print("\n" + "="*70)
            print("FULL ANALYSIS REPORT")
            print("="*70)
            
            print("\n📝 SUMMARY:")
            summary = analyzer.summarize_note(text)
            print(f"  {summary}\n")
            
            print("🧠 KEY MEDICAL CONCEPTS:")
            concepts = analyzer.extract_medical_concepts(text, top_n=5)
            for i, (concept, score) in enumerate(concepts, 1):
                print(f"  {i}. {concept}")
            
            print("\n🚨 URGENCY LEVEL:")
            urgency = analyzer.classify_urgency(text)
            top_urgency = max(urgency.items(), key=lambda x: x[1])
            print(f"  Primary: {top_urgency[0].upper()} ({top_urgency[1]:.1%} confidence)")
            
            print("\n🏥 RECOMMENDED SPECIALTY:")
            routing = analyzer.route_to_specialty(text)
            if routing:
                top_specialty = max(routing.items(), key=lambda x: x[1])
                print(f"  Primary: {top_specialty[0].title()} ({top_specialty[1]:.1%} confidence)")
            
            print("="*70)
        
        elif choice == "7":
            print(f"\n📊 Analyzing all {len(analyzer.clinical_notes)} note(s)...\n")
            
            for i, note in enumerate(analyzer.clinical_notes, 1):
                print(f"\n--- Note #{i} ---")
                print(f"Preview: {note[:100]}...")
                
                urgency = analyzer.classify_urgency(note)
                top_urgency = max(urgency.items(), key=lambda x: x[1])
                print(f"Urgency: {top_urgency[0]} ({top_urgency[1]:.1%})")
                
                routing = analyzer.route_to_specialty(note)
                if routing:
                    top_specialty = max(routing.items(), key=lambda x: x[1])
                    print(f"Specialty: {top_specialty[0]} ({top_specialty[1]:.1%})")
                print()
        
        elif choice == "8":
            new_path = input("Enter new file path: ").strip()
            try:
                analyzer = HealthcareAIAnalyzer(new_path, device=device)
            except:
                print("❌ Failed to load file. Keeping current file.")
        
        elif choice == "9":
            print_help()
        
        else:
            print("\n❌ Invalid option. Please select 0-9.")


if __name__ == "__main__":
    main()
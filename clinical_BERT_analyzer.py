from __future__ import annotations

"""
Healthcare Text Analyzer - hardened local prototype

Important:
- This script is designed to be safer and more production-oriented than the original,
  but code alone does not make a deployment HIPAA compliant.
- Organizational controls, BAAs, infrastructure hardening, IAM, endpoint encryption,
  backup strategy, monitoring, incident response, and formal validation are still required.
- Advanced NLP functions are disabled unless approved model artifacts are available locally.
- Network model downloads are intentionally disabled.
"""

import base64
import getpass
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken

# Optional scientific / ML dependencies
try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional
    AutoModel = None
    AutoTokenizer = None
    pipeline = None


# ---------------------------------------------------------------------------
# Environment hardening
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_TELEMETRY", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    app_dir: Path = field(default_factory=lambda: Path(os.getenv("HTA_APP_DIR", "./hta_secure")).resolve())
    input_base_dir: Path = field(default_factory=lambda: Path(os.getenv("HTA_INPUT_BASE_DIR", ".")).resolve())
    session_timeout_minutes: int = int(os.getenv("HTA_SESSION_TIMEOUT_MINUTES", "10"))
    max_login_attempts: int = int(os.getenv("HTA_MAX_LOGIN_ATTEMPTS", "5"))
    max_file_size_mb: int = int(os.getenv("HTA_MAX_FILE_SIZE_MB", "10"))
    pbkdf2_iterations: int = int(os.getenv("HTA_PBKDF2_ITERATIONS", "600000"))
    min_password_length: int = int(os.getenv("HTA_MIN_PASSWORD_LENGTH", "14"))
    lockout_minutes: int = int(os.getenv("HTA_LOCKOUT_MINUTES", "15"))
    audit_log_file: str = os.getenv("HTA_AUDIT_LOG_FILE", "audit_log.enc")
    cache_file: str = os.getenv("HTA_CACHE_FILE", "embeddings_cache.enc")
    user_db_file: str = os.getenv("HTA_USER_DB_FILE", "users.json")
    encryption_key_file: str = os.getenv("HTA_KEY_FILE", "master.key")
    allowed_extensions: Tuple[str, ...] = (".txt", ".md", ".json")
    enable_deidentification: bool = os.getenv("HTA_ENABLE_DEIDENTIFICATION", "1") == "1"
    offline_only: bool = os.getenv("HTA_OFFLINE_ONLY", "1") == "1"
    model_name_or_path: Optional[str] = os.getenv("HTA_ENCODER_MODEL_PATH")
    summarizer_name_or_path: Optional[str] = os.getenv("HTA_SUMMARIZER_MODEL_PATH")
    classifier_name_or_path: Optional[str] = os.getenv("HTA_CLASSIFIER_MODEL_PATH")
    device_preference: str = os.getenv("HTA_DEVICE", "cpu").lower()
    abstain_threshold: float = float(os.getenv("HTA_ABSTAIN_THRESHOLD", "0.72"))

    def __post_init__(self) -> None:
        self.app_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.app_dir, 0o700)
        except Exception:
            pass

    @property
    def audit_log_path(self) -> Path:
        return self.app_dir / self.audit_log_file

    @property
    def cache_path(self) -> Path:
        return self.app_dir / self.cache_file

    @property
    def user_db_path(self) -> Path:
        return self.app_dir / self.user_db_file

    @property
    def key_path(self) -> Path:
        return self.app_dir / self.encryption_key_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _b64e(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _b64d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


def write_secure_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------
class KeyManager:
    """Manages the local master key.

    For a true production deployment, replace this with KMS / HSM / OS keychain.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def load_or_create_master_key(self) -> bytes:
        env_key = os.getenv("HTA_MASTER_KEY")
        if env_key:
            raw = env_key.encode("ascii")
            # Fernet keys are already urlsafe-base64-encoded 32-byte keys.
            Fernet(raw)
            return raw

        if self.config.key_path.exists():
            raw = self.config.key_path.read_bytes().strip()
            Fernet(raw)
            return raw

        raw = Fernet.generate_key()
        atomic_write_bytes(self.config.key_path, raw)
        return raw


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------
class UserStore:
    def __init__(self, config: AppConfig):
        self.config = config
        if not self.config.user_db_path.exists():
            self._initialize_empty_db()

    def _initialize_empty_db(self) -> None:
        db = {"users": {}, "failed_attempts": {}}
        write_secure_text(self.config.user_db_path, json.dumps(db, indent=2))

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.config.user_db_path.read_text(encoding="utf-8"))

    def _save(self, db: Dict[str, Any]) -> None:
        write_secure_text(self.config.user_db_path, json.dumps(db, indent=2))

    def has_any_users(self) -> bool:
        db = self._load()
        return bool(db.get("users"))

    def create_user(self, username: str, password: str, role: str = "admin") -> None:
        username = username.strip().lower()
        if not username:
            raise ValueError("Username cannot be empty")
        if len(password) < self.config.min_password_length:
            raise ValueError(f"Password must be at least {self.config.min_password_length} characters")

        db = self._load()
        if username in db["users"]:
            raise ValueError("User already exists")

        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.config.pbkdf2_iterations,
        )
        db["users"][username] = {
            "username": username,
            "salt": _b64e(salt),
            "password_hash": _b64e(digest),
            "iterations": self.config.pbkdf2_iterations,
            "role": role,
            "created_at": utc_now().isoformat(),
            "disabled": False,
        }
        self._save(db)

    def verify_user(self, username: str, password: str) -> bool:
        username = username.strip().lower()
        db = self._load()
        user = db["users"].get(username)
        if not user or user.get("disabled"):
            return False

        salt = _b64d(user["salt"])
        expected = _b64d(user["password_hash"])
        calc = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(user["iterations"]),
        )
        return hmac.compare_digest(calc, expected)

    def get_role(self, username: str) -> Optional[str]:
        db = self._load()
        user = db["users"].get(username.strip().lower())
        return user.get("role") if user else None

    def register_failure(self, username: str) -> int:
        username = username.strip().lower()
        db = self._load()
        failures = db.setdefault("failed_attempts", {})
        entry = failures.get(username, {"count": 0, "last_failed_at": None, "locked_until": None})
        now = utc_now()

        locked_until = entry.get("locked_until")
        if locked_until and now < datetime.fromisoformat(locked_until):
            self._save(db)
            return entry["count"]

        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_failed_at"] = now.isoformat()
        if entry["count"] >= self.config.max_login_attempts:
            entry["locked_until"] = (now + timedelta(minutes=self.config.lockout_minutes)).isoformat()
        failures[username] = entry
        self._save(db)
        return entry["count"]

    def clear_failures(self, username: str) -> None:
        username = username.strip().lower()
        db = self._load()
        db.setdefault("failed_attempts", {}).pop(username, None)
        self._save(db)

    def is_locked(self, username: str) -> Tuple[bool, Optional[datetime]]:
        username = username.strip().lower()
        db = self._load()
        entry = db.setdefault("failed_attempts", {}).get(username)
        if not entry or not entry.get("locked_until"):
            return False, None
        locked_until = datetime.fromisoformat(entry["locked_until"])
        if utc_now() >= locked_until:
            db["failed_attempts"].pop(username, None)
            self._save(db)
            return False, None
        return True, locked_until


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
class EncryptedAuditLogger:
    """Encrypted append-only JSONL log with chained integrity hash."""

    def __init__(self, path: Path, fernet: Fernet, signing_key: bytes):
        self.path = path
        self.fernet = fernet
        self.signing_key = signing_key
        if not self.path.exists():
            atomic_write_bytes(self.path, b"")

    def _last_chain(self) -> str:
        try:
            lines = [ln for ln in self.path.read_bytes().splitlines() if ln.strip()]
            if not lines:
                return "GENESIS"
            last_plain = self.fernet.decrypt(lines[-1])
            record = json.loads(last_plain.decode("utf-8"))
            return str(record["chain_hash"])
        except Exception:
            return "CORRUPTED"

    def log(self, username: Optional[str], action: str, details: Dict[str, Any]) -> None:
        previous = self._last_chain()
        record = {
            "timestamp": utc_now().isoformat(),
            "user": username,
            "action": action,
            "details": details,
            "previous_chain_hash": previous,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        chain_hash = hmac.new(self.signing_key, payload + previous.encode("utf-8"), hashlib.sha256).hexdigest()
        record["chain_hash"] = chain_hash
        encrypted = self.fernet.encrypt(json.dumps(record, separators=(",", ":")).encode("utf-8"))
        with self.path.open("ab") as handle:
            handle.write(encrypted + b"\n")
        try:
            os.chmod(self.path, 0o600)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Security manager
# ---------------------------------------------------------------------------
class SecurityManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.master_key = KeyManager(config).load_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        self.user_store = UserStore(config)
        signing_key = hashlib.sha256(self.master_key + b"audit-signing").digest()
        self.audit = EncryptedAuditLogger(config.audit_log_path, self.fernet, signing_key)
        self.current_user: Optional[str] = None
        self.current_role: Optional[str] = None
        self.session_start: Optional[datetime] = None
        self.last_activity: Optional[datetime] = None

    def bootstrap_admin_if_needed(self) -> None:
        if self.user_store.has_any_users():
            return
        print("\n[BOOTSTRAP] No users found. Create the initial administrator account.")
        while True:
            username = input("Admin username: ").strip().lower()
            password = getpass.getpass("Admin password: ")
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                print("[ERROR] Passwords do not match.")
                continue
            try:
                self.user_store.create_user(username, password, role="admin")
                self.audit.log(username, "BOOTSTRAP_ADMIN_CREATED", {"role": "admin"})
                print("[SUCCESS] Initial admin user created.")
                return
            except Exception as exc:
                print(f"[ERROR] {exc}")

    def authenticate(self) -> bool:
        self.bootstrap_admin_if_needed()
        print("\n[AUTH] Secure login required")
        username = input("Username: ").strip().lower()

        locked, locked_until = self.user_store.is_locked(username)
        if locked:
            self.audit.log(username, "LOGIN_BLOCKED_LOCKOUT", {"locked_until": locked_until.isoformat() if locked_until else None})
            print(f"[ERROR] Account locked until {locked_until.isoformat() if locked_until else 'unknown'}")
            return False

        password = getpass.getpass("Password: ")
        if self.user_store.verify_user(username, password):
            self.current_user = username
            self.current_role = self.user_store.get_role(username)
            self.session_start = utc_now()
            self.last_activity = utc_now()
            self.user_store.clear_failures(username)
            self.audit.log(username, "LOGIN_SUCCESS", {"role": self.current_role})
            print("[SUCCESS] Authentication successful")
            return True

        attempts = self.user_store.register_failure(username)
        self.audit.log(username, "LOGIN_FAILURE", {"attempts": attempts})
        print("[ERROR] Invalid credentials")
        return False

    def touch(self) -> None:
        self.last_activity = utc_now()

    def check_session_timeout(self) -> bool:
        if not self.last_activity:
            return True
        if utc_now() - self.last_activity > timedelta(minutes=self.config.session_timeout_minutes):
            self.audit.log(self.current_user, "SESSION_TIMEOUT", {})
            print("[WARNING] Session timed out.")
            return True
        return False

    def encrypt_text(self, text: str) -> bytes:
        return self.fernet.encrypt(text.encode("utf-8"))

    def decrypt_text(self, blob: bytes) -> str:
        return self.fernet.decrypt(blob).decode("utf-8")

    def secure_cleanup(self) -> None:
        self.audit.log(self.current_user, "SESSION_END", {})
        self.current_user = None
        self.current_role = None
        self.session_start = None
        self.last_activity = None


# ---------------------------------------------------------------------------
# PHI de-identification
# ---------------------------------------------------------------------------
class Deidentifier:
    """Rule-based de-identification.

    This is materially broader than the original implementation but still not a
    substitute for a validated de-identification program.
    """

    PATTERNS: List[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        (re.compile(r"\b(?:DOB|Date of Birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I), "[DOB]"),
        (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
        (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
        (re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", re.I), "[DATE]"),
        (re.compile(r"\b(?:MRN|Medical Record Number|Acct|Account)[:#\s-]*[A-Z0-9-]{5,}\b", re.I), "[MRN]"),
        (re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
        (re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"), "[EMAIL]"),
        (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[ZIP]"),
        (re.compile(r"\b(?:https?://|www\.)\S+\b", re.I), "[URL]"),
        (re.compile(r"\b(?:Room|Rm|Apt|Apartment|Unit)\s+#?\w+\b", re.I), "[LOCATION]"),
        (re.compile(r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\s(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way)\b", re.I), "[ADDRESS]"),
    ]

    NAME_LABEL_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"\b(?:Patient Name|Name|Provider|Physician|Doctor|Dr\.?|Seen by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"), "[NAME]"),
    ]

    @classmethod
    def scrub(cls, text: str) -> str:
        cleaned = text
        for pattern, repl in cls.PATTERNS:
            cleaned = pattern.sub(repl, cleaned)
        for pattern, repl in cls.NAME_LABEL_PATTERNS:
            cleaned = pattern.sub(lambda m: m.group(0).replace(m.group(1), repl), cleaned)
        return cleaned


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------
class SecureFileLoader:
    def __init__(self, config: AppConfig, security: SecurityManager):
        self.config = config
        self.security = security

    def validate_path(self, input_path: str) -> Path:
        candidate = Path(input_path).expanduser().resolve()
        base = self.config.input_base_dir.resolve()
        if base not in candidate.parents and candidate != base:
            raise PermissionError(f"Path must be under allowed base directory: {base}")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError("File not found")
        if candidate.suffix.lower() not in self.config.allowed_extensions:
            raise ValueError(f"Disallowed file extension: {candidate.suffix}")
        if candidate.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
            raise ValueError("File exceeds configured size limit")
        st_mode = candidate.stat().st_mode
        if stat.S_ISLNK(st_mode):
            raise ValueError("Symlinks are not allowed")
        return candidate

    def load_text(self, input_path: str) -> str:
        path = self.validate_path(input_path)
        raw = path.read_text(encoding="utf-8")
        cleaned = Deidentifier.scrub(raw) if self.config.enable_deidentification else raw
        self.security.audit.log(self.security.current_user, "FILE_LOADED", {"path": str(path), "bytes": len(raw)})
        return cleaned


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
class ModelManager:
    """Loads only local, offline model artifacts."""

    def __init__(self, config: AppConfig, security: SecurityManager):
        self.config = config
        self.security = security
        self.encoder_tokenizer = None
        self.encoder_model = None
        self.summarizer = None
        self.classifier = None
        self.device = self._get_device()
        self.models_loaded = False
        self.load_errors: List[str] = []

    def _get_device(self) -> int:
        if self.config.device_preference == "cuda" and torch is not None and torch.cuda.is_available():
            return 0
        return -1

    def _validate_local_model_path(self, path_str: Optional[str], label: str) -> Optional[str]:
        if not path_str:
            self.load_errors.append(f"{label}: not configured")
            return None
        p = Path(path_str).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            self.load_errors.append(f"{label}: path not found -> {p}")
            return None
        return str(p)

    def load(self) -> None:
        if AutoTokenizer is None or AutoModel is None or pipeline is None:
            self.load_errors.append("transformers is not installed")
            return

        enc_path = self._validate_local_model_path(self.config.model_name_or_path, "encoder")
        sum_path = self._validate_local_model_path(self.config.summarizer_name_or_path, "summarizer")
        cls_path = self._validate_local_model_path(self.config.classifier_name_or_path, "classifier")

        if not any([enc_path, sum_path, cls_path]):
            self.security.audit.log(self.security.current_user, "MODEL_LOAD_SKIPPED", {"reason": "no local paths configured"})
            return

        try:
            if enc_path:
                self.encoder_tokenizer = AutoTokenizer.from_pretrained(enc_path, local_files_only=True, trust_remote_code=False)
                self.encoder_model = AutoModel.from_pretrained(enc_path, local_files_only=True, trust_remote_code=False)
                if self.device == 0 and torch is not None and torch.cuda.is_available():
                    self.encoder_model.to("cuda")
                else:
                    self.encoder_model.to("cpu")

            if sum_path:
                self.summarizer = pipeline("summarization", model=sum_path, tokenizer=sum_path, device=self.device, local_files_only=True)

            if cls_path:
                self.classifier = pipeline("zero-shot-classification", model=cls_path, tokenizer=cls_path, device=self.device, local_files_only=True)

            self.models_loaded = any([self.encoder_model, self.summarizer, self.classifier])
            self.security.audit.log(self.security.current_user, "MODEL_LOAD_SUCCESS", {"loaded": self.models_loaded, "device": self.device})
        except Exception as exc:
            self.load_errors.append(str(exc))
            self.security.audit.log(self.security.current_user, "MODEL_LOAD_ERROR", {"error": str(exc)})


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class SecureHealthcareTextAnalyzer:
    def __init__(self, config: AppConfig, security: SecurityManager, filepath: str):
        self.config = config
        self.security = security
        self.loader = SecureFileLoader(config, security)
        self.model_manager = ModelManager(config, security)
        self.data = self.loader.load_text(filepath)
        self.notes = [chunk.strip() for chunk in re.split(r"\n\s*\n", self.data) if chunk.strip()]
        self.cache: Dict[str, Any] = {}
        self._load_cache()
        self.model_manager.load()

    # --------------------- cache ---------------------
    def _load_cache(self) -> None:
        if not self.config.cache_path.exists():
            self.cache = {"version": 1, "entries": {}}
            return
        try:
            blob = self.config.cache_path.read_bytes()
            plain = self.security.decrypt_text(blob)
            loaded = json.loads(plain)
            if not isinstance(loaded, dict) or "entries" not in loaded:
                raise ValueError("Invalid cache schema")
            self.cache = loaded
        except (InvalidToken, ValueError, json.JSONDecodeError):
            self.cache = {"version": 1, "entries": {}}
            self.security.audit.log(self.security.current_user, "CACHE_RESET", {"reason": "invalid_or_corrupt"})

    def save_cache(self) -> None:
        payload = json.dumps(self.cache, separators=(",", ":"))
        encrypted = self.security.encrypt_text(payload)
        atomic_write_bytes(self.config.cache_path, encrypted)

    # --------------------- utilities ---------------------
    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _softmax_dict(labels: List[str], scores: List[float]) -> Dict[str, float]:
        return {label: round(float(score), 6) for label, score in zip(labels, scores)}

    @staticmethod
    def _truncate(text: str, max_words: int = 600) -> str:
        words = text.split()
        return " ".join(words[:max_words])

    def extract_medical_concepts(self, text: str, top_n: int = 15) -> List[Dict[str, Any]]:
        """Heuristic extraction only. Not a substitute for validated clinical NER."""
        self.security.touch()
        self.security.audit.log(self.security.current_user, "CONCEPT_EXTRACTION", {"chars": len(text)})

        pattern = re.compile(
            r"\b(?:[A-Za-z]+(?:itis|osis|emia|pathy|algia|oma|ectomy|otomy|scopy|gram|genic|static|dynamic)|"
            r"hypertension|diabetes|asthma|copd|pneumonia|stroke|sepsis|arrhythmia|tachycardia|bradycardia|"
            r"atrial fibrillation|heart failure|chronic kidney disease|coronary artery disease|myocardial infarction)\b",
            re.I,
        )
        counts: Dict[str, int] = {}
        for match in pattern.finditer(text):
            term = match.group(0).strip().lower()
            counts[term] = counts.get(term, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
        return [{"concept": term, "count": count, "confidence": min(0.99, 0.55 + 0.08 * count)} for term, count in ranked]

    def summarize_note(self, text: str) -> Dict[str, Any]:
        self.security.touch()
        self.security.audit.log(self.security.current_user, "SUMMARIZATION", {"chars": len(text)})
        if not self.model_manager.summarizer:
            return {
                "status": "disabled",
                "reason": "No approved local summarization model configured",
                "summary": None,
            }

        clipped = self._truncate(text, 700)
        try:
            result = self.model_manager.summarizer(clipped, max_length=180, min_length=40, do_sample=False)
            summary = result[0]["summary_text"]
            return {"status": "ok", "summary": summary, "human_review_required": True}
        except Exception as exc:
            self.security.audit.log(self.security.current_user, "SUMMARIZATION_ERROR", {"error": str(exc)})
            return {"status": "error", "reason": str(exc), "summary": None}

    def classify_urgency(self, text: str) -> Dict[str, Any]:
        self.security.touch()
        self.security.audit.log(self.security.current_user, "URGENCY_CLASSIFICATION", {"chars": len(text)})
        labels = ["emergency", "urgent", "routine", "follow-up"]
        if not self.model_manager.classifier:
            return {
                "status": "disabled",
                "reason": "No approved local classifier configured",
                "prediction": None,
                "scores": None,
            }
        try:
            result = self.model_manager.classifier(self._truncate(text, 450), labels, multi_label=False)
            scores = self._softmax_dict(result["labels"], result["scores"])
            top_label = result["labels"][0]
            top_score = float(result["scores"][0])
            abstain = top_score < self.config.abstain_threshold
            return {
                "status": "ok",
                "prediction": None if abstain else top_label,
                "scores": scores,
                "abstained": abstain,
                "threshold": self.config.abstain_threshold,
                "human_review_required": True,
            }
        except Exception as exc:
            self.security.audit.log(self.security.current_user, "URGENCY_CLASSIFICATION_ERROR", {"error": str(exc)})
            return {"status": "error", "reason": str(exc), "prediction": None, "scores": None}

    def route_to_specialty(self, text: str) -> Dict[str, Any]:
        self.security.touch()
        self.security.audit.log(self.security.current_user, "SPECIALTY_ROUTING", {"chars": len(text)})
        labels = [
            "cardiology",
            "neurology",
            "pulmonology",
            "gastroenterology",
            "orthopedics",
            "emergency medicine",
            "internal medicine",
            "pediatrics",
        ]
        if not self.model_manager.classifier:
            return {
                "status": "disabled",
                "reason": "No approved local classifier configured",
                "prediction": None,
                "scores": None,
            }
        try:
            result = self.model_manager.classifier(self._truncate(text, 450), labels, multi_label=False)
            scores = self._softmax_dict(result["labels"], result["scores"])
            top_label = result["labels"][0]
            top_score = float(result["scores"][0])
            abstain = top_score < self.config.abstain_threshold
            return {
                "status": "ok",
                "prediction": None if abstain else top_label,
                "scores": scores,
                "abstained": abstain,
                "threshold": self.config.abstain_threshold,
                "human_review_required": True,
            }
        except Exception as exc:
            self.security.audit.log(self.security.current_user, "SPECIALTY_ROUTING_ERROR", {"error": str(exc)})
            return {"status": "error", "reason": str(exc), "prediction": None, "scores": None}

    def full_report(self, note_index: int = 0) -> Dict[str, Any]:
        if not self.notes:
            raise ValueError("No notes loaded")
        if note_index < 0 or note_index >= len(self.notes):
            raise IndexError("Invalid note index")
        note = self.notes[note_index]
        concepts = self.extract_medical_concepts(note)
        summary = self.summarize_note(note)
        urgency = self.classify_urgency(note)
        specialty = self.route_to_specialty(note)
        return {
            "note_index": note_index,
            "note_length_chars": len(note),
            "medical_concepts": concepts,
            "summary": summary,
            "urgency": urgency,
            "specialty": specialty,
            "disclaimer": "Outputs are assistive only and require qualified human review.",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_menu() -> None:
    print("\n" + "=" * 64)
    print("Healthcare Text Analyzer - hardened local prototype")
    print("=" * 64)
    print("[1] Extract medical concepts")
    print("[2] Summarize first note")
    print("[3] Classify urgency of first note")
    print("[4] Route first note to specialty")
    print("[5] Full report for first note")
    print("[0] Exit")
    print("=" * 64)


def pretty(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main() -> int:
    config = AppConfig()
    security = SecurityManager(config)

    try:
        if not security.authenticate():
            return 1

        filepath = input(f"Enter input file path under {config.input_base_dir}: ").strip()
        if not filepath:
            print("[ERROR] No file path provided")
            return 1

        analyzer = SecureHealthcareTextAnalyzer(config, security, filepath)
        print(f"[SUCCESS] Loaded {len(analyzer.notes)} note block(s)")
        if analyzer.model_manager.load_errors:
            print("[INFO] Model status:")
            for err in analyzer.model_manager.load_errors:
                print(f"  - {err}")

        while True:
            if security.check_session_timeout():
                break
            print_menu()
            choice = input("Select option: ").strip()
            security.touch()

            try:
                if choice == "0":
                    break
                elif choice == "1":
                    pretty(analyzer.extract_medical_concepts(analyzer.notes[0] if analyzer.notes else analyzer.data))
                elif choice == "2":
                    pretty(analyzer.summarize_note(analyzer.notes[0] if analyzer.notes else analyzer.data))
                elif choice == "3":
                    pretty(analyzer.classify_urgency(analyzer.notes[0] if analyzer.notes else analyzer.data))
                elif choice == "4":
                    pretty(analyzer.route_to_specialty(analyzer.notes[0] if analyzer.notes else analyzer.data))
                elif choice == "5":
                    pretty(analyzer.full_report(0))
                else:
                    print("[ERROR] Invalid option")
            except Exception as exc:
                security.audit.log(security.current_user, "CLI_ACTION_ERROR", {"error": str(exc), "choice": choice})
                print(f"[ERROR] {exc}")
            finally:
                analyzer.save_cache()
    finally:
        security.secure_cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


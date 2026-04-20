"""
pipeline.py — ModerationPipeline
Production-grade three-layer content moderation guardrail.

Layer 1: Regex pre-filter  (instant, no model call)
Layer 2: Calibrated model  (DistilBERT fine-tuned + isotonic calibration)
Layer 3: Human review queue (uncertain confidence band)
"""

import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


# ── LAYER 1: REGEX BLOCKLIST ───────────────────────────────────────────────
# Organised by category. Each value is a list of compiled regex patterns.
# input_filter() returns the matched category name so the block is auditable.

BLOCKLIST = {

    "direct_threat": [
        # Pattern 1: "I will/gonna/going to [threat verb] you"
        re.compile(
            r"\b(i|we)\s+(will|gonna|going\s+to|shall)\s+"
            r"(kill|murder|shoot|stab|hurt|harm|destroy|end)\s+(you|u|ye|yall|y'all)\b",
            re.IGNORECASE
        ),
        # Pattern 2: "you are going to die / you will die"
        re.compile(
            r"\byou('re|\s+are|\s+will)?\s+(going\s+to\s+)?die\b",
            re.IGNORECASE
        ),
        # Pattern 3: "someone should [threat verb] you"
        re.compile(
            r"\bsomeone\s+should\s+(kill|shoot|stab|hurt|murder)\s+(you|u)\b",
            re.IGNORECASE
        ),
        # Pattern 4: "I'll find where you live / I know where you sleep"
        re.compile(
            r"\b(i'll|i\s+will|i\s+am\s+going\s+to)\s+find\s+"
            r"(where|out\s+where)\s+you\s+(live|sleep|work|stay)\b",
            re.IGNORECASE
        ),
        # Pattern 5: capturing group for variable threat verb — "I'll [verb] you"
        re.compile(
            r"\b(i'll|i\s+will|i'm\s+going\s+to|ima|imma)\s+"
            r"(?P<verb>kill|murder|shoot|stab|hurt|beat|end)\s+(you|u|your\s+\w+)\b",
            re.IGNORECASE
        ),
    ],

    "self_harm_directed": [
        # Pattern 1: "you should kill/hang yourself"
        re.compile(
            r"\byou\s+should\s+(kill|hang|hurt|harm)\s+yourself\b",
            re.IGNORECASE
        ),
        # Pattern 2: "go kill/hang yourself"
        re.compile(
            r"\bgo\s+(kill|hang|hurt|harm|end)\s+yourself\b",
            re.IGNORECASE
        ),
        # Pattern 3: "nobody would miss you if you died/were gone"
        re.compile(
            r"\bnobody\s+(would|will|cares?|misses?)\s+.{0,20}"
            r"(miss|care).{0,10}(you|if\s+you\s+(died?|were\s+gone))\b",
            re.IGNORECASE
        ),
        # Pattern 4: "do everyone a favour and disappear/die"
        re.compile(
            r"\bdo\s+every(one|body)\s+a\s+favo(u)?r\s+and\s+"
            r"(disappear|die|end\s+it|kill\s+yourself)\b",
            re.IGNORECASE
        ),
    ],

    "doxxing_stalking": [
        # Pattern 1: "I know where you live/work/sleep"
        re.compile(
            r"\b(i|we)\s+know\s+where\s+you\s+(live|work|sleep|stay|go)\b",
            re.IGNORECASE
        ),
        # Pattern 2: "I will/I'll post your address/number/info"
        re.compile(
            r"\b(i'll|i\s+will|i'm\s+going\s+to|gonna)\s+post\s+"
            r"your\s+(address|home\s+address|phone|number|info|details|location)\b",
            re.IGNORECASE
        ),
        # Pattern 3: "I found your real name / I found out who you are"
        re.compile(
            r"\b(i\s+found|i've\s+found|found\s+out)\s+"
            r"(your\s+real\s+name|who\s+you\s+(really\s+)?are|your\s+identity)\b",
            re.IGNORECASE
        ),
        # Pattern 4: "everyone will know who you really are / your real identity"
        re.compile(
            r"\beveryone\s+(will|is\s+going\s+to)\s+know\s+"
            r"(who\s+you\s+(really\s+)?are|your\s+(real\s+)?(name|identity))\b",
            re.IGNORECASE
        ),
    ],

    "dehumanization": [
        # Pattern 1: "[group] are not human/people/persons"
        re.compile(
            r"\b\w+\s+are\s+not\s+(?:human|people|persons?|real\s+people)\b",
            re.IGNORECASE
        ),
        # Pattern 2: "[group] are animals/vermin/parasites/subhuman"
        re.compile(
            r"\b\w+\s+are\s+(animals?|vermin|parasites?|subhuman|savages?|beasts?)\b",
            re.IGNORECASE
        ),
        # Pattern 3: "[group] should be exterminated/eliminated/wiped out"
        re.compile(
            r"\b\w+\s+should\s+be\s+"
            r"(exterminated|eliminated|wiped\s+out|eradicated|purged|cleansed)\b",
            re.IGNORECASE
        ),
        # Pattern 4: "[group] are a disease/plague/cancer/infestation"
        re.compile(
            r"\b\w+\s+are\s+a\s+(disease|plague|cancer|infestation|virus|pest)\b",
            re.IGNORECASE
        ),
    ],

    "coordinated_harassment": [
        # Pattern 1: "everyone report [username/this account]"
        re.compile(
            r"\beveryone\s+(please\s+)?(report|flag|block)\s+"
            r"(this\s+)?(user|account|profile|@\w+)\b",
            re.IGNORECASE
        ),
        # Pattern 2: "let's all go after / target [user]"
        re.compile(
            r"\blet'?s\s+(all\s+)?(go\s+after|target|attack|mass\s+report|dogpile)\b",
            re.IGNORECASE
        ),
        # Pattern 3: "raid their profile/server/stream" with lookahead
        re.compile(
            r"\braid\b(?=.{0,30}(profile|server|stream|channel|page|account))",
            re.IGNORECASE
        ),
    ],
}


def input_filter(text: str) -> dict | None:
    """
    Layer 1: fast regex pre-filter.
    Returns a block decision dict if matched, else None.
    The category name is included so blocks are auditable.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision":   "block",
                    "layer":      "input_filter",
                    "category":   category,
                    "confidence": 1.0
                }
    return None


# ── LAYER 2 + 3: CALIBRATED MODEL ─────────────────────────────────────────

class _HFWrapper(BaseEstimator, ClassifierMixin):
    """Wraps HuggingFace model for sklearn CalibratedClassifierCV."""

    def __init__(self, model, tokenizer, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.classes_  = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, texts):
        self.model.eval()
        enc = self.tokenizer(
            list(texts),
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device)
            )
        p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        return np.column_stack([1 - p, p])

    def predict(self, texts):
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)


class ModerationPipeline:
    """
    Three-layer production content moderation pipeline.

    Layer 1 — Regex pre-filter : instant block for high-signal patterns
    Layer 2 — Calibrated model : DistilBERT with isotonic calibration
    Layer 3 — Human review     : uncertain band routed to review queue

    Usage:
        pipeline = ModerationPipeline(model_path, device)
        result   = pipeline.predict("some comment text")
    """

    BLOCK_THRESHOLD  = 0.6
    ALLOW_THRESHOLD  = 0.4

    def __init__(self, model_path: str, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_model       = AutoModelForSequenceClassification.from_pretrained(
                             model_path
                         ).to(device)
        hf_model.eval()
        self._wrapper   = _HFWrapper(hf_model, self.tokenizer, device)
        self._calibrator = None   # fitted lazily or via fit_calibrator()

    def fit_calibrator(self, texts, labels):
        """
        Fit isotonic calibration on a held-out labelled set.
        Call once after construction before running predict().
        """
        self._calibrator = CalibratedClassifierCV(
            self._wrapper, method="isotonic", cv="prefit"
        )
        self._calibrator.fit(list(texts), labels)
        return self

    def _model_prob(self, text: str) -> float:
        """Return calibrated toxic probability for a single text."""
        if self._calibrator is not None:
            prob = self._calibrator.predict_proba([text])[0, 1]
        else:
            prob = self._wrapper.predict_proba([text])[0, 1]
        return float(prob)

    def predict(self, text: str) -> dict:
        """
        Run the three-layer pipeline on a single comment.

        Returns a dict with keys:
            decision    : "block" | "allow" | "review"
            layer       : "input_filter" | "model" | "model_uncertain"
            confidence  : float 0-1
            category    : str (only present for input_filter blocks)
        """
        # ── Layer 1 ──────────────────────────────────────────────────────
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result

        # ── Layer 2 ──────────────────────────────────────────────────────
        prob = self._model_prob(text)

        if prob >= self.BLOCK_THRESHOLD:
            return {
                "decision":   "block",
                "layer":      "model",
                "confidence": round(prob, 4)
            }

        if prob <= self.ALLOW_THRESHOLD:
            return {
                "decision":   "allow",
                "layer":      "model",
                "confidence": round(prob, 4)
            }

        # ── Layer 3 ──────────────────────────────────────────────────────
        return {
            "decision":   "review",
            "layer":      "model_uncertain",
            "confidence": round(prob, 4)
        }

    def predict_batch(self, texts: list, batch_size: int = 64) -> list:
        """Run predict() on a list of texts. Returns list of decision dicts."""
        results = []
        for i, text in enumerate(texts):
            results.append(self.predict(text))
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(texts)}...")
        return results

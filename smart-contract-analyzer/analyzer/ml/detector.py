"""
Smart Contract Security Analyzer - ML Detection Module

LSTM-based vulnerability detection for Solidity contracts.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from typing import List, Dict
import re


class VulnerabilityDetector:
    """ML-based vulnerability detection."""
    
    def __init__(self, model_path: str = None):
        self.max_length = 500
        self.vocab_size = 10000
        self.model = self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load pre-trained model."""
        if path and os.path.exists(path):
            return load_model(path)
        return self._build_model()
    
    def _build_model(self):
        """Build LSTM model for vulnerability detection."""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 128, input_length=self.max_length),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='sigmoid'),  # 5 vulnerability types
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def tokenize(self, source_code: str) -> List[int]:
        """Tokenize source code into integers."""
        # Simple tokenizer for Solidity
        tokens = re.findall(r'\b\w+\b|[{}()\[\];,.=]', source_code.lower())
        return [hash(t) % self.vocab_size for t in tokens]
    
    def detect(self, source_code: str) -> List[Dict]:
        """Detect vulnerabilities in source code."""
        tokens = self.tokenize(source_code)
        padded = pad_sequences([tokens], maxlen=self.max_length)
        
        predictions = self.model.predict(padded, verbose=0)[0]
        
        vulnerability_types = [
            'reentrancy',
            'integer_overflow',
            'access_control',
            'unchecked_call',
            'denial_of_service',
        ]
        
        findings = []
        for i, (vuln_type, score) in enumerate(zip(vulnerability_types, predictions)):
            if score > 0.5:
                findings.append({
                    'type': vuln_type,
                    'confidence': float(score),
                    'severity': self._score_to_severity(score),
                })
        
        return findings
    
    def _score_to_severity(self, score: float) -> str:
        """Convert confidence score to severity."""
        if score > 0.9:
            return 'critical'
        elif score > 0.7:
            return 'high'
        elif score > 0.5:
            return 'medium'
        return 'low'
    
    def train(self, X, y, epochs: int = 10):
        """Train the model."""
        self.model.fit(X, y, epochs=epochs, validation_split=0.2)
    
    def save(self, path: str):
        """Save model."""
        self.model.save(path)

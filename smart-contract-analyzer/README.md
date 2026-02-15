# Smart Contract Security Analyzer

A comprehensive security analysis tool for Ethereum smart contracts using static analysis, symbolic execution, and ML-based vulnerability detection.

## 🔒 Overview

This tool provides:

- **Static Analysis**: Slither, Mythril, and custom analyzers
- **Symbolic Execution**: Echidna, Manticore integration
- **ML Vulnerability Detection**:-trained models for vulnerability classification
- **Gas Optimization**: Cost analysis and suggestions
- **Compliance Checking**: ERC standards verification

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Analyze a contract
python -m analyzer analyze ./contracts/MyContract.sol

# Run ML detector
python -m analyzer ml-detect ./contracts/MyContract.sol
```

## 📦 Dependencies

- Python 3.9+
- solc (Solidity compiler)
- Slither (Trail of Bits)
- Mythril
- Echidna (optional)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Smart Contract Security Analyzer                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │   Solidity  │───▶│  AST Parser  │───▶│  Static Analyzer   │  │
│  │   Source    │    │  (Slither)   │    │  (Rules Engine)    │  │
│  └─────────────┘    └──────────────┘    └────────────────────┘  │
│                                                   │               │
│         ┌────────────────────────────────────────┘               │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ML Vulnerability Detector                   │    │
│  │  ┌─────────────┐  ┌────────────┐  ┌──────────────────┐  │    │
│  │  │ Tokenizer   │→ │  LSTM/Transformer │  │ Vulnerability │  │    │
│  │  │ (AST-based) │  │    Model        │  │   Classifier  │  │    │
│  │  └─────────────┘  └────────────┘  └──────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Report Generator                      │    │
│  │  - JSON/XML/SARIF output                                │    │
│  │  - Severity classification                               │    │
│  │  - Remediation suggestions                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
smart-contract-analyzer/
├── analyzer/
│   ├── static/           # Static analysis modules
│   │   ├── ast_parser.py
│   │   ├── rules_engine.py
│   │   └── taint_analysis.py
│   ├── symbolic/          # Symbolic execution
│   │   ├── echidna_wrapper.py
│   │   └── manticore_wrapper.py
│   ├── ml/               # ML-based detection
│   │   ├── models/
│   │   │   ├── tokenizer.py
│   │   │   ├── detector.py
│   │   │   └── vulnerabilities.h5
│   │   └── training/
│   ├── gas/             # Gas optimization
│   │   ├── analyzer.py
│   │   └── optimizer.py
│   └── cli.py           # Command-line interface
├── contracts/           # Test contracts
├── tests/              # Unit tests
├── requirements.txt
├── README.md
└── setup.py
```

## 🔍 Detectable Vulnerabilities

| Vulnerability | Severity | Description |
|---------------|-----------|-------------|
| Reentrancy | Critical | Recursive call attack |
| Integer Overflow | High | Arithmetic overflow |
| Access Control | High | Unrestricted access |
| Unchecked CALL | Medium | Unchecked low-level call |
| Front-Running | Medium | Transaction ordering |
| DoS | Medium | Denial of service |
| Bad Randomness | High | Weak randomness source |
| Time Manipulation | High | Block timestamp abuse |

## 💻 Usage

### CLI Usage

```bash
# Static analysis only
python -m analyzer static ./contracts/Token.sol

# Symbolic execution
python -m analyzer symbolic ./contracts/Token.sol

# ML-based detection
python -m analyzer ml ./contracts/Token.sol

# Full analysis
python -m analyzer full ./contracts/Token.sol -o report.json
```

### Python API

```python
from analyzer import ContractAnalyzer

# Create analyzer
analyzer = ContractAnalyzer()

# Load contract
analyzer.load_contract("./contracts/MyContract.sol")

# Run all analyses
results = analyzer.analyze()

# Print findings
for finding in results.findings:
    print(f"[{finding.severity}] {finding.name}: {finding.description}")
```

### Gas Optimization

```python
from analyzer.gas import GasAnalyzer

analyzer = GasAnalyzer()
report = analyzer.analyze(contract_source)

print(f"Total gas cost: {report.total_gas}")
print(f"Optimizations: {report.suggestions}")
```

## 🤖 ML Model

The ML detector uses an LSTM-based model trained on:
- 50,000+ contracts from Etherscan
- Labeled vulnerability dataset
- AST-based tokenization

### Training

```bash
python -m analyzer.ml.train --data-dir ./training_data --epochs 50
```

## 📊 Output Formats

### JSON

```json
{
  "contract": "MyContract",
  "findings": [
    {
      "type": "Reentrancy",
      "severity": "Critical",
      "line": 42,
      "description": "Potential reentrancy vulnerability"
    }
  ],
  "score": 75
}
```

### SARIF (for CI/CD integration)

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  ...
}
```

## 🔧 Integration

### GitHub Actions

```yaml
- name: Security Analysis
  uses: ./smart-contract-analyzer
  with:
    contracts: ./contracts/**/*.sol
    severity: medium
```

### CI/CD Pipeline

```bash
# Fail on critical vulnerabilities
analyzer full ./contracts/*.sol --fail-on critical
```

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

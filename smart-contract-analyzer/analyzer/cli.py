# Smart Contract Security Analyzer CLI

Command-line interface for smart contract security analysis.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
from analyzer.static.rules_engine import RulesEngine
from analyzer.ml.detector import MLVulnerabilityDetector
from analyzer.gas.analyzer import GasAnalyzer
from analyzer.report import ReportGenerator


class ContractAnalyzer:
    """Main analyzer orchestrator."""
    
    def __init__(self):
        self.rules_engine = RulesEngine()
        self.ml_detector = MLVulnerabilityDetector()
        self.gas_analyzer = GasAnalyzer()
        self.report_gen = ReportGenerator()
    
    def load_contract(self, path: str):
        """Load contract source code."""
        with open(path, 'r') as f:
            self.source = f.read()
        self.contract_name = Path(path).stem
        return self
    
    def analyze(self, full: bool = True, ml: bool = True) -> dict:
        """Run analysis and return findings."""
        findings = []
        
        if full:
            # Static analysis
            findings.extend(self.rules_engine.analyze(self.source))
            
            # Gas analysis
            gas_report = self.gas_analyzer.analyze(self.source)
            findings.extend(gas_report.findings)
        
        if ml:
            # ML-based detection
            ml_findings = self.ml_detector.detect(self.source)
            findings.extend(ml_findings)
        
        return {
            'contract': self.contract_name,
            'findings': [f.to_dict() for f in findings],
            'gas_report': gas_report.to_dict() if full else None,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Smart Contract Security Analyzer'
    )
    parser.add_argument(
        'contract', type=str, help='Path to Solidity contract'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='json',
        choices=['json', 'sarif', 'text'],
        help='Output format'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run full analysis (static + gas)'
    )
    parser.add_argument(
        '--ml', action='store_true', default=True,
        help='Run ML-based detection'
    )
    parser.add_argument(
        '--severity', type=str, choices=['critical', 'high', 'medium', 'low'],
        help='Minimum severity level'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ContractAnalyzer()
    analyzer.load_contract(args.contract)
    results = analyzer.analyze(full=args.full, ml=args.ml)
    
    # Generate report
    report = analyzer.report_gen.generate(results, args.output)
    
    print(report)


if __name__ == '__main__':
    main()

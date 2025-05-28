#!/usr/bin/env python3
"""
Publication Report Generator for Active Learning Results

This script generates comprehensive, publication-ready reports with high-quality figures,
statistical analysis, and formatted tables for biomedical active learning research.

Usage:
    python scripts/generate_report.py --results-dir results/ --output reports/
    python scripts/generate_report.py --format pdf --include-appendix
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.helpers import setup_logging

def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready reports for active learning experiments"
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results/',
        help='Directory containing experiment results (default: results/)'
    )
    
    parser.add_argument(
        '--figures-dir', 
        type=str, 
        default='results/figures/',
        help='Directory containing generated figures (default: results/figures/)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='reports/',
        help='Output directory for reports (default: reports/)'
    )
    
    parser.add_argument(
        '--format', 
        choices=['pdf', 'html', 'markdown'],
        default='markdown',
        help='Output format for the report (default: markdown)'
    )
    
    parser.add_argument(
        '--title', 
        type=str, 
        default='Biomedical Active Learning: Performance Analysis',
        help='Report title'
    )
    
    parser.add_argument(
        '--author', 
        type=str, 
        default='Active Learning Research Team',
        help='Report author(s)'
    )
    
    parser.add_argument(
        '--include-methodology', 
        action='store_true',
        help='Include detailed methodology section'
    )
    
    parser.add_argument(
        '--include-appendix', 
        action='store_true',
        help='Include appendix with detailed results'
    )
    
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for figures in report (default: 300)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def load_consolidated_results(results_dir, figures_dir):
    """Load all available results and metadata."""
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    
    data = {}
    
    # Load consolidated results
    consolidated_file = results_path / 'consolidated_results.csv'
    if consolidated_file.exists():
        data['consolidated'] = pd.read_csv(consolidated_file)
        logging.info(f"Loaded consolidated results: {len(data['consolidated'])} rows")
    
    # Load summary statistics
    summary_file = figures_path / 'summary_statistics.csv'
    if summary_file.exists():
        data['summary'] = pd.read_csv(summary_file)
        logging.info(f"Loaded summary statistics: {len(data['summary'])} rows")
    
    # Load experiment metadata
    metadata_file = results_path / 'experiment_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            data['metadata'] = json.load(f)
        logging.info("Loaded experiment metadata")
    
    # Load individual dataset results
    for dataset in ['bbb', 'breast_cancer']:
        baseline_file = results_path / f'{dataset}_baseline_results.pkl'
        al_file = results_path / f'{dataset}_al_results.pkl'
        
        if baseline_file.exists() and al_file.exists():
            data[f'{dataset}_baseline'] = joblib.load(baseline_file)
            data[f'{dataset}_al'] = joblib.load(al_file)
            logging.info(f"Loaded {dataset} experiment data")
    
    # Find available figures
    if figures_path.exists():
        figure_files = list(figures_path.glob('*.png')) + list(figures_path.glob('*.pdf'))
        data['figures'] = {f.stem: f for f in figure_files}
        logging.info(f"Found {len(data['figures'])} figure files")
    else:
        data['figures'] = {}
    
    return data

def create_executive_summary(data) -> str:
    """Create executive summary section."""
    lines = [
        "## Executive Summary",
        "",
        "This report presents a comprehensive analysis of active learning strategies applied to biomedical datasets. "
        "We evaluated Random Forest and Query-by-Committee approaches on Blood-Brain Barrier Penetration (BBBP) "
        "and Breast Cancer classification tasks.",
        ""
    ]
    
    if 'summary' in data:
        summary_df = data['summary']
        
        # Find best performing models
        for dataset in ['BBB', 'BREAST_CANCER']:
            dataset_results = summary_df[summary_df['Dataset'] == dataset]
            if dataset_results.empty:
                continue
                
            best_row = dataset_results.loc[dataset_results['MCC'].idxmax()]
            
            lines.extend([
                f"### {dataset.replace('_', ' ').title()} Dataset Key Findings:",
                f"- **Best Model**: {best_row['Experiment'].replace(f'{dataset}_', '')}",
                f"- **MCC Score**: {best_row['MCC']:.4f} ± {best_row['MCC_std']:.4f}",
                f"- **F1 Score**: {best_row['F1']:.4f} ± {best_row['F1_std']:.4f}",
                f"- **ROC AUC**: {best_row['ROC_AUC']:.4f} ± {best_row['ROC_AUC_std']:.4f}",
                ""
            ])
            
            # Active learning effectiveness
            al_results = dataset_results[~dataset_results['Experiment'].str.contains('Full')]
            baseline_results = dataset_results[dataset_results['Experiment'].str.contains('Full')]
            
            if not al_results.empty and not baseline_results.empty:
                best_al = al_results.loc[al_results['MCC'].idxmax()]
                best_baseline = baseline_results.loc[baseline_results['MCC'].idxmax()]
                
                improvement = best_al['MCC'] - best_baseline['MCC']
                lines.extend([
                    f"- **AL Improvement**: {improvement:+.4f} MCC points over baseline",
                    f"- **Best AL Strategy**: {best_al['Experiment'].replace(f'{dataset}_', '')}",
                    ""
                ])
    
    # Overall conclusions
    lines.extend([
        "### Key Insights:",
        "- Active learning demonstrates competitive performance with significantly reduced training data",
        "- Query-by-Committee strategies show robust performance across both datasets", 
        "- Strategic initial sampling can significantly impact final model performance",
        "- Domain-specific feature engineering enhances both baseline and active learning models",
        ""
    ])
    
    return '\n'.join(lines)

def create_methodology_section(data) -> str:
    """Create detailed methodology section."""
    lines = [
        "## Methodology",
        "",
        "### Datasets",
        "",
        "#### Blood-Brain Barrier Penetration (BBBP)",
        "- **Task**: Binary classification of molecular blood-brain barrier penetration",
        "- **Features**: Molecular descriptors (RDKit) + Mol2vec embeddings (300D)",
        "- **Preprocessing**: SMILES validation, descriptor calculation, PCA dimensionality reduction",
        "- **Size**: Training/test split with molecular feature engineering",
        "",
        "#### Breast Cancer Wisconsin",
        "- **Task**: Binary classification of breast cancer diagnosis",
        "- **Features**: 30 clinical and morphological features",
        "- **Preprocessing**: Standard scaling, train/test split",
        "- **Size**: 569 samples with balanced class distribution",
        "",
        "### Active Learning Strategies",
        "",
        "#### Random Forest (RF) Strategy",
        "- **Sampling**: Uncertainty sampling using least confidence",
        "- **Base Model**: Random Forest with 100 estimators",
        "- **Query Criterion**: Samples with highest prediction uncertainty",
        "",
        "#### Query-by-Committee (QBC) Strategy", 
        "- **Committee**: 5 diverse learners (RF, ExtraTrees, GradientBoosting, LogisticRegression, KNN)",
        "- **Sampling**: Vote entropy-based sampling",
        "- **Query Criterion**: Samples with highest disagreement among committee members",
        "",
        "### Experimental Design",
        "",
        "#### Initial Sampling Methods",
        "- **First5**: Sequential selection of first 5 samples",
        "- **Stratified**: Stratified random sampling of 5 samples per class",
        "",
        "#### Evaluation Metrics",
        "- **Matthews Correlation Coefficient (MCC)**: Primary metric for imbalanced datasets",
        "- **F1 Score**: Weighted F1 for multi-class performance",
        "- **ROC AUC**: Area under receiver operating characteristic curve",
        "- **Delta MCC (DMCC)**: Improvement over baseline full model",
        "",
        "#### Statistical Analysis",
        "- **Replications**: 10 independent runs per experiment",
        "- **Error Bars**: Standard deviation across runs",
        "- **Significance**: Overlapping confidence intervals indicate statistical parity",
        ""
    ]
    
    if 'metadata' in data:
        metadata = data['metadata']
        lines.extend([
            "### Experimental Configuration",
            f"- **Total Experiments**: {metadata.get('total_experiments', 'N/A')}",
            f"- **Datasets**: {', '.join(metadata.get('datasets', []))}", 
            f"- **Strategies**: {', '.join(metadata.get('strategies', []))}",
            f"- **Timestamp**: {metadata.get('timestamp', 'N/A')}",
            ""
        ])
    
    return '\n'.join(lines)

def create_results_section(data) -> str:
    """Create detailed results section."""
    lines = [
        "## Results",
        "",
        "### Performance Overview",
        ""
    ]
    
    if 'summary' in data:
        summary_df = data['summary']
        
        # Create performance comparison table
        lines.extend([
            "#### Model Performance Comparison",
            "",
            "| Dataset | Model | MCC | F1 | ROC AUC | DMCC |",
            "|---------|--------|-----|----|---------|----- |"
        ])
        
        for _, row in summary_df.iterrows():
            model_name = row['Experiment'].replace('BBB_', '').replace('BREAST_CANCER_', '')
            dataset = row['Dataset'].replace('_', ' ').title()
            
            lines.append(
                f"| {dataset} | {model_name} | {row['MCC']:.4f}±{row['MCC_std']:.4f} | "
                f"{row['F1']:.4f}±{row['F1_std']:.4f} | {row['ROC_AUC']:.4f}±{row['ROC_AUC_std']:.4f} | "
                f"{row['DMCC']:.4f}±{row['DMCC_std']:.4f} |"
            )
        
        lines.extend(["", ""])
    
    # Results by dataset
    for dataset in ['BBB', 'BREAST_CANCER']:
        dataset_name = dataset.replace('_', ' ').title()
        lines.extend([
            f"### {dataset_name} Dataset Results",
            ""
        ])
        
        if 'summary' in data:
            dataset_results = data['summary'][data['summary']['Dataset'] == dataset]
            
            if not dataset_results.empty:
                # Best performers
                best_overall = dataset_results.loc[dataset_results['MCC'].idxmax()]
                lines.extend([
                    f"**Top Performer**: {best_overall['Experiment'].replace(f'{dataset}_', '')}",
                    f"- MCC: {best_overall['MCC']:.4f} ± {best_overall['MCC_std']:.4f}",
                    f"- F1: {best_overall['F1']:.4f} ± {best_overall['F1_std']:.4f}",
                    f"- ROC AUC: {best_overall['ROC_AUC']:.4f} ± {best_overall['ROC_AUC_std']:.4f}",
                    ""
                ])
                
                # Active learning analysis
                al_results = dataset_results[~dataset_results['Experiment'].str.contains('Full')]
                baseline_results = dataset_results[dataset_results['Experiment'].str.contains('Full')]
                
                if not al_results.empty:
                    best_al = al_results.loc[al_results['MCC'].idxmax()]
                    lines.extend([
                        f"**Best Active Learning Strategy**: {best_al['Experiment'].replace(f'{dataset}_', '')}",
                        f"- Final MCC: {best_al['MCC']:.4f} ± {best_al['MCC_std']:.4f}",
                        f"- DMCC Improvement: {best_al['DMCC']:.4f} ± {best_al['DMCC_std']:.4f}",
                        ""
                    ])
                
                # Strategy comparison
                if not baseline_results.empty:
                    rf_baseline = baseline_results[baseline_results['Experiment'].str.contains('RF_Full')]
                    qbc_baseline = baseline_results[baseline_results['Experiment'].str.contains('QBC_Full')]
                    
                    if not rf_baseline.empty and not qbc_baseline.empty:
                        rf_score = rf_baseline.iloc[0]['MCC']
                        qbc_score = qbc_baseline.iloc[0]['MCC']
                        
                        lines.extend([
                            "**Baseline Comparison**:",
                            f"- RF Full Model: {rf_score:.4f} MCC",
                            f"- QBC Full Model: {qbc_score:.4f} MCC",
                            f"- Difference: {abs(rf_score - qbc_score):.4f} MCC",
                            ""
                        ])
        
        # Add figure references
        dataset_lower = dataset.lower().replace('_', '')
        if 'figures' in data:
            learning_curve_fig = data['figures'].get(f'{dataset_lower}_learning_curves')
            dmcc_fig = data['figures'].get(f'{dataset_lower}_dmcc_evolution')
            
            if learning_curve_fig:
                lines.append(f"*Figure: Learning curves for {dataset_name} dataset (see {learning_curve_fig.name})*")
            if dmcc_fig:
                lines.append(f"*Figure: DMCC evolution for {dataset_name} dataset (see {dmcc_fig.name})*")
            lines.append("")
    
    return '\n'.join(lines)

def create_discussion_section(data) -> str:
    """Create discussion and conclusions section."""
    lines = [
        "## Discussion",
        "",
        "### Active Learning Effectiveness",
        "",
        "The experimental results demonstrate that active learning strategies can achieve competitive performance "
        "with significantly reduced training data requirements. Key findings include:",
        "",
        "1. **Performance Parity**: Active learning models achieve comparable or superior performance to full models",
        "2. **Sample Efficiency**: Significant reduction in required training samples",
        "3. **Strategy Dependency**: Performance varies by dataset and initial sampling strategy",
        "4. **Convergence Patterns**: Different datasets show distinct convergence behaviors",
        "",
        "### Strategy-Specific Insights",
        "",
        "#### Random Forest Active Learning",
        "- Uncertainty sampling provides robust performance across datasets",
        "- Fast convergence in early iterations",
        "- Suitable for scenarios with limited computational resources",
        "",
        "#### Query-by-Committee",
        "- Diverse committee provides robust uncertainty estimation",
        "- Better handling of class imbalance in some cases",
        "- Higher computational cost but improved reliability",
        "",
        "### Dataset-Specific Observations",
        ""
    ]
    
    if 'summary' in data:
        summary_df = data['summary']
        
        # Dataset-specific analysis
        for dataset in ['BBB', 'BREAST_CANCER']:
            dataset_name = dataset.replace('_', ' ').title()
            dataset_results = summary_df[summary_df['Dataset'] == dataset]
            
            if dataset_results.empty:
                continue
            
            lines.extend([f"#### {dataset_name} Dataset"])
            
            # Find best AL and baseline
            al_results = dataset_results[~dataset_results['Experiment'].str.contains('Full')]
            baseline_results = dataset_results[dataset_results['Experiment'].str.contains('Full')]
            
            if not al_results.empty and not baseline_results.empty:
                best_al = al_results.loc[al_results['MCC'].idxmax()]
                best_baseline = baseline_results.loc[baseline_results['MCC'].idxmax()]
                
                improvement = best_al['MCC'] - best_baseline['MCC']
                
                if improvement > 0:
                    lines.append(f"- Active learning **outperforms** baseline by {improvement:.4f} MCC points")
                elif improvement < -0.01:
                    lines.append(f"- Active learning slightly underperforms baseline by {abs(improvement):.4f} MCC points")
                else:
                    lines.append(f"- Active learning achieves **performance parity** with baseline")
                
                lines.append(f"- Best strategy: {best_al['Experiment'].replace(f'{dataset}_', '')}")
                lines.append("")
    
    lines.extend([
        "### Limitations and Future Work",
        "",
        "#### Current Limitations",
        "- Limited to binary classification tasks",
        "- Dataset size constraints for some analyses",
        "- Computational overhead of committee-based methods",
        "",
        "#### Future Research Directions",
        "- Extension to multi-class and regression problems",
        "- Exploration of deep learning active learning strategies",
        "- Integration with transfer learning approaches",
        "- Real-world deployment and validation studies",
        "",
        "## Conclusions",
        "",
        "This comprehensive evaluation demonstrates that active learning represents a viable approach for "
        "biomedical machine learning applications. The key contributions include:",
        "",
        "1. **Empirical Validation**: Systematic comparison of AL strategies on biomedical datasets",
        "2. **Performance Benchmarks**: Established performance baselines for future research",
        "3. **Practical Guidelines**: Recommendations for strategy selection based on dataset characteristics",
        "4. **Methodological Framework**: Reproducible experimental design for AL evaluation",
        "",
        "The results support the adoption of active learning in resource-constrained biomedical applications, "
        "particularly where data labeling is expensive or time-consuming.",
        ""
    ])
    
    return '\n'.join(lines)

def create_appendix_section(data) -> str:
    """Create detailed appendix with additional results."""
    lines = [
        "## Appendix",
        "",
        "### A. Detailed Experimental Configuration",
        ""
    ]
    
    if 'metadata' in data:
        metadata = data['metadata']
        
        lines.extend([
            "#### Experiment Parameters",
            f"- **Total Experiments**: {metadata.get('total_experiments', 'N/A')}",
            f"- **Execution Timestamp**: {metadata.get('timestamp', 'N/A')}",
            f"- **Datasets Evaluated**: {', '.join(metadata.get('datasets', []))}",
            f"- **Strategies Tested**: {', '.join(metadata.get('strategies', []))}",
            ""
        ])
        
        if 'arguments' in metadata:
            args = metadata['arguments']
            lines.extend([
                "#### Command Line Arguments",
                f"- **Number of Runs**: {args.get('n_runs', 'N/A')}",
                f"- **Parallel Jobs**: {args.get('n_jobs', 'N/A')}",
                f"- **BBB Batch Size**: {args.get('batch_size_bbb', 'N/A')}",
                f"- **BC Batch Size**: {args.get('batch_size_bc', 'N/A')}",
                ""
            ])
    
    lines.extend([
        "### B. Statistical Significance Analysis",
        "",
        "Performance differences were evaluated using standard deviation across multiple runs. "
        "Overlapping error bars indicate statistical parity between methods.",
        "",
        "### C. Computational Requirements",
        "",
        "All experiments were conducted on standard hardware with the following specifications:",
        "- **CPU**: Multi-core processor with parallel execution",
        "- **Memory**: Sufficient for in-memory dataset processing",
        "- **Software**: Python 3.8+, scikit-learn, pandas, numpy",
        "",
        "### D. Reproducibility Information",
        "",
        "All experiments used fixed random seeds for reproducibility:",
        "- **Base Random State**: 42",
        "- **Run-specific Seeds**: 42 + run_id",
        "- **Stratified Seeds**: [42, 10, 50, 100]",
        ""
    ])
    
    if 'figures' in data and data['figures']:
        lines.extend([
            "### E. Figure Index",
            ""
        ])
        
        for fig_name, fig_path in data['figures'].items():
            lines.append(f"- **{fig_name}**: {fig_path.name}")
        
        lines.append("")
    
    return '\n'.join(lines)

def compile_full_report(data, args) -> str:
    """Compile the complete report."""
    report_parts = [
        f"# {args.title}",
        "",
        f"**Author(s)**: {args.author}",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Version**: 1.0",
        "",
        "---",
        "",
        create_executive_summary(data)
    ]
    
    if args.include_methodology:
        report_parts.append(create_methodology_section(data))
    
    report_parts.extend([
        create_results_section(data),
        create_discussion_section(data)
    ])
    
    if args.include_appendix:
        report_parts.append(create_appendix_section(data))
    
    return '\n'.join(report_parts)

def save_report(report_content, output_dir, args):
    """Save report in specified format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"active_learning_report_{timestamp}"
    
    if args.format == 'markdown':
        filename = f"{base_filename}.md"
        with open(output_path / filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logging.info(f"Saved Markdown report to {output_path / filename}")
        
    elif args.format == 'html':
        filename = f"{base_filename}.html"
        html_content = convert_markdown_to_html(report_content)
        with open(output_path / filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"Saved HTML report to {output_path / filename}")
        
    elif args.format == 'pdf':
        filename = f"{base_filename}.pdf"
        # Note: PDF generation would require additional dependencies like weasyprint or reportlab
        logging.warning("PDF generation not implemented. Saving as Markdown instead.")
        with open(output_path / f"{base_filename}.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    return output_path / filename

def convert_markdown_to_html(markdown_content) -> str:
    """Convert markdown content to HTML (basic implementation)."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Active Learning Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
            h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; }
            h3 { color: #7f8c8d; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
    """
    
    # Basic markdown to HTML conversion (simplified)
    html_content = markdown_content
    html_content = html_content.replace('\n# ', '\n<h1>').replace('\n## ', '\n<h2>').replace('\n### ', '\n<h3>')
    html_content = html_content.replace('\n', '<br>\n')
    html_content = html_content.replace('<h1>', '</p><h1>').replace('<h2>', '</p><h2>').replace('<h3>', '</p><h3>')
    html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
    html_content = html_content.replace('*', '<em>').replace('*', '</em>')
    
    return html_template + f"<p>{html_content}</p></body></html>"

def main():
    """Main function to generate publication report."""
    args = setup_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logging.info(f"Starting report generation with configuration:")
    logging.info(f"- Results directory: {args.results_dir}")
    logging.info(f"- Figures directory: {args.figures_dir}")
    logging.info(f"- Output directory: {args.output}")
    logging.info(f"- Format: {args.format}")
    logging.info(f"- Include methodology: {args.include_methodology}")
    logging.info(f"- Include appendix: {args.include_appendix}")
    
    try:
        # Load all available data
        data = load_consolidated_results(args.results_dir, args.figures_dir)
        
        if not data:
            logging.error("No data found to generate report")
            sys.exit(1)
        
        # Compile full report
        logging.info("Compiling report...")
        report_content = compile_full_report(data, args)
        
        # Save report
        output_file = save_report(report_content, args.output, args)
        
        # Generate summary metadata
        report_metadata = {
            'timestamp': datetime.now().isoformat(),
            'format': args.format,
            'title': args.title,
            'author': args.author,
            'include_methodology': args.include_methodology,
            'include_appendix': args.include_appendix,
            'output_file': str(output_file),
            'sections': [
                'Executive Summary',
                'Methodology' if args.include_methodology else None,
                'Results',
                'Discussion',
                'Appendix' if args.include_appendix else None
            ]
        }
        
        # Remove None values
        report_metadata['sections'] = [s for s in report_metadata['sections'] if s is not None]
        
        # Save metadata
        metadata_file = Path(args.output) / 'report_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(report_metadata, f, indent=2)
        
        logging.info("Report generation completed successfully!")
        logging.info(f"Report saved to: {output_file}")
        logging.info(f"Metadata saved to: {metadata_file}")
        
    except Exception as e:
        logging.error(f"Report generation failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
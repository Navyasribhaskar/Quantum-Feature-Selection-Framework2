🧬 Quantum Feature Selection Framework
Hybrid Quantum-Classical Machine Learning for Gene Expression Analysis

A production-quality Streamlit application that reduces high-dimensional gene-expression data (thousands of genes) down to a handful of statistically significant features, encodes them into a quantum circuit, and compares a classical SVM against a quantum-enhanced SVM — all through an interactive, research-grade web interface.

The project began as a research notebook exploring hybrid quantum-classical classification on gene-expression data, and was re-architected into reusable, unit-tested Python modules (src/) with a 9-page Streamlit dashboard (dashboard/) suitable for demos, coursework, and stakeholder presentations.

📑 Table of Contents
Features
Project Structure
Installation
How to Run
Bringing Your Own Dataset
Requirements
Testing
Future Scope
Screenshots
Contributing
License
Acknowledgements
✨ Features
📤 Upload your own gene-expression CSV, or explore the bundled Golub et al. leukemia dataset
🧹 Transparent, step-by-step preprocessing (cleaning, encoding, scaling, splitting)
🎯 ANOVA F-test feature selection with adjustable top-K
⚛️ Live, inspectable quantum circuits (angle encoding + entanglement), Bloch spheres, and measurement histograms, powered by Qiskit Aer
🧠 Side-by-side Classical SVM vs. Quantum-Enhanced SVM training & evaluation
📊 Rich, interactive Plotly visualizations: confusion matrices, ROC curves, PCA projections, metric comparisons
📥 One-click exports: predictions, metrics, classification reports, selected features, PNG graphs, and a consolidated PDF report
🖥️ A polished, modern, scientific UI suitable for demos, coursework, and stakeholder presentations
📁 Project Structure
Quantum-Feature-Selection-Framework/
│
├── dashboard/
│   ├── app.py                     # Streamlit entry point
│   └── pages/
│       ├── 1_Home.py              # Overview, workflow diagram, tech stack
│       ├── 2_Dataset.py           # Upload / inspect dataset
│       ├── 3_Preprocessing.py     # Cleaning, encoding, scaling, splitting
│       ├── 4_Feature_Selection.py # ANOVA (SelectKBest) feature selection
│       ├── 5_Quantum_Analysis.py  # Quantum circuits, Bloch spheres, encoding
│       ├── 6_Classification.py    # Train & evaluate Classical + Quantum SVM
│       ├── 7_Results.py           # Confusion matrices, ROC curves, comparisons
│       ├── 8_Download.py          # Export CSVs, graphs, and PDF report
│       └── 9_About.py             # Methodology, references, future scope
│
├── src/
│   ├── data_loader.py             # Dataset ingestion & validation
│   ├── preprocessing.py           # Cleaning, encoding, scaling, splitting
│   ├── feature_selection.py       # ANOVA F-test feature selection
│   ├── quantum_pipeline.py        # Qiskit circuits, simulation, feature extraction
│   ├── classifier.py              # SVM training & evaluation
│   ├── visualization.py           # Plotly chart builders
│   ├── ui_components.py           # Shared Streamlit UI building blocks
│   ├── report_generator.py        # PDF report generation (ReportLab)
│   └── utils.py                   # Constants, session-state keys, helpers
│
├── assets/
│   └── style.css                  # Custom dashboard theme
│
├── data/
│   ├── raw/golub.csv              # Bundled Golub et al. leukemia dataset
│   └── sample/                    # Space for additional sample datasets
│
├── tests/
│   ├── test_data_loader.py        # Dataset loading & validation tests
│   ├── test_feature_selection.py  # ANOVA feature selection tests
│   └── test_utils.py              # Shared helper-function tests
│
├── docs/
│   └── screenshots/                # Dashboard screenshots for this README
│
├── outputs/                        # Exported results land here (gitignored)
├── .streamlit/config.toml          # Streamlit theme configuration
├── .github/workflows/ci.yml        # GitHub Actions: install deps + run tests
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md
🚀 Installation
Requirements: Python 3.10 or later.

# 1. Clone / unzip the project, then move into it
cd Quantum-Feature-Selection-Framework
▶️ How to Run
Launch the app from the project root (so relative paths to data/ and assets/ resolve correctly):

streamlit run dashboard/app.py
Then open the URL Streamlit prints (typically http://localhost:8501) in your browser.

Recommended walkthrough
Home — read the project overview and workflow.
Dataset — load the bundled Golub dataset (or upload your own CSV), confirm the target and metadata columns.
Preprocessing — run cleaning, encoding, scaling, and the train/test split.
Feature Selection — choose top-K and run the ANOVA F-test.
Quantum Analysis — inspect a sample's quantum circuit, then run full-batch quantum feature extraction.
Classification — train and evaluate both the Classical and Quantum-Enhanced SVMs.
Results — review confusion matrices, ROC curves, and comparison charts.
Download — export predictions, metrics, reports, graphs, and a PDF summary.
About — methodology, references, and future scope.
🗂️ Bringing Your Own Dataset
The uploaded CSV should have:

One row per sample
One column per gene/feature (numeric)
One target/label column (categorical, ≥2 classes)
Optionally, extra metadata columns (patient ID, gender, etc.) that you can exclude on the Dataset page
The app will attempt to auto-detect a sensible target column and any metadata columns matching the original Golub dataset's schema (Samples, BM.PB, Gender, Source, tissue.mf), but both are fully adjustable in the UI.

🧪 Requirements (Python packages)
See requirements.txt for exact version constraints. Key libraries:

Library	Purpose
streamlit	Interactive multi-page web application
qiskit, qiskit-aer	Quantum circuit design & local simulation
scikit-learn, scipy	Preprocessing, ANOVA feature selection, SVM classification
pandas, numpy	Data manipulation & numerical computing
plotly	Interactive charts
matplotlib, pylatexenc	Quantum circuit diagrams & Bloch spheres
kaleido	PNG export of Plotly charts
reportlab	PDF report generation
✅ Testing
The src/ layer is unit-tested with pytest, covering dataset validation, ANOVA feature selection, and shared helper functions.

pip install pytest
pytest tests/ -v
Tests run automatically on every push and pull request via the GitHub Actions workflow in .github/workflows/ci.yml.

🔮 Future Scope
Larger quantum circuits and advanced feature maps (ZZFeatureMap, PauliFeatureMap)
Quantum Kernel Estimation / Quantum SVM using an explicit quantum kernel matrix
Execution on real IBM Quantum hardware via Qiskit Runtime
Larger, multi-omics datasets for more statistically robust conclusions
Hybrid quantum neural networks trained end-to-end
See the in-app About page for full details and references.

🤝 Contributing
Contributions are welcome. See CONTRIBUTING.md for setup instructions, coding guidelines, and how to run the test suite before opening a pull request.

📄 License
This project is licensed under the MIT License.

🙏 Acknowledgements
Golub, T. R., et al. (1999) for the leukemia gene-expression dataset
The Qiskit and scikit-learn open-source communities
Streamlit for the application framework powering this dashboard

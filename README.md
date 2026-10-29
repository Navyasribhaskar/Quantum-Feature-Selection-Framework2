🧬 Quantum Feature Selection Framework
Hybrid Quantum-Classical Machine Learning for Gene Expression Analysis

A production-quality Streamlit application for high-dimensional gene-expression analysis. The framework selects statistically significant features, encodes them into a quantum circuit, and compares classical SVM with quantum-enhanced SVM through an interactive Streamlit dashboard.

📑 Table of Contents
- ✨ Features
- 📁 Project Structure
- 🚀 Installation
- ▶️ How to Run
- 🗂️ Bringing Your Own Dataset
- 🧪 Requirements
- ✅ Testing
- 🔮 Future Scope
- 🤝 Contributing
- 📄 License
- 🙏 Acknowledgements
✨ Features
- 📤 Upload a gene-expression CSV or explore the bundled Golub leukemia dataset.
- 🧹 Inspect preprocessing steps, including cleaning, encoding, scaling, and splitting.
- 🎯 Select the top-K features using the ANOVA F-test.
- ⚛️ Explore quantum circuits using angle encoding and entanglement.
- 🧠 Compare classical SVM and quantum-enhanced SVM results.
- 📊 View interactive charts, confusion matrices, ROC curves, and metric comparisons.
- 📥 Export predictions, metrics, selected features, charts, and a PDF report.
- 🖥️ Use a multi-page dashboard designed for research demos and coursework.
  
📁 Project Structure
The project code is inside the Quantum-Feature-Selection-Framework/ folder in this repository.
Quantum-Feature-Selection-Framework/
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_Home.py
│       ├── 2_Dataset.py
│       ├── 3_Preprocessing.py
│       ├── 4_Feature_Selection.py
│       ├── 5_Quantum_Analysis.py
│       ├── 6_Classification.py
│       ├── 7_Results.py
│       ├── 8_Download.py
│       └── 9_About.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── quantum_pipeline.py
│   ├── classifier.py
│   ├── visualization.py
│   ├── ui_components.py
│   ├── report_generator.py
│   └── utils.py
├── assets/
│   └── style.css
├── data/
│   ├── raw/
│   │   └── golub.csv
│   └── sample/
├── tests/
├── docs/
│   └── screenshots/
├── outputs/
├── .streamlit/
│   └── config.toml
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md

🚀 Installation
Requirements: Python 3.10 or later.
1. Clone the repository:
   git clone https://github.com/Navyasribhaskar/Quantum-Feature-Selection-Framework2.git
2. Move into the project folder:
   cd Quantum-Feature-Selection-Framework2/Quantum-Feature-Selection-Framework
3. Install dependencies:
   pip install -r requirements.txt
   
▶️ How to Run
Run the command from inside the Quantum-Feature-Selection-Framework folder so the app can resolve its data/ and assets/ paths.
streamlit run dashboard/app.py
Then open the local URL shown in the terminal, usually http://localhost:8501.
Recommended walkthrough
1. Home: Review the project overview and workflow.
2. Dataset: Load the sample dataset or upload a compatible CSV.
3. Preprocessing: Inspect cleaning, encoding, scaling, and splitting.
4. Feature Selection: Choose the number of features and run ANOVA selection.
5. Quantum Analysis: Inspect the circuit and quantum feature extraction.
6. Classification: Train and evaluate the classical and quantum-enhanced SVMs.
7. Results: Review confusion matrices, ROC curves, and comparison charts.
8. Download: Export results and reports.
9. About: Read the methodology, references, and future scope.
    
🗂️ Bringing Your Own Dataset
For best compatibility, your CSV should contain:
- One row per sample.
- Numeric columns for genes or other features.
- One target/label column with at least two classes.
- Optional metadata columns that can be excluded before training.
On the Dataset page, select the correct target column and exclude any identifier or metadata columns that should not be used as model features.

🧪 Requirements
See [`requirements.txt`](Quantum-Feature-Selection-Framework/requirements.txt) for the project's dependency list.
Library	Purpose
streamlit	Interactive web application
qiskit, qiskit-aer	Quantum circuit design and simulation
scikit-learn, scipy	Preprocessing, feature selection, and SVM classification
pandas, numpy	Data handling and numerical computing
plotly	Interactive charts
matplotlib, pylatexenc	Circuit diagrams and visualizations
kaleido	Plotly chart image export
reportlab	PDF report generation


✅ Testing
If the test suite is included in your checkout, run it from the project folder:
pip install pytest
pytest tests/ -v
The repository also includes a GitHub Actions workflow at [`ci.yml`](Quantum-Feature-Selection-Framework/.github/workflows/ci.yml) for automated checks.

🔮 Future Scope
- Explore advanced feature maps such as ZZFeatureMap and PauliFeatureMap.
- Investigate quantum kernel estimation and explicit quantum kernel matrices.
- Explore execution on IBM Quantum hardware through Qiskit Runtime.
- Evaluate larger multi-omics datasets.
- Investigate hybrid quantum neural networks.
  
🤝 Contributing
Contributions are welcome. See [`CONTRIBUTING.md`](Quantum-Feature-Selection-Framework/CONTRIBUTING.md) for contribution and testing guidance.

📄 License
This project is licensed under the MIT License. See LICENSE.

🙏 Acknowledgements
- Golub, T. R., et al. (1999), for the leukemia gene-expression dataset.
- The Qiskit and scikit-learn open-source communities.
- Streamlit for the application framework.

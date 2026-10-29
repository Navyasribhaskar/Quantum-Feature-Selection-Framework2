🧬 Quantum Feature Selection Framework
Hybrid Quantum-Classical Machine Learning for Gene Expression
Analysis

A production-quality Streamlit application that reduces high-dimensional
gene-expression data (thousands of genes) down to a handful of
statistically significant features, encodes them into a quantum circuit,
and compares a classical SVM against a quantum-enhanced SVM --- all
through an interactive, research-grade web interface.

The project began as a research notebook exploring hybrid
quantum-classical classification on gene-expression data, and was
re-architected into reusable, unit-tested Python modules (src/) with a
9-page Streamlit dashboard (dashboard/) suitable for demos,
coursework, and stakeholder presentations.
📑 Table of Contents
- Features
- Project
  Structure
- Installation
- How to
  Run
- Bringing Your Own
  Dataset
- Requirements
- Testing
- Future
  Scope
- Screenshots
- Contributing
- License
- Acknowledgements
✨ Features
- 📤 Upload your own gene-expression CSV, or explore the bundled Golub
  et al. leukemia dataset
- 🧹 Transparent, step-by-step preprocessing (cleaning, encoding,
  scaling, splitting)
- 🎯 ANOVA F-test feature selection with adjustable top-K
- ⚛️ Live, inspectable quantum circuits (angle encoding +
  entanglement), Bloch spheres, and measurement histograms, powered by
  Qiskit Aer
- 🧠 Side-by-side Classical SVM vs. Quantum-Enhanced SVM training &
  evaluation
- 📊 Rich, interactive Plotly visualizations: confusion matrices, ROC
  curves, PCA projections, metric comparisons
- 📥 One-click exports: predictions, metrics, classification reports,
  selected features, PNG graphs, and a consolidated PDF report
- 🖥️ A polished, modern, scientific UI suitable for demos, coursework,
  and stakeholder presentations

## 📁 Project Structure

The project code lives inside the `Quantum-Feature-Selection-Framework/` folder.

```text
Quantum-Feature-Selection-Framework/
├── dashboard/
│   ├── app.py
│   └── pages/
├── src/
├── assets/
├── data/
├── tests/
├── docs/
├── outputs/
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md



**Important:** The inner triple-backtick line closes the folder-tree block. The final triple-backtick line in the example closes the outer Markdown example; **don't include that final outer line in your README**. In your actual README, the `## 🚀 Installation` heading should come after the folder-tree block has been closed.

You can also use the corrected file I prepared earlier: [Download the corrected README](sandbox:/mnt/data/README_outer_fixed.md).

Once the Project Structure displays in a separate shaded code block, your README will be much cleaner. Then we'll get back to deploying your live dashboard! 🚀

🚀 Installation
Requirements: Python 3.10 or later.
# 1. Clone the repository, then enter the project folder
# git clone https://github.com/Navyasribhaskar/Quantum-Feature-Selection-Framework2.git
cd Quantum-Feature-Selection-Framework2/Quantum-Feature-Selection-Framework
▶️ How to Run
Launch the app from the project root (so relative paths to data/
and assets/ resolve correctly):
streamlit run dashboard/app.py
Then open the URL Streamlit prints (typically http://localhost:8501)
in your browser.
Recommended walkthrough
1. Home --- read the project overview and workflow.
2. Dataset --- load the bundled Golub dataset (or upload your own
   CSV), confirm the target and metadata columns.
3. Preprocessing --- run cleaning, encoding, scaling, and the
   train/test split.
4. Feature Selection --- choose top-K and run the ANOVA F-test.
5. Quantum Analysis --- inspect a sample's quantum circuit, then
   run full-batch quantum feature extraction.
6. Classification --- train and evaluate both the Classical and
   Quantum-Enhanced SVMs.
7. Results --- review confusion matrices, ROC curves, and
   comparison charts.
8. Download --- export predictions, metrics, reports, graphs, and a
   PDF summary.
9. About --- methodology, references, and future scope.

🗂️ Bringing Your Own Dataset
The uploaded CSV should have:
- One row per sample
- One column per gene/feature (numeric)
- One target/label column (categorical, ≥2 classes)
- Optionally, extra metadata columns (patient ID, gender, etc.) that
  you can exclude on the Dataset page
The app will attempt to auto-detect a sensible target column and any
metadata columns matching the original Golub dataset's schema
(Samples, BM.PB, Gender, Source, tissue.mf), but both are
fully adjustable in the UI.

🧪 Requirements (Python packages)
See
requirements.txt
for exact version constraints. Key libraries:
  Library           Purpose
  streamlit           Interactive multi-page web application

  qiskit,             Quantum circuit design & local simulation
  qiskit-aer          

  scikit-learn,       Preprocessing, ANOVA feature selection, SVM
  scipy               classification

  pandas, numpy     Data manipulation & numerical computing

  plotly              Interactive charts

  matplotlib,         Quantum circuit diagrams & Bloch spheres
  pylatexenc          

  kaleido             PNG export of Plotly charts
  reportlab           PDF report generation

✅ Testing
The src/ layer is unit-tested with pytest, covering dataset
validation, ANOVA feature selection, and shared helper functions.
pip install pytest
pytest tests/ -v
Tests run automatically on every push and pull request via the GitHub
Actions workflow in
.github/workflows/ci.yml.

🔮 Future Scope
- Larger quantum circuits and advanced feature maps (ZZFeatureMap,
  PauliFeatureMap)
- Quantum Kernel Estimation / Quantum SVM using an explicit quantum
  kernel matrix
- Execution on real IBM Quantum hardware via Qiskit Runtime
- Larger, multi-omics datasets for more statistically robust
  conclusions
- Hybrid quantum neural networks trained end-to-end
See the in-app About page for full details and references.

🤝 Contributing
Contributions are welcome. See
CONTRIBUTING.md
for setup instructions, coding guidelines, and how to run the test suite
before opening a pull request.

📄 License
This project is licensed under the MIT
License.

🙏 Acknowledgements
- Golub, T. R., et al. (1999) for the leukemia gene-expression dataset
- The Qiskit and scikit-learn open-source communities
- Streamlit for the application framework powering this dashboard

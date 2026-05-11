# Hybrid SVM-RF Framework for DDoS Detection in SDN
### VTU Major Project | KLS Vishwanathrao Deshpande Institute of Technology

---

## WHAT TO DO AFTER DOWNLOADING THIS ZIP

### ──────────────────────────────────────
### STEP 1 — Extract the ZIP
### ──────────────────────────────────────
Right-click the zip file → Extract Here (or use terminal):
```bash
unzip hybrid-svm-rf-ddos.zip -d ~/
cd ~/hybrid-svm-rf-ddos
```

---

### ──────────────────────────────────────
### STEP 2 — Install Ubuntu 22.04 LTS
### ──────────────────────────────────────
This project REQUIRES a Linux environment.

Options:
- Install Ubuntu 22.04 natively (recommended)
- Use VirtualBox / VMware with Ubuntu 22.04 ISO
- Use WSL2 on Windows (limited Mininet support)

Download Ubuntu 22.04: https://ubuntu.com/download/desktop

---

### ──────────────────────────────────────
### STEP 3 — Run the Setup Script
### ──────────────────────────────────────
Open terminal inside the project folder and run:

```bash
chmod +x setup.sh
sudo ./setup.sh
```

This script will automatically install:
- Python 3 + pip
- Mininet (network emulator)
- Ryu SDN Controller
- All Python libraries (scikit-learn, pandas, numpy, etc.)
- hping3 (attack simulation tool)
- Scapy (packet generation)

---

### ──────────────────────────────────────
### STEP 4 — Download the Datasets
### ──────────────────────────────────────
You must manually download two datasets and place them in data/raw/

1. SDN-DDoS Traffic Dataset:
   → https://data.mendeley.com/datasets/b7vw628825/1
   → Save as: data/raw/sdn_ddos.csv

2. NSL-KDD Dataset:
   → https://www.unb.ca/cic/datasets/nsl.html
   → Download KDDTrain+.txt and KDDTest+.txt
   → Save both in: data/raw/

---

### ──────────────────────────────────────
### STEP 5 — Preprocess the Data
### ──────────────────────────────────────
```bash
cd ml_model
python3 preprocess.py
```
This cleans the data, encodes labels, and saves processed files to data/processed/

---

### ──────────────────────────────────────
### STEP 6 — Train the Hybrid Model
### ──────────────────────────────────────
```bash
python3 train_model.py
```
This will:
- Run Random Forest feature selection
- Train SVM and RF classifiers
- Save models to ml_model/saved_model/
- Print accuracy and F1-score

Expected output: Accuracy > 99%

---

### ──────────────────────────────────────
### STEP 7 — Evaluate and See Results
### ──────────────────────────────────────
```bash
python3 evaluate.py
```
Generates:
- Confusion matrix → saved to results/plots/
- Classification report (precision, recall, F1)
- ROC curve

---

### ──────────────────────────────────────
### STEP 8 — Start the SDN Network
### ──────────────────────────────────────
Open TWO terminals:

Terminal 1 — Start Ryu Controller:
```bash
cd ryu_controller
ryu-manager ddos_controller.py
```

Terminal 2 — Start Mininet Topology:
```bash
cd mininet_topology
sudo python3 topology.py
```

---

### ──────────────────────────────────────
### STEP 9 — Simulate a DDoS Attack
### ──────────────────────────────────────
Inside the Mininet CLI (after topology starts), run:
```
mininet> attacker hping3 -S --flood -V -p 80 10.0.0.10
```

Watch Terminal 1 — you should see:
[ALERT] DDoS detected! Flow blocked.

---

### ──────────────────────────────────────
### STEP 10 — View All Results
### ──────────────────────────────────────
All graphs and plots are saved in: results/plots/
- confusion_matrix.png
- roc_curve.png
- feature_importance.png
- accuracy_comparison.png

---

## Project Structure
```
hybrid-svm-rf-ddos/
├── data/
│   ├── raw/                  ← Put downloaded datasets here
│   └── processed/            ← Auto-generated after preprocess.py
├── mininet_topology/
│   └── topology.py           ← SDN network with hosts + attacker
├── ryu_controller/
│   └── ddos_controller.py    ← Main SDN controller with ML integration
├── ml_model/
│   ├── preprocess.py         ← Data cleaning and normalization
│   ├── feature_selection.py  ← RF-based feature importance
│   ├── train_model.py        ← Trains SVM + RF hybrid model
│   ├── evaluate.py           ← Metrics, plots, confusion matrix
│   └── saved_model/          ← Trained model files saved here
├── detection/
│   └── hybrid_detector.py    ← Hybrid SVM+RF prediction logic
├── mitigation/
│   └── mitigation_engine.py  ← Installs flow rules to block attacks
├── alerts/
│   └── alert_system.py       ← Email/SMS alert on attack detection
├── results/
│   └── plots/                ← All generated graphs saved here
├── setup.sh                  ← Auto-installs all dependencies
└── README.md                 ← This file
```

---

## Team Members
- Mr. Sumukh Chavan (2VD23CS076)
- Ms. Supriya Kammar (2VD23CS077)
- Ms. Vidyashree Patil (2VD23CS088)
- Mr. Vishal Magwadkar (2VD23CS090)

Guide: Dr. Venkatesh Shankar
Institution: KLS VDIT, Haliyal | VTU Belagavi
Academic Year: 2025-26

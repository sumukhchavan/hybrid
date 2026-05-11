#!/bin/bash
# ============================================================
# Hybrid SVM-RF DDoS Detection - Auto Setup Script
# Run with: sudo ./setup.sh
# ============================================================

echo "=============================================="
echo " Hybrid SVM-RF DDoS Detection - Setup Script"
echo "=============================================="
echo ""

# Update system
echo "[1/7] Updating system packages..."
apt update -y && apt upgrade -y

# Install system tools
echo "[2/7] Installing system tools..."
apt install -y python3 python3-pip git curl net-tools hping3 openvswitch-switch

# Install Mininet
echo "[3/7] Installing Mininet..."
apt install -y mininet
echo "Mininet version: $(mn --version 2>&1)"

# Install Python libraries
echo "[4/7] Installing Python ML libraries..."
pip3 install scikit-learn pandas numpy matplotlib seaborn joblib scapy flask

# Install Ryu SDN Controller
echo "[5/7] Installing Ryu SDN Controller..."
pip3 install ryu
echo "Ryu version: $(ryu-manager --version 2>&1)"

# Create placeholder in data/raw
echo "[6/7] Setting up data directories..."
mkdir -p data/raw data/processed ml_model/saved_model results/plots

# Make all Python files executable
echo "[7/7] Finalizing permissions..."
chmod +x mininet_topology/topology.py
chmod +x ryu_controller/ddos_controller.py

echo ""
echo "=============================================="
echo " Setup COMPLETE!"
echo "=============================================="
echo ""
echo " NEXT STEP: Download datasets manually."
echo " See README.md Step 4 for download links."
echo ""
echo " Then run: cd ml_model && python3 train_model.py"
echo "=============================================="

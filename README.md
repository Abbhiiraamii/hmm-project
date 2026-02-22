# HMM Project – Baum-Welch Visualization

## Student Information
Name: Abhirami S,
Registration Number: TCR24CS003

---

## Description

This project implements the Baum-Welch algorithm to train a Hidden Markov Model (HMM) using Python and Flask.

The example uses a two-state coin toss model:
- Fair
- Biased

The application:
- Runs Forward and Backward algorithms
- Performs Baum-Welch parameter updates
- Tracks log-likelihood across iterations
- Visualizes state transitions in a web interface

---

## Requirements

Install Python packages:

pip install flask numpy state-transition-diagrams graphviz

---

## Run the Project

python app.py

Open in browser:
http://127.0.0.1:7000

---

## Note

Click on the state diagram to inspect transition and emission matrices.  
The final iteration shows the trained model parameters.
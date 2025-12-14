# MalariaSpecNet

MalariaSpecNet is a CNN-based malaria parasite classification software for automated recognition of different Plasmodium species and uninfected blood smear images.  
This project was developed by **Group 1** and includes both a graphical user interface (GUI) and reusable Python scripts for inference and model training.

---

## Set up environment

```bash
# 1. Create and activate conda environment (Python 3.10)
conda create -n malariaspecnet python=3.10
conda activate malariaspecnet

# 2. Move to project folder
cd MalariaSpecNet

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Use

### Option 1 · Terminal (Command Line)

```bash
# General usage:
python predict.py [-c CHECKPOINT] -i INPUT_FOLDER [-o OUTPUT_CSV]

# Example:
# python predict.py -i ../Test -o ../Test_predictions_result.csv

```

### Option 2 · GUI (Graphical User Interface)
```bash
python GUI.py
```

## Project Structure

```text
.
├── MalariaSpecNet/               # Core application directory
│   ├── GUI.py                    # Main script to launch the GUI
│   ├── train.py                  # Script for model training
│   ├── malaria_predict.py        # Inference logic and utility functions
│   ├── model.ckpt                # Pre-trained model checkpoint
│   └── requirements.txt          # Python dependencies
├── Test                          # Example images for quick testing
├── Documentation.pdf             # Instruction of MalariaSpecNet
└── README.md
```
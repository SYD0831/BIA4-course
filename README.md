# MalariaSpecNet

MalariaSpecNet is a CNN-based malaria parasite classification software for automated recognition of different Plasmodium species and uninfected blood smear images.  
This project was developed by **Group 1** and includes both a graphical user interface (GUI) and reusable Python scripts for inference and model training.

---

## Set up environment

```bash
# Move to project folder
cd MalariaSpecNet

# Install dependencies
pip install -r requirements.txt
```

## How to Use

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
├── Test/                         # Example images for quick testing
├── Malaria_Species...Documentation.pdf  # Detailed project report
└── README.md                     # Project documentation
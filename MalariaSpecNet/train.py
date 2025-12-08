import os
import sys
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# 1. Environment and Path Setup
# -----------------------------------------------------------------------------
# Add current directory to system path to ensure 'data' and 'model' modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import custom modules (Ensure 'data/' and 'model/' folders exist in current directory)
try:
    from data import DInterface
    from model.standard_net import StandardNetLightning
except ImportError as e:
    print("Error: Could not import project modules. Please ensure 'data' and 'model' folders are in the current directory.")
    raise e

def main():
    # -----------------------------------------------------------------------------
    # 2. Argument Parsing
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train Malaria Species Identification Model")
    
    # Add argument for data directory
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='data/MPIDB', 
        help='Path to the dataset root directory (default: data/MPIDB)'
    )
    
    # You can add more arguments here (e.g., batch_size, epochs) if needed
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs for training')

    args = parser.parse_args()

    # Check if data path exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return

    # -----------------------------------------------------------------------------
    # 3. Initialization
    # -----------------------------------------------------------------------------
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    print(f"Working Directory: {os.getcwd()}")
    print(f"Data Directory: {args.data_dir}")
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Accelerator used: {accelerator}")

    # -----------------------------------------------------------------------------
    # 4. Data Module (DInterface)
    # -----------------------------------------------------------------------------
    # Initialize DataModule using the argument path
    dm = DInterface(
        num_workers=4,
        dataset='mpidb_dataset',
        batch_size=args.batch_size,
        root=args.data_dir,  # Use the path from arguments
        classes=['falciparum', 'vivax', 'ovale', 'negative'],
        img_size=100,
        aug=True,
    )

    # -----------------------------------------------------------------------------
    # 5. Model Module
    # -----------------------------------------------------------------------------
    # Initialize the Lightning Model
    model = StandardNetLightning(
        in_channels=7,
        num_classes=4,
        lr=5e-4,
        weight_decay=1e-4,
        dropout=0.3,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready. Total parameters: {total_params/1e6:.3f} M")

    # -----------------------------------------------------------------------------
    # 6. Callbacks & Logger
    # -----------------------------------------------------------------------------
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="mpidb_7ch_cnn",
    )

    # Save the best model based on minimum validation loss
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="mpidb-7ch-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        save_weights_only=False,
    )

    # Early Stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8,
        verbose=True,
    )

    # Monitor Learning Rate changes
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -----------------------------------------------------------------------------
    # 7. Trainer & Training Loop
    # -----------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=[ckpt_callback, early_stop, lr_monitor],
        log_every_n_steps=1,
        deterministic=True,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    print("Training finished.")

    # -----------------------------------------------------------------------------
    # 8. Testing
    # -----------------------------------------------------------------------------
    print("Testing with the best model...")
    # Automatically load the best checkpoint from training for testing
    test_results = trainer.test(datamodule=dm, ckpt_path='best')
    print("Test Results:", test_results)

    # -----------------------------------------------------------------------------
    # 9. Detailed Evaluation (Confusion Matrix & Report)
    # -----------------------------------------------------------------------------
    print("\nGenerating detailed evaluation report...")
    
    # Retrieve best model path from the callback
    best_model_path = ckpt_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    
    if best_model_path:
        # Load the best model
        best_model = StandardNetLightning.load_from_checkpoint(best_model_path)
        best_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model.to(device)

        all_preds = []
        all_labels = []

        # Ensure the datamodule is set up for the test stage
        dm.setup(stage='test') 
        
        # Manual inference loop to gather all predictions
        with torch.no_grad():
            for xb, yb in dm.test_dataloader():
                xb = xb.to(device)
                yb = yb.to(device)
                logits = best_model(xb)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        class_names = ['falciparum', 'vivax', 'ovale', 'negative']

        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    else:
        print("Best model path not found. Skipping detailed evaluation.")

if __name__ == "__main__":
    main()

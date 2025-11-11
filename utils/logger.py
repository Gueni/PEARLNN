import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name='pearlnn', log_level=logging.INFO, log_file=None):
    """Setup logger for PEARLNN"""
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name='pearlnn'):
    """Get existing logger or create new one"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        setup_logger(name)
    
    return logger

class ProgressLogger:
    """Progress logging with rich formatting"""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        
    def update(self, step=1, message=""):
        """Update progress"""
        self.current_step += step
        progress = self.current_step / self.total_steps
        percent = progress * 100
        
        if message:
            print(f"\r{self.description}: {percent:.1f}% - {message}", end="", flush=True)
        else:
            print(f"\r{self.description}: {percent:.1f}%", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete

class TrainingLogger:
    """Specialized logger for training progress"""
    
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch_losses = []
        
    def log_epoch(self, epoch, train_loss, val_loss=None, lr=None):
        """Log training progress for an epoch"""
        self.epoch_losses.append((train_loss, val_loss))
        
        if val_loss is not None:
            if lr is not None:
                print(f"Epoch {epoch:4d}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.2e}")
            else:
                print(f"Epoch {epoch:4d}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            if lr is not None:
                print(f"Epoch {epoch:4d}/{self.epochs} | Train Loss: {train_loss:.6f} | LR: {lr:.2e}")
            else:
                print(f"Epoch {epoch:4d}/{self.epochs} | Train Loss: {train_loss:.6f}")
    
    def get_best_epoch(self):
        """Get epoch with best validation loss"""
        if not any(val_loss is not None for _, val_loss in self.epoch_losses):
            return 0, self.epoch_losses[0][0]
        
        best_epoch = 0
        best_loss = float('inf')
        
        for epoch, (train_loss, val_loss) in enumerate(self.epoch_losses):
            loss = val_loss if val_loss is not None else train_loss
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
        
        return best_epoch, best_loss
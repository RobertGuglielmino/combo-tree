
from datetime import timedelta
import sys
import time
import psutil
import torch


class ConsoleProgress:
    def __init__(self, update_interval=0.5):
        self.start_time = None
        self.last_update = time.time()

        self.total_samples = 0
        self.batch_size = 0
        self.n_batches = 1
        self.line_count = 0
        self.update_interval = update_interval  # Update interval in seconds

        self.epochs = 0
        self.current_loss = 0.0
        self.epoch_loss = 0.0
        
        
    def print_header(self, text):
        """Print section header"""
        sys.stdout.write('\n')  # Add blank line
        sys.stdout.write("="*50 + '\n')
        sys.stdout.write(f"⚡ {text}\n")
        sys.stdout.write("="*50 + '\n')
        sys.stdout.flush()
    
    def print_configuration(self):
        print(f"📊 Training Configuration:")
        print(f"  • Samples: {self.total_samples:,}")
        print(f"  • Batch Size: {self.batch_size}")
        print(f"  • Epochs: {self.epochs}")
        print("\n")  # Extra space for dynamic updates
        status_lines = 4 
        sys.stdout.write('\n' * status_lines)
        self.clear_lines(status_lines)

    def reset_epoch_stats(self):
        """Reset accumulated stats for new epoch"""
        self.epoch_loss = 0
        self.n_batches = 0
        self.start_time = time.time()



    def update_status(self, loss, curr_count, curr_epoch):
        # Calculate progress stats
        self.epoch_loss += loss.item()
        self.n_batches += 1
        
        batch_size_actual = min(self.batch_size, self.total_samples - curr_count)
        progress = (curr_count + batch_size_actual) / self.total_samples * 100
        
        elapsed = time.time() - self.start_time
        eta = (elapsed / (curr_count + batch_size_actual)) * (self.total_samples - (curr_count + batch_size_actual))
        
        ram = psutil.Process().memory_info().rss / 1024 / 1024
        gpu_mem = f", GPU: {torch.cuda.memory_allocated()/1024**2:.1f}MB" if torch.cuda.is_available() else ""

        
        if not self.should_update():
            return
        
        # Move cursor up 4 lines and clear them
        sys.stdout.write('\033[4A\033[K')
        
        
        # Print all lines at once
        status = [
            f"🔄 Epoch: {curr_epoch+1}/{self.epochs}",
            f"📈 Progress: [{progress:6.2f}%] [{curr_count+batch_size_actual:,}/{self.total_samples//self.batch_size:,}]",
            f"⏱️  Time: {elapsed:.1f}s elapsed, {eta:.1f}s remaining",
            f"💻 RAM: {ram:.1f}MB{gpu_mem} | Loss: {loss:.4f} | Avg Loss: {self.epoch_loss/self.n_batches:.4f} | Batch: {self.n_batches}/{self.total_samples//self.batch_size}"
        ]
        
        print('\n'.join(status))
        sys.stdout.flush()


           
    def should_update(self):
        """Check if enough time has passed for an update"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False 
    
    
    def format_time(self, seconds):
        return str(timedelta(seconds=int(seconds)))

    def update_progress(self, current, total, start_time):
        elapsed_time = time.time() - start_time
        if current > 0:
            estimated_total_time = elapsed_time * total / current
            estimated_remaining = estimated_total_time - elapsed_time
            eta = self.format_time(estimated_remaining)
        else:
            eta = "calculating..."

        # Clear the current line and move cursor to beginning
        sys.stdout.write('\033[K')
        progress = f"\rProcessing: {current}/{total} files ({(current/total*100):.1f}%) | "
        progress += f"Elapsed: {self.format_time(elapsed_time)} | ETA: {eta}"
        sys.stdout.write(progress)
        sys.stdout.flush()
                

    # FACTORY FUNCTIONS

    def set_total_samples(self, int): 
        self.total_samples = int
        return self
    
    def set_batch_size(self, int):
        self.batch_size = int
        return self
    
    def set_start_time(self, int):
        self.start_time = int
        return self
    
    def set_epochs(self, int):
        self.epochs = int
        return self
    

    # HELPERS
    def clear_lines(self, n=4):
        """Clear n lines up"""
        for i in range(n):
            sys.stdout.write('\033[F')  # Move cursor up
            sys.stdout.write('\033[K')  # Clear line
        sys.stdout.flush()

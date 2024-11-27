from datetime import timedelta
import sys
import time

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def update_progress(current, total, start_time):
    elapsed_time = time.time() - start_time
    if current > 0:
        estimated_total_time = elapsed_time * total / current
        estimated_remaining = estimated_total_time - elapsed_time
        eta = format_time(estimated_remaining)
    else:
        eta = "calculating..."

    # Clear the current line and move cursor to beginning
    sys.stdout.write('\033[K')
    progress = f"\rProcessing: {current}/{total} files ({(current/total*100):.1f}%) | "
    progress += f"Elapsed: {format_time(elapsed_time)} | ETA: {eta}"
    sys.stdout.write(progress)
    sys.stdout.flush()
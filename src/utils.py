from pathlib import Path
import matplotlib.pyplot as plt

def save_plot(base_dir, filename):
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300)
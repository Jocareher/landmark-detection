from __future__ import annotations

from pathlib import Path

import torch


def save_model_summary(
    model: torch.nn.Module,
    output_dir: Path,
    input_size: tuple[int, int, int, int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "model_summary.txt"

    try:
        from torchinfo import summary

        summary_text = str(
            summary(
                model,
                input_size=input_size,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                row_settings=["var_names"],
                col_width=20,
                depth=3,
                verbose=0,
                device="cpu",
            )
        )
    except Exception as error:
        summary_text = f"torchinfo summary unavailable: {error}\n"

    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(f"[INFO] Saved model summary to {summary_path}")
    return summary_path

"""Run the released all-atom ESMFold2 model from a JSON-safe input."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-loops", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=200)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm-dropout", type=float, default=0.3)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--precision", choices=("default", "bf16", "fp32"), default="default"
    )
    return parser.parse_args()


def _confidence(result: Any, filename: str) -> dict[str, Any]:
    return {
        "file": filename,
        "mean_plddt": (
            float(result.plddt.float().mean().item())
            if result.plddt is not None
            else None
        ),
        "ptm": float(result.ptm) if result.ptm is not None else None,
        "iptm": float(result.iptm) if result.iptm is not None else None,
    }


def main() -> None:
    args = _arguments()

    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model
    from esm.models.hub import read_safetensors_dir
    from esm.utils.structure.input_builder import (
        deserialize_structure_prediction_input,
    )

    if "dtype" not in inspect.signature(read_safetensors_dir).parameters:
        raise RuntimeError(
            "ESMFold2's low-memory checkpoint loader is not installed. "
            "Reinstall it with `python install_tools.py esmfold2`."
        )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")

    model_options: dict[str, Any] = {"device": device}
    if args.precision != "default":
        model_options["dtype"] = (
            torch.bfloat16 if args.precision == "bf16" else torch.float32
        )
        model_options["esmc_precision"] = args.precision

    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", **model_options).eval()
    model.set_chunk_size(args.chunk_size or None)

    document = json.loads(args.input.read_text(encoding="utf-8"))
    prediction_input = deserialize_structure_prediction_input(document)
    with torch.inference_mode():
        folded = ESMFold2InputBuilder().fold(
            model,
            prediction_input,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            num_diffusion_samples=args.num_diffusion_samples,
            seed=args.seed,
            lm_dropout=args.lm_dropout,
            complex_id=args.name,
        )

    results = folded if isinstance(folded, list) else [folded]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confidence = []
    for index, result in enumerate(results, start=1):
        suffix = f"_sample_{index}" if len(results) > 1 else ""
        filename = f"{args.name}{suffix}.cif"
        (args.output_dir / filename).write_text(
            result.complex.to_mmcif(), encoding="utf-8"
        )
        confidence.append(_confidence(result, filename))

    (args.output_dir / "confidence.json").write_text(
        json.dumps(confidence, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(results)} ESMFold2 prediction(s) to {args.output_dir}")


if __name__ == "__main__":
    main()

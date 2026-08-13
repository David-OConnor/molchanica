#!/usr/bin/env python3
"""Convert a ProteinMPNN checkpoint for Molchanica's native ΔΔG scanner.

Molchanica implements the ProteinMPNN network in Rust (see `src/therapeutic_misc/ddg/mpnn.rs`) so a
saturation-mutagenesis scan needs no Python at run time. It still needs the published weights, and
a PyTorch `.pt` is a zip of pickled objects — reading that in Rust would mean writing a pickle
interpreter and pointing it at a downloaded file, which is both a lot of work and a poor idea. So
the checkpoint is converted once, here, in the environment that already has Torch because
`install_tool proteinmpnn` built it.

Two things are written, into one file:

1. Every tensor of the `state_dict`, under its original key. Keeping upstream's names means a
   checkpoint whose layout has changed fails on a named missing key rather than silently loading
   into the wrong slots.
2. A reference forward pass: a fixed synthetic backbone, and the log-probabilities upstream's own
   code produces for it. `molchanica --verify-mpnn` replays that through the Rust implementation
   and reports the largest disagreement, which is how the port is checked against the original
   rather than merely assumed to match.

Usage:

    python scripts/convert_mpnn_weights.py \\
        --checkpoint <ProteinMPNN>/vanilla_model_weights/v_48_020.pt \\
        --output     <ProteinMPNN>/converted/v_48_020.mcnn \\
        --repo       <ProteinMPNN>

`--repo` is optional: without it the tensors are still converted, but the reference pass is
skipped, since generating it means importing upstream's model code.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MAGIC = b"MCNN"
FORMAT_VERSION = 1

# Deliberately small, and deliberately not a real protein: the point is a fixed input both
# implementations can agree on, not a biologically meaningful structure. 24 residues is enough that
# the k-nearest-neighbour graph is not simply "everything", which is what would hide an indexing
# bug in the neighbour gather.
REFERENCE_LENGTH = 24


def build_reference_backbone(length: int = REFERENCE_LENGTH):
    """A deterministic synthetic backbone, as two chains so the cross-chain positional bucket is
    exercised too."""

    import numpy as np

    coords = np.zeros((length, 4, 3), dtype=np.float32)
    for index in range(length):
        angle = index * 100.0 * np.pi / 180.0
        rise = index * 1.5
        ca = np.array([2.3 * np.cos(angle), 2.3 * np.sin(angle), rise], dtype=np.float32)
        coords[index, 0] = ca + np.array([-1.0, 0.2, -0.5], dtype=np.float32)  # N
        coords[index, 1] = ca                                                   # CA
        coords[index, 2] = ca + np.array([0.9, 0.9, 0.4], dtype=np.float32)     # C
        coords[index, 3] = ca + np.array([1.2, 2.0, 0.4], dtype=np.float32)     # O

    # Two chains, and a numbering gap inside the first, so the positional encoding sees a
    # discontinuity as well as a chain break.
    residue_idx = np.arange(length, dtype=np.int32)
    residue_idx[length // 2 :] += 40
    chain_idx = np.zeros(length, dtype=np.int32)
    chain_idx[length // 2 :] = 1
    return coords, residue_idx, chain_idx


def reference_log_probs(checkpoint_path: Path, repo: Path):
    """Run upstream ProteinMPNN's own `unconditional_probs` on the reference backbone."""

    import numpy as np
    import torch

    sys.path.insert(0, str(repo))
    try:
        from protein_mpnn_utils import ProteinMPNN
    except ImportError as error:  # pragma: no cover - depends on the checkout
        raise SystemExit(
            f"Could not import ProteinMPNN from {repo}: {error}\n"
            "Point --repo at the ProteinMPNN checkout, or omit it to skip the reference pass."
        ) from error

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_dim = 128
    num_layers = 3
    model = ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    coords, residue_idx, chain_idx = build_reference_backbone()
    X = torch.from_numpy(coords).unsqueeze(0)
    mask = torch.ones(1, coords.shape[0])
    residue = torch.from_numpy(residue_idx).unsqueeze(0).long()
    chain = torch.from_numpy(chain_idx).unsqueeze(0).long()

    with torch.no_grad():
        log_probs = model.unconditional_probs(X, mask, residue, chain)

    return coords, residue_idx, chain_idx, log_probs[0].numpy().astype(np.float32)


def write_tensor(handle, name: str, array) -> None:
    import numpy as np

    array = np.ascontiguousarray(array, dtype=np.float32)
    encoded = name.encode("utf-8")
    handle.write(struct.pack("<I", len(encoded)))
    handle.write(encoded)
    handle.write(struct.pack("<I", array.ndim))
    for dimension in array.shape:
        handle.write(struct.pack("<I", dimension))
    handle.write(array.tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="A ProteinMPNN .pt file.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the .mcnn file.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="The ProteinMPNN checkout, used to generate the reference forward pass.",
    )
    arguments = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
        import torch
    except ImportError as error:
        raise SystemExit(
            f"{error}. Run this with the interpreter install_tool built for ProteinMPNN, e.g.\n"
            "  <data dir>/molchanica/proteinmpnn-venv/bin/python scripts/convert_mpnn_weights.py ..."
        ) from error

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)

    tensors = {name: value.detach().cpu().numpy() for name, value in state.items()}

    reference = {}
    if arguments.repo is not None:
        coords, residue_idx, chain_idx, log_probs = reference_log_probs(
            arguments.checkpoint, arguments.repo
        )
        reference = {
            "reference.N": coords[:, 0],
            "reference.CA": coords[:, 1],
            "reference.C": coords[:, 2],
            "reference.O": coords[:, 3],
            # Written as floats like everything else; the reader casts them back to integers, which
            # is exact for values this small and keeps the file format to a single element type.
            "reference.residue_idx": residue_idx.astype("float32"),
            "reference.chain_idx": chain_idx.astype("float32"),
            "reference.log_probs": log_probs,
        }
        print(f"Recorded a reference pass over {coords.shape[0]} residues.")
    else:
        print("No --repo given; skipping the reference pass (verification will be unavailable).")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.output, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", FORMAT_VERSION))
        handle.write(struct.pack("<I", len(tensors) + len(reference)))
        for name, array in tensors.items():
            write_tensor(handle, name, array)
        for name, array in reference.items():
            write_tensor(handle, name, array)

    size_mb = arguments.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(tensors)} tensors to {arguments.output} ({size_mb:.1f} MB).")
    if reference:
        print("Check the port with:  molchanica --verify-mpnn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

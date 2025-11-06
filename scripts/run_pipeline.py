"""
Pipeline runner CLI.
- Imports modular components without changing computation logic.
- Configure subjects, paths, K, split, and folder/file templates via CLI.
"""

import argparse
from STF_CA.framework.runner import PipelineRunner


def parse_args():
    p = argparse.ArgumentParser(description="Run EEG φ (phi) pipeline with windowed FFT backend.")
    p.add_argument("--input_base_dir", type=str, default="./data",
                   help="Base directory containing subject folders (default: ./data)")
    p.add_argument("--save_base_dir", type=str, default="./outputs",
                   help="Directory where results will be written (default: ./outputs)")
    p.add_argument("--subjects", type=int, nargs="+", required=True,
                   help="List of subject IDs, e.g., --subjects 1 2 3")
    p.add_argument("--M", type=int, default=16, help="Number of cells (default: 16)")
    p.add_argument("--K", type=int, default=10, help="Coalitions per cell (default: 10)")
    p.add_argument("--seed", type=int, default=23, help="Random seed (default: 23)")
    p.add_argument("--split", type=str, default="test", choices=["train", "test", "both"],
                   help="Which split(s) to process (default: test)")
    p.add_argument("--subject_dir_template", type=str, default="subject_{id}_results_0_7",
                   help="Subject folder name pattern (default: subject_{id}_results_0_7)")
    p.add_argument("--model_filename_template", type=str, default="model_subject_{id}.h5",
                   help="Model filename pattern (default: model_subject_{id}.h5)")
    return p.parse_args()


def main():
    args = parse_args()
    runner = PipelineRunner(
        input_base_dir=args.input_base_dir,
        save_base_dir=args.save_base_dir,
        M=args.M,
        K=args.K,
        seed=args.seed,
        split=args.split,
        subject_dir_template=args.subject_dir_template,
        model_filename_template=args.model_filename_template,
    )
    runner.run_on_subjects(subject_list=args.subjects)


if __name__ == "__main__":
    main()

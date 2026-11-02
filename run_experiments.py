import argparse
from src.experiments import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PneumoniaMNIST ViT and FedAvg experiments")
    parser.add_argument(
        "--experiment",
        choices=["all", "primary", "matched", "repeats"],
        default="all",
        help="Which notebook experiment track to run",
    )
    args = parser.parse_args()
    run_all(args.experiment)

from scripts.full_sim import do_full_simulation
import sys
from scripts.config import get_config

def main():
    config = get_config()

    do_full_simulation(config.dataset_name, sample_sizes=config.sample_sizes, redundancy=config.redundancy, columns_to_use=config.columns_to_use, weights = config.weights)
if __name__ == "__main__":
    main()
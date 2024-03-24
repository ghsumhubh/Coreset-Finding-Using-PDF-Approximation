from scripts.full_sim import do_full_simulation
import sys

def main():
    dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_name += " "+sys.argv[2]
    REDUNDANCY = 40
    SAMPLE_SIZES = [10,50, 100,150,200,250,500]
    #REDUNDANCY = 40
    #SAMPLE_SIZES = [10]
    do_full_simulation(dataset_name, sample_sizes=SAMPLE_SIZES, redundancy=REDUNDANCY)

if __name__ == "__main__":
    main()
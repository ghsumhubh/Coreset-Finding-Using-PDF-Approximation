import argparse

ALL_DATASET_NAMES = [
    'Abalone',
    'Insurance',
    'Melbourne Housing',
    'Seoul Bike',
    'Sleep Efficiency',
    'Wine Quality'
]


dataset_names_to_sizes = {
    'Abalone': [50, 100, 150, 200, 250, 500],
    'Insurance': [10,25,50,100,150],
    'Melbourne Housing': [50, 100, 150, 200, 250, 500],
    'Seoul Bike': [50, 100, 150, 200, 250, 500],
    'Sleep Efficiency': [10,25,50,100,150],
    'Wine Quality': [100, 150, 200, 250,350, 500],
}

def get_config():
    parser = argparse.ArgumentParser(description='Script to run simulations on different datasets. Choose a specific dataset or "All" to run on all available datasets.')

    
    parser.add_argument('--dataset_name', type=str, required=True, choices=ALL_DATASET_NAMES,
                        help=f"Name of the dataset to run the simulation on. Choices: {', '.join(ALL_DATASET_NAMES)}.")
 
    parser.add_argument('--redundancy', type=int, default=20, help='Number of times to sample the data')
    #parser.add_argument('--sample_sizes', type=int, nargs='+', default=[50, 100, 150, 200, 250, 500], help='Sizes of the samples to take')

    args =  parser.parse_args()

    args.sample_sizes = dataset_names_to_sizes[args.dataset_name] 

    return args


def get_sample_sizes_for_dataset(dataset_name):
    return dataset_names_to_sizes[dataset_name]

def get_all_dataset_names():
    return ALL_DATASET_NAMES
import argparse

ALL_DATASET_NAMES = [
    'Abalone',
    'Insurance',
    'Melbourne Housing',
    'Seoul Bike',
    'Sleep Efficiency',
    'Wine Quality'
]

def get_config():
    parser = argparse.ArgumentParser(description='Script to run simulations on different datasets. Choose a specific dataset or "All" to run on all available datasets.')

    # Add 'ALL' to the list of choices for dataset_name
    valid_choices = ALL_DATASET_NAMES + ['All']
    
    parser.add_argument('--dataset_name', type=str, required=True, choices=valid_choices,
                        help=f"Name of the dataset to run the simulation on. Choices: {', '.join(valid_choices)}.")

    parser.add_argument('--redundancy', type=int, default=40, help='Number of times to sample the data')
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[50, 100, 150, 200, 250, 500], help='Sizes of the samples to take')

    return parser.parse_args()
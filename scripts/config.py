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

# Uses the top 4 feautres with the highest absolute pearson correlation 
dataset_to_pearson_features_weight= {
    'Abalone': [1,1,1,0,0,0,0,1,0,0,1],
    'Insurance': [1,0,1,1,1,0,0,0,0,1],
    'Melbourne Housing': [1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    'Seoul Bike': [1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
    'Sleep Efficiency': [0,0,0,0,0,1,1,1,0,1,0,1],
    'Wine Quality': [0,0,1,0,0,1,0,0,1,0,0,1,1],
}

dataset_to_shap_features_weight= {
    'Abalone': [0,1,0,1,1,0,1,0,0,0,1],
    'Insurance': [1,0,1,1,1,0,0,0,0,1],
    'Melbourne Housing': [0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    'Seoul Bike': [1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1],
    'Sleep Efficiency': [0,0,0,0,0,1,1,1,0,0,1,1],
    'Wine Quality': [0,0,1,0,0,0,1,1,0,0,0,1,1],
}





def get_config():
    parser = argparse.ArgumentParser(description='Script to run simulations on different datasets. Choose a specific dataset or "All" to run on all available datasets.')

    
    parser.add_argument('--dataset_name', type=str, required=True, choices=ALL_DATASET_NAMES,
                        help=f"Name of the dataset to run the simulation on. Choices: {', '.join(ALL_DATASET_NAMES)}.")
 
    parser.add_argument('--redundancy', type=int, default=20, help='Number of times to sample the data')
    #parser.add_argument('--sample_sizes', type=int, nargs='+', default=[50, 100, 150, 200, 250, 500], help='Sizes of the samples to take')

    parser.add_argument('--columns_to_use', type=str , required=True, help='ALL for all columns, FEATURES for only features', choices=['ALL', 'FEATURES'])
    # Add weight type : options are pearson, shap
    
    parser.add_argument('--weight_type', type=str , required=False, help='Type of weight to use for the features', choices=['pearson', 'shap'], default=None)

    args =  parser.parse_args()

    args.sample_sizes = dataset_names_to_sizes[args.dataset_name] 

    if args.weight_type == None:
        args.weights = None
    elif args.weight_type == 'pearson':
        args.weights = dataset_to_pearson_features_weight[args.dataset_name]
    elif args.weight_type == 'shap':
        args.weights = dataset_to_shap_features_weight[args.dataset_name]
    else:
        raise ValueError(f"Invalid weight type: {args.weight_type}")

    return args


def get_sample_sizes_for_dataset(dataset_name):
    return dataset_names_to_sizes[dataset_name]

def get_all_dataset_names():
    return ALL_DATASET_NAMES


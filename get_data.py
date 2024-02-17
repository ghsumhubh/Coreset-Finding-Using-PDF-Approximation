from ucimlrepo import fetch_ucirepo 
from utils import make_categorical_into_onehot

UCI_DATASETS_TO_ID = {
        'Abalone' : 1,
        'Parkinsons Telemonitoring': 189,
        'AIDS Clinical Trials Group Study 175': 890,
        'SUPPORT2': 880,
        'Infared Thermography Temperature': 925
    }

UCI_DATASETS_TO_CATEGORICAL_FEATURES= {
        'Abalone' : ['Sex'],
        'Parkinsons Telemonitoring': [],
        'AIDS Clinical Trials Group Study 175': [],
        'SUPPORT2': ['sex', 'dzgroup', 'dzclass', 'edu', 'income', 'race', 'ca', 'prg6m', 'dnr', 'adlp', 'sfdm2'],
        'Infared Thermography Temperature': 925
}


def get_uci_dataset(index):
    # order uci datasets by index

    key_list = list(UCI_DATASETS_TO_ID.keys())
    val_list = list(UCI_DATASETS_TO_ID.values())


    dataset = fetch_ucirepo(id=val_list[index])

    X = dataset.data.features
    Y = dataset.data.targets

    types= dataset.variables['type']
    names = dataset.variables['name']

    categorical_names = []
    for i in range(len(types)):
        if types[i] == 'Categorical':
            categorical_names.append(names[i])

    X, Y = make_categorical_into_onehot(X=X, y=Y, columns_to_onehot=categorical_names)

    description = {
        'dataset name': key_list[index],
        'number of features': X.shape[1],
        'number of training examples': X.shape[0],
        'number of test examples': Y.shape[0],
        'variables information': dataset.variables,
        'categorical features': categorical_names
    }

    return X, Y, description

    




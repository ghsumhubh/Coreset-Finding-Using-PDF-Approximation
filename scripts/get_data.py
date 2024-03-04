from ucimlrepo import fetch_ucirepo 
from scripts.split import split_data
from scripts.utils import *

UCI_DATASETS_TO_ID = {
        'Abalone' : 1,
        'Parkinsons Telemonitoring': 189,
        'AIDS Clinical Trials Group Study 175': 890,
        'SUPPORT2': 880,
        'Infared Thermography Temperature': 925
    }

def get_dataset(index):
    return get_uci_dataset(index)


def get_uci_dataset(index):
    # order uci datasets by index

    key_list = list(UCI_DATASETS_TO_ID.keys())
    val_list = list(UCI_DATASETS_TO_ID.values())


    dataset = fetch_ucirepo(id=val_list[index])

    X = dataset.data.features
    Y = dataset.data.targets

    # remove nan values
    X = X.dropna(axis=1)
    Y = Y.dropna(axis=1)

    # if 2 targets pick the first one TODO: this is a temporary fix
    if Y.shape[1] > 1:
        Y = Y.iloc[:,0]

    types= dataset.variables['type']
    names = dataset.variables['name']

    categorical_names = []
    for i in range(len(types)):
        if types[i] == 'Categorical':
            categorical_names.append(names[i])

    # TODO: REMOVE NEXT LINE
    #print(dataset.variables)
    X, Y = make_categorical_into_onehot(X=X, y=Y, columns_to_onehot=categorical_names)
    
    description = {
        'dataset name': key_list[index],
        'number of features': X.shape[1],
        'number of samples': X.shape[0],
        'variables information': dataset.variables,
        'categorical features': categorical_names
    }
    x_train, x_test, y_train, y_test = split_data(X, Y)
    x_train, x_test, y_train, y_test = convert_to_numpy_dataset(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test, description

    




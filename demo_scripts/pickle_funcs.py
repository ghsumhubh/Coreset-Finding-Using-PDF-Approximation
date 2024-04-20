import pickle
def save_sample(x_sample, y_sample, history, index):
    # save as pickle
    with open(f'demo_data/x_sample_{index}.pkl', 'wb') as f:
        pickle.dump(x_sample, f)
    with open(f'demo_data/y_sample_{index}.pkl', 'wb') as f:
        pickle.dump(y_sample, f)
    with open(f'demo_data/history_{index}.pkl', 'wb') as f:
        pickle.dump(history, f)

def load_sample(index):
    # load from pickle
    with open(f'demo_data/x_sample_{index}.pkl', 'rb') as f:
        x_sample = pickle.load(f)
    with open(f'demo_data/y_sample_{index}.pkl', 'rb') as f:
        y_sample = pickle.load(f)
    with open(f'demo_data/history_{index}.pkl', 'rb') as f:
        history = pickle.load(f)
    return x_sample, y_sample, history


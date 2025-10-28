def train_val_split(dataset):
    train = dataset.sample(frac=0.8, random_state=2)
    val = dataset.drop(train.index)
    y_train = train['Diagnosis']
    y_val = val['Diagnosis']
    x_train = train.drop('Diagnosis', axis=1)
    x_val = val.drop('Diagnosis', axis=1)
    return x_train, y_train, x_val, y_val

def normalize(dataset):
    return dataset/255
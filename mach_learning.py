'''
If you are doing the Machine Learning challenge, create and train your models in this file. 
Be ready to do more than just create a model.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch as t
import torch.utils.data as td
import torch.nn.functional as F


def train_rf_model(df, features, target_col='IC50'):
    # select relevant columns - include both categorical and numeric predictors, as well as the target
    # variable
    # selected_columns = ['Tissue', 'Genetic Feature', 'msi_pval', 'Recurrent Gain Loss', 'fdr', target_col]
    """Trains a Random Forest Regressor model to predict the target.

    Args:
        df: The input DataFrame.
        target_col (str): The name of the target column. Defaults to 'IC50'.

    Returns:
        tuple: A tuple containing:
            - y_test: target values from the test set.
            - y_pred: Predicted target values for the test set.
            - model (RandomForestRegressor): The trained Random Forest model.
            - X (DataFrame): The features used for training.
    """
    # select relevant columns
    df_selected = df[features + [target_col]]
    # convert categorical columns into binary dummy variables
    df_encoded = pd.get_dummies(df_selected, drop_first=True, dummy_na=False)
    print(df_encoded)

    # separate features and labels
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # split data into train (70%), temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # split temp into dev (66.6% of temp ~ 20% of total) and test (33.3% of temp ~ 10% of total)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    print(f"Training set size: {len(X_train)}")
    print(f"Development set size: {len(X_dev)}")
    print(f"Testing set size: {len(X_test)}")

    # train the Random Forest model
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=None, min_samples_split=5)
    # fit model on training data
    model.fit(X_train, y_train)
    print("model fit finished")

    # predict on test set
    y_pred = model.predict(X_test)

    # evaluate model performance on the test set
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation on Test Set:")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # return outputs for further plots and analysis
    return y_test, y_pred, model, X_test


class DrugDataset(td.Dataset):
    def __init__(self, x, y):
        self._x = x.values
        self._y = y.values.astype("float32").squeeze()

    def __getitem__(self, idx):
        return t.from_numpy(self._x[idx, :].astype("float32")), self._y[idx]

    def __len__(self):
        return len(self._y)


class RFTorch(t.nn.Module):
    def __init__(self, cols):
        super().__init__()
        self.fc1 = t.nn.Linear(cols, 500)
        self.fc2 = t.nn.Linear(500, 100)
        self.fc3 = t.nn.Linear(100, 10)
        self.fc4 = t.nn.Linear(10, 1)

    def forward(self, batch_x):
        x = F.relu(self.fc1(batch_x))
        x2 = F.relu(self.fc2(x))
        x3 = F.relu(self.fc3(x2))
        batch_y_hat = self.fc4(x3)
        return t.squeeze(batch_y_hat)


class Hyperparams:
    def __init__(self, batch_size=10, epochs=10, learning_rate=0.0001):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def to_dict(self):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate
        }


def setup(df, features, labels):
    df = pd.get_dummies(df, sparse=True)
    features = list(df.columns)
    features.remove(labels[0])
    train_size = int(len(df) * 0.7)
    dev_size = int(len(df) * 0.2)
    train_X = df.loc[:train_size, features]
    train_y = df.loc[:train_size, labels]
    trainset = DrugDataset(train_X, train_y)
    dev_X = df.loc[train_size:train_size + dev_size, features]
    dev_y = df.loc[train_size:train_size + dev_size, labels]
    devset = DrugDataset(dev_X, dev_y)
    test_X = df.loc[train_size + dev_size:, features]
    test_y = df.loc[train_size + dev_size:, labels]
    testset = DrugDataset(test_X, test_y)
    hparams = Hyperparams(batch_size=16, epochs=40, learning_rate=0.00004)
    model = RFTorch(len(features))
    optim = t.optim.SGD(model.parameters(), lr=hparams.learning_rate)
    loss_fn = t.nn.MSELoss(reduction="mean")
    trainloader = td.DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True)
    devloader = td.DataLoader(devset, batch_size=5000)
    train(model, optim, loss_fn, hparams.epochs, trainloader, devloader)
    testloader = td.DataLoader(testset, batch_size=len(testset))
    X, y = next(iter(testloader))
    model.eval()
    with t.no_grad():
        y_hat = model(X)
        test_rmse = t.sqrt(F.mse_loss(y_hat, y))
    return y, y_hat, test_rmse


def train(model, optim, loss_fn, epochs, trainloader, devloader):
    for epoch in range(epochs):
        train_losses = []
        train_outputs = t.tensor([], dtype=t.float32)
        train_targets = t.tensor([], dtype=t.float32)
        model.train()
        with t.enable_grad():
            for batch_X, batch_y in trainloader:
                optim.zero_grad()
                batch_y_hat = model.forward(batch_X)
                loss = loss_fn(batch_y_hat, batch_y)
                loss.backward()
                optim.step()

                train_losses.append(loss.detach())
                train_outputs = t.cat((train_outputs, batch_y_hat.detach()))
                train_targets = t.cat((train_targets, batch_y.detach()))                    

        train_loss = np.mean(train_losses)
        train_rmse = t.sqrt(F.mse_loss(train_outputs, train_targets))

        dev_losses = []
        dev_outputs = t.tensor([], dtype=t.float32)
        dev_targets = t.tensor([], dtype=t.float32)
        model.eval()
        with t.no_grad():
            for batch_X, batch_y in devloader:
                batch_y_hat = model(batch_X)
                loss = loss_fn(batch_y_hat, batch_y)
                dev_losses.append(loss.detach())
                dev_outputs = t.cat((dev_outputs, batch_y_hat.detach()))
                dev_targets = t.cat((dev_targets, batch_y.detach()))
        dev_loss = np.mean(dev_losses)
        dev_rmse = t.sqrt(F.mse_loss(dev_outputs, dev_targets))

        print(f"\nEpoch {epoch}:")
        print(f"Loss: train={train_loss:.4f}, dev={dev_loss:.4f}")
        print(f"RMSE: train={train_rmse:.4f}, dev={dev_rmse:.4f}")

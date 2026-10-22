import numpy as np


def predict(houses, weights, bias):
    # y = mx + h, with one m per column (m = weights, h = bias)
    # houses  : array (number of houses, number of columns), one row per house
    # weights : array (number of columns), one weight per column (the "m")
    # bias    : one number, the starting price (the "h")
    # @ is the matrix product: for each house, multiply each column by its weight and add it all up
    return houses @ weights + bias


def mse(y_true, y_pred):
    # mean squared error
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    # mean absolute error
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    # R^2 score
    model_error = np.sum((y_true - y_pred) ** 2)
    mean_error = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (model_error / mean_error)


def train_sgd(weights, bias, houses, prices, learning_rate=0.01, epochs=100):
    # stochastic gradient descent
    n = len(prices)
    for epoch in range(epochs):
        # shuffle the houses every epoch (this is the "stochastic" part)
        order = np.random.permutation(n)
        for i in order:
            # predict the price of the i-th house (one house -> one number)
            y_pred = predict(houses[i], weights, bias)
            # calculate the error
            error = y_pred - prices[i]
            # update weights and bias
            weights -= learning_rate * error * houses[i]
            bias -= learning_rate * error
        # show the loss on all houses: it should go down
        print("epoch", epoch, "mse", mse(prices, predict(houses, weights, bias)))
    return weights, bias


def save_model(weights, bias, filename):
    # save the model to a file
    np.savez(filename, weights=weights, bias=bias)


def load_model(filename):
    # load the model from a file
    data = np.load(filename)
    return data["weights"], data["bias"]


if __name__ == "__main__":
    from kkd_data import load_dataset, train_test_split

    # load the cleaned data from kkd_data.py
    houses, prices, stats = load_dataset()
    prices = prices / 1e6  # prices in millions of NOK, keeps the numbers small for SGD

    # 90% to train, 10% to test
    houses_train, houses_test, prices_train, prices_test = train_test_split(
        houses, prices, test_frac=0.1
    )

    # train, starting from zero
    weights, bias = train_sgd(
        np.zeros(houses.shape[1]),
        0.0,
        houses_train,
        prices_train,
        learning_rate=0.001,
        epochs=30,
    )

    # evaluate on houses the model has never seen
    predictions = predict(houses_test, weights, bias)
    print("mae (millions NOK)", mae(prices_test, predictions))
    print("r2", r2(prices_test, predictions))

    # the key drivers: biggest weights first
    for name, weight in sorted(
        zip(stats["feature_names"], weights), key=lambda t: -abs(t[1])
    ):
        print(name, round(weight, 2))

    save_model(weights, bias, "model.npz")

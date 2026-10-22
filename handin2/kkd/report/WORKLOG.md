# KKD work log

Team: Dylan, Cem, Simon

## The task

Knut (KKD) wants a price estimate for a house before it goes on the market. We had to make a price model
trained with our own SGD (only Python and numpy), a Dash dashboard that uses it, find some insights in the
data, and present everything in a short pitch using the Fremtiden template.

## Who did what

We split it so we could work at the same time:

- Cem: the data part (`kkd_data.py`): reading the files, joining them, cleaning, one-hot encoding and
  normalizing. He also made the pitch.
- Dylan: the model (`model.py`): predict, the SGD training loop, MSE/MAE/R², saving and loading the model.
- Simon: the dashboard (`kkd_dashboard.py`).
- Connecting everything, testing and the insights we did together.

The data part gives the model three things: `load_dataset()` returns `houses` (one normalized row per
house), `prices` and `stats` (means, stds and column names). The dashboard uses
`encode_house()` so a new house is encoded exactly like the training data. The trained model is saved in
`model.npz` (`weights` and `bias`).

## The data

There are 4 files (houses, agents, districts, schools) linked by ids, 2552 houses in total. While going
through them we found a few problems:

- 2 houses have the price 2147483646, which is just the int32 max and not a real price, so we removed them.
- 34 houses have no year. We used the median (1993).
- `remodeled = -1` means the house was never remodeled. We turned it into two columns, `was_remodeled` and
  `years_since_remodel`.
- `rooms` is empty for about 71 % of the houses, so we dropped it. `size` says more or less the same thing.
- One house has the color "reds", changed to "red". "unknown" is kept as its own color.
- `kitchens` is sometimes 1 and sometimes 1.0, so we convert it to int.
- One line in houses.jsonl has a stray character before the `{`, so the reader starts at the first `{`.

We left out `sold`, `days_on_marked` and `sold_in_month` because they are only known after the house is put
on the market, so using them to predict the price would be cheating. (All unsold houses also have
`days_on_marked = 100`, which looks like a filler value.) The agent is left out too, since it says nothing
about the house itself.

For the encoding, `year` became `age`, yes/no columns became 0/1, and advertisement, color and district are
one-hot encoded (one category per group is left out as the reference). School rating and capacity come from
the school file through `school_id`. Then every column is normalized to mean 0 and std 1, so the same learning
rate works for all of them and we can compare the weights afterwards. In the end we have 2549 houses and 26
columns.

## The model

It is a linear model with one weight per column, so basically y = m·x + h with many m's:

```
price = houses @ weights + bias
```

We train it with SGD. The weights start at zero, and in each epoch we shuffle the houses and update the
weights after every single house. The loss is the MSE, and with the chain rule the update becomes:

```
weights -= learning_rate * (prediction - price) * house
bias    -= learning_rate * (prediction - price)
```

We divide the prices by one million during training, because with prices in the millions the steps were way
too big.

To make sure the loop actually worked before using the real data, we first tried it on fake data generated
with weights [2, -1, 0.5] and bias 3. It found [2.00, -0.99, 0.51] and 2.99, so we knew the code was fine.

On the real data a learning rate of 0.01 made the loss jump around, 0.001 worked. The loss stops going down
after around 10 epochs, we kept 30 to be safe.

## Results

We kept 10 % of the houses aside as a test set. On those:

- R² = 0.935, so the model explains about 94 % of the price differences
- MAE = 1.37 million NOK on average (the median house is around 8.5 million)

The MSE is only used for training, MAE and R² are just to see how good it is.

The biggest weights (in million NOK per standard deviation) are size (4.83) and lot width (2.57), far ahead
of external storage (1.01), storage rating (0.60), condition (0.37) and bathrooms (0.37). Premium
advertisement gives +0.17. Color, sun factor and parking are almost 0.

The parking one surprised us: if you just compare median prices, houses with parking are about 13 % more
expensive per m², but the model gives it almost nothing. That's because houses with parking also tend to be
bigger, and once size is in the model, parking doesn't add much. The small negative weight on kitchens is the
same kind of thing.

## Dashboard

You fill in the house, click "Look for a house" and it shows the predicted price. It encodes the inputs with
`encode_house` and uses the weights from `model.npz`.

When we connected it, it kept saying "Awaiting model.npz". The dashboard was looking for keys called `w` and
`b` but the model saves them as `weights` and `bias`, so we fixed the names.

We then tried it with different inputs: default values, empty fields, every district and school, every color
and ad type. The price goes up with size (50 m² gives 3.6 M, 400 m² gives 25.4 M) and with condition
(5.4 M at 2, 7.3 M at 10), and premium ads give the highest price, which matches the weights.

## Limitations

- Since it's linear, a very small house in very bad condition can get a negative price. The dashboard shows
  0 in that case.
- We don't use the month of sale, even though summer sales look more expensive in the raw data.
- The test set is only 254 houses, and because of the random order in SGD the scores change a little each
  run.

## Running it

```bash
cd handin2/kkd
../.venv/bin/python kkd_data.py        # checks the data
../.venv/bin/python model.py           # trains, prints the results and saves model.npz
../.venv/bin/python report/plot_weights.py   # makes report/key_drivers.png
../.venv/bin/python kkd_dashboard.py   # then go to http://127.0.0.1:8050
```

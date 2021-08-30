from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
from tinbasic import mlpreg_tibasic

xs = np.array([
    0., 0.,
    0., 1.,
    1., 0.,
    1., 1.
]).reshape(4, 2)

ys = np.array([0., 1., 1., 0.])

model = MLPRegressor(activation='relu', max_iter=10000, hidden_layer_sizes=(4,4), random_state=1, verbose=False, early_stopping=False, n_iter_no_change=100, batch_size=1)

model = model.fit(xs, ys)

print('score:', model.score(xs, ys)) # outputs 0.5
print('predictions:', model.predict(xs)) # outputs [0, 0, 0, 0]
print('expected:', np.array([0, 1, 1, 0]))

# print(f"{mlpreg_tibasic(model)=}")
out = mlpreg_tibasic(model, "formula")

print("================")
print(out)
print("================")

print()
print()

print(f"{model.coefs_=}")
print(f"{model.intercepts_=}")
import os
import warnings
import logging
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from models.mpnn_model import MPNNModel
from models.data_load import MPNNDataset
import params_mpnn
import argparse
import pickle


# float 32
parser = argparse.ArgumentParser(description='task & seed')
parser.add_argument('--task', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

task = args.task
seed = args.seed

np.random.seed(seed)
tf.random.set_seed(seed)

# dataset 80:10:10
data_path = './data/preprocessed_dataset/' + str(task) + '/seed_' + str(seed)

with open(data_path + '/train_data.pkl', 'rb') as f:
    x_train, y_train = pickle.load(f)

with open(data_path + '/val_data.pkl', 'rb') as f:
    x_val, y_val = pickle.load(f)

with open(data_path + '/test_data.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

train_dataset = MPNNDataset(x_train, y_train)
val_dataset = MPNNDataset(x_val, y_val)
test_dataset = MPNNDataset(x_test, y_test)
print('Dataset preparation done!')

# checkpoints path
path = './checkpoints/' + str(task) + '/seed_' + str(seed)
if not os.path.exists(path):
    os.makedirs(path)

# model
model = MPNNModel()

tf.keras.utils.plot_model(model, to_file=path + '/model.png', show_dtype=True, show_shapes=True, dpi=800)

optimizer = tf.keras.optimizers.Adam(learning_rate=params_mpnn.learning_rate)


# loss function mse
def loss_fn(y, pred):
    if y.shape != pred.shape:
        print(y, pred)
    return tf.reduce_mean((y-pred)**2)


@tf.function(reduce_retracing=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        pred = tf.squeeze(pred, axis=-1)
        loss = tf.py_function(loss_fn, [y, pred], tf.float32)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss, pred


@tf.function(reduce_retracing=True)
def test_step(x):
    pred = model(x)
    pred = tf.squeeze(pred, axis=-1)
    return pred


epochs = params_mpnn.max_epochs
loss = None
r2 = -10
monitor = 0
train_rmse_list = []
val_rmse_list = []

for epoch in range(epochs):
    y_true = []
    y_pred = []
    for _, (x, y) in enumerate(train_dataset):
        loss, pred = train_step(x, y)
        y_true.append(y.numpy())
        y_pred.append(pred.numpy())
    y_true = np.float32(np.concatenate(y_true, axis=0).reshape(-1))
    y_pred = np.float32(np.concatenate(y_pred, axis=0).reshape(-1))
    train_rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    train_rmse_list.append(train_rmse)

    y_true = []
    y_pred = []
    for x, y in val_dataset:
        pred = test_step(x)
        y_true.append(y.numpy())
        y_pred.append(pred.numpy())
    y_true = np.float32(np.concatenate(y_true, axis=0).reshape(-1))
    y_pred = np.float32(np.concatenate(y_pred, axis=0).reshape(-1))
    val_rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    val_rmse_list.append(val_rmse)

    # val r2 cal
    val_r2 = metrics.r2_score(y_true, y_pred)

    print('task:', str(task),
          'seed:', seed,
          'epoch:', epoch,
          'loss: {:.4f}'.format(loss.numpy()),
          'rmse: {:.4f}'.format(train_rmse),
          'val rmse: {:.4f}'.format(val_rmse),
          'val r2: {:.4f}'.format(val_r2),
          )

    if val_r2 > r2:
        r2 = val_r2
        monitor = 0

        # model save weights
        model.save_weights(path + '/model_weights.h5')
    else:
        monitor += 1
    if monitor > 0:
        print('monitor:', monitor)
    if monitor > 10:
        print('Training done!')
        break

# plot and save train val mse
plt.plot(train_rmse_list, label="train")
plt.plot(val_rmse_list, label="val")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.savefig(path + '/rmse.png', dpi=800)
# plt.show()
plt.close()

train_rmse_list = pd.DataFrame(train_rmse_list)
val_rmse_list = pd.DataFrame(val_rmse_list)
df_rmse = pd.concat([train_rmse_list, val_rmse_list], axis=1)
df_rmse.to_csv(path + '/rmse.txt', index=False, header=['train_rmse', 'val_rmse'])

# load model to predict
model = MPNNModel()
model.load_weights(path + '/model_weights.h5')

y_true = []
y_pred = []
for x, y in test_dataset:
    pred = test_step(x)
    y_true.append(y.numpy())
    y_pred.append(pred.numpy())
y_true = np.float32(np.concatenate(y_true, axis=0).reshape(-1))
y_pred = np.float32(np.concatenate(y_pred, axis=0).reshape(-1))

plt.plot(y_true, y_true, "-")
plt.plot(y_true, y_pred, ".")
plt.xlabel("TD-DFT Value")
plt.ylabel("Predicted Value")
plt.savefig(path + '/test_prediction.png', dpi=800)
# plt.show()
plt.close()


y_true = pd.DataFrame(y_true)
y_pred = pd.DataFrame(y_pred)
df_test = pd.concat([y_true, y_pred], axis=1)
df_test.to_csv(path + '/test_prediction.txt', index=False, header=['y_true', 'y_pred'])

# test performance
test_pf = []
mae = metrics.mean_absolute_error(y_true, y_pred)
mse = metrics.mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_true, y_pred)
test_pf.append([mae, mse, rmse, r2])
test_pf = pd.DataFrame(test_pf)
test_pf.to_csv(path + '/test_result.txt', index=False, header=['MAE', 'MSE', 'RMSE', 'R2'])


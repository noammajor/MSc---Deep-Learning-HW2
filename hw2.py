#start
# Part 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path="./train.csv"):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)
def reShape_data(data):
    size = 28
    return data.values.reshape(-1, size, size)

def plot_sample_images(path="./train.csv"):
    x_data = load_data("./train.csv")
    y_data = x_data.pop('label').values
    X_train = reShape_data(x_data)
    y_train = y_data
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    # Increase figure size for better visibility since we now have 10 rows
    plt.figure(figsize=(10, 20)) 
    
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, 4, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = y * 4 + i + 1
            plt.subplot(len(classes), 4, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.text(-10, 14, cls, fontsize=10, ha='right', va='center', fontweight='bold')               
    plt.tight_layout()
    plt.show()
plot_sample_images()
# Part 2 preprocessing
def split_data(x, y, train_ratio=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    split_indx= int(n_samples * train_ratio)
    train_indices = indices[:split_indx]
    val_indices = indices[split_indx:]
    X_train = x[train_indices]
    y_train = y[train_indices]
    X_val = x[val_indices]
    y_val = y[val_indices]
    return X_train, y_train, X_val, y_val
def normalize_data(X, mean, std):
    X_norm = (X - mean) / (std + 1e-8)
    return X_norm
# Soft max loss functions
def softmax_loss_vectorized_L2(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #used for numerical stability (overflow) + turn into probabilities
    scores_prob = X.dot(W) - np.max(X.dot(W), axis=1, keepdims=True)
    probs = np.exp(scores_prob)/np.sum(np.exp(scores_prob), axis=1, keepdims=True)
    loss = (-np.sum(np.log(probs[np.arange(num_train), y]))/num_train) + reg* np.sum(W * W)
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores/num_train) + 2 * reg * W
    return loss, dW

def softmax_loss_vectorized_L1(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #used for numerical stability (overflow) + turn into probabilities
    scores_prob = X.dot(W) - np.max(X.dot(W), axis=1, keepdims=True)
    probs = np.exp(scores_prob)/np.sum(np.exp(scores_prob), axis=1, keepdims=True)
    loss = (-np.sum(np.log(probs[np.arange(num_train), y]))/num_train) + reg* np.sum(np.abs(W))
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores/num_train) + reg * np.sign(W)
    return loss, dW
# model
class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train_val(
        self,
        X,
        y,
        X_val,
        y_val,
        learning_rate=1e-3,
        reg=1e-5,
        epochs=10,
        batch_size=200,
        verbose=False,
        penalty='l2',
    ):
        self.loss_func = softmax_loss_vectorized_L1 if penalty == 'l1' else softmax_loss_vectorized_L2
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        dat_hist = {'loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        for epoch in range(epochs):
            epoch_l = 0.0
            indices = np.random.permutation(num_train)
            for i in range(0, num_train, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                loss, grad = self.loss_func(self.W,X_batch, y_batch, reg)
                self.W -= learning_rate*grad
                epoch_l += loss
            
            epoch_l = epoch_l / (num_train // batch_size)
            train_acc = np.mean(self.predict(X) == y)
            val_acc = np.mean(self.predict(X_val) == y_val)
            v_loss, _ = softmax_loss_vectorized_L2(self.W, X_val, y_val, reg)
            
            dat_hist['loss'].append(epoch_l)
            dat_hist['val_loss'].append(v_loss)
            dat_hist['train_acc'].append(train_acc)
            dat_hist['val_acc'].append(val_acc)


            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: loss {epoch_l:.4f}, val_acc {val_acc:.4f}")

        return dat_hist

    def predict(self, X):
        y_pred =np.argmax(X.dot(self.W),axis=1)
        return y_pred
    
    def loss(self, X_batch, y_batch, reg, loss_func=softmax_loss_vectorized_L2):
        return loss_func(self.W, X_batch, y_batch, reg)

# Part 2 model
results = {}
best_val = -1
best_vals = None
best_softmax = None
x_data = load_data()
y_data = x_data.pop('label').values
X_train, y_train, X_val, y_val = split_data(reShape_data(x_data), y_data, train_ratio=0.8, seed=42)
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)
X_train = normalize_data(X_train, train_mean, train_std)
X_val = normalize_data(X_val, train_mean, train_std)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

learning_rates = [0.05, 0.1, 0.15, 0.2, 1e-2] 
regularization_strengths = [1e-4, 5e-4, 1e-3, 5e-3]
batch_size = [100,200,300]
loss_func = ['l1','l2']
for penalty in loss_func:
    for bs in batch_size:
        for lr in learning_rates:
            for reg in regularization_strengths:
                lin = LinearClassifier()
                hist_vals = lin.train_val(X_train, y_train, X_val, y_val, learning_rate=lr, reg=reg, penalty=penalty,
                            epochs=20, batch_size=bs, verbose=True)
                results[(lr, reg)] = (hist_vals['train_acc'][-1], hist_vals['val_acc'][-1])
                if best_val < hist_vals['val_acc'][-1]:
                    best_val = hist_vals['val_acc'][-1]
                    best_softmax = lin
                    best_vals = hist_vals

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e batch_size %d loss %s train accuracy: %f val accuracy: %f' % (
                lr, reg, bs, penalty, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)
# Plotting the loss and accuracy curves
positions = np.arange(len(best_vals['loss']))
labels = positions + 1
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(best_vals['loss'], label='Training Loss')
plt.plot(best_vals['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(positions, labels)
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(best_vals['train_acc'], label='Training Accuracy')
plt.plot(best_vals['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(positions, labels)
plt.legend()
plt.grid()
plt.show()
print

#Test answers
test_data = load_data("./test.csv")
test_data = normalize_data(reShape_data(test_data), train_mean, train_std)
test_data = np.reshape(test_data, (test_data.shape[0], -1))
test_data = np.hstack([test_data, np.ones((test_data.shape[0], 1))])
test_preds = best_softmax.predict(test_data)
np.savetxt("lr_pred.csv", test_preds, fmt='%d', delimiter='\n')


# Part 3 - two layer net class helpers
def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], -1) @ w + b
    cache = (x, w, b)
    return out, cache
def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = (dout@w.T).reshape(x.shape), x.reshape(x.shape[0], -1).T @ dout , np.sum(dout, axis=0)
    return dx, dw, db
def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache
def relu_backward(dout, cache):
    x = cache
    return dout*(x>0)
def leaky_relu_forward(x):
    out = np.maximum(0.01*x, x)
    cache = x
    return out, cache
def leaky_relu_backward(dout, cache):
    x = cache
    return dout * np.where(x > 0, 1, 0.01)
def sigmoid_forward(x):
    out = 1/(1+np.exp(-x))
    cache = x
    return out, cache
def sigmoid_backward(dout, cache):
    x = cache
    return dout*(1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))

def dropout_forward(x, dropout_param):
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p).astype(int)
        if p > 0:
            mask= mask/p
        out = (x*mask)
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = mask*dout
    elif mode == "test":
        dx = dout
    return dx

def softmax_loss(x, y):
    loss, dx = None, None
    num_train = x.shape[0]
    #used for numerical stability (overflow) + turn into probabilities
    scores_prob = x - np.max(x, axis=1, keepdims=True)
    probs = np.exp(scores_prob)/np.sum(np.exp(scores_prob), axis=1, keepdims=True)
    loss = (-np.sum(np.log(probs[np.arange(x.shape[0]), y])))/num_train
    dx = probs.copy()
    dx[np.arange(num_train), y] -= 1
    dx /= num_train
    return loss, dx

# Two layer net class
class TwoLayerNet(object):
    def __init__(
        self,
        input_dim=28*28,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        activation_function_forward = relu_forward,
        activation_function_backward = relu_backward,
        dropout_param = {'use': True, 'mode': 'train', 'p':1.0, 'seed': 42},
    ):
        self.params = {}
        self.reg = reg
        import numpy as np

        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        self.activation_function_forward = activation_function_forward
        self.activation_function_backward = activation_function_backward
        self.dropout_param = dropout_param

    def loss(self, X, y=None):
        self.dropout_param['mode'] = 'train' if y is not None else 'test'
        scores = None
        #forward pass
        forward_aff, cache_1_for = affine_forward(X,self.params['W1'], self.params['b1'])
        relu_for, cache_relu = self.activation_function_forward(forward_aff)
        if self.dropout_param['use']:
            relu_for, cache_dropout = dropout_forward(relu_for, self.dropout_param)
        scores, cache_2_for = affine_forward(relu_for ,self.params['W2'], self.params['b2'])

        if y is None:
            return scores

        loss, grads = 0, {}
        loss , DL1 = softmax_loss(scores, y)
        loss += self.reg*0.5*(np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        #backward pass
        DL2,dw2,grads['b2'] = affine_backward(DL1, cache_2_for)
        if self.dropout_param['use']:
            DL2 = dropout_backward(DL2, cache_dropout)
        DL3 = self.activation_function_backward(DL2, cache_relu )
        DL4,dw1,grads['b1'] = affine_backward(DL3, cache_1_for )
        grads['W1'] = dw1 + self.reg*self.params['W1']
        grads['W2'] = dw2 + self.reg*self.params['W2']
        return loss, grads
    
    def train_val(
        self,
        X,
        y,
        X_val,
        y_val,
        learning_rate=1e-3,
        reg=1e-5,
        epochs=10,
        batch_size=200,
        verbose=False,
    ):
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )
        dat_hist = {'loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        for epoch in range(epochs):
            epoch_l = 0.0
            indices = np.random.permutation(num_train)
            for i in range(0, num_train, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                loss, grad = self.loss(X_batch, y_batch)
                for param_name in self.params:
                    self.params[param_name] -= learning_rate * grad[param_name]
                epoch_l += loss
            
            epoch_l = epoch_l / (num_train // batch_size)
            train_acc = np.mean(self.predict(X) == y)
            val_acc = np.mean(self.predict(X_val) == y_val)
            v_loss, _ = self.loss(X_val, y_val)
            
            dat_hist['loss'].append(epoch_l)
            dat_hist['val_loss'].append(v_loss)
            dat_hist['train_acc'].append(train_acc)
            dat_hist['val_acc'].append(val_acc)


            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: loss {epoch_l:.4f}, val_acc {val_acc:.4f}")

        return dat_hist
    
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc
    
    def predict(self, X):
        scores = self.loss(X)
        return np.argmax(scores, axis=1)

# training two layer net
best_model = None
highest_acc = 0
top_50_configs = [
    (0.1, 256, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.5}),
    (0.1, 256, 20, 0.001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.5}),
    (0.1, 256, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 256, 20, 0.0001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.001, 100, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 128, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 128, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 256, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.5}),
    (0.1, 256, 20, 0.0001, 200, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 256, 20, 0.001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.05, 256, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.5}),
    (0.05, 128, 20, 0.001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.05, 256, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 200, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 256, 20, 0.001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 128, 20, 0.001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.0001, 200, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.001, 200, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.05, 256, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.05, 128, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.5}),
    (0.1, 128, 20, 0.001, 200, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.0001, 200, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 256, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 256, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8}),
    (0.05, 256, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.05, 256, 20, 0.001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 128, 20, 0.0001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 128, 20, 0.001, 100, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.5}),
    (0.1, 256, 20, 0.005, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.05, 128, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.001, 200, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8}),
    (0.05, 128, 20, 0.001, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.0001, 200, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 256, 20, 0.0001, 300, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8}),
    (0.1, 256, 20, 0.0001, 100, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.5}),
    (0.1, 256, 20, 0.005, 100, (leaky_relu_forward, leaky_relu_backward, "Leaky ReLU"), {'use': False, 'p': 1.0}),
    (0.1, 128, 20, 0.001, 200, (relu_forward, relu_backward, "ReLU"), {'use': True, 'p': 0.8})
]
x_data_raw = load_data("./train.csv")
y_data = x_data_raw.pop('label').values
X_raw = x_data_raw.values
X_train_raw, y_train, X_val_raw, y_val = split_data(X_raw, y_data, train_ratio=0.8, seed=42)
train_mean = np.mean(X_train_raw, axis=0)
train_std = np.std(X_train_raw, axis=0)
X_train = normalize_data(X_train_raw, train_mean, train_std)
X_val = normalize_data(X_val_raw, train_mean, train_std)
test_data_2 = load_data("./test.csv").values
test_data_2 = normalize_data(test_data_2, train_mean, train_std)

for lr, hs, ep, rs, bs, act, drop in top_50_configs:
# Set seed for reproducibility as mentioned in Tip 1 [cite: 36, 42]
    dropout_param = {'use': drop['use'], 'mode': 'train', 'p': drop['p'], 'seed': 42}
    model = TwoLayerNet(28*28, hs, 10, reg=rs,activation_function_forward=act[0],activation_function_backward=act[1], dropout_param=dropout_param)
    hist = model.train_val(X_train, y_train, X_val, y_val, learning_rate=lr, reg=rs, epochs=ep, batch_size=bs)
    acc = model.check_accuracy(X_val, y_val)
    activationNAME = act[2]
    dropout = drop['p'] if drop['use'] else "None"
    if acc > highest_acc:
        highest_acc = acc
        best_model = model
        beststats = hist
        activation_best = activationNAME
        dropout_best = dropout
        learning_rate_best = lr
        hidden_size_best = hs
        epochs_best = ep
        reg_best = rs
        batch_size_best = bs
if best_model:
    test_preds = best_model.predict(test_data_2)
    np.savetxt("NN_pred.csv", test_preds, fmt='%d', delimiter='\n')
    plt.figure(figsize=(14, 6))  
    plt.subplot(1, 2, 1)
    plt.plot(beststats['loss'], label='Train Loss')
    plt.plot(beststats['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(beststats['train_acc'], label='Train Acc')
    plt.plot(beststats['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
#start
#%% Part 1
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
    num_classes = len(classes)
    samples_per_class = 4
    for y, cls in enumerate(classes):
            idxs = np.flatnonzero(y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(X_train[idx].astype('uint8'), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title(cls, fontsize=8)
    plt.show()
plot_sample_images()
#%% Part 2 preprocessing
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
#%% Soft max loss functions
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
#%% model
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

#%% Part 2 model
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
for lr in learning_rates:
    for reg in regularization_strengths:
        lin = LinearClassifier()
        hist_vals = lin.train_val(X_train, y_train, X_val, y_val, learning_rate=lr, reg=reg,
                      epochs=20, verbose=True)
        results[(lr, reg)] = (hist_vals['train_acc'][-1], hist_vals['val_acc'][-1])
        if best_val < hist_vals['val_acc'][-1]:
            best_val = hist_vals['val_acc'][-1]
            best_softmax = lin
            best_vals = hist_vals

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)
#%% Plotting the loss and accuracy curves
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
#%% Test answers
test_data = load_data("./test.csv")
test_data = normalize_data(reShape_data(test_data), train_mean, train_std)
test_data = np.reshape(test_data, (test_data.shape[0], -1))
test_data = np.hstack([test_data, np.ones((test_data.shape[0], 1))])
test_preds = best_softmax.predict(test_data)
np.savetxt("lr_pred.csv", test_preds, fmt='%d', delimiter='\n')
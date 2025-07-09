# load pkl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

pkls = [
    # 'base-None-None.pkl',
    # 'base-ReLU-None.pkl',
    # 'base-LeakyReLU02-None.pkl',
    # 'base-Sigmoid-None.pkl',
    # 'base-Tanh-None.pkl',
    
    'base-None-None-onehot.pkl',
    'base-ReLU-None-onehot.pkl',
    'base-LeakyReLU02-None-onehot.pkl',
    'base-Sigmoid-None-onehot.pkl',
    'base-Tanh-None-onehot.pkl',
    
    # 'base-Tanh-None-500.pkl',
    # 'base-Tanh-Gaussian001.pkl',
    # 'base-Tanh-Gaussian005.pkl',
    # 'base-Tanh-Gaussian01.pkl',
    # 'base-Tanh-Gaussian02.pkl',
    # 'base-Tanh-Gaussian03.pkl',
]
pkls_name = [
    'w/o activation function',
    'rectified linear unit',
    'leaky rectified linear unit with 0.2 slope',
    'sigmoid',
    'hyperbolic tangent',
    # 'w/o data augmentation',
    # 'with gaussian data augmentation (σ=0.01)',
    # 'with gaussian data augmentation (σ=0.05)',
    # 'with gaussian data augmentation (σ=0.1)',
    # 'with gaussian data augmentation (σ=0.2)',
    # 'with gaussian data augmentation (σ=0.3)',
]
colors = ['red', 'orange', 'green', 'blue', 'purple', 'black']
epochs = np.arange(1, 201)
results = {}
for pkl_id, pkl in enumerate(pkls):
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    train_losses_mean = np.mean(data['TRAIN_LOSSES'], axis=0)
    test_losses_mean = np.mean(data['TEST_LOSSES'], axis=0)
    train_accs_mean = np.mean(data['TRAIN_ACCS'], axis=0)
    test_accs_mean = np.mean(data['TEST_ACCS'], axis=0)
    train_losses_std = np.std(data['TRAIN_LOSSES'], axis=0)
    test_losses_std = np.std(data['TEST_LOSSES'], axis=0)
    train_accs_std = np.std(data['TRAIN_ACCS'], axis=0)
    test_accs_std = np.std(data['TEST_ACCS'], axis=0)

    model_name = pkls_name[pkl_id]
        
    results[model_name] = {
        'train_losses_mean': train_losses_mean,
        'test_losses_mean': test_losses_mean,
        'train_accs_mean': train_accs_mean,
        'test_accs_mean': test_accs_mean,
        'train_losses_std': train_losses_std,
        'test_losses_std': test_losses_std,
        'train_accs_std': train_accs_std,
        'test_accs_std': test_accs_std,
    }
    
    results[model_name]['FC1_WEIGHTS'] = np.array(data['FC1_WEIGHTS'])
    results[model_name]['FC2_WEIGHTS'] = np.array(data['FC2_WEIGHTS'])
    results[model_name]['FC1_BIASES'] = np.array(data['FC1_BIASES'])
    results[model_name]['FC2_BIASES'] = np.array(data['FC2_BIASES'])
    
    print(f"{model_name} - Final Test Acc: {test_accs_mean[-1]:.4f} ± {test_accs_std[-1]:.4f}")



# 1. Train Loss
plt.figure(figsize=(8, 6.5))
plt.title('Model Performance Comparison (128 experiments average)\nTraining Loss Traversal', fontsize = 16, fontweight = 'bold')
# plt.title('Model Performance Comparison (128 experiments average)\nTesting Loss Traversal', fontsize = 16, fontweight = 'bold')
for plt_idx, (model_name, data) in enumerate(results.items()):
    line = plt.plot(epochs, data['train_losses_mean'], label = f"{model_name}\n{results[model_name]['train_losses_mean'][-1]:.4f}±{results[model_name]['train_losses_std'][-1]:.4f}", linewidth = 2)
    # line = plt.plot(epochs, data['test_losses_mean'], label = f"{model_name}\n{results[model_name]['test_losses_mean'][-1]:.4f}±{results[model_name]['test_losses_std'][-1]:.4f}", linewidth = 2)
    color = line[0].get_color()
    
    plt.fill_between(epochs, 
                     data['train_losses_mean'] - data['train_losses_std'],
                     data['train_losses_mean'] + data['train_losses_std'],
                     alpha = 0.05, color = colors[plt_idx])
    '''
    plt.fill_between(epochs, 
                     data['test_losses_mean'] - data['test_losses_std'],
                     data['test_losses_mean'] + data['test_losses_std'],
                     alpha = 0.05, color = colors[plt_idx])
    '''

plt.xlabel('Epochs', fontsize = 14, fontweight = 'bold')
plt.ylabel('Loss Function Mean Absolute Error (MAE)', fontsize = 14, fontweight = 'bold')
plt.legend(loc='upper right', fontsize = 10) # bbox_to_anchor=(1.05, 1), 
plt.grid('--', 'both', 'both', alpha=0.4)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim(0, len(epochs) + 1)
plt.ylim(0, 1.15)
plt.show()
plt.cla()
plt.clf()
plt.close()

# 2. Train Accuracy
plt.figure(figsize=(8, 6.5))
plt.title('Model Performance Comparison (128 experiments average)\nTraining Accuracy Traversal', fontsize = 16, fontweight = 'bold')
# plt.title('Model Performance Comparison (128 experiments average)\nTesting Accuracy Traversal', fontsize = 16, fontweight = 'bold')
for plt_idx, (model_name, data) in enumerate(results.items()):
    line = plt.plot(epochs, data['train_accs_mean'], label = f"{model_name}\n{results[model_name]['train_accs_mean'][-1]:.4f}±{results[model_name]['train_accs_std'][-1]:.4f}", linewidth = 2)
    color = line[0].get_color()
    
    plt.fill_between(epochs, 
                     data['train_accs_mean'] - data['train_accs_std'],
                     data['train_accs_mean'] + data['train_accs_std'],
                     alpha = 0.05, color = colors[plt_idx])
    '''
    plt.fill_between(epochs,
                    data['test_accs_mean'] - data['test_accs_std'],
                    data['test_accs_mean'] + data['test_accs_std'],
                    alpha = 0.05, color = colors[plt_idx])
    '''
plt.xlabel('Epochs', fontsize = 14, fontweight = 'bold')
plt.ylabel('Training Accuracy', fontsize = 14, fontweight = 'bold')
plt.legend(loc='lower right', fontsize = 10)
plt.grid('--', 'both', 'both', alpha=0.4)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim(0, len(epochs) + 1)
plt.ylim(0, 1.15)
plt.show()
plt.cla()
plt.clf()
plt.close()


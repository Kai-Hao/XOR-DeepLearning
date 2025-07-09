import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from model import STANDARD_MODEL

from tqdm import tqdm
# Set random seed for reproducibility
random.seed(42)
SEEDS             = random.sample(range(2**31 - 1), 128)
EPOCHS            = 500
HIDDEN_LAYER_BIAS = True
ONE_HOT_ENCODING  = False
ACTIVATION        = nn.Tanh()
OUTPUT_ACTIVATION = nn.Sigmoid()
INIT_WEIGHT       = None
# None, nn.init.xavier_uniform_, nn.init.xavier_normal_, nn.init.kaiming_uniform_, nn.init.kaiming_normal_
GAUSSIAN_SAMPLE   = 1
LEARNING_RATE     = 0.2
GAUSSIAN_SIGMA    = 0.01
RECORS_NAME       = f"base-Tanh-Gaussian001.pkl"

def GAUSSIAN_DATA_AUGMENTATION(data, sigma = GAUSSIAN_SIGMA, sample_number = GAUSSIAN_SAMPLE):
    data = np.array(data)
    augmented_data = []
    for _ in range(sample_number):
        noise = np.random.normal(0, sigma, size = data.shape)
        augmented_data.append(data + noise)
    return np.array(augmented_data)

def SET_SEED(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DATA
EVALUATION_INPUT = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
if ONE_HOT_ENCODING:
    EVALUATION_LABEL = [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ]
else:
    EVALUATION_LABEL = [
        [0],
        [1],
        [1],
        [0]
    ]
RECORDS = {
    "TRAIN_LOSSES"  : [],
    "TEST_LOSSES"   : [],
    "TRAIN_ACCS"    : [],
    "TEST_ACCS"     : [],
    "FC1_WEIGHTS"   : [],
    "FC2_WEIGHTS"   : [],
    "FC1_BIASES"    : [],
    "FC2_BIASES"    : []
}
for SEED in tqdm(SEEDS):
    SET_SEED(SEED)
    # MODEL
    CLASSIFIER = STANDARD_MODEL(
        activation         = ACTIVATION,
        output_activation  = OUTPUT_ACTIVATION,
        hidden_layer_bias  = HIDDEN_LAYER_BIAS,
        one_hot_encoding   = ONE_HOT_ENCODING,
        init_weight        = INIT_WEIGHT,
        seed               = SEED
    )
    # RECORDS INIT
    for key in RECORDS:
        RECORDS[key].append([])
    print(f"Model initialized with seed {SEED}")
    OPTIMIZER   = torch.optim.SGD(CLASSIFIER.parameters(), lr = LEARNING_RATE)
    CRITICAL    = nn.L1Loss().cuda()
    CLASSIFIER  = CLASSIFIER.cuda()
    
    for EPOCH in range(1, EPOCHS + 1):
        CLASSIFIER.train()
        DATA_AUG = []
        for D in EVALUATION_INPUT:
            DATA_AUG.append(np.squeeze(GAUSSIAN_DATA_AUGMENTATION(D, sample_number = 1)))
        DATA_AUG  = torch.tensor(DATA_AUG, dtype = torch.float32).cuda()
        LABEL = torch.tensor(EVALUATION_LABEL, dtype = torch.float32).cuda()
        # MODEL FORWORD
        PREDICT = CLASSIFIER(DATA_AUG)
        LOSS    = CRITICAL(PREDICT, LABEL)
        # MODEL BACKWORD
        OPTIMIZER.zero_grad()
        LOSS.backward()
        OPTIMIZER.step()
        ACCURACY = ((PREDICT > 0.5) == LABEL).float().mean()
        # RECORDS
        RECORDS["TRAIN_LOSSES"][-1].append(LOSS.item())
        RECORDS["TRAIN_ACCS"][-1].append(ACCURACY.item())
        RECORDS["FC1_WEIGHTS"][-1].append(CLASSIFIER.fc1.weight.data.cpu().numpy())
        RECORDS["FC2_WEIGHTS"][-1].append(CLASSIFIER.fc2.weight.data.cpu().numpy())
        RECORDS["FC1_BIASES"][-1].append(CLASSIFIER.fc1.bias.data.cpu().numpy())
        RECORDS["FC2_BIASES"][-1].append(CLASSIFIER.fc2.bias.data.cpu().numpy())
        # EVALUATION
        CLASSIFIER.eval()
        with torch.no_grad():
            DATA = torch.tensor(EVALUATION_INPUT, dtype = torch.float32).cuda()
            PREDICT  = CLASSIFIER(DATA)
            LOSS     = CRITICAL(PREDICT, LABEL)
            ACCURACY = ((PREDICT > 0.5) == LABEL).float().mean()
        RECORDS["TEST_LOSSES"][-1].append(LOSS.item())
        RECORDS["TEST_ACCS"][-1].append(ACCURACY.item())
    tqdm.write(f"Seed {SEED}, Loss: {LOSS.item()}, Accuracy: {ACCURACY.item()}")
    tqdm.write("fc1 weights: " + str(CLASSIFIER.fc1.weight.data.cpu().numpy()))
    tqdm.write("fc1 bias: " + str(CLASSIFIER.fc1.bias.data.cpu().numpy()))
    tqdm.write("fc2 weights: " + str(CLASSIFIER.fc2.weight.data.cpu().numpy()))
    tqdm.write("fc2 bias: " + str(CLASSIFIER.fc2.bias.data.cpu().numpy()))
    
# Save the records
import pickle
with open(RECORS_NAME, 'wb') as f:
    pickle.dump(RECORDS, f)

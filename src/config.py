import torch

class Config:
    # Paths
    RAW_TRAIN_PATH = "../data/CONDA_train.csv"
    RAW_VALID_PATH = "../data/CONDA_valid.csv"
    PROCESSED_TRAIN_PATH = "../data/train_context.csv"
    PROCESSED_TEST_PATH = "../data/test_context.csv"
    MODEL_SAVE_PATH = "../data/hybrid_model_best.pth"
    
    # Data Parameters
    MAX_LEN_BERT = 128
    MAX_LEN_CHAR = 150
    
    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 4
    LR_BERT = 2e-5
    LR_CNN = 1e-3
    
    # Robustness
    AUGMENT_PROB = 0.5
    
    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed():
        import random
        import numpy as np
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
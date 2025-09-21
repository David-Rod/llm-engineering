## Results from Week 7 Training Exercise:
- Avg price error of $69, RMSLE of 0.50 , and Hits 55.6% using default values from course in Google collab
- Avg price error of $68.45, RMSLE of 0.57 , and Hits 57.2% using the following in Google collab:
    ```
    # Hyperparameters for QLoRA

    LORA_R = 32
    LORA_ALPHA = 128
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.4
    QUANT_4_BIT = True

    # Hyperparameters for Training

    EPOCHS = 2
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-4
    LR_SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.03
    OPTIMIZER = "paged_adamw_32bit"
    ```
- Avg price error of $78.03, RMSLE of 0.64, and Hits 52.8% using the following in Google collab:
    ```
    # Hyperparameters for QLoRA

    LORA_R = 64
    LORA_ALPHA = 128
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    LORA_DROPOUT = 0.05
    QUANT_4_BIT = True

    # Hyperparameters for Training

    EPOCHS = 1
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 3e-4
    LR_SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.1
    OPTIMIZER = "paged_adamw_32bit"
    ```
- Avg price error of $70.83, RMSLE of 0.51, and Hits 54.4% using the following in Google collab:
    ```
    # Hyperparameters for QLoRA

    LORA_R = 32
    LORA_ALPHA = 64
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 1e-4
    WARMUP_RATIO = 0.03
    QUANT_4_BIT = True


    # Hyperparameters for Training

    EPOCHS = 1
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.03
    OPTIMIZER = "paged_adamw_32bit"
    ```
- Avg price error of $62.55, RMSLE of 0.54, and Hits 62.4% using the following in Google collab:
    ```
    # Hyperparameters for QLoRA

    LORA_R = 32
    LORA_ALPHA = 128
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.1
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.05
    QUANT_4_BIT = True


    # Hyperparameters for Training

    EPOCHS = 1
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    LR_SCHEDULER_TYPE = 'cosine'
    OPTIMIZER = "paged_adamw_32bit"
    ```
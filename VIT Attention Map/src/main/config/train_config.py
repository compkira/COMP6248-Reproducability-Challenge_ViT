from src.main.config.config_base import ConfigBase


class TrainConfig(ConfigBase):
    device = 'cuda:0'

    batch_size = 1
    num_workers = 1
    max_epoch = 100
    print_freq = 200

    loss_function = "CrossEntropyLoss"

    # parameters for the optimizer
    optimizer_name = "SGD"
    lr = 0.03
    momentum = 0.9
    weight_decay = 1e-5
    nesterov = True
    

    # parameters for the scheduler
    scheduler_name = "StepLR"
    step_size = 20
    gamma = 0.3

    use_visdom = False
    initialize_model = False
    load_model = False
    save_model = False

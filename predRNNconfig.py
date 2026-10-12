import torch


class PredRNNConfig:
    #PredRNN-V2 architecture
    input_length = 5
    output_length = 5
    total_length = 10
    
    img_width = 64
    img_channel = 1

    patch_size = 4
    num_hidden = "64,64,64,64"
    filter_size = 5
    stride = 1
    layer_norm = 1
    decouple_beta = 0.1
    reverse_scheduled_sampling = 1


    #Training
    batch_size = 10
    num_epochs = 100
    num_workers = 0

    lr = 1e-4
    lr_beta1 = 0.5
    lr_beta2 = 0.9
    optimizer_eps = 1e-7

    start_epoch_saving = 1

    seed = 42
    use_seed = False

    #Dataset 
    dataset = "taasrad19"
    dataset_size = 1.0
    normalize_values = True
    split_validation = False

    working_dir = "./"
    result_folder_suffix = ""
    
    #Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

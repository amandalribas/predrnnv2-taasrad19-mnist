Ambiente virtual: 
python -m venv.venv

#### Informações sobre os argumentos de entrada: 

python trainer.py --help
python evaluator.py --help

Datasets disponíveis: 

taasrad19
moving_mnist


#### Exemplos de comando:
###### Teste de 1 Época com Moving Mnist:

python trainer.py --dataset moving_mnist --num_epochs 1 --batch_size 2 --result_folder_suffix teste_1_epoca

python predictor.py --dataset moving_mnist --dataset_mode test --batch_size 1 --num_workers 0 --normalize_values --model_weight_file_path "results/moving_mnist_teste_1_epoca/weights/moving_mnist_predrnn_v2_1.pth" 

python trainer.py --dataset taasrad19 --num_epochs 1 --batch_size 2 --result_folder_suffix teste_1_epoca

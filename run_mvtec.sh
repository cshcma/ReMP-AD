
python train.py --config_path ./config/mvtec.yaml
python test.py --config_path ./config/mvtec.yaml --k_shot 4
python test.py --config_path ./config/mvtec.yaml --k_shot 2
python test.py --config_path ./config/mvtec.yaml --k_shot 1

python train.py --config_path ./config/visa.yaml
python test.py --config_path ./config/visa.yaml --k_shot 4
python test.py --config_path ./config/visa.yaml --k_shot 2
python test.py --config_path ./config/visa.yaml --k_shot 1



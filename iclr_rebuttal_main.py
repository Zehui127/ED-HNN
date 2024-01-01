import subprocess

Datasets = [ "musae_Twitch_FR", "musae_Twitch_EN","musae_Twitch_ES",
            "musae_Twitch_PT", "musae_Twitch_RU", "musae_Twitch_DE",
            "grand_ArteryAorta", "grand_ArteryCoronary", "grand_Breast", "grand_Brain",
            "grand_Leukemia", "grand_Lung", "grand_Stomach", "grand_Lungcancer", "grand_Stomachcancer",
            "grand_KidneyCancer", "amazon_Photo", "amazon_Computer",
            "musae_Facebook", "musae_Github"]


# for dataset in Datasets:
#     data_dir = f"/home/zl6222/repositories/ED-HNN/data/{dataset.lower()}_local"
#     raw_data_dir = "/home/zl6222/repositories/ED-HNN/raw_data/coauthorship/cora"  # Assuming the same raw_data_dir for all datasets

#     command = f"CUDA_VISIBLE_DEVICES='2' python train.py --method EDGNN --dname {dataset} " \
#               f"--All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 " \
#               f"--Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean " \
#               f"--restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 5 --cuda 0 " \
#               f"--data_dir {data_dir} --raw_data_dir {raw_data_dir}"

#     subprocess.run(command, shell=True, check=True)

# Datasets = ["musae_Twitch_ES",
#             "grand_ArteryAorta",
#             "grand_Leukemia","amazon_Photo",
#             "musae_Facebook","musae_Github"]

# Update the dataset names and directories as needed
for dataset in Datasets:
    data_dir = f"/home/zl6222/repositories/ED-HNN/data/{dataset.lower()}_local"
    raw_data_dir = "/home/zl6222/repositories/ED-HNN/raw_data/coauthorship/cora"
    command = f"CUDA_VISIBLE_DEVICES='2' python train.py --method EDGNN --dname {dataset} " \
              f"--All_num_layers 2 --MLP_num_layers 2 --MLP2_num_layers 2 --MLP3_num_layers 2 " \
              f"--Classifier_num_layers 2 --MLP_hidden 512 --Classifier_hidden 256 --aggregate mean " \
              f"--restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 2 --feature_noise 1.0 --cuda 0 " \
              f"--data_dir {data_dir} --raw_data_dir {raw_data_dir}"
    subprocess.run(command, shell=True, check=True)

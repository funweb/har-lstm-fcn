dataset_dir_format = "\'../datasets/casas/ende/{dataname}/{distant}/npy/3/{dataname}-{type}-{data_type}-{k}.npy\',  # {index}"

type = "test"  # train or test
data_type = "x"  # X or Y
index = 0
for dataname in ["cairo", "milan", "kyoto7", "kyoto8", "kyoto11"]:
    for distant in ["9999", "999", "1", "2", "3", "4", "5"]:
        for k in range(3):
            print(dataset_dir_format.format(dataname=dataname, distant=distant, type=type, data_type=data_type, k=k, index=index))
            index = index + 1


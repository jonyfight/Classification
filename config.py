class DefaultConfigs(object):
    #1.string parameters
    train_data = "E:/Smart_Image_Project/Images_Classification/pytorch_hand_classifier-master/data/train1/"  # /home/user/zcj/tutorials/data/train/
    test_data = "F:/data_images2/test_origin1/"
    val_data = "E:/Smart_Image_Project/Images_Classification/pytorch_hand_classifier-master/data/val1/"  # /home/user/zcj/tutorials/data/val/
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"  # 1
    augmen_level = "medium"  # "none", "light", "medium", "hard","hard2"

    #2.numeric parameters
    epochs = 150  #100
    batch_size = 4  # 原本最多设置为3
    img_height = 512  # 300
    img_weight = 512  # 300
    num_classes = 2  # 62
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()

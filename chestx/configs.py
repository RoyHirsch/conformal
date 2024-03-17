import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Train/test medical imaging models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--model_name", type=str, default="resnet18", dest="model_name",
                        help="model name to train/test")
    parser.add_argument("--n_classes", type=int, default=15, help="number of classes")

    # Chexport args
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.0, help="noise ratio")

    # ---- Relevant for T est ---- #
    parser.add_argument("--lam", type=float, default=0.0001, help="noise ratio")
    parser.add_argument("--lr_lam", type=float, default=0.01, help="noise ratio")
    parser.add_argument("--opt", type=str, default='Adam')


    return parser.parse_args()
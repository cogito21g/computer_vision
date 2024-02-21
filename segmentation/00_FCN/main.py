import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--verbose", help="show status", default=1, type=int)
    args.add_argument("--epoch", help="select epoch", default=20, type=int)
    args.add_argument("--seed", help="fix random seed", default=2024, type=int)
    args.add_argument("--lr", help="select learning rate", default=0.0001, type=float)
    args.add_argument("--batch_size", help="select batch szie", default=4, type=int)
    args.add_argument("--mode", help="select mode(train/test)", default="test", type=str)
    args.add_argument("--save_dir", help="select save directory", default="mymodel", type=str)
    
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args.verbose)
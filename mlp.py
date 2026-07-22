from src.parsing import parsing_command_line
from src.split.split import split_dataset
from src.train.train import train
from src.predict.predict import predict


def main():
    args = parsing_command_line()

    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "split-dataset":
        split_dataset(args)


if __name__ == "__main__":
    main()

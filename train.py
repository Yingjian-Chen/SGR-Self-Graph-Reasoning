from src.seed import seed_everything
from data.utils.dataset import ReasoningDataset
from torch.utils.data import DataLoader
from model.reasoningLLM_ori import ReasoningLLM
from src.config import parse_args
from model.trainer import Trainer
from src.collate import collate_fn
import os


def main(args):
    # 1. set up
    seed = args.seed
    seed_everything(seed)
    print(args)

    # 2. Dataset Building
    dataset = ReasoningDataset(args.dataset_name)
    split = dataset.get_splits()

    train_dataset = [dataset[i] for i in split['train']]
    val_dataset = [dataset[i] for i in split['val']]

    # 3. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Model Building
    model = ReasoningLLM(args)

    # checkpoint_path = os.path.join(
    #     args.output_dir, f'{args.project_name}_{args.seed}_{args.llm_model_name}/best_model'
    # )
    # model.load_lora_train(lora_dir=checkpoint_path)

    # 5. Training
    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.train()
    

if __name__ == "__main__":
    args = parse_args()
    print("===== Configuration =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=========================")
    
    main(args)
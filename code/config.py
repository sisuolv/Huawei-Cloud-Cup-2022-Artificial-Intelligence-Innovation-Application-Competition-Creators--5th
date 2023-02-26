import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="XXXXXXXXXXXXXXXXX")

    parser.add_argument("--seed", type=int, default=2021, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--traindata_path', type=str, default='./train.json')

    parser.add_argument('--prefetch', default=8, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='./save')
    parser.add_argument('--ckpt_file', type=str, default='./save/model_double_be.bin')
    parser.add_argument('--best_score', default=0.01, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================

    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='./medbertbase/') #medbert-base-wwm-chinese
    
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=1024)
    
    # ========================== Awp =============================
    parser.add_argument('--attack_func', type=str, default='nono')
    parser.add_argument('--ema', type=bool, default=True)

    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--bert_learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=21, help='How many epochs')
    parser.add_argument('--batch_size', default=8, type=int, help="use for training duration per worker")
    parser.add_argument('--print_steps', type=int, default=50, help="Number of steps to log training metrics.")
    
    parser.add_argument('--pre_ckpt_file', type=str, default='')
    
    

    return parser.parse_args()

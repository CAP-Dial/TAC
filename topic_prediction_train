import argparse
import os
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from transformers import BertTokenizer, BertForSequenceClassification

# ======================
# 1. 话题列表 & 映射
# ======================

topics = [
    "不上学/间断上学", "专业", "亲人生病或离世", "优势科目", "偏科情况",
    "做事时乐趣是否比平时少", "入睡困难", "劣势科目", "学校类型", "家庭结构",
    "寄宿", "年级", "成绩", "抚养人", "整体感受", "是否交一些家人认为不太好的朋友",
    "是否受欺负", "是否尝试自杀", "是否悲伤或抑郁数小时", "是否感到躯体疼痛",
    "是否担心健康或生病", "是否担心碰的东西是脏的或有细菌或被投毒",
    "是否无法停止担忧", "是否易愤怒或发脾气", "是否暴饮暴食", "是否有想法持续进入脑中",
    "是否有离家出走", "是否有自伤行为", "是否由于紧张不能做想做的事或应该做的事",
    "是否看见他人看不见的事物或人", "是否难以集中注意力", "母亲专注力是否集中",
    "母亲文化程度", "母亲是否有产后焦虑抑郁", "母亲是否独生", "母亲生产年龄",
    "沉迷手机", "沉迷游戏", "父亲专注力是否集中", "父亲文化程度", "父母离异或关系不好",
    "特长或爱好", "生活习惯", "睡眠", "睡眠维持困难/早醒", "结束语", "记忆障碍",
    "是否饮酒", "父亲不良嗜好", "父亲是否独生", "偏差行为", "是否有时特别爱花钱",
    "沉迷动漫", "囤积卡片手办等", "成绩下降", "是否想到自杀或自杀相关的事",
    "是否曾打砸东西或他人", "家族史", "是否分不清梦境还是现实", "是否感到需要反复检查某些事",
    "是否觉得虚无缥缈不真实", "母亲不良嗜好", "家庭氛围", "学校", "是否吸烟",
    "是否容易被激惹或烦恼", "母亲排行第几", "沉迷追星", "父亲排行第几？", "背叛",
    "转学经历", "是否觉得恍惚", "是否为了防止坏事发生不得不以某种方式做事",
    "是否说谎", "其他"
]

topic_to_id = {topic: idx for idx, topic in enumerate(topics)}
num_topics = len(topics)


# ======================
# 2. 数据集定义
# ======================

class TopicDataset(Dataset):
    """
    每一行数据格式：
    {
        "history_topics": ["成绩", "偏科情况", ...],
        "current_topic": "成绩",
        "next_topic": "偏科情况"
    }
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        history = item['history_topics']
        current = item['current_topic']
        next_topic = item['next_topic']

        current_topic_id = topic_to_id[current]
        label_id = topic_to_id[next_topic]

        # 文本输入：history + current_topic
        input_text = " ".join(history) + " " + current

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),       # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0), # [max_length]
            'labels': torch.tensor(label_id, dtype=torch.long),
            'current_topic_id': torch.tensor(current_topic_id, dtype=torch.long)
        }


# ======================
# 3. 转移矩阵加载
# ======================

def load_transition_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0)

    # 只检查尺寸是否正确（75×75）
    if df.shape != (len(topics), len(topics)):
        raise ValueError(f"转移矩阵形状不对，期望 {(len(topics), len(topics))}，实际 {df.shape}")

    mat = torch.tensor(df.values, dtype=torch.float32)
    return mat



# ======================
# 4. Loss 计算：交叉熵 / 交叉熵 + KL
# ======================

def compute_loss(mask_logits, labels, prior_probs, kl_weight=0.0, loss_type='cross_entropy'):
    """
    mask_logits: [B, num_topics]  已加权后的 logits
    labels:      [B]
    prior_probs: [B, num_topics]  当前话题对应的先验转移概率（每行已经归一化）
    """
    eps = 1e-8

    # 交叉熵部分
    ce_loss = F.cross_entropy(mask_logits, labels)

    if loss_type == 'cross_entropy':
        return ce_loss

    elif loss_type == 'cross_entropy_kl':
        # 模型预测分布 p_model
        p_model = F.softmax(mask_logits, dim=-1)                       # [B, num_topics]
        log_p_model = torch.log(p_model + eps)                         # [B, num_topics]

        # 先验分布 p_prior
        p_prior = prior_probs                                         # 已归一化 [B, num_topics]
        log_p_prior = torch.log(p_prior + eps)

        # KL(p_model || p_prior) = sum p_model * (log p_model - log p_prior)
        kl_per_sample = torch.sum(p_model * (log_p_model - log_p_prior), dim=-1)  # [B]
        kl_loss = kl_per_sample.mean()

        return ce_loss + kl_weight * kl_loss

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ======================
# 5. 训练与验证
# ======================

def train_one_epoch(args, model, dataloader, optimizer, scheduler, transition_matrix, epoch, writer=None):
    model.train()
    device = args.device
    transition_matrix = transition_matrix.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    eps = 1e-8

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        current_topic_id = batch['current_topic_id'].to(device)  # [B]

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, num_topics]

        # —— 取出当前话题对应的先验转移概率行 —— #
        prior_rows = transition_matrix[current_topic_id]  # [B, num_topics]

        # 行归一化为概率
        row_sums = prior_rows.sum(dim=-1, keepdim=True)   # [B,1]
        prior_probs = prior_rows / torch.clamp(row_sums, min=eps)

        # log 先验
        log_prior = torch.log(prior_probs + eps)          # [B, num_topics]

        # —— 先验加权（控制模块核心）：mask_logits = logits + alpha * log A[i,*] —— #
        mask_logits = logits + args.alpha * log_prior     # [B, num_topics]

        # —— 计算 loss —— #
        loss = compute_loss(
            mask_logits=mask_logits,
            labels=labels,
            prior_probs=prior_probs,
            kl_weight=args.kl_weight,
            loss_type=args.loss_type
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        # 计算准确率（基于 mask_logits 的最终预测）
        preds = torch.argmax(mask_logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # TensorBoard 迭代日志
        global_step = epoch * len(dataloader) + step
        if writer is not None:
            writer.add_scalar('Iter/train_loss', loss.item(), global_step)

        if (step + 1) % args.print_interval == 0:
            print(f"[Epoch {epoch+1} Step {step+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    scheduler.step()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    if writer is not None:
        writer.add_scalar('Epoch/train_loss', avg_loss, epoch)
        writer.add_scalar('Epoch/train_acc', avg_acc, epoch)

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(args, model, dataloader, transition_matrix, epoch, writer=None):
    model.eval()
    device = args.device
    transition_matrix = transition_matrix.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    eps = 1e-8

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        current_topic_id = batch['current_topic_id'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, num_topics]

        prior_rows = transition_matrix[current_topic_id]  # [B, num_topics]
        row_sums = prior_rows.sum(dim=-1, keepdim=True)
        prior_probs = prior_rows / torch.clamp(row_sums, min=eps)

        log_prior = torch.log(prior_probs + eps)

        mask_logits = logits + args.alpha * log_prior

        loss = compute_loss(
            mask_logits=mask_logits,
            labels=labels,
            prior_probs=prior_probs,
            kl_weight=args.kl_weight,
            loss_type=args.loss_type
        )

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(mask_logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    if writer is not None:
        writer.add_scalar('Epoch/val_loss', avg_loss, epoch)
        writer.add_scalar('Epoch/val_acc', avg_acc, epoch)

    return avg_loss, avg_acc


# ======================
# 6. 工具函数
# ======================

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================
# 7. 参数解析
# ======================

def parse_args():
    parser = argparse.ArgumentParser(description="Topic prediction with BERT + transition matrix (control module).")

    # 核心超参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pretrained BERT model name or path.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to training dataset (.jsonl).')
    parser.add_argument('--val_dataset_path', type=str, default=None,
                        help='Optional validation dataset path (.jsonl). If None,不做验证。')

    parser.add_argument('--transition_matrix_path', type=str, default='data/transition_matrix.csv',
                        help='CSV file path of topic transition matrix.')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'cross_entropy_kl'],
                        help='Loss type: cross_entropy or cross_entropy_kl')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for log prior from transition matrix.')
    parser.add_argument('--kl_weight', type=float, default=0.5,
                        help='Weight for KL divergence term when using cross_entropy_kl.')

    # 训练细节
    parser.add_argument('--device', type=str, default=None,
                        help='cuda or cpu. If None, auto-detect.')
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 70],
                        help='Milestones for MultiStepLR.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR decay factor for MultiStepLR.')
    parser.add_argument('--print_interval', type=int, default=10)

    # 日志 & 保存
    parser.add_argument('--log_tensorboard', action='store_true',
                        help='Whether to log to TensorBoard.')
    parser.add_argument('--tensorboard_path', type=str, default='./logs',
                        help='TensorBoard log dir.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints.')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save model every N epochs.')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ======================
# 8. 主函数
# ======================

def main():
    args = parse_args()

    # 设备
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    # 固定随机种子
    set_seed(args.seed)

    # 日志与保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    writer = None
    if args.log_tensorboard:
        writer = SummaryWriter(log_dir=args.tensorboard_path)

    # tokenizer & model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_topics
    )

    if torch.cuda.device_count() > 1 and 'cuda' in args.device:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    model.to(args.device)

    # 数据
    train_data = load_jsonl(args.dataset_path)
    print(f"Loaded train data: {len(train_data)} samples.")
    train_dataset = TopicDataset(train_data, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_dataset_path is not None:
        val_data = load_jsonl(args.val_dataset_path)
        print(f"Loaded val data: {len(val_data)} samples.")
        val_dataset = TopicDataset(val_data, tokenizer, max_length=args.max_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 转移矩阵
    transition_matrix = load_transition_matrix(args.transition_matrix_path)
    print(f"Transition matrix shape: {transition_matrix.shape}")  # [75, 75]

    # 优化器 & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

        train_loss, train_acc = train_one_epoch(
            args, model, train_loader, optimizer, scheduler,
            transition_matrix, epoch, writer
        )
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

        if val_loader is not None:
            val_loss, val_acc = evaluate(
                args, model, val_loader, transition_matrix, epoch, writer
            )
            print(f"Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.4f}")

            # 保存最优
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model to {best_path}")
        else:
            val_loss = None

        # 按间隔保存当前 epoch
        if (epoch + 1) % args.save_interval == 0:
            epoch_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), epoch_path)
            print(f"Saved checkpoint to {epoch_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()


""" 纯交叉熵
python train_kl_loss.py \
  --model_name bert-base-uncased \
  --dataset_path data/train_dataset_topic_split.jsonl \
  --transition_matrix_path data/transition_matrix.csv \
  --loss_type cross_entropy \
  --alpha 0.0 \
  --kl_weight 0.0 \
  --save_dir ./ce_loss_no_alpha \
  --log_tensorboard
"""

""" 交叉熵+KL
python train_kl_loss.py \
  --model_name bert-base-uncased \
  --dataset_path data/train_dataset_topic_split.jsonl \
  --transition_matrix_path data/transition_matrix.csv \
  --loss_type cross_entropy_kl \
  --alpha 0.0 \
  --kl_weight 0.5 \
  --save_dir ./kl_loss_no_alpha \
  --log_tensorboard

"""

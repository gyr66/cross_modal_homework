import torch

from loss import LanguageModelLoss, ContrastiveLoss
from evaluate import eval
from data import F30kDataset, collate_fn
from tqdm import tqdm


def train_epoch(
    epoch, config, dataloader, model, optimizer, lm_loss_fn, contrastive_loss_fn
):
    model.train()
    for train_data in tqdm(
        dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{config.epochs}"
    ):
        images, captions, masks, valid_length, _ = train_data
        images = images.to(config.device)
        captions = captions.to(config.device)
        masks = masks.to(config.device)

        img_feats, caption_feats, caption_generator_ouput = model(
            images, captions, valid_length
        )

        lm_loss = lm_loss_fn(caption_generator_ouput, captions[:, 1:], masks[:, 1:])
        contrastive_loss = contrastive_loss_fn(img_feats, caption_feats)
        loss = lm_loss + contrastive_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lm_loss.item(), contrastive_loss.item()


def train(config, model, optimizer, train_dataloader, val_dataloader):
    lm_loss_fn = LanguageModelLoss()
    contrastive_loss_fn = ContrastiveLoss(config.margin)

    best_score = 0
    for epoch in range(config.epochs):
        print(f"\nTrain on epoch {epoch + 1 } / {config.epochs}\n")
        lm_loss, contrastive_loss = train_epoch(
            epoch,
            config,
            train_dataloader,
            model,
            optimizer,
            lm_loss_fn,
            contrastive_loss_fn,
        )
        print(
            "Epoch %d, lm_loss: %.2f, contrastive_loss: %.2f"
            % (epoch, lm_loss, contrastive_loss)
        )
        print(f"\nValidation on epoch {epoch + 1} / {config.epochs}\n")
        score = validate(config, val_dataloader, model)
        if score > best_score:
            best_score = score
            torch.save(
                {"config": config, "model_state_dict": model.state_dict()},
                "best_model.pth",
            )


def validate(config, val_loader, model):
    result, score = eval(config, val_loader, model)
    print(
        "Image to text: %.1f, %.1f, %.1f"
        % (
            result["image2text"]["r1"],
            result["image2text"]["r5"],
            result["image2text"]["r10"],
        )
    )
    print(
        "Text to image: %.1f, %.1f, %.1f"
        % (
            result["text2image"]["r1"],
            result["text2image"]["r5"],
            result["text2image"]["r10"],
        )
    )
    return score


if __name__ == "__main__":
    import argparse
    import torch
    from torch.utils.data import DataLoader

    from data import F30kDataset, collate_fn
    from models import VSRN
    from vocab import load_vocab, Vocabulary

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vocab_path", type=str, default="vocab.pkl")
    parser.add_argument("--dim_word", type=int, default=300)
    parser.add_argument("--dim_embed", type=int, default=2048)
    parser.add_argument("--dim_hidden", type=int, default=512)
    parser.add_argument("--dim_image", type=int, default=2048)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--do_resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")

    args = parser.parse_args()

    vocab = load_vocab(args.vocab_path)

    config = args
    config.vocab_size = len(vocab)

    train_dataset = F30kDataset(args.data_path, "train", vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataset = F30kDataset(args.data_path, "dev", vocab)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = VSRN(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.do_resume:
        if config.checkpoint is None:
            raise ValueError("checkpoint is required when do_resume is True")
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resume from {config.checkpoint}")

    train(config, model, optimizer, train_loader, val_loader)

import numpy as np
import torch
from tqdm import tqdm


def eval(config, dataloader, model):
    model.to(config.device)
    model.eval()

    img_feats = np.zeros((len(dataloader.dataset), config.dim_embed))
    cap_feats = np.zeros((len(dataloader.dataset), config.dim_embed))
    for eval_data in tqdm(dataloader, total=len(dataloader)):
        images, captions, masks, valid_length, index = eval_data
        images = images.to(config.device)
        captions = captions.to(config.device)
        masks = masks.to(config.device)
        with torch.no_grad():
            img_feats_batch, cap_feats_batch, _ = model(images, captions, valid_length)
        img_feats[np.array(index)] = img_feats_batch.cpu().numpy().copy()
        cap_feats[np.array(index)] = cap_feats_batch.cpu().numpy().copy()

    r_i2t = i2t(img_feats, cap_feats)
    r_t2i = t2i(img_feats, cap_feats)

    result = {
        "text2image": {"r1": r_t2i[0], "r5": r_t2i[1], "r10": r_t2i[2]},
        "image2text": {"r1": r_i2t[0], "r5": r_i2t[1], "r10": r_i2t[2]},
    }

    score = r_i2t[0] + r_i2t[1] + r_t2i[0] + r_t2i[1]

    return result, score


def i2t(
    images,
    captions,
):
    image_cnt = images.shape[0] // 5
    ranks = np.zeros(image_cnt)
    for index in range(image_cnt):
        im = images[5 * index].reshape(1, images.shape[1])
        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        rank = 1e20
        for i in range(5 * index, 5 * index + 5):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return r1, r5, r10


def t2i(
    images,
    captions,
):
    image_cnt = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])
    ranks = np.zeros(5 * image_cnt)
    for index in range(image_cnt):
        queries = captions[5 * index : 5 * index + 5]
        d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return r1, r5, r10


if __name__ == "__main__":
    import argparse
    import json
    import torch
    from torch.utils.data import DataLoader
    from data import F30kDataset, collate_fn
    from vocab import load_vocab, Vocabulary

    from models import VSRN

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", type=str, default="vocab.pkl")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--split", type=str, default="dev")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    config = checkpoint["config"]
    model = VSRN(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    vocab = load_vocab(args.vocab_path)

    dataset = F30kDataset(args.data_path, args.split, vocab)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    result, _ = eval(config, dataloader, model)
    print(json.dumps(result, indent=4))
    with open("result.json", "w") as f:
        json.dump(result, f)

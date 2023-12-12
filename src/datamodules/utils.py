import torch


def get_example_data(dataloader, device, n=5):
    """
    Samples the dataloader and returns a zipped list of examples
    """
    images = []
    img_embeddings = []
    text_embeddings = []
    captions = []
    for img, emb, txt in dataloader:
        img_emb, text_emb = emb.get("img"), emb.get("text")
        if img_emb is not None:
            img_emb = img_emb.to(device=device, dtype=torch.float)
            img_embeddings.extend(list(img_emb))
        else:
            # Then we add None img.shape[0] times
            img_embeddings.extend([None] * img.shape[0])
        if text_emb is not None:
            text_emb = text_emb.to(device=device, dtype=torch.float)
            text_embeddings.extend(list(text_emb))
        else:
            # Then we add None img.shape[0] times
            text_embeddings.extend([None] * img.shape[0])
        img = img.to(device=device, dtype=torch.float)
        images.extend(list(img))
        captions.extend(list(txt))
        if len(images) >= n:
            break
    return list(zip(images[:n], img_embeddings[:n], text_embeddings[:n], captions[:n]))


def split_test_full_data(batch, device):
    images = []
    img_embeddings = []
    text_embeddings = []
    captions = []
    img, emb, txt = batch
    img_emb, text_emb = emb.get("img"), emb.get("text")
    img_emb = img_emb.to(device=device, dtype=torch.float)
    img_embeddings.extend(list(img_emb))
    text_emb = text_emb.to(device=device, dtype=torch.float)
    text_embeddings.extend(list(text_emb))
    img = img.to(device=device, dtype=torch.float)
    images.extend(list(img))
    captions.extend(list(txt))

    test_example_data = list(zip(images, img_embeddings, text_embeddings, captions))
    return test_example_data

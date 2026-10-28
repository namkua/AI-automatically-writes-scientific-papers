from model.convneur import ConvNeur

def build_model(scale="m1", num_classes=1000, img_size=224):
    """
    Factory function for ConvNeur models.
    Supports ImageNet (img_size=224, chunk_size=196) and CIFAR/toy (img_size=32, chunk_size=16).
    """
    chunk_size = 196 if img_size >= 128 else 16
    use_mem_batch = True if img_size >= 128 else False

    scales = {
        "m1": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320], "mem_dim": 80},
        "m2": {"depths": [2, 2, 6, 2], "dims": [48, 96, 192, 384], "mem_dim": 96},
        "m3": {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512], "mem_dim": 128},
        "m4": {"depths": [2, 2, 8, 2], "dims": [80, 160, 320, 640], "mem_dim": 160},
    }

    if scale not in scales:
        raise ValueError(f"Scale {scale} not supported. Choices: {list(scales.keys())}")

    cfg = scales[scale]
    return ConvNeur(
        num_classes=num_classes,
        depths=cfg["depths"],
        dims=cfg["dims"],
        mem_dim=cfg["mem_dim"],
        chunk_size=chunk_size,
        use_mem_batch_size=use_mem_batch,
    )

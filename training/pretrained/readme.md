# Pretrained weights

Place the following weight files in this folder (`./training/pretrained/`).

| File | Source | Needed for |
|------|--------|------------|
| `ms1mv3_arcface_r100_fp16.pth` | [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) (ArcFace, MS1MV3, R100) | **Required** — the IAFM identity branch |
| `xception-b5690688.pth` | [DeepfakeBench `pretrained.zip`](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip) | Xception backbone variant only |
| `efficientnet-b4-6ed6700e.pth` | [DeepfakeBench `pretrained.zip`](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip) | EfficientNet-B4 backbone variant only |

The default SELFI configuration uses the **CLIP ViT-B/16** backbone
(`openai/clip-vit-base-patch16`), which is downloaded automatically from HuggingFace on first run,
so only `ms1mv3_arcface_r100_fp16.pth` is strictly required.

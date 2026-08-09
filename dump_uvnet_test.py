# dump_uvnet_test.py
# S6 推理适配器(UV-Net)——本文件为本方新增脚本, 官方仓库零改动。
# v1.0(2026-08-09): 用官方组件拼装全量测试集逐面 dump:
#   模型/数据/前向 = 官方(Segmentation.load_from_checkpoint + MFCADDataset + 官方 permute 前向,
#   逐字复刻 uvnet/models.py training_step 的输入处理);
#   唯一刻意偏离 = DataLoader(shuffle=False, drop_last=False) 全量 8922
#   (官方 get_dataloader 硬编码 drop_last=True 会丢尾批 26 样本, 见协议 §26 发现 10)。
# 输出: <out_dir>/<stem>.pred 与 <stem>.gt, 一行一个类别索引, 面顺序=数据集图节点顺序。
import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from datasets.mfcad import MFCADDataset
from uvnet.models import Segmentation


def main():
    parser = argparse.ArgumentParser(description="S6: dump UV-Net per-face predictions on full test split")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S6] device = {device}")

    # 官方权重加载路径(超参数 num_classes/crv_in_channels 随 ckpt 恢复)
    model = Segmentation.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    # 官方数据集(test split, 无随机旋转 —— 与官方 test 分支一致)
    test_data = MFCADDataset(root_dir=args.dataset_path, split="test", random_rotate=False)
    print(f"[S6] test samples loaded = {len(test_data)}")

    # 唯一偏离点: 全量评测(不丢尾批), collate 仍用官方 _collate(携带 filename)
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_data._collate,
        drop_last=False,
    )

    n_samples = 0
    n_faces = 0
    with torch.no_grad():
        for batch in loader:
            filenames = batch["filename"]
            graphs = batch["graph"].to(device)
            # 官方前向输入处理(uvnet/models.py:146-147 同款 permute)
            graphs.ndata["x"] = graphs.ndata["x"].permute(0, 3, 1, 2)
            graphs.edata["x"] = graphs.edata["x"].permute(0, 2, 1)
            logits = model(graphs)                      # [total_nodes, num_classes]
            preds = torch.argmax(logits, dim=-1)        # [total_nodes]
            labels = graphs.ndata["y"]

            counts = graphs.batch_num_nodes().tolist()  # 每图节点数, 顺序与 filenames 一致
            offset = 0
            for fn, n in zip(filenames, counts):
                p = preds[offset:offset + n].cpu().tolist()
                g = labels[offset:offset + n].cpu().tolist()
                offset += n
                with open(out_dir / f"{fn}.pred", "w") as f:
                    f.write("\n".join(str(int(v)) for v in p) + "\n")
                with open(out_dir / f"{fn}.gt", "w") as f:
                    f.write("\n".join(str(int(v)) for v in g) + "\n")
                n_samples += 1
                n_faces += n

    print(f"[S6] UV-Net dump done: samples={n_samples}, faces={n_faces}, out={out_dir}")


if __name__ == "__main__":
    main()

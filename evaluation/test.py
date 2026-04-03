import matplotlib.pyplot as plt
from ortholoc.dataset import OrthoLoC
import torch
import numpy as np

from evaluation.dataset import SyntheticDatasetGenerator


def main():
    print("Loading dataset...")

  
    dataset = OrthoLoC(limit_size=3)

    sample = dataset[0]

    print("Available keys:", sample.keys())

   
    map_img = sample["image_dop"]
    if isinstance(map_img, torch.Tensor):
        map_img = map_img.detach().cpu().numpy()


    if map_img.ndim == 3 and map_img.shape[0] in (1, 3):
        map_img = np.transpose(map_img, (1, 2, 0))

    print("Map shape:", map_img.shape)


    plt.figure(figsize=(6, 6))
    map_img_vis = (map_img - map_img.min()) / (map_img.max() - map_img.min() + 1e-8)
    plt.imshow(map_img_vis)
    plt.title("Orthophoto (map_img)")
    plt.axis("off")

    generator = SyntheticDatasetGenerator(map_img)
    frames = generator.generate(5, crop_size=(128, 128))


    n_show = min(5, len(frames))
    print(f"Generated frames: {len(frames)} (showing {n_show})")

    plt.figure(figsize=(3 * n_show, 3))
    for i in range(n_show):
        frame = frames[i]
        print(f"Frame {i}: shape={frame.uav_img.shape}, gt={frame.ground_truth_px}")
        plt.subplot(1, n_show, i + 1)
        plt.imshow(frame.uav_img)
        plt.title(f"UAV #{i}")
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()

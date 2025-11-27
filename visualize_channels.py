import os
import cv2
import glob
import matplotlib.pyplot as plt

# Base folders
folder_rgb = r"dataset\Spunta\leaf_images\RGBFilter"
folder_pgm = os.path.join(folder_rgb, "channels_pgm")

# Collect both healthy and unhealthy patches
healthy = sorted(glob.glob(os.path.join(folder_rgb, "*healthybox*RGBpatch*.png")))
unhealthy = sorted(glob.glob(os.path.join(folder_rgb, "*unhealthy*RGBpatch*.png")))

print("Found healthy:", len(healthy))
print("Found unhealthy:", len(unhealthy))

# Pick 2 samples of each (if available)
samples = healthy[:2] + unhealthy[:2]

print("\nUsing files:")
for s in samples:
    print("  ", os.path.basename(s))


def show_channels(rgb_path):
    base = os.path.basename(rgb_path).replace(".png", "")
    
    # Load RGB
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    channel_map = {
        "R":  base.replace("RGBpatch", "Rpatch") + ".pgm",
        "G":  base.replace("RGBpatch", "Gpatch") + ".pgm",
        "B":  base.replace("RGBpatch", "Bpatch") + ".pgm",
        "H":  base.replace("RGBpatch", "Hpatch") + ".pgm",
        "S":  base.replace("RGBpatch", "Spatch") + ".pgm",
        "V":  base.replace("RGBpatch", "Vpatch") + ".pgm",
        "L":  base.replace("RGBpatch", "Lpatch") + ".pgm",
        "u":  base.replace("RGBpatch", "upatch") + ".pgm",
        "v":  base.replace("RGBpatch", "vupatch") + ".pgm",
    }

    # Load each channel
    channels = {}
    for cname, fname in channel_map.items():
        path = os.path.join(folder_pgm, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        channels[cname] = img

    # ---- PLOT ALL CHANNELS ----

    plt.figure(figsize=(16, 10))
    plt.suptitle(f"Channels for: {os.path.basename(rgb_path)}", fontsize=14)

    # RGB
    plt.subplot(3, 4, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    # Each channel
    idx = 2
    for cname in ["R","G","B","H","S","V","L","u","v"]:
        plt.subplot(3, 4, idx)
        plt.title(cname)
        plt.axis("off")
        if channels[cname] is None:
            plt.text(0.5, 0.5, "Missing", ha="center", va="center")
        else:
            plt.imshow(channels[cname], cmap="gray")
        idx += 1

    plt.tight_layout()
    plt.show()


for sample in samples:
    show_channels(sample)

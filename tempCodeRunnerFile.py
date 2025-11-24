def save_histogram(G, out_path, title="Histogram"):
    """
    Plots the original binned granulometry histogram from the .m file.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(
        G,
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(label="Count")
    plt.xlabel("Size index")
    plt.ylabel("Shape index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

from config import FIGURE_DIR

def ensure_dir(path):
    """주어진 path 디렉토리가 없으면 생성"""
    import os
    os.makedirs(path, exist_ok=True)


def plot_loss(epochs_list, total_losses, final_losses, elbo_losses, filename='training_losses.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, total_losses, label='Total Loss')
    plt.plot(epochs_list, final_losses, label='Final (MSE) Loss')
    plt.plot(epochs_list, elbo_losses, label='ELBO Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR + filename, dpi=600)
    plt.close()


def plot_inducing_points(inducing_points_list, epochs, num_inducing_points, filename='inducing_points_evolution.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    for i in range(num_inducing_points):
        plt.scatter(
            range(1, epochs + 1),
            [points[i, 0] for points in inducing_points_list if i < points.shape[0]],
            label=f'Inducing Point {i+1}',
            s=10,
        )
    plt.xlabel('Epoch')
    plt.ylabel('Inducing Point Value')
    plt.title('Evolution of Inducing Points During Training')
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR + filename, dpi=600)
    plt.close()

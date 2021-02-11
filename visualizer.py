import matplotlib.pyplot as plt
import config


def visualizer(img, points=None, pred_points=None):
    plt.subplots()
    plt.imshow(img)
    if points is not None:
        points_re = points.reshape(config.num_fiber, config.num_points, -1)
        for i in range(config.num_fiber):
            plt.plot(points_re[i][:, 0], points_re[i][:, 1], linewidth=1, color='b')
    if pred_points is not None:
        pred_points_re = pred_points.reshape(config.num_fiber, config.num_points, -1)
        for i in range(config.num_fiber):
            plt.plot(pred_points_re[i][:, 0], pred_points_re[i][:, 1], linewidth=1, color='r')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def analysis(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    acc = history_dict['mae']
    val_acc = history_dict['mae']

    epochs = range(1, len(loss_values) + 1)

    # subplot( m , n , p ) divides the current figure into an m -by- n grid and
    # creates axes in the position specified by p
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_values, 'bo', label='Train loss')
    plt.plot(epochs, val_loss_values, 'b', label='Val loss')
    plt.title('Train and val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.plot(epochs, acc, 'bo', label='train acc')
    plt.plot(epochs, val_acc, 'b', label='val acc')
    plt.title('Train and val accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

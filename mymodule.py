from matplotlib import pyplot as plt
import numpy as np

# Функция для отрисовки результатов
def plot_res(hist, max_line=False, max_stable_line=None, main_metric=None, print_loss=False, figsize=(8, 6)):
    plt.figure(figsize=figsize)

    keys = hist.history.keys()
    keys = list(keys)

    if not print_loss:
        keys.remove('loss')
        keys.remove('val_loss')

    for k in keys:
        plt.plot(hist.history[k], label=k)

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')

    if main_metric is None:
        main_metric = keys[-1]

    if max_line:
        max_acc = round(max(hist.history[main_metric]) * 100)
        plt.hlines(max_acc / 100, 0, len(hist.history[main_metric]), label=f'Max {main_metric}: {max_acc}%', colors='g')

    if max_stable_line:
        avg_acc = round((np.mean(hist.history[main_metric][-max_stable_line:])) * 100)
        plt.hlines(avg_acc / 100, 0, len(hist.history[main_metric]),
                   label=f'Max stable {main_metric}: {avg_acc}%', colors='r')
    plt.legend()
    plt.show()

x = 100

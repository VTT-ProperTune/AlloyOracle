"""Visualising model performance."""

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def save_r2_plot(Y, Y_pred, config, only_show=False):
    """Save scatter plots comparing simulated vs predicted outputs.

    Produces one subplot per output and dataset and writes `r2_combined.png`.
    """
    key = list(Y.keys())[0]

    n_outputs = Y[key].shape[1]

    n_cols, n_rows = n_outputs, len(list(Y.keys()))

    fig = plt.figure(figsize=(4*n_cols,4*n_rows), constrained_layout=True)

    ax_idx = 1
    for key in Y:
        for i in range(n_outputs):

            true, pred = Y[key][:,i].reshape(-1, 1), Y_pred[key]['pred'][:,i].reshape(-1, 1)
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)
            ax = fig.add_subplot(n_rows, n_cols, ax_idx)
            ax.set_title('MAE {:.2%}, R2 {:.2%}'.format(mae, r2))
            ax.scatter(true, pred, marker='o', edgecolor='tab:orange', alpha=0.5, s=2, label='Samples') # , c='lime'
            ax.plot([true.min(), true.max()], [true.min(), true.max()], c='k', lw=2, label='$R^2=100%$')
            ax.set_xlabel('Simulated {}'.format(config['outputs'][i]))
            ax.set_ylabel('Predicted {}'.format(config['outputs'][i]))
            ax.plot(true, LinearRegression().fit(true, pred).predict(true), linestyle='dashed', c='k', lw=2, label='Linear Fit')
            ax.grid(alpha=0.5)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),ncol=2, fancybox=True)
            ax_idx += 1

    if only_show:
        plt.show()
    else:
        plt.savefig(config['path_res'] + '//r2_combined.png', dpi=100)
        plt.close('all')
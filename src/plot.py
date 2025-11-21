import matplotlib.pyplot as plt


def make_plot(trajectoryHistory,
              timeHistory,
              prediction_train,
              timeHistory_train,
              prediction,
              timeHistory_test,
              dim,
              plot_path=None):

    fig, axes = plt.subplots(nrows=dim,
                             ncols=1,
                             figsize=(12, 8),
                             sharex=True)
    fig.subplots_adjust(hspace=0, wspace=.1)

    # make title dependant on regression method?
    axes[0].set_title('NGRC results')

    for i in range(0, dim):
        axes[i].plot(timeHistory,
                     trajectoryHistory[i],
                     'b--',
                     label='solution')
        axes[i].plot(timeHistory_train,
                     prediction_train[i],
                     'g-',
                     label='training fit')
        axes[i].plot(timeHistory_test,
                     prediction[i],
                     'r-',
                     label='prediction')
        axes[i].set_xlabel('time')
        axes[i].set_ylabel(f'var {i+1}')

    axes[0].legend(loc="upper left",
                   bbox_to_anchor=(1.01, 0.99),
                   borderaxespad=0.0)

    fig.subplots_adjust(right=0.85)

    plt.savefig('result.png')

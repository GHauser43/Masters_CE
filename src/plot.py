import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def make_plot(trajectoryHistory,
              timeHistory,
              prediction_train,
              timeHistory_train,
              prediction,
              timeHistory_test,
              dim,
              plotPath):

    ### TEMP
    print('-')
    print('full')
    print(timeHistory.shape)
    print(timeHistory[0],timeHistory[-1])
    print(trajectoryHistory.shape)
    print(trajectoryHistory[0,0],trajectoryHistory[0,-1])
    print('-')
    print('train')
    print(timeHistory_train.shape)
    print(timeHistory_train[0],timeHistory_train[-1])
    print(prediction_train.shape)
    print(prediction_train[0,0],prediction_train[0,-1])
    print('-')
    print('test')
    print(timeHistory_test.shape)
    print(timeHistory_test[0],timeHistory_test[-1])
    print(prediction.shape)
    print(prediction[0,0],prediction[0,-1])
    print('-')
    ###

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

    plt.savefig(plotPath)
    print('plot saved to: ', plotPath)

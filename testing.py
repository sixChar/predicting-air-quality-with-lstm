from data_utils import *
from lstm_model import AirPredictor

from matplotlib import pyplot as plt

# Number of hours ahead to test predictions on. This many hours of data is fed into the network for prediction and then
# it tries to predict this many hours into the future.
TEST_HOURS = 128

SAVE_PLOTS_PATH = "plots/"


# Load all the data
test_data,test_dates = load_all_preprocessed()
test_data = test_data[TRAIN_EXAMPLES:]
test_dates = test_dates[TRAIN_EXAMPLES:]


def get_test_batch(size):
    test_batches = []
    for i in range(size):
        start_i = np.random.randint(0, test_data.shape[0] - 2 * TEST_HOURS)
        test_batches.append(test_data[np.newaxis, start_i: start_i + 2 * TEST_HOURS, :])
    return np.concatenate(test_batches, axis=0)

'''
    Plots the mean absolute error of all attributes over time.
'''
def test_err_mean_attrs(model,num_tests=64):
    test_batches = get_test_batch(num_tests)

    pred_out = model.predict_seq(test_batches[:,:TEST_HOURS,:],TEST_HOURS)

    test_err = np.mean(np.abs(pred_out - test_batches[:,TEST_HOURS:,:]), axis=0)

    test_mean_attr = np.mean(test_err,axis=-1)

    plt.plot(np.arange(test_mean_attr.shape[0]),test_mean_attr,'bo')
    plt.show()

'''
    Plots the error over time of a single attribute averaged over "num_tests" test batches.
'''
def test_err_single_attr(model,test_attr,num_tests=64):
    test_batches = get_test_batch(num_tests)

    pred_out = model.predict_seq(test_batches[:,:TEST_HOURS,:],TEST_HOURS)
    test_err = np.mean(np.abs(pred_out - test_batches[:,TEST_HOURS:,:]), axis=0)

    attr_err = test_err[:,prop_to_index(test_attr)]
    plt.plot(attr_err)
    plt.show()

'''
    Plots the error over time of all non nan flag attributes excluding wind direction, averaged
    over "num_tests" test batches. All of these attributes have been normalized with
    mean 0 and std 1. The attribute plots show the error values from each station on seperate lines.
'''
def test_attrs_group_station(model, num_tests=64, save_plots=False):
    test_batches = get_test_batch(num_tests)

    pred_out = model.predict_seq(test_batches[:, :TEST_HOURS, :], TEST_HOURS)
    test_err = np.mean(np.abs(pred_out - test_batches[:, TEST_HOURS:, :]), axis=0)


    for i,attr in enumerate(ATTRIBUTES):
        if is_wind_dir(attr):
            continue

        fig = plt.figure(i)
        ax = plt.gca()

        fig.suptitle(attr)

        for station in STATIONS:
            index = prop_to_index("_".join((attr,station)))
            line, = ax.plot(np.arange(test_err.shape[0]),test_err[:,index],'-')
            line.set_label(station)
        ax.legend()

        if save_plots:
            plt.savefig("".join((SAVE_PLOTS_PATH,attr,".png")))
    plt.show()


if __name__=="__main__":
    model = AirPredictor(load_weights=True)
    test_attrs_group_station(model,num_tests=128,save_plots=True)




















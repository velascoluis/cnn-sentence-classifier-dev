#@to-commit
import tensorflow_data_validation as tfdv
print('TFDV version: {}'.format(tfdv.version.__version__))
train_stats = tfdv.generate_statistics_from_csv(data_location='/Users/velascoluis/PycharmProjects/cnn-sentence-classifier-dev/local/data/airplaneData.csv')
tfdv.visualize_statistics(train_stats)



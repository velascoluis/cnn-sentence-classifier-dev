import argparse
from datetime import datetime
import logging
import json
from kubeflow.metadata import metadata
import retrying
import tensorflow_data_validation as tfdv
print('TFDV version: {}'.format(tfdv.version.__version__))




import tensorflow_data_validation as tfdv
import argparse
import pandas as pd
import logging
import sys

def skew_drift_validator(mode,
                         gcp_bucket,
                         control_set_path,
                         treatment_set_path,
                         feature_list_str,
                         Linf_value):
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting skew drift validator ..')
    logging.info('Input data:')
    logging.info('mode:{}'.format(mode))
    logging.info('gcp_bucket:{}'.format(gcp_bucket))
    logging.info('control_set_path:{}'.format(control_set_path))
    logging.info('treatment_set_path:{}'.format(treatment_set_path))
    logging.info('Linf_value:{}'.format(Linf_value))



    feature_list = eval(feature_list_str)
    control_set_df = pd.read_csv("gs://" + gcp_bucket + "/" + control_set_path, sep=',')
    treat_set_df = pd.read_csv("gs://" + gcp_bucket + "/" + treatment_set_path, sep=',')
    control_stats = tfdv.generate_statistics_from_dataframe(dataframe=control_set_df)
    treat_stats = tfdv.generate_statistics_from_dataframe(dataframe=treat_set_df)
    control_schema = tfdv.infer_schema(control_stats)
    treat_schema = tfdv.infer_schema(treat_stats)

    for feature in feature_list:
        if (mode == "skew"):
            if(tfdv.get_feature(control_schema, feature).domain): # if we have domain it is a categorical variable
                tfdv.get_feature(control_schema, feature).skew_comparator.infinity_norm.threshold = Linf_value
            else:
                logging.critical("feature: {} is not categorical".format(feature))
                sys.exit(1)
        elif (mode == "drift"):
            tfdv.get_feature(control_schema, feature).drift_comparator.infinity_norm.threshold = Linf_value
        else:
            logging.critical("mode: {} not supported".format(mode))
            sys.exit(1)
    anomalies = tfdv.validate_statistics(
        statistics=control_stats, schema=control_schema, serving_statistics=treat_stats)
    if(anomalies.anomaly_info):
        logging.info("Data-{} detected:".format(anomalies))
        return anomalies
    else:
        logging.info("No data-{} detected".format(mode))




def main(params):
    skew_drift_validator(params.mode,
                         params.gcp_bucket,
                         params.control_set_path,
                         params.treatment_set_path,
                         params.feature_list,
                         params.Linf_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='01. Skew-drift validator')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--gcp_bucket', type=str, required=True)
    parser.add_argument('--control_set_path', required=True)
    parser.add_argument('--treatment_set_path', type=str, required=True)
    parser.add_argument('--feature_list', type=str, required=True)
    parser.add_argument('--Linf_value', type=float, required=True)
    params = parser.parse_args()
    main(params)

import argparse
import datetime
import json
import logging
import time
import yaml
from distutils.util import strtobool
from kubernetes import client as k8s_client
from kubernetes import config
from kubernetes.client import rest

TFJobGroup = "kubeflow.org"
TFJobPlural = "tfjobs"


def generateEvaluatorSpec(workdir,params):
    mount_dir = workdir
    command_exec = ['/usr/bin/python3', 'src/04_train_model.py']
    command_args = ['--keras_model_path=' + mount_dir + '/model.bin',
                    '--x_train_path=' + mount_dir + '/x_train.bin',
                    '--x_val_path=' + mount_dir + '/x_val.bin',
                    '--y_train_path=' + mount_dir + '/y_train.bin',
                    '--y_val_path=' + mount_dir + '/y_val.bin',
                    '--epochs=' + str(params.epochs),
                    '--batch_size=' + str(params.batch_size),
                    '--output_trained_model_path=' + mount_dir]

    command_cmd = command_exec + command_args
    pvc_spec = {
        "claimName": "kfpipeline-data-pvc"
    }

    volumes_spec = [{

        "name": "pvolumes-tfjob",
        "persistentVolumeClaim": pvc_spec
    }]

    volumeMonts_spec = [{
        "mountPath": workdir,
        "name": "pvolumes-tfjob"
    }]

    containers_spec = [{
        "command": command_cmd,
        "image": "gcr.io/velascoluis-test/04_train_model:latest",
        "name": "tensorflow",
        "volumeMounts": volumeMonts_spec
    }]

    template_spec_body = {
        "volumes": volumes_spec,
        "containers": containers_spec,
    }

    template_spec = {
        "spec": template_spec_body
    }
    evaluator_spec = {
        "replicas": 1,
        "restartPolicy": "Never",
        "template": template_spec
    }
    return evaluator_spec





def generateMasterSpec(workdir,params):
    mount_dir = workdir
    command_exec = ['/usr/bin/python3', 'src/04_train_model.py']
    command_args = ['--keras_model_path=' + mount_dir + '/model.bin',
                    '--x_train_path=' + mount_dir + '/x_train.bin',
                    '--x_val_path=' + mount_dir + '/x_val.bin',
                    '--y_train_path=' + mount_dir + '/y_train.bin',
                    '--y_val_path=' + mount_dir + '/y_val.bin',
                    '--epochs=' + str(params.epochs),
                    '--batch_size=' + str(params.batch_size),
                    '--output_trained_model_path=' + mount_dir]

    command_cmd = command_exec + command_args
    pvc_spec = {
        "claimName": "kfpipeline-data-pvc"
    }

    volumes_spec = [{

        "name": "pvolumes-tfjob",
        "persistentVolumeClaim": pvc_spec
    }]

    volumeMonts_spec = [{
        "mountPath": workdir,
        "name": "pvolumes-tfjob"
    }]

    containers_spec = [{
        "command": command_cmd,
        "image": "gcr.io/velascoluis-test/04_train_model:latest",
        "name": "tensorflow",
        "volumeMounts": volumeMonts_spec
    }]

    template_spec_body = {
        "volumes": volumes_spec,
        "containers": containers_spec,
    }

    template_spec = {
        "spec": template_spec_body
    }
    master_spec = {
        "replicas": 1,
        "restartPolicy": "Never",
        "template": template_spec
    }
    return master_spec




def generatePSSpec(workdir,ps_num_replicas,params):
    mount_dir = workdir
    command_exec = ['/usr/bin/python3', 'src/04_train_model.py']
    command_args = ['--keras_model_path=' + mount_dir + '/model.bin',
                    '--x_train_path=' + mount_dir + '/x_train.bin',
                    '--x_val_path=' + mount_dir + '/x_val.bin',
                    '--y_train_path=' + mount_dir + '/y_train.bin',
                    '--y_val_path=' + mount_dir + '/y_val.bin',
                    '--epochs=' + str(params.epochs),
                    '--batch_size=' + str(params.batch_size),
                    '--output_trained_model_path=' + mount_dir]

    command_cmd = command_exec + command_args
    pvc_spec = {
        "claimName": "kfpipeline-data-pvc"
    }

    volumes_spec = [{

        "name": "pvolumes-tfjob",
        "persistentVolumeClaim": pvc_spec
    }]

    volumeMonts_spec = [{
        "mountPath": workdir,
        "name": "pvolumes-tfjob"
    }]

    containers_spec = [{
        "command": command_cmd,
        "image": "gcr.io/velascoluis-test/04_train_model:latest",
        "name": "tensorflow",
        "volumeMounts": volumeMonts_spec
    }]

    template_spec_body = {
        "volumes": volumes_spec,
        "containers": containers_spec,
    }

    template_spec = {
        "spec": template_spec_body
    }
    ps_spec = {
        "replicas": ps_num_replicas,
        "restartPolicy": "Never",
        "template": template_spec
    }
    return ps_spec







def generateWorkerSpec(workdir,worker_num_replicas,params):
    mount_dir = workdir
    command_exec = ['/usr/bin/python3', 'src/04_train_model.py']
    command_args = ['--keras_model_path=' + mount_dir + '/model.bin',
                    '--x_train_path=' + mount_dir + '/x_train.bin',
                    '--x_val_path=' + mount_dir + '/x_val.bin',
                    '--y_train_path=' + mount_dir + '/y_train.bin',
                    '--y_val_path=' + mount_dir + '/y_val.bin',
                    '--epochs=' + str(params.epochs),
                    '--batch_size=' + str(params.batch_size),
                    '--output_trained_model_path=' + mount_dir]

    command_cmd = command_exec + command_args
    pvc_spec = {
        "claimName": "kfpipeline-data-pvc"
    }

    volumes_spec = [{

        "name": "pvolumes-tfjob",
        "persistentVolumeClaim": pvc_spec
    }]

    volumeMonts_spec = [{
        "mountPath": workdir,
        "name": "pvolumes-tfjob"
    }]

    containers_spec = [{
        "command": command_cmd,
        "image": "gcr.io/velascoluis-test/04_train_model:latest",
        "name": "tensorflow",
        "volumeMounts": volumeMonts_spec
    }]

    template_spec_body = {
        "volumes": volumes_spec,
        "containers": containers_spec,
    }

    template_spec = {
        "spec": template_spec_body
    }
    worker_spec = {
        "replicas": worker_num_replicas,
        "restartPolicy": "Never",
        "template": template_spec
    }
    return worker_spec




def generateJSONTFJobSpec(params):
    #Generate common def
    inst = {
        "apiVersion": "%s/%s" % (TFJobGroup, params.version),
        "kind": "TFJob",
        "metadata": {
            "name": params.name,
            "namespace": params.namespace,
        },
        "spec": {
            "cleanPodPolicy": params.cleanPodPolicy,
            "tfReplicaSpecs": {
            },
        },
    }
    if params.ttlSecondsAfterFinished >= 0:
        inst["spec"]["ttlSecondsAfterFinished"] = params.ttlSecondsAfterFinished
    if params.backoffLimit >= 0:
        inst["spec"]["backoffLimit"] = params.backoffLimit
    if params.activeDeadlineSeconds >= 0:
        inst["spec"]["activeDeadlineSecond"] = params.activeDeadlineSeconds

    #Generate ReplicaSpecs
    if(params.worker_num_replicas > 0):
        inst["spec"]["tfReplicaSpecs"]["Worker"] = generateWorkerSpec(params.workdir,params.worker_num_replicas,params)
    if (params.ps_num_replicas > 0):
        inst["spec"]["tfReplicaSpecs"]["PS"] = generatePSSpec(params.workdir,params.ps_num_replicas,params)
    if(params.master == 'True'):
        inst["spec"]["tfReplicaSpecs"]["Master"] = generateMasterSpec(params.workdir,params)
    if (params.evaluator == 'True'):
        inst["spec"]["tfReplicaSpecs"]["Evaluator"] = generateEvaluatorSpec(params.workdir,params)
    return inst






class K8sCR(object):
    def __init__(self, group, plural, version, client):
        self.group = group
        self.plural = plural
        self.version = version
        self.client = k8s_client.CustomObjectsApi(client)

    def wait_for_condition(self,
                           namespace,
                           name,
                           expected_conditions=[],
                           timeout=datetime.timedelta(days=365),
                           polling_interval=datetime.timedelta(seconds=30),
                           status_callback=None):
        """Waits until any of the specified conditions occur.
    Args:
      namespace: namespace for the CR.
      name: Name of the CR.
      expected_conditions: A list of conditions. Function waits until any of the
        supplied conditions is reached.
      timeout: How long to wait for the CR.
      polling_interval: How often to poll for the status of the CR.
      status_callback: (Optional): Callable. If supplied this callable is
        invoked after we poll the CR. Callable takes a single argument which
        is the CR.
    """
        end_time = datetime.datetime.now() + timeout
        while True:
            try:
                results = self.client.get_namespaced_custom_object(
                    self.group, self.version, namespace, self.plural, name)
            except Exception as e:
                logging.error("There was a problem waiting for %s/%s %s in namespace %s; Exception: %s",
                              self.group, self.plural, name, namespace, e)
                raise

            if results:
                if status_callback:
                    status_callback(results)
                expected, condition = self.is_expected_conditions(results, expected_conditions)
                if expected:
                    logging.info("%s/%s %s in namespace %s has reached the expected condition: %s.",
                                 self.group, self.plural, name, namespace, condition)
                    return results
                else:
                    if condition:
                        logging.info("Current condition of %s/%s %s in namespace %s is %s.",
                                     self.group, self.plural, name, namespace, condition)

            if datetime.datetime.now() + polling_interval > end_time:
                raise Exception(
                    "Timeout waiting for {0}/{1} {2} in namespace {3} to enter one of the "
                    "conditions {4}.".format(self.group, self.plural, name, namespace, expected_conditions))

            time.sleep(polling_interval.seconds)

    def is_expected_conditions(self, cr_object, expected_conditions):
        return False, ""

    def create(self, spec):
        """Create a CR.
    Args:
      spec: The spec for the CR.
    """
        try:
            # Create a Resource
            namespace = spec["metadata"].get("namespace", "default")
            logging.info("Creating %s/%s %s in namespace %s.",
                         self.group, self.plural, spec["metadata"]["name"], namespace)
            api_response = self.client.create_namespaced_custom_object(
                self.group, self.version, namespace, self.plural, spec)
            logging.info("Created %s/%s %s in namespace %s.",
                         self.group, self.plural, spec["metadata"]["name"], namespace)
            return api_response
        except rest.ApiException as e:
            self._log_and_raise_exception(e, "create")

    def delete(self, name, namespace):
        try:
            body = {
                # Set garbage collection so that CR won't be deleted until all
                # owned references are deleted.
                "propagationPolicy": "Foreground",
            }
            logging.info("Deleteing %s/%s %s in namespace %s.",
                         self.group, self.plural, name, namespace)
            api_response = self.client.delete_namespaced_custom_object(
                self.group,
                self.version,
                namespace,
                self.plural,
                name,
                body)
            logging.info("Deleted %s/%s %s in namespace %s.",
                         self.group, self.plural, name, namespace)
            return api_response
        except rest.ApiException as e:
            self._log_and_raise_exception(e, "delete")

    def _log_and_raise_exception(self, ex, action):
        message = ""
        if ex.message:
            message = ex.message
        if ex.body:
            try:
                body = json.loads(ex.body)
                message = body.get("message")
            except ValueError:
                logging.error("Exception when %s %s/%s: %s", action, self.group, self.plural, ex.body)
                raise

        logging.error("Exception when %s %s/%s: %s", action, self.group, self.plural, ex.body)
        raise ex


class TFJob(K8sCR):
    def __init__(self, version="v1", client=None):
        super(TFJob, self).__init__(TFJobGroup, TFJobPlural, version, client)

    def is_expected_conditions(self, inst, expected_conditions):
        conditions = inst.get('status', {}).get("conditions")
        if not conditions:
            return False, ""
        if conditions[-1]["type"] in expected_conditions and conditions[-1]["status"] == "True":
            return True, conditions[-1]["type"]
        else:
            return False, conditions[-1]["type"]


def main(params):

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Starting TFJOB Distributed Launcher step.')
    logging.info('Input data ..')
    logging.info('name:{}'.format(params.name))
    logging.info('namespace:{}'.format(params.namespace))
    logging.info('activeDeadlineSeconds:{}'.format(params.activeDeadlineSeconds))
    logging.info('backoffLimit:{}'.format(params.backoffLimit))
    logging.info('cleanPodPolicy:{}'.format(params.cleanPodPolicy))
    logging.info('ttlSecondsAfterFinished:{}'.format(params.ttlSecondsAfterFinished))
    logging.info('deleteAfterDone:{}'.format(params.deleteAfterDone))
    logging.info('tfjobTimeoutMinutes:{}'.format(params.tfjobTimeoutMinutes))
    logging.info('epochs:{}'.format(params.epochs))
    logging.info('batch_size:{}'.format(params.batch_size))
    logging.info('workdir:{}'.format(params.workdir))
    logging.info('worker_num_replicas:{}'.format(params.worker_num_replicas))
    logging.info('ps_num_replicas:{}'.format(params.ps_num_replicas))
    logging.info('master:{}'.format(params.master))
    logging.info('evaluator:{}'.format(params.evaluator))

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    tfjob = TFJob(version=params.version, client=api_client)
    logging.info('STEP: DISTR TRAIN (1/3) Generating TFJOBSpec ...')
    inst = generateJSONTFJobSpec(params)
    logging.info('STEP: DISTR TRAIN (2/3) Launching containers ...')
    create_response = tfjob.create(inst)
    expected_conditions = ["Succeeded", "Failed"]
    tfjob.wait_for_condition(
        params.namespace, params.name, expected_conditions,
        timeout=datetime.timedelta(minutes=params.tfjobTimeoutMinutes))
    logging.info('STEP: DISTR TRAIN (3/3) Expected condition reached (Success, Fail or timeout) ...')
    if bool(params.deleteAfterDone):
        tfjob.delete(params.name, params.namespace)
    logging.info('TFJOB Distributed Launcher step finished.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kubeflow TFJob launcher')
    parser.add_argument('--name', type=str, help='TFJob name.')
    parser.add_argument('--namespace', type=str,default='kubeflow',help='TFJob namespace.')
    parser.add_argument('--version', type=str,default='v1',help='TFJob version.')
    parser.add_argument('--activeDeadlineSeconds', type=int,default=-1,help='Specifies the duration (in seconds) since startTime during which the job can remain active before it is terminated. Must be a positive integer. This setting applies only to pods where restartPolicy is OnFailure or Always.')
    parser.add_argument('--backoffLimit', type=int,default=-1,help='Number of retries before marking this job as failed.')
    parser.add_argument('--cleanPodPolicy', type=str,default="Running",help='Defines the policy for cleaning up pods after the TFJob completes.')
    parser.add_argument('--ttlSecondsAfterFinished', type=int,default=-1,help='Defines the TTL for cleaning up finished TFJobs.')
    parser.add_argument('--deleteAfterDone', type=str,default=True,help='When tfjob done, delete the tfjob automatically if it is True.')
    parser.add_argument('--tfjobTimeoutMinutes', type=int,default=60 * 24,help='Time in minutes to wait for the TFJob to reach end')
    parser.add_argument('--epochs', type=int,default='None')
    parser.add_argument('--batch_size', type=int, default='None')
    parser.add_argument('--workdir', type=str, default='None')
    parser.add_argument('--worker_num_replicas', type=int, default='None')
    parser.add_argument('--ps_num_replicas',     type=int, default='None')
    parser.add_argument('--master',    type=str, default='False')
    parser.add_argument('--evaluator', type=str, default='False')
    params = parser.parse_args()
    main(params)



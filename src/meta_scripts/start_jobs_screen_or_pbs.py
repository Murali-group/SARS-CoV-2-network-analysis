# Script to run methods by submitting them to a screen session, or to a PBS job scheduler (using qsub) 

import yaml
import argparse
from collections import defaultdict
import os
import sys
import re
import socket
import subprocess
import time
import itertools
import copy

# add a folder up from this file to the path so these imports work from anywhere
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from annotation_prediction.src import main as run_eval_algs
from annotation_prediction.src.algorithms import runner
from annotation_prediction.src.utils import config_utils
from annotation_prediction.src.evaluate import cross_validation as cv


"""
This file contains the functions I use for submitting jobs to the baobab cluster
More info about the baobab cluster is available here: https://github.com/Murali-group/utils/wiki/Baobab-info
"""
def copyToBaobabNodes(file_to_copy):
    """ 
    Copy a given file to each of the 6 baobab nodes. 
    Keeps the same filepath, but replaces /data to /localdisk which is unique to each of the nodes.
    Useful to speed up reading input files if many jobs are using the same file (such as an interactome)
    """
    print("Copying %s to the 6 baobab nodes" % (file_to_copy))
    print("\tTip: If you are being asked for a password, setup ssh keys to allow passwordless ssh and scp")
    ip_template = "192.168.200.%s"
    # use the localdisk storage on the baobab nodes to speed up read/write times
    copy_to_dir = re.sub("^/data", "/localdisk", os.path.dirname(os.path.abspath(file_to_copy))) 
    # loop through the 6 nodes
    #for i in tqdm(range(1,7)):
    for i in range(1,7):
        command = "ssh %s 'mkdir -p %s'; scp %s %s:%s" % (ip_template%i, copy_to_dir, os.path.abspath(file_to_copy), ip_template%i, copy_to_dir)
        runCommandOnBaobab(command)


def runCommandOnBaobab(command):
    """ 
    Run a given command on baobab. 
    Useful for submitting a job to the baobab cluster or copying a file from baobab to each of the nodes 
    """
    print("Running: %s" % (command))
    if 'baobab' in socket.gethostname():
        subprocess.check_call(command, shell=True)
    else:
        command = "ssh -t baobab.cbb.lan \"%s\"" % (command)
        subprocess.check_call(command, shell=True)


def submitQsubFile(qsub_file):
    """ Submit a qsub file to the baobab cluster using the qsub command
    """
    command = "qsub " + qsub_file
    runCommandOnBaobab(command)


def writeQsubFile(jobs, qsub_file, submit=False, log_file=None, err_log_file=None, name=None, nodes=1, ppn=24, walltime='10:00:00'):
    """ Function to write a qsub bash script which can be submitted to the baobab cluster.
    *jobs*: a list or set of commands/jobs to be run in this job. 
            This could include a 'cd' to move to your project directory, or 'export PYTHONPATH' to setup environmental variables for example
    *qsub_file*: path/to/file to write. Really just a bash script with special headers recognized by PBS. 
    *submit*: option to submit the written qsub file to the baobab cluster
    *log_file*: file which will contain the stdout output of the submitted qsub file. If None, -out.log will be appended to qsub_file
    *err_log_file*: file which will contain the stderr output of the submitted qsub file. If None, -err.log will be appended to qsub_file
    *name*: name to give the job 
    *nodes*: total number of nodes you need
    *ppn*: processors per node that you will need. Max is 24
    *walltime*: amount of time your job will be allowed before being forcefully removed. 'HH:MM:SS'
    """
    if log_file is None:
        std_out_log = "%s-out.log" % qsub_file
        std_err_log = "%s-err.log" % qsub_file
    else:
        std_out_log = log_file
        std_err_log = log_file
    # start the qsub file 
    with open(qsub_file, 'w') as out:
        out.write('#PBS -l nodes=%d:ppn=%d,walltime=%s\n' % (nodes, ppn, walltime))
        # set the job name
        out.write('#PBS -N %s\n' % (name))
        out.write('#PBS -o %s\n' % (std_out_log))
        out.write('#PBS -e %s\n' % (std_err_log))
        out.write('#####################################################################################\n')
        out.write('echo "Job Started at: `date`"\n')
        # write each job, as well as an echo (print) statement of the job to be run to the qsub file
        out.write('\n'.join(['echo """%s"""\n%s' % (cmd, cmd) for cmd in jobs]) + '\n')
        out.write('echo "Job Ended at: `date`"\n')
        # TODO some kind of email or notification if any of the jobs failed

    if submit:
        submitQsubFile(qsub_file)


def get_algs_to_run(alg_settings, **kwargs):
    # if there aren't any algs specified by the command line (i.e., kwargs),
    # then use whatever is in the config file
    if kwargs['algs'] is None:
        algs_to_run = run_eval_algs.get_algs_to_run(alg_settings)
        kwargs['algs'] = [a.lower() for a in algs_to_run]
        print("\nNo algs were specified. Using the algorithms in the yaml file:")
        print(str(kwargs['algs']) + '\n')
        if len(algs_to_run) == 0:
            print("ERROR: Must specify algs with --alg or by setting 'should_run' to [True] in the config file")
            sys.exit("Quitting")
    else:
        # make the alg names lower so capitalization won't make a difference
        kwargs['algs'] = [a.lower() for a in kwargs['algs']]
    return kwargs['algs']


def main(config_map, **kwargs):
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)
    algs = config_map['algs']
    config_map = copy.deepcopy(config_map)
    kwargs['algs'] = config_utils.get_algs_to_run(algs, **kwargs)

    yaml_file_pref = "" 
    if kwargs.get('cross_validation_folds'):
        cv_out_pref = cv.get_output_prefix(
            folds=kwargs['cross_validation_folds'], rep=kwargs.get('num_reps',1),
            sample_neg_examples_factor=kwargs.get('sample_neg_examples_factor'),
            curr_seed=kwargs.get('cv_seed'))
        yaml_file_pref += cv_out_pref
    run_jobs(algs, config_map, yaml_file_pref, **kwargs)


def run_jobs(alg_settings, config_map, 
        yaml_file_pref='', postfix='', **kwargs):
    curr_config_map = copy.deepcopy(config_map)
    # setup the config file so that the specified algs have "True" for the should_run flag, and the others have false
    for alg, params in alg_settings.items():
        if alg.lower() in kwargs['algs']:
            #print('Running %s' % (alg))
            print(alg, params)
            params['should_run'] = [True]
        else:
            params['should_run'] = [False]
            continue
        # start one job per param combination
        if kwargs.get('job_per_param'):
            # get the parameter combinations
            combos = [dict(zip(params, val))
                for val in itertools.product(
                    *(params[param] for param in params))]
            for param_combo in combos:
                # only write the current alg, param settings in this yaml file
                curr_config_map['algs'] = {alg: {p: [val] for p, val in param_combo.items()}}
                # get the param str from the alg's runner 
                params_str = runner.get_runner_params_str(alg, param_combo)
                curr_yaml_pref = yaml_file_pref+alg+params_str+postfix
                run_job_wrapper(curr_config_map, curr_yaml_pref, **kwargs)
    if not kwargs.get('job_per_param'):
        # run the specified algs together
        algs = kwargs['algs'] 
        alg_name = '-'+'-'.join(algs) if len(algs) < 4 else '-%dalgs' % len(algs)
        curr_yaml_pref = yaml_file_pref+alg_name+postfix
        run_job_wrapper(curr_config_map, curr_yaml_pref, **kwargs)


def run_job_wrapper(config_map, alg_name, **kwargs):
    # start a separate job for each dataset
    if kwargs.get('job_per_dataset'):
        for dataset in config_map['input_settings']['datasets']:
            curr_config_map = copy.deepcopy(config_map)
            # the exp_name is unique for each dataset.
            # just get the name and remove any folders(??)
            exp_name = dataset['exp_name'].split('/')[-1]
            net_version = '-'.join(dataset['net_version'].split('/')[1:])
            # add a folder for each dataset using the exp_name
            curr_alg_name = "%s-%s/%s" % (net_version, exp_name, alg_name)
            curr_config_map['input_settings']['datasets'] = [dataset]
            run_job(curr_config_map, curr_alg_name, **kwargs)
    else:
        run_job(config_map, alg_name, **kwargs)


def run_job(config_map, alg_name, **kwargs):
    # make a directory for this config, and then put a config file for each method inside
    yaml_base = kwargs['config'].replace('.yaml','')
    yaml_file = "%s/%s.yaml" % (yaml_base, alg_name)
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    cmd_file = os.path.abspath("%s/%s.sh" % (yaml_base, alg_name))
    log_file = os.path.abspath("%s/%s.log" % (yaml_base, alg_name))
    write_yaml_file(yaml_file, config_map)
    # now run it. Submit it to screen
    if kwargs.get('job_per_dataset'):
        name = alg_name
    else:
        name = "%s-%s" % (alg_name, config_map['input_settings']['datasets'][0]['exp_name'].split('/')[-1])
    # option for the python environment to use
    if kwargs.get('python'):
        python = kwargs['python']
    else:
        python = "source /data/jeff-law/tools/anaconda3/bin/activate covid19; \npython"
    # pass the arguments specified when calling this script to this command
    str_args = get_script_args(**kwargs)
    command = "%s -u %s --config %s %s >> %s 2>&1" % (
        python, kwargs['script_to_run'], os.path.abspath(yaml_file), str_args, log_file)
    jobs = ["cd %s" % (os.getcwd()), command]
    if kwargs['qsub'] is True:
        submit = not kwargs['test_run']
        writeQsubFile(
            jobs, cmd_file, name=name, submit=submit,  # log_file=log_file, # can't pass the log file since the PBS output file will overwrite 
            nodes=1, ppn=kwargs.get('cores',2), walltime=kwargs.get('walltime', '200:00:00'))
        # start a sleep job with the specified # cores and time to sleep
        # helps to manage RAM consumption, and for timing methods
        if kwargs.get('sleep'):
            cores, sleep_time = kwargs['sleep']
            print("\tstarting sleep job with %s cores for %s time" % (cores, sleep_time))
            jobs = ["sleep %s" % sleep_time]
            sleep_file = "%s/sleep.sh" % (yaml_base)
            log_file = sleep_file.replace('.sh','.log')
            writeQsubFile(
                jobs, sleep_file, name="sleep-%s"%sleep_time, submit=submit, log_file=log_file,
                nodes=1, ppn=int(cores), walltime=kwargs.get('walltime', '200:00:00'))
        if kwargs['test_run']:
            print(cmd_file)
            #sys.exit()
    else:
        # write the bash file
        write_bash_file(cmd_file, jobs)
        submit_to_screen(cmd_file, name, log_file, **kwargs)


def get_script_args(**kwargs):
    # get everything after the --pass-to-script option and pass it on
    if kwargs['pass_to_script']:
        args_to_pass = ' '.join(sys.argv[sys.argv.index('--pass-to-script')+1:])
    else:
        args_to_pass = ""
    return args_to_pass


def write_bash_file(cmd_file, jobs):
    print("\twriting to %s" % (cmd_file))
    with open(cmd_file, 'w') as out:
        out.write('echo "Job Started at: `date`"\n')
        # write each job, as well as an echo (print) statement of the job to be run to the qsub file
        out.write('\n'.join(['echo """%s"""\n%s' % (cmd, cmd) for cmd in jobs]) + '\n')
        out.write('echo "Job Ended at: `date`"\n')


def submit_to_screen(cmd_file, name, log_file, **kwargs):
    # looks like the job won't start if the name is too long, so truncate it here if needed
    name = name[:80] if len(name) > 80 else name
    # can't have '/' in screen name apparently 
    name = name.replace('/','-')
    print("\tsubmitting '%s' %s to screen" % (name, cmd_file))
    cmd = "screen -S %s -d -m /bin/sh -c \"bash %s >> %s 2>&1\"" % (name, cmd_file, log_file)
    print(cmd+'\n')
    if kwargs['test_run']:
        return
    else:
        subprocess.check_call(cmd, shell=True)


def write_yaml_file(yaml_file, config_map):
    print("\twriting to %s" % (yaml_file))
    with open(yaml_file, 'w') as out:
        yaml.dump(config_map, out, default_flow_style=False)


# this is a list of the arguments that are unique to this script and should not be passed
# because the script we're running won't recognize them
meta_args = [
    "config",
    "script_to_run",
    "alg",
    "qsub",
    "test_run",
]


def setup_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Script for running algorithms in parallel either using screen or by submitting an HPC job (using the PBS scheduler). ' +
            'The options here include all of the options for run_eval_algs.py ' +
            'and everything specified after --pass-to-script will be passed on to that script')

        parser.add_argument('--config', required=True,
            help='Configuration file')
        parser.add_argument('--taxon', '-T', dest="taxons", type=str, action='append',
                help="Specify the species taxonomy ID for which to evaluate. Multiple may be specified. Otherwise, all species will be used")
        #parser.add_argument('--string-

        parser.add_argument('--forced', action="store_true", default=False,
            help='Overwrite the ExpressionData.csv file if it already exists.')

    group = parser.add_argument_group('masterscript options')
    group.add_argument('--script-to-run', default="src/annotation_prediction/run_eval_algs.py",
            help="script to run when submitting to screen / qsub")
    group.add_argument('--alg', dest="algs", action="append", 
            help="Name of algorithm to run. May specify multiple. Default is whatever is set to true in the config file")
    group.add_argument('--job-per-param', action='store_true', default=False,
            help="Each parameter set combination per alg will get its own job")
    group.add_argument('--job-per-dataset', action='store_true', default=False,
            help="Each dataset will get its own job")
    group.add_argument('--qsub', action='store_true', default=False,
            help="submit the jobs to a PBS queue with qsub.")
    group.add_argument('--test-run', action='store_true', default=False,
            help="Just print out the first command generated")
    group.add_argument('--cores', type=int, default=2,
            help="Number of cores to use per job submitted. Default: 2")
    group.add_argument('--sleep', type=str, nargs=2,
            help="<num-cores> <time-to-sleep> Amount of time to sleep with the specified number of cores between submitted jobs. " + \
                    "Useful if jobs are RAM intensive at the start of the run, or timing methods on their own nodes")
    group.add_argument('--python', 
            help="Path to python bin executable to use. If qsub is specified, default is to use the the current environment (TODO).")
    group.add_argument('--pass-to-script', action='store_true', default=False,
            help="All options specified after this option will be passed to the --script-to-run")
    # TODO add a machine option

    return parser


if __name__ == "__main__":
    # first load the run_eval_algs parser
    parser = run_eval_algs.setup_opts()
    parser = setup_parser(parser)
    opts = parser.parse_args()
    kwargs = vars(opts)
    kwargs['postfix'] = '' if kwargs['postfix'] is None else kwargs['postfix']
    config_file = opts.config
    with open(config_file, 'r') as conf:
        # for some reason this isn't recognized in other versions of PYyaml
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    main(config_map, **kwargs)

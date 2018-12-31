"""
The ``profilers`` module
========================

Contains various tools to trace and analyze, and retrieve the results of a run.
In particular, contains the class ``RunSaver`` which is widely used in the
``bullseye_graph`` module.
"""

import tensorflow as tf
import numpy as np
import json
import time
import os
import shutil
import re

from tensorflow.python.client import timeline

from .warning_handler import *

#name of the result directory
data_dir = "bullseye_data"
#prefix of the epoch directories names
epoch_keyword = "epoch_"
#name of the final results directory
final_dir = "final_results"
#name of the evolution directory
evolution_dir = "evolution"
#name of the profiler directory
profiler_dir = "profiler"
#name of the timeliner file
timeliner_file = "timeliner"

def read_results(run_id):
    """
    Once a run is completed, reads and returns the associated results, namely
    the mu, cov and the associated final ELBO.

    Parameters
    ----------
    run_id : str
        The ID of the run of interest

    Returns
    -------
    mu : np.array [p]
        μ
    cov : np.array [p,p]
        Σ
    elbo : np.array []
        final ELBO
    """

    run_dir = os.path.join(data_dir, run_id, final_dir)
    mu = np.load(os.path.join(run_dir,"mu.npy"))
    cov = np.load(os.path.join(run_dir,"cov.npy"))
    elbo = np.load(os.path.join(run_dir,"best_elbo.npy"))
    return mu, cov, elbo

def trace_results(run_id):
    """
    Once a run is completed, reads and returns the results of each iteration,
    namely the mu's, cov's and associated ELBO's.

    Parameters
    ----------
    run_id : str
        The ID of the run of interest

    Returns
    -------
    mu : list of np.arrays [p]
        μ's
    cov : list of np.arrays [p,p]
        Σ's
    elbo : np.array []
        ELBO's
    """
    mus = []
    covs = []
    elbos = []
    dirs = all_epoch_dirs(run_id)
    for dir in dirs:
        mu = np.load(os.path.join(dir,"mu.npy"))
        cov = np.load(os.path.join(dir,"cov.npy"))
        elbo = np.load(os.path.join(dir,"elbo.npy"))

        mus.append(mu)
        covs.append(cov)
        elbos.append(elbo)

    return mus, covs, elbos

def all_epoch_dirs(run_id):
    """
    Lists the different epoch dirs for a given run ID.

    Parameters
    ----------
    run_id : str
        The ID of the run of interest.

    Returns
    -------
    list of stf
        The different epoch directories names.
    """

    run_dir = os.path.join(data_dir, run_id)
    dirs = [os.path.join(run_dir,dir) for dir in os.listdir(run_dir)
            if re.match("^"+epoch_keyword,dir)]
    return dirs

class RunSaver():
    """
    The ``RunSaver`` class
    ======================
    This class is used to handle the results of the Bullseye algorithm, but also
    to profile the performed operations. The different results will be written
    in "<data_dir>/<run_id>", we will then consider it as the working directory
    below.

    Traces the run and records the time and space required for each of the
    tensorflow operations performed, and can save the results of each iterations
    of the bullseye algorithm. Make use of the ``Profiler`` and ``TimeLiner``
    defined below.

    If used, the ``RunSaver`` class will always save :
        -the different options used for the run in "config.json",
        -the final results of the run in the "./<final_dir>" directory,
        -the informations relative to the evolutions of the algorithm in the
         "./<evolution_dir>" directory.

    If ``keep_track`` is set to true when the class is instanced :
        -directories following the "./<epoch_keyword>*" pattern will be created,
         containing the intermediate results of each iterations.

    If ``timeliner`` is set to true when the class is instanced :
        -the ``TimeLiner`` class defined below will be used, and will generate a
         "./<timeliner_file>.json" result file.

    If ``profiler`` is set to true when the class is instanced :
        -the ``Profiler`` class defined below will be used, and will generate a
        "./<profiler_dir>" directory.
    """
    def __init__(self, G, run_id, run_kwargs, timeliner, profiler, keep_track):
        """
        Initialize the saver.
        """
        self.start_time = 0
        self.keep_track = keep_track
        self.status = []
        self.elbos = []
        self.times = []
        self.best_elbos = []

        #set run_dir
        run_dir = os.path.join(data_dir, run_id)
        self.run_dir = run_dir

        #create the <data_dir> directory if it does not exists
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        #if the directory related to the given run_id already exists, remove it
        if os.path.isdir(run_dir):
            warn_removing_dir(run_dir)
            shutil.rmtree(run_dir)
        os.mkdir(run_dir)

        #handle timeliner
        self.runs_timeline = None
        if timeliner:
            self.runs_timeline = TimeLiner()
            run_kwargs['options'] =\
                tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_kwargs['run_metadata'] = tf.RunMetadata()

        #handle profiler
        self.profiler = None
        if profiler:
            self.profiler = Profiler(run_dir)

        #save the Graph configuration into "config.json"
        self.save_config(G)

    def save_config(self, G):
        """
        Save the config file of the graph into "config.json".
        """
        config = {}
        for option in G.option_list:
            config[option]=getattr(G,option)

        config_path = os.path.join(self.run_dir,"config.json")
        with open(config_path, "w", encoding = 'utf-8') as f:
            json.dump(config, f)


    def start_epoch(self):
        """
        Prepare the saver to start an epoch
        """
        self.start_time = time.time()

    def finish_epoch(self,statu, elbo, best_elbo):
        """
        Finish the epoch and save statistics
        """
        self.times.append(time.time()-self.start_time)
        self.status.append(statu)
        self.elbos.append(elbo)
        self.best_elbos.append(best_elbo)

        #handle profiler
        if self.profiler is not None:
            self.profiler.profile_operations()

    def save_step(self, mu, cov, epoch):
        """
        To call at the end of an iteration, to save the results of one iteration
        when ``keep_track`` is set to true.
        """
        if self.keep_track:
            width = 5
            dirname_ = epoch_keyword + '{epoch:0>{width}}'
            dirname = dirname_.format(epoch=epoch, width=width)

            elbo = self.elbos[-1]
            best_elbo = self.best_elbos[-1]

            infos = {'time' : self.times[-1],
                    'status' : self.status[-1]}

            self.__save_values(dirname, mu, cov, best_elbo, elbo, **infos)

    def save_final_results(self, mu, cov):
        """
        To call at the end of an iteration. Saves the results of one iteration
        when ``keep_track`` is set to true.
        """
        best_elbo = self.best_elbos[-1]

        infos = {'total_time' : sum(self.times)}

        self.__save_values(final_dir, mu, cov, best_elbo, **infos)

        if self.runs_timeline is not None:
            file = os.path.join(self.run_dir, '{}.json'.format(timeliner_file))
            self.runs_timeline.save(file)
        #handle evolution_path
        evolution_path = os.path.join(self.run_dir, evolution_dir)
        os.mkdir(evolution_path)

        #saving with numpy
        to_save = ['times','elbos','best_elbos','status']
        for key in to_save:
            value = getattr(self,key)
            np.save(os.path.join(evolution_path, key), np.asarray(value))

    def __save_values(self, dirname, mu, cov, best_elbo, elbo=None, **infos):
        """
        Private method, saving function to make the code more flexible.
        """
        dir_to_save = os.path.join(self.run_dir, dirname)
        os.mkdir(dir_to_save)

        to_save = {"mu" : mu, "cov": cov, "best_elbo": best_elbo, "elbo": elbo}
        for key, value in to_save.items():
            if value is not None:
                np.save(os.path.join(dir_to_save,key), value)

        infos_path = os.path.join(dir_to_save,"infos.json")
        with open(infos_path, "w", encoding = 'utf-8') as f:
            json.dump(infos, f)

    def before_run(self):
        """
        Prepare to profile upcoming operations
        """
        if self.profiler is not None:
            self.profiler.prepare_next_step()

    def after_run(self, kwargs):
        """
        Handle results of the profiled operations
        """
        #handle timeline
        if self.runs_timeline is not None:
            self.runs_timeline.update_timeline(kwargs["run_metadata"])

    def final_stats(self):
        """
        Is called at the end of the run. Return simple statistics.
        """
        stats = ["status","elbos","times","best_elbos"]
        r = {}
        for stat in stats:
            r[stat] = getattr(self,stat)
        return r

class TimeLiner:
    """
    The ``TimeLiner`` class
    =======================
    This class is very inspired by a question on stackoverflow addressed by
    frank_wang87. See : https://stackoverflow.com/questions/46374793/who-can-ex\
    plain-the-profiling-result-of-tensorflow

    Traces the run and records the time required for each of the tensorflow
    operations performed. Make use of ``timeline`` from
    ``tensorflow.python.client``.
    """
    def __init__(self):
        """
        Does not take any parameters.
        """
        self._timeline_dict = None

    def update_timeline(self, run_metadata):
        """
        Update the timeliner metadata.
        """
        #added lines
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()

        #convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict

        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        """
        Save the timeliner data into a .json file.
        """
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

class Profiler:
    """
    The ``Profiler`` class
    ======================
    Traces the run and records the time and space required for each of the
    tensorflow operations performed. Make use of ``ProfileOptionBuilder`` from
    ``tf.profiler``.
    """
    def __init__(self, run_dir):
        """
        Initialize the profiler
        """
        profiler_path = os.path.join(run_dir,profiler_dir)
        self.builder = tf.profiler.ProfileOptionBuilder
        self.opts = self.builder(self.builder.time_and_memory())\
            .order_by('micros').build()

        self.pctx = tf.contrib.tfprof.ProfileContext(profiler_path,
                                                trace_steps=[],
                                                dump_steps=[])

        #→ using __enter__ is not a good practice
        self.pctx.__enter__()
    def prepare_next_step(self):
        """
        Prepare the profiler to observe the run.
        """
        self.pctx.trace_next_step()
        self.pctx.dump_next_step()

    def profile_operations(self):
        """
        Profile the observed operations.
        """
        self.pctx.profiler.profile_operations(options=self.opts)

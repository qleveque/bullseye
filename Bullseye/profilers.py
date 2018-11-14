import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
import json
import time
import os
import shutil
import re

from .warning_handler import *

data_dir = "bullseye_data"
epoch_keyword = "epoch_"
final_dir = "final_results"
evolution_dir = "evolution"

def read_results(run_id):
    run_dir = os.path.join(data_dir, run_id, final_dir)
    mu = np.load(os.path.join(run_dir,"mu.npy"))
    cov = np.load(os.path.join(run_dir,"cov.npy"))
    elbo = np.load(os.path.join(run_dir,"best_elbo.npy"))
    return mu, cov, elbo

def trace_results(run_id):
    mus = []
    covs = []
    elbos = []
    dirs = all_epoch_dirs(run_id)
    for dir in dirs:
        mu = np.load(os.path.join(run_dir,"mu.npy"))
        cov = np.load(os.path.join(run_dir,"cov.npy"))
        elbo = np.load(os.path.join(run_dir,"elbo.npy"))

        mus.append(mu)
        covs.append(cov)
        elbos.append(elbo)

    return mus, covs, elbos

def all_epoch_dirs(run_id):
    run_dir = os.path.join(data_dir, run_id)
    dirs = [os.path.join(run_dir,dir)
            for dir in os.listdir(run_dir)
            if re.match("^"+keyword,dir)]
    return

class TimeLiner:
    """
    class very inspired by a question in stackoverflow addressed
     by frank_wang87
    https://stackoverflow.com/questions/46374793
     /who-can-explain-the-profiling-result-of-tensorflow
    """
    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, run_metadata):
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
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

class Profiler:
    def __init__(self, run_dir):
        profiler_dir = os.path.join(run_dir,"profiler")
        self.builder = tf.profiler.ProfileOptionBuilder
        self.opts = self.builder(self.builder.time_and_memory())\
            .order_by('micros').build()

        self.pctx = tf.contrib.tfprof.ProfileContext(profiler_dir,
                                                trace_steps=[],
                                                dump_steps=[])
        #using __enter__ with is not good practice,
        #but I remain consistent with the code this way
        self.pctx.__enter__()
    def prepare_next_step(self):
        self.pctx.trace_next_step()
        self.pctx.dump_next_step()

    def profile_operations(self):
        self.pctx.profiler.profile_operations(options=self.opts)

class RunSaver():
    def __init__(self, G, run_id, run_kwargs, timeliner, profiler, keep_track):
        run_dir = os.path.join(data_dir, run_id)

        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        if os.path.isdir(run_dir):
            warn_removing_dir(run_dir)
            shutil.rmtree(run_dir)
        os.mkdir(run_dir)

        self.run_dir = run_dir
        self.start_time = 0

        self.keep_track = keep_track

        self.status = []
        self.elbos = []
        self.times = []
        self.best_elbos = []

        self.runs_timeline = None
        if timeliner:
            self.runs_timeline = TimeLiner()
            run_kwargs['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_kwargs['run_metadata'] = tf.RunMetadata()

        self.profiler = None
        if profiler:
            self.profiler = Profiler(run_dir)

        self.save_config(G)

    def save_config(self, G):
        config = {}
        for option in G.option_list:
            config[option]=getattr(G,option)

        config_path = os.path.join(self.run_dir,"config.json")
        with open(config_path, "w", encoding = 'utf-8') as f:
            json.dump(config, f)


    def start_epoch(self):
        self.start_time = time.time()

    def finish_epoch(self,statu, elbo, best_elbo):
        self.times.append(time.time()-self.start_time)
        self.status.append(statu)
        self.elbos.append(elbo)
        self.best_elbos.append(best_elbo)

        #handle profiler
        if self.profiler is not None:
            self.profiler.profile_operations()

    def save_step(self, mu, cov, epoch):
        if not self.keep_track:
            width = 5
            dirname = epoch_keyword + '{epoch:0>{width}}'.format(epoch=epoch, width=width)
            elbo = self.elbos[-1]
            best_elbo = self.best_elbos[-1]

            infos = {'time' : self.times[-1],
                    'status' : self.status[-1]}

            self.save_values(dirname, mu, cov, best_elbo, elbo, **infos)

    def save_final_results(self, mu, cov):
        best_elbo = self.best_elbos[-1]

        infos = {'total_time' : sum(self.times)}

        self.save_values(final_dir, mu, cov, best_elbo, **infos)

        if self.runs_timeline is not None:
            file = os.path.join(self.run_dir, 'timeliner.json')
            self.runs_timeline.save(file)

        evolution_path = os.path.join(self.run_dir, evolution_dir)
        os.mkdir(evolution_path)
        np.save(os.path.join(evolution_path, 'times'), np.asarray(self.times))
        np.save(os.path.join(evolution_path, 'elbos'), np.asarray(self.elbos))
        np.save(os.path.join(evolution_path, 'best_elbos'), np.asarray(self.best_elbos))
        np.save(os.path.join(evolution_path, 'status'), np.asarray(self.status))

    def save_values(self, dirname, mu, cov, best_elbo, elbo=None, **infos):
        dir_to_save = os.path.join(self.run_dir, dirname)
        os.mkdir(dir_to_save)
        np.save(os.path.join(dir_to_save,"mu"), mu)
        np.save(os.path.join(dir_to_save,"cov"), cov)
        np.save(os.path.join(dir_to_save,"best_elbo"), best_elbo)
        if elbo is not None:
            np.save(os.path.join(dir_to_save,"elbo"), elbo)

        infos_path = os.path.join(dir_to_save,"infos.json")
        with open(infos_path, "w", encoding = 'utf-8') as f:
            json.dump(infos, f)

    def before_run(self):
        if self.profiler is not None:
            self.profiler.prepare_next_step()

    def after_run(self, kwargs):
        #handle timeline
        if self.runs_timeline is not None:
            self.runs_timeline.update_timeline(kwargs["run_metadata"])

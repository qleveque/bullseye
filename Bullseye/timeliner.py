#code very inspired by a question in stackoverflow addressed by frank_wang87
#https://stackoverflow.com/questions/46374793/who-can-explain-the-profiling-result-of-tensorflow

from tensorflow.python.client import timeline
import json

class TimeLiner:
    _timeline_dict = None

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
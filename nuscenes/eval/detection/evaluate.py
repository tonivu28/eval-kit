# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# Adapted by Toni Vu (VinAI), 2021.

"""
    To my colleagues:
    This code is based on the nuScenes detection evaluation code and adapted to evaluate our 3D Object Detection task at VinAI.
    Results ("metrics_summary.json" and plots) are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    (values of distance thresholds are specified in "nuscenes/eval/detection/configs/detection_cvpr_2019.json")
    - Five True Positive (TP) metrics: Average of {translation, velocity, scale, orientation, attribute} errors (best=0).
    (some modifications WILL*** be needed to adapt to our data, e.g. we don't have annotations for velocity and attributes?)
    - nuScenes Detection Score (NDS): The weighted sum of the above (best=1).
    Please see https://www.nuscenes.org/object-detection for more details.
    
    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format. (and filters the boxes***).
    - run: Performs evaluation, dumps the metric data to disk, and print results to the screen.
    - render: Renders 
    - (reder some graphs/detected images various examples for demonstration***)
    - (extend the code to implement other metrics***)
    
    (***) means future work, the code for these features are not yet implemented.

    All the code might seem complicated but it will become handy as the evaluation metrics getting more complex.
    If you just want to get the results, don't bother with the details, just give input to the function "nusc_eval_kit()" at the end of this page.
    You might need to install some packages: pyquaternion, motmetrics if required.
    
    If you wants to read the code, let's me give you a short description of the terminology in nuScenes world which you will find it useful soon:
        1. scene: a clip/video, each scene is associated with an unique id called "scene_token"
        2. sample: a frame which has annotations, each sample is associated with an unique id called "sample_token"
        3. instance: an object (car, cyclist, pedestrian...) appeared in multiple frames/samples
        4. sample_annotation: a 3D bounding box (with other attributes) of an object in a specific (time)frame/sample
    
    It's assumed that:
    - Every "sample_token" is given in the results, although there may be not predictions for that sample.
    
    I created dummy groundtruth and prediction files for illustration of the format of these files:
        data/sets/demo/gt.json
        data/sets/demo/submission.json
    
"""

# import argparse
import json
import os
# os.chdir('C:/Users/nghiavt5/Documents/VinAI/eval-kit')
# import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes.eval.common.config import config_factory
# from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_groundtruth #, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve#, visualize_sample

class DetectionEval:
    
    def __init__(self,
                 config: DetectionConfig,
                 result_path: str,
                 groundtruth_path: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the JSON result/prediction file.
        :param groundtruth_path: Path of the JSON ground truth file.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        # self.nusc = nusc
        self.result_path = result_path
        self.groundtruth_path = groundtruth_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'
        assert os.path.exists(groundtruth_path), 'Error: The grounth truth file does not exist!'
        
        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)  # pred_boxes is a Dict(key = sample_token, value = list[Detection Box])                                 
        # self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
        self.gt_boxes = load_groundtruth(self.groundtruth_path, DetectionBox,
                                                     verbose=verbose)
        
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        self.sample_tokens = self.gt_boxes.sample_tokens    
    

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                      dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                            savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                            savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
      
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary



def nusc_eval_kit(
        result_path: str = 'data/sets/demo/submission.json',    # The submission as a JSON file.
        groundtruth_path: str = 'data/sets/demo/gt.json',       # The ground truth as a JSON file.
        output_dir: str = 'data/sets/demo/',                    # Folder to store result metrics
        config_path: str = '',  #'Path to the configuration file. If no path given, the CVPR 2019 configuration will be used.
        render_curves: bool = False,                            # Whether to render PR and TP curves to disk
        verbose: bool = True):                                  # 'Whether to print to stdout.'
    
    if config_path == '':
        cfg = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg = DetectionConfig.deserialize(json.load(_f))

    nusc_eval = DetectionEval(config=cfg, result_path=result_path, groundtruth_path = groundtruth_path, output_dir=output_dir, verbose=verbose)
    nusc_eval.main(render_curves=render_curves)
  
    

if __name__ == "__main__":
    nusc_eval_kit()

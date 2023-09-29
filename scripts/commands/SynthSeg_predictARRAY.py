"""
This script enables to launch predictions with SynthSeg from the terminal.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# python imports
import os
import sys
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
import tensorflow as tf

# add main folder to python path and import ./SynthSeg/predict_synthseg.py
synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
model_dir = os.path.join(synthseg_home, 'models')
labels_dir = os.path.join(synthseg_home, 'data/labels_classes_priors')
from SynthSeg.predict_synthseg import predict


# parse arguments
parser = ArgumentParser(description="SynthSeg", epilog='\n')

# input/outputs
def predictArray(input_array, affine=np.eye(4), qc=None, crop=None, path_model_parcelation=None, vol=None, resample=None, post=None, robust=False, fast=False, cpu=False, v1=False, threads=1, parc=False, ct=False):

    # print SynthSeg version and checks boolean params for SynthSeg-robust
    if robust:
        fast = True
        assert not v1, 'The flag --v1 cannot be used with --robust since SynthSeg-robust only came out with 2.0.'
        version = 'SynthSeg-robust 2.0'
    else:
        version = 'SynthSeg 1.0' if v1 else 'SynthSeg 2.0'
        if fast:
            version += ' (fast)'
    print('\n' + version + '\n')

    # enforce CPU processing if necessary
    if cpu:
        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # limit the number of threads to be used if running on CPU
    if threads == 1:
        print('using 1 thread')
    else:
        print('using %s threads' % threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    # path models
    if robust:
        path_model_segmentation = os.path.join(model_dir, 'synthseg_robust_2.0.h5')
    else:
        path_model_segmentation = os.path.join(model_dir, 'synthseg_2.0.h5')
    path_model_parcellation = os.path.join(model_dir, 'synthseg_parc_2.0.h5')
    path_model_qc = os.path.join(model_dir, 'synthseg_qc_2.0.h5')

    # path labels
    labels_segmentation = os.path.join(labels_dir, 'synthseg_segmentation_labels_2.0.npy')
    labels_denoiser = os.path.join(labels_dir, 'synthseg_denoiser_labels_2.0.npy')
    labels_parcellation = os.path.join(labels_dir, 'synthseg_parcellation_labels.npy')
    labels_qc = os.path.join(labels_dir, 'synthseg_qc_labels_2.0.npy')
    names_segmentations_labels = os.path.join(labels_dir, 'synthseg_segmentation_names_2.0.npy')
    names_parcellation_labels = os.path.join(labels_dir, 'synthseg_parcellation_names.npy')
    names_qc_labels = os.path.join(labels_dir, 'synthseg_qc_names_2.0.npy')
    topology_classes = os.path.join(labels_dir, 'synthseg_topological_classes_2.0.npy')
    n_neutral_labels = 19

    # use previous model if needed
    if v1:
        path_model_segmentation = os.path.join(model_dir, 'synthseg_1.0.h5')
        labels_segmentation = labels_segmentation.replace('_2.0.npy', '.npy')
        labels_qc = labels_qc.replace('_2.0.npy', '.npy')
        names_segmentations_labels = names_segmentations_labels.replace('_2.0.npy', '.npy')
        names_qc_labels = names_qc_labels.replace('_2.0.npy', '.npy')
        topology_classes = topology_classes.replace('_2.0.npy', '.npy')
        n_neutral_labels = 18

    converted = np.array(input_array, dtype=np.float32)
    temp_file = nib.Nifti1Image(converted, affine)
    nib.save(temp_file, './tempOriginal.nii.gz')

    # run prediction
    predict(path_images='./tempOriginal.nii.gz',
            path_segmentations='./tempLabels.nii.gz',
            path_model_segmentation=path_model_segmentation,
            labels_segmentation=labels_segmentation,
            robust=robust,
            fast=fast,
            v1=v1,
            do_parcellation=parc,
            n_neutral_labels=n_neutral_labels,
            names_segmentation=names_segmentations_labels,
            labels_denoiser=labels_denoiser,
            path_posteriors=post,
            path_resampled=resample,
            path_volumes=vol,
            path_model_parcellation=path_model_parcelation,
            labels_parcellation=labels_parcellation,
            names_parcellation=names_parcellation_labels,
            path_model_qc=path_model_qc,
            labels_qc=labels_qc,
            path_qc_scores=qc,
            names_qc=names_qc_labels,
            cropping=crop,
            topology_classes=topology_classes,
            ct=ct)
    
    label_array = nib.load("./tempLabels.nii.gz").get_fdata()

    os.remove("./tempOriginal.nii.gz")
    #os.remove("./tempLabels.nii.gz")
      
    return label_array

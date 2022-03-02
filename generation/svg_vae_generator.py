from __future__ import print_function

import copy
import logging
import re
import warnings

from bezmerizing import Path
# for drawing the paths
from flat import document
from flat import rgb
from flat import shape
from flat import svg
from magenta.models import svg_vae
from magenta.models.svg_vae import svg_utils
import numpy as np
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
import tensorflow as tf

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()




#### tfe = tf.contrib.eager
Modes = tf.estimator.ModeKeys

tf.enable_eager_execution()
tf.get_logger().setLevel('WARN')

logging.getLogger("mlperf_compliance").setLevel(logging.ERROR)
warnings.simplefilter("ignore")

from IPython.display import display
from IPython.display import SVG

#--------------------------------------------------------------------------------------------------

class SVG_VAE_Generator():
    def __init__(self, problem_name, hparam_set, hparams, model_name, ckpt_dir, split=Modes.TRAIN) -> None:
        self.svg_model, self.svg_hparams = self.__initialize_model_svg(problem_name, hparam_set, hparams, model_name, ckpt_dir, split=Modes.TRAIN)

    def __initialize_model_svg(self, problem_name, hparam_set, hparams, model_name, ckpt_dir, split=Modes.TRAIN):
        """Returns an initialized model, dataset iterator and hparams."""
        tf.reset_default_graph()

        # Create hparams and get glyphazzn problem definition
        hparams = trainer_lib.create_hparams(hparam_set, hparams,
                                            problem_name=problem_name)
        problem = registry.problem(problem_name)

        # Get model definition
        ModelClass = registry.model(model_name)
        model = ModelClass(hparams, mode=Modes.PREDICT,
                            problem_hparams=hparams.problem_hparams)

        # The model apparently needs to be "fed" a sample before you can
        # initialize it from the checkpoint. as far as I can tell, it
        # doesn't matter what's in this sample—it can be completely made up.
        # I'm not sure if I got these dimensions right, not sure if it matters?
        features = {}
        features['bottleneck'] = tf.random_normal([64, 32])
        features['targets_cls'] = [[11]*64]
        features['inputs'] = tf.zeros([64, 51, 1, hparams.problem.feature_dim])
        features['targets_psr'] = tf.zeros([64, 1, 64*64, 1])
        features['targets'] = tf.zeros([64, 51, 1, hparams.problem.feature_dim])
        output, extra_losses = model(features)
        model.initialize_from_ckpt(ckpt_dir)  # initializes ops

        return model, hparams

    def __get_means_stdevs(self):
        with tf.gfile.Open('../data/mean.npz', 'rb') as f:
            mean_npz = np.load(f)
        with tf.gfile.Open('../data/stdev.npz', 'rb') as f:
            stdev_npz = np.load(f)
        return mean_npz, stdev_npz


    def __show(self, page) -> None:
        display(SVG(page.svg()))



def initialize_model_svg(problem_name, hparam_set, hparams, model_name, ckpt_dir, split=Modes.TRAIN):
    """Returns an initialized model, dataset iterator and hparams."""
    tf.reset_default_graph()

    # create hparams and get glyphazzn problem definition
    hparams = trainer_lib.create_hparams(hparam_set, hparams,
                                        problem_name=problem_name)
    problem = registry.problem(problem_name)

    # get model definition
    ModelClass = registry.model(model_name)
    model = ModelClass(hparams, mode=Modes.PREDICT,
                        problem_hparams=hparams.problem_hparams)

    # the model apparently needs to be "fed" a sample before you can
    # initialize it from the checkpoint. as far as I can tell, it
    # doesn't matter what's in this sample—it can be completely made up.
    # I'm not sure if I got these dimensions right, not sure if it matters?
    features = {}
    features['bottleneck'] = tf.random_normal([64, 32])
    features['targets_cls'] = [[11]*64]
    features['inputs'] = tf.zeros([64, 51, 1, hparams.problem.feature_dim])
    features['targets_psr'] = tf.zeros([64, 1, 64*64, 1])
    features['targets'] = tf.zeros([64, 51, 1, hparams.problem.feature_dim])
    output, extra_losses = model(features)
    model.initialize_from_ckpt(ckpt_dir)  # initializes ops

    return model, hparams

#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    svg_vae_generator = SVG_VAE_Generator(
        problem_name='glyph_azzn_problem',
        hparam_set='svg_decoder',
        hparams=(
            "vae_ckpt_dir=../models/image_vae_external,"
            "vae_hparam_set=image_vae,mix_temperature=0.0001,gauss_temperature=0.0001"
        ),
        model_name='svg_decoder',
        ckpt_dir='../models/svg_decoder_external/'
    )

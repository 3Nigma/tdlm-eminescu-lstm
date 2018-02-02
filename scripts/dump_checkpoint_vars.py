# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This script is an entry point for dumping checkpoints for various deeplearning frameworks.
"""

from __future__ import print_function

import argparse
from tensorflow_checkpoint_dumper import TensorflowCheckpointDumper

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint_file',
      type=str,
      required=True,
      help='Path to the model checkpoint')
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='The output directory where to store the converted weights')
  parser.add_argument(
      '--remove_variables_regex',
      type=str,
      default='',
      help='A regular expression to match against variable names that should '
      'not be included')
  FLAGS, unparsed = parser.parse_known_args()

  if unparsed:
    parser.print_help()
    print('Unrecognized flags: ', unparsed)
    exit(-1)

  checkpoint_dumper = TensorflowCheckpointDumper(FLAGS.checkpoint_file, FLAGS.output_dir, FLAGS.remove_variables_regex)
  checkpoint_dumper.build_and_dump_vars()

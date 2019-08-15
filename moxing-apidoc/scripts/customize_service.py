# Copyright 2019 ModelArts Authors from Huawei Cloud. All Rights Reserved.
# https://www.huaweicloud.com/product/modelarts.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from model_service.tfserving_model_service import TfServingBaseService


class CnnService(TfServingBaseService):
  def _preprocess(self, data):
    preprocessed_data = {}
    for k, v in data.items():
      for file_name, file_content in v.items():
        image = Image.open(file_content)
        image = image.convert('RGB')
        image = np.asarray(image, dtype=np.float32)
        image = image[np.newaxis, :, :, :]
        preprocessed_data[k] = image
    return preprocessed_data

  def _postprocess(self, data):
    
    with open(os.path.join(self.model_path, 'labels.txt')) as f:
      labels_list = f.read().split('\n')
      
    outputs = {}

    def softmax(x):
      x = np.array(x)
      orig_shape = x.shape

      if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)
        if len(denominator.shape) == 1:
          denominator = denominator.reshape((denominator.shape[0], 1))
        x = x * denominator
      else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
      assert x.shape == orig_shape

      return x

    predictions_list = softmax(data['logits'][0])
    predictions_list = ['%.3f' % p for p in predictions_list]

    scores = dict(zip(labels_list, predictions_list))
    scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(labels_list) > 5:
      scores = scores[:5]
    label_index = predictions_list.index(max(predictions_list))
    predicted_label = str(labels_list[label_index])
    outputs['predicted_label'] = predicted_label
    outputs['scores'] = scores
    return outputs

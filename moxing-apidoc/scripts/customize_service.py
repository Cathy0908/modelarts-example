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
import h5py
import numpy as np
from PIL import Image
from model_service.tfserving_model_service import TfServingBaseService


def normalize_feature(feature):
  return feature / np.linalg.norm(feature)


def get_topk_result(query, all_data, query_included=True, metric_type='cosine_similarity', top_k=1):
  """Get the nearset data to query in all_data

  :param query: having shape of (1, k) where k is the number of feature dimension.
  :param all_data: having shape of (m, k) where k is the number of feature dimension and m is the number of samples.
  :param query_included: whether the query is included in all_data.
  :param metric_type: metric on how to compute the similarities between any two smaples.
  :param top_k: the top k similar examples will be returned.

  :return an index or a list of index indicating the location of the most similar samples in all_data.
  """
  if metric_type == 'cosine_similarity':
    similarity_scores = np.dot(np.array(query), (np.array(all_data)).T)
    rankings = np.argsort(similarity_scores)[::-1]  # larger score means the samples are more similar
  elif metric_type == 'euclidean_distance':
    distances = [np.sqrt(np.sum(np.square(query, data))) for data in all_data]
    rankings = np.argsort(distances)  # smaller distance means the samples are more similar
  else:
    print('No metric_type is provided!')
  if top_k > 1:
    if query_included:
      return rankings[1:top_k + 1]
    return rankings[0:top_k]
  elif top_k == 1:
    if query_included:
      return rankings[1]
    return rankings[0]
  else:
    print('Please set valid top_k number!')


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
    h5f = h5py.File(os.path.join(self.model_path, 'index'), 'r')
    labels_list = h5f['labels_list'][:]
    is_multilabel = h5f['is_multilabel'].value
    h5f.close()
    print(labels_list)
    print(data)
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

    if is_multilabel:
      predictions_list = [1 / (1 + np.exp(-p)) for p in data['logits'][0]]
    else:
      predictions_list = softmax(data['logits'][0])
    predictions_list = ['%.3f' % p for p in predictions_list]

    scores = dict(zip(labels_list, predictions_list))
    scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(labels_list) > 5:
      scores = scores[:5]
    label_index = predictions_list.index(max(predictions_list))
    # predicted_label = labels_list[label_index].decode('utf-8')
    predicted_label = str(labels_list[label_index])
    print('predicted label is: %s ' % predicted_label)
    outputs['predicted_label'] = predicted_label
    outputs['scores'] = scores
    return outputs

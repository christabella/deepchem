"""
NCI molecules DB dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem

logger = logging.getLogger(__name__)


def load_nci_logp(featurizer='ECFP', shard_size=1000, split='random', reload=True):
  # Load nci dataset
  logger.info("About to load NCI dataset.")
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "nci/" + featurizer + "/" + str(split))

  # dataset_file = os.path.join(data_dir, "nci_unique.csv")
  dataset_file = os.path.join(os.getcwd(), "../datasets/nci_logp.csv")
  if not os.path.exists(dataset_file):
    raise ValueError("Dataset file not found at {}".format(dataset_file))

  all_nci_logp_tasks = (['logp'])

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return all_nci_logp_tasks, all_dataset, transformers

  # Featurize nci dataset
  logger.info("About to featurize nci dataset.")
  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=all_nci_logp_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file, shard_size=shard_size)

  # Initialize transformers
  logger.info("About to transform data")
  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=dataset)
  ]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return all_nci_logp_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  logger.info("Performing new split.")
  train, valid, test = splitter.train_valid_test_split(dataset)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return all_nci_logp_tasks, (train, valid, test), transformers

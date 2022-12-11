Contains classes that help build the pipeline, with DataSets, DataLoaders, Metrics, Split data...

# Running dataset

1. Set the python path to the root of the repository: 

```sh
export PYTHONPATH=$PWD
```

2. Run the python job: 

```sh
python src/commons/dataset.py
```

# Understanding `modded_recorder.py`

The package `pytorch_vit`, whose source code can be seen [here](https://github.com/lucidrains/vit-pytorch), implements many different Vision Transformer architectures.

In particular, it has a `pytorch_vit.recorder` module which allows transforming the base model `pytorch_vit.ViT` into a model which returns attention maps in the forward pass. This is really useful for extracting attention maps from a trained model.

However, it was chosen to implement in `notebooks/nb_3_ViT_pretraining.ipynb` the model `pytorch_vit.SimpleViT`, which is a simplification (and optimization) of the original architecture. Unfortunately, `pytorch_vit.recorder` is not compatible with this module.

Thus, in order to still leverage the existing code, it was moddified (`modded_recorder.py`) as to work with `pytorch_vit.SimpleViT` instead.


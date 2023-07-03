# dependencies

following packages are required to run the code in this folder:

```Bash
tensorflow keras-tuner numpy pandas matplotlib
``` 

# Recommended environment

```Bash
conda create -n tf2 python=3.9 tensorflow-gpu=2.4.1 numpy=1.19.5 pandas=1.5.2
conda activate tf2
pip install keras-tuner==1.3.5
pip install gast==0.3.3
conda deactivate
```


# Common issue
## Tensorflow AutoGraph warning:

 > WARNING:tensorflow:AutoGraph could not transform <bound method SamplingLayer.call of <keras_custom.sampling.SamplingLayer object at 0x7f06d9e5beb0>> and will run it as-is.
 > Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
 > Cause: module 'gast' has no attribute 'Index'
 > To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

 > Keras error: Cause: module 'gast' has no attribute 'Index'

solution:
```Bash
pip install gast==0.3.3
```

## RaggedTensor ReduceSum Error: 

error message:
 > Cannot convert a symbolic Tensor (gradient_tape/model_1/tf.math.reduce_mean_1/RaggedReduceMean/RaggedReduceSum/sub:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported

solution:
```Bash
pip install numpy=1.19.5
```
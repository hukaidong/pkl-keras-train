# dependencies

following packages are required to run the code in this folder:

```
tensorflow keras-tuner numpy pandas pyyaml
(optional) matplotlib
``` 

# Recommended environment

```Bash
pip install tensorflow==2.12.0
pip install keras-tuner numpy pandas pyyaml
```


# Common issue

Following are issues happened with tensorflow 2.6.0. Solutions are not applicable to recommended 2.12.0 version.

## Tensorflow AutoGraph warning:

 > WARNING:tensorflow:AutoGraph could not transform <bound method SamplingLayer.call of <keras_custom.sampling.SamplingLayer object at 0x7f06d9e5beb0>> and will run it as-is.
 > Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
 > Cause: module 'gast' has no attribute 'Index'
 > To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

 > Keras error: Cause: module 'gast' has no attribute 'Index'

If you are using tensorflow=2.6.0, solution is to downgrade gast to 0.3.3:
```Bash
pip install gast==0.3.3
```

## RaggedTensor ReduceSum Error: 

error message:
 > Cannot convert a symbolic Tensor (gradient_tape/model_1/tf.math.reduce_mean_1/RaggedReduceMean/RaggedReduceSum/sub:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported

Solution is to downgrade numpy to 1.19.5:
```Bash
pip install numpy=1.19.5
```
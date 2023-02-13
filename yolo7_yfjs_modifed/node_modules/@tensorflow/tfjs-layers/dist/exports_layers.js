/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import { InputLayer } from './engine/input_layer';
import { Layer } from './engine/topology';
import { input } from './exports';
import { ELU, LeakyReLU, PReLU, ReLU, Softmax, ThresholdedReLU } from './layers/advanced_activations';
import { Conv1D, Conv2D, Conv2DTranspose, Conv3D, Cropping2D, SeparableConv2D, UpSampling2D, Conv3DTranspose } from './layers/convolutional';
import { DepthwiseConv2D } from './layers/convolutional_depthwise';
import { ConvLSTM2D, ConvLSTM2DCell } from './layers/convolutional_recurrent';
import { Activation, Dense, Dropout, Flatten, Masking, Permute, RepeatVector, Reshape, SpatialDropout1D } from './layers/core';
import { Embedding } from './layers/embeddings';
import { Add, Average, Concatenate, Dot, Maximum, Minimum, Multiply } from './layers/merge';
import { AlphaDropout, GaussianDropout, GaussianNoise } from './layers/noise';
import { BatchNormalization, LayerNormalization } from './layers/normalization';
import { ZeroPadding2D } from './layers/padding';
import { AveragePooling1D, AveragePooling2D, AveragePooling3D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPooling1D, MaxPooling2D, MaxPooling3D } from './layers/pooling';
import { GRU, GRUCell, LSTM, LSTMCell, RNN, RNNCell, SimpleRNN, SimpleRNNCell, StackedRNNCells } from './layers/recurrent';
import { Bidirectional, TimeDistributed } from './layers/wrappers';
// TODO(cais): Add doc string to all the public static functions in this
//   class; include exectuable JavaScript code snippets where applicable
//   (b/74074458).
// Input Layer.
/**
 * An input layer is an entry point into a `tf.LayersModel`.
 *
 * `InputLayer` is generated automatically for `tf.Sequential`` models by
 * specifying the `inputshape` or `batchInputShape` for the first layer.  It
 * should not be specified explicitly. However, it can be useful sometimes,
 * e.g., when constructing a sequential model from a subset of another
 * sequential model's layers. Like the code snippet below shows.
 *
 * ```js
 * // Define a model which simply adds two inputs.
 * const model1 = tf.sequential();
 * model1.add(tf.layers.dense({inputShape: [4], units: 3, activation: 'relu'}));
 * model1.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
 * model1.summary();
 * model1.predict(tf.zeros([1, 4])).print();
 *
 * // Construct another model, reusing the second layer of `model1` while
 * // not using the first layer of `model1`. Note that you cannot add the second
 * // layer of `model` directly as the first layer of the new sequential model,
 * // because doing so will lead to an error related to the fact that the layer
 * // is not an input layer. Instead, you need to create an `inputLayer` and add
 * // it to the new sequential model before adding the reused layer.
 * const model2 = tf.sequential();
 * // Use an inputShape that matches the input shape of `model1`'s second
 * // layer.
 * model2.add(tf.layers.inputLayer({inputShape: [3]}));
 * model2.add(model1.layers[1]);
 * model2.summary();
 * model2.predict(tf.zeros([1, 3])).print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Inputs', namespace: 'layers'}
 */
export function inputLayer(args) {
    return new InputLayer(args);
}
// Advanced Activation Layers.
/**
 * Exponetial Linear Unit (ELU).
 *
 * It follows:
 * `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
 * `f(x) = x for x >= 0`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Fast and Accurate Deep Network Learning by Exponential Linear Units
 * (ELUs)](https://arxiv.org/abs/1511.07289v1)
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function elu(args) {
    return new ELU(args);
}
/**
 * Rectified Linear Unit activation function.
 *
 * Input shape:
 *   Arbitrary. Use the config field `inputShape` (Array of integers, does
 *   not include the sample axis) when using this layer as the first layer
 *   in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function reLU(args) {
    return new ReLU(args);
}
/**
 * Leaky version of a rectified linear unit.
 *
 * It allows a small gradient when the unit is not active:
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function leakyReLU(args) {
    return new LeakyReLU(args);
}
/**
 * Parameterized version of a leaky rectified linear unit.
 *
 * It follows
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 * wherein `alpha` is a trainable weight.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function prelu(args) {
    return new PReLU(args);
}
/**
 * Softmax activation layer.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function softmax(args) {
    return new Softmax(args);
}
/**
 * Thresholded Rectified Linear Unit.
 *
 * It follows:
 * `f(x) = x for x > theta`,
 * `f(x) = 0 otherwise`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Zero-Bias Autoencoders and the Benefits of Co-Adapting
 * Features](http://arxiv.org/abs/1402.3337)
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function thresholdedReLU(args) {
    return new ThresholdedReLU(args);
}
// Convolutional Layers.
/**
 * 1D convolution layer (e.g., temporal convolution).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input over a single spatial (or temporal) dimension
 * to produce a tensor of outputs.
 *
 * If `use_bias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model, provide an
 * `inputShape` argument `Array` or `null`.
 *
 * For example, `inputShape` would be:
 * - `[10, 128]` for sequences of 10 vectors of 128-dimensional vectors
 * - `[null, 128]` for variable-length sequences of 128-dimensional vectors.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional',  namespace: 'layers'}
 */
export function conv1d(args) {
    return new Conv1D(args);
}
/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of outputs.
 *
 * If `useBias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 3]` for 128x128 RGB pictures
 * in `dataFormat='channelsLast'`.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function conv2d(args) {
    return new Conv2D(args);
}
/**
 * Transposed convolutional layer (sometimes called Deconvolution).
 *
 * The need for transposed convolutions generally arises
 * from the desire to use a transformation going in the opposite direction of
 * a normal convolution, i.e., from something that has the shape of the output
 * of some convolution to something that has the shape of its input while
 * maintaining a connectivity pattern that is compatible with said
 * convolution.
 *
 * When using this layer as the first layer in a model, provide the
 * configuration `inputShape` (`Array` of integers, does not include the
 * sample axis), e.g., `inputShape: [128, 128, 3]` for 128x128 RGB pictures in
 * `dataFormat: 'channelsLast'`.
 *
 * Input shape:
 *   4D tensor with shape:
 *   `[batch, channels, rows, cols]` if `dataFormat` is `'channelsFirst'`.
 *   or 4D tensor with shape
 *   `[batch, rows, cols, channels]` if `dataFormat` is `'channelsLast`.
 *
 * Output shape:
 *   4D tensor with shape:
 *   `[batch, filters, newRows, newCols]` if `dataFormat` is
 * `'channelsFirst'`. or 4D tensor with shape:
 *   `[batch, newRows, newCols, filters]` if `dataFormat` is `'channelsLast'`.
 *
 * References:
 *   - [A guide to convolution arithmetic for deep
 * learning](https://arxiv.org/abs/1603.07285v1)
 *   - [Deconvolutional
 * Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function conv2dTranspose(args) {
    return new Conv2DTranspose(args);
}
/**
 * 3D convolution layer (e.g. spatial convolution over volumes).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of outputs.
 *
 * If `useBias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 128, 1]` for 128x128x128 grayscale volumes
 * in `dataFormat='channelsLast'`.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function conv3d(args) {
    return new Conv3D(args);
}
export function conv3dTranspose(args) {
    return new Conv3DTranspose(args);
}
/**
 * Depthwise separable 2D convolution.
 *
 * Separable convolution consists of first performing
 * a depthwise spatial convolution
 * (which acts on each input channel separately)
 * followed by a pointwise convolution which mixes together the resulting
 * output channels. The `depthMultiplier` argument controls how many
 * output channels are generated per input channel in the depthwise step.
 *
 * Intuitively, separable convolutions can be understood as
 * a way to factorize a convolution kernel into two smaller kernels,
 * or as an extreme version of an Inception block.
 *
 * Input shape:
 *   4D tensor with shape:
 *     `[batch, channels, rows, cols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, rows, cols, channels]` if data_format='channelsLast'.
 *
 * Output shape:
 *   4D tensor with shape:
 *     `[batch, filters, newRows, newCols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, newRows, newCols, filters]` if data_format='channelsLast'.
 *     `rows` and `cols` values might have changed due to padding.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function separableConv2d(args) {
    return new SeparableConv2D(args);
}
/**
 * Cropping layer for 2D input (e.g., image).
 *
 * This layer can crop an input
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, croppedRows, croppedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, croppedRows, croppedCols]`.
 *
 * Examples
 * ```js
 *
 * const model = tf.sequential();
 * model.add(tf.layers.cropping2D({cropping:[[2, 2], [2, 2]],
 *                                inputShape: [128, 128, 3]}));
 * //now output shape is [batch, 124, 124, 3]
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function cropping2D(args) {
    return new Cropping2D(args);
}
/**
 * Upsampling layer for 2D inputs.
 *
 * Repeats the rows and columns of the data
 * by size[0] and size[1] respectively.
 *
 *
 * Input shape:
 *    4D tensor with shape:
 *     - If `dataFormat` is `"channelsLast"`:
 *         `[batch, rows, cols, channels]`
 *     - If `dataFormat` is `"channelsFirst"`:
 *        `[batch, channels, rows, cols]`
 *
 * Output shape:
 *     4D tensor with shape:
 *     - If `dataFormat` is `"channelsLast"`:
 *        `[batch, upsampledRows, upsampledCols, channels]`
 *     - If `dataFormat` is `"channelsFirst"`:
 *         `[batch, channels, upsampledRows, upsampledCols]`
 *
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function upSampling2d(args) {
    return new UpSampling2D(args);
}
// Convolutional(depthwise) Layers.
/**
 * Depthwise separable 2D convolution.
 *
 * Depthwise Separable convolutions consists in performing just the first step
 * in a depthwise spatial convolution (which acts on each input channel
 * separately). The `depthMultplier` argument controls how many output channels
 * are generated per input channel in the depthwise step.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function depthwiseConv2d(args) {
    return new DepthwiseConv2D(args);
}
// Basic Layers.
/**
 * Applies an activation function to an output.
 *
 * This layer applies element-wise activation function.  Other layers, notably
 * `dense` can also apply activation functions.  Use this isolated activation
 * function to extract the values before and after the
 * activation. For instance:
 *
 * ```js
 * const input = tf.input({shape: [5]});
 * const denseLayer = tf.layers.dense({units: 1});
 * const activationLayer = tf.layers.activation({activation: 'relu6'});
 *
 * // Obtain the output symbolic tensors by applying the layers in order.
 * const denseOutput = denseLayer.apply(input);
 * const activationOutput = activationLayer.apply(denseOutput);
 *
 * // Create the model based on the inputs.
 * const model = tf.model({
 *     inputs: input,
 *     outputs: [denseOutput, activationOutput]
 * });
 *
 * // Collect both outputs and print separately.
 * const [denseOut, activationOut] = model.predict(tf.randomNormal([6, 5]));
 * denseOut.print();
 * activationOut.print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function activation(args) {
    return new Activation(args);
}
/**
 * Creates a dense (fully connected) layer.
 *
 * This layer implements the operation:
 *   `output = activation(dot(input, kernel) + bias)`
 *
 * `activation` is the element-wise activation function
 *   passed as the `activation` argument.
 *
 * `kernel` is a weights matrix created by the layer.
 *
 * `bias` is a bias vector created by the layer (only applicable if `useBias`
 * is `true`).
 *
 * **Input shape:**
 *
 *   nD `tf.Tensor` with shape: `(batchSize, ..., inputDim)`.
 *
 *   The most common situation would be
 *   a 2D input with shape `(batchSize, inputDim)`.
 *
 * **Output shape:**
 *
 *   nD tensor with shape: `(batchSize, ..., units)`.
 *
 *   For instance, for a 2D input with shape `(batchSize, inputDim)`,
 *   the output would have shape `(batchSize, units)`.
 *
 * Note: if the input to the layer has a rank greater than 2, then it is
 * flattened prior to the initial dot product with the kernel.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function dense(args) {
    return new Dense(args);
}
/**
 * Applies
 * [dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) to
 * the input.
 *
 * Dropout consists in randomly setting a fraction `rate` of input units to 0 at
 * each update during training time, which helps prevent overfitting.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function dropout(args) {
    return new Dropout(args);
}
/**
 * Spatial 1D version of Dropout.
 *
 * This Layer type performs the same function as the Dropout layer, but it drops
 * entire 1D feature maps instead of individual elements. For example, if an
 * input example consists of 3 timesteps and the feature map for each timestep
 * has a size of 4, a `spatialDropout1d` layer may zero out the feature maps
 * of the 1st timesteps and 2nd timesteps completely while sparing all feature
 * elements of the 3rd timestep.
 *
 * If adjacent frames (timesteps) are strongly correlated (as is normally the
 * case in early convolution layers), regular dropout will not regularize the
 * activation and will otherwise just result in merely an effective learning
 * rate decrease. In this case, `spatialDropout1d` will help promote
 * independence among feature maps and should be used instead.
 *
 * **Arguments:**
 *   rate: A floating-point number >=0 and <=1. Fraction of the input elements
 *     to drop.
 *
 * **Input shape:**
 *   3D tensor with shape `(samples, timesteps, channels)`.
 *
 * **Output shape:**
 *   Same as the input shape.
 *
 * References:
 *   - [Efficient Object Localization Using Convolutional
 *      Networks](https://arxiv.org/abs/1411.4280)
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function spatialDropout1d(args) {
    return new SpatialDropout1D(args);
}
/**
 * Flattens the input. Does not affect the batch size.
 *
 * A `Flatten` layer flattens each batch in its inputs to 1D (making the output
 * 2D).
 *
 * For example:
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const flattenLayer = tf.layers.flatten();
 * // Inspect the inferred output shape of the flatten layer, which
 * // equals `[null, 12]`. The 2nd dimension is 4 * 3, i.e., the result of the
 * // flattening. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(flattenLayer.apply(input).shape));
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function flatten(args) {
    return new Flatten(args);
}
/**
 * Repeats the input n times in a new dimension.
 *
 * ```js
 *  const model = tf.sequential();
 *  model.add(tf.layers.repeatVector({n: 4, inputShape: [2]}));
 *  const x = tf.tensor2d([[10, 20]]);
 *  // Use the model to do inference on a data point the model hasn't see
 *  model.predict(x).print();
 *  // output shape is now [batch, 2, 4]
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function repeatVector(args) {
    return new RepeatVector(args);
}
/**
 * Reshapes an input to a certain shape.
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const reshapeLayer = tf.layers.reshape({targetShape: [2, 6]});
 * // Inspect the inferred output shape of the Reshape layer, which
 * // equals `[null, 2, 6]`. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(reshapeLayer.apply(input).shape));
 * ```
 *
 * Input shape:
 *   Arbitrary, although all dimensions in the input shape must be fixed.
 *   Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 *
 * Output shape:
 *   [batchSize, targetShape[0], targetShape[1], ...,
 *    targetShape[targetShape.length - 1]].
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function reshape(args) {
    return new Reshape(args);
}
/**
 * Permutes the dimensions of the input according to a given pattern.
 *
 * Useful for, e.g., connecting RNNs and convnets together.
 *
 * Example:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.permute({
 *   dims: [2, 1],
 *   inputShape: [10, 64]
 * }));
 * console.log(model.outputShape);
 * // Now model's output shape is [null, 64, 10], where null is the
 * // unpermuted sample (batch) dimension.
 * ```
 *
 * Input shape:
 *   Arbitrary. Use the configuration field `inputShape` when using this
 *   layer as the first layer in a model.
 *
 * Output shape:
 *   Same rank as the input shape, but with the dimensions re-ordered (i.e.,
 *   permuted) according to the `dims` configuration of this layer.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function permute(args) {
    return new Permute(args);
}
/**
 * Maps positive integers (indices) into dense vectors of fixed size.
 * eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
 *
 * **Input shape:** 2D tensor with shape: `[batchSize, sequenceLength]`.
 *
 * **Output shape:** 3D tensor with shape: `[batchSize, sequenceLength,
 * outputDim]`.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function embedding(args) {
    return new Embedding(args);
}
// Merge Layers.
/**
 * Layer that performs element-wise addition on an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). The inputs are specified as an
 * `Array` when the `apply` method of the `Add` layer instance is called. For
 * example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const addLayer = tf.layers.add();
 * const sum = addLayer.apply([input1, input2]);
 * console.log(JSON.stringify(sum.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function add(args) {
    return new Add(args);
}
/**
 * Layer that performs element-wise averaging on an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const averageLayer = tf.layers.average();
 * const average = averageLayer.apply([input1, input2]);
 * console.log(JSON.stringify(average.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function average(args) {
    return new Average(args);
}
/**
 * Layer that concatenates an `Array` of inputs.
 *
 * It takes a list of tensors, all of the same shape except for the
 * concatenation axis, and returns a single tensor, the concatenation
 * of all inputs. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 3]});
 * const concatLayer = tf.layers.concatenate();
 * const output = concatLayer.apply([input1, input2]);
 * console.log(JSON.stringify(output.shape));
 * // You get [null, 2, 5], with the first dimension as the undetermined batch
 * // dimension. The last dimension (5) is the result of concatenating the
 * // last dimensions of the inputs (2 and 3).
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function concatenate(args) {
    return new Concatenate(args);
}
/**
 * Layer that computes the element-wise maximum an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const maxLayer = tf.layers.maximum();
 * const max = maxLayer.apply([input1, input2]);
 * console.log(JSON.stringify(max.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function maximum(args) {
    return new Maximum(args);
}
/**
 * Layer that computes the element-wise minimum of an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const minLayer = tf.layers.minimum();
 * const min = minLayer.apply([input1, input2]);
 * console.log(JSON.stringify(min.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function minimum(args) {
    return new Minimum(args);
}
/**
 * Layer that multiplies (element-wise) an `Array` of inputs.
 *
 * It takes as input an Array of tensors, all of the same
 * shape, and returns a single tensor (also of the same shape).
 * For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const input3 = tf.input({shape: [2, 2]});
 * const multiplyLayer = tf.layers.multiply();
 * const product = multiplyLayer.apply([input1, input2, input3]);
 * console.log(product.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function multiply(args) {
    return new Multiply(args);
}
/**
 * Layer that computes a dot product between samples in two tensors.
 *
 * E.g., if applied to a list of two tensors `a` and `b` both of shape
 * `[batchSize, n]`, the output will be a tensor of shape `[batchSize, 1]`,
 * where each entry at index `[i, 0]` will be the dot product between
 * `a[i, :]` and `b[i, :]`.
 *
 * Example:
 *
 * ```js
 * const dotLayer = tf.layers.dot({axes: -1});
 * const x1 = tf.tensor2d([[10, 20], [30, 40]]);
 * const x2 = tf.tensor2d([[-1, -2], [-3, -4]]);
 *
 * // Invoke the layer's apply() method in eager (imperative) mode.
 * const y = dotLayer.apply([x1, x2]);
 * y.print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function dot(args) {
    return new Dot(args);
}
// Normalization Layers.
/**
 * Batch normalization layer (Ioffe and Szegedy, 2014).
 *
 * Normalize the activations of the previous layer at each batch,
 * i.e. applies a transformation that maintains the mean activation
 * close to 0 and the activation standard deviation close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape` (Array of integers, does
 *   not include the sample axis) when calling the constructor of this class,
 *   if this layer is used as a first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Batch Normalization: Accelerating Deep Network Training by Reducing
 * Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
 *
 * @doc {heading: 'Layers', subheading: 'Normalization', namespace: 'layers'}
 */
export function batchNormalization(args) {
    return new BatchNormalization(args);
}
/**
 * Layer-normalization layer (Ba et al., 2016).
 *
 * Normalizes the activations of the previous layer for each given example in a
 * batch independently, instead of across a batch like in `batchNormalization`.
 * In other words, this layer applies a transformation that maintanis the mean
 * activation within each example close to0 and activation variance close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the argument `inputShape` when using this layer as the first
 *   layer in a model.
 *
 * Output shape:
 *   Same as input.
 *
 * References:
 *   - [Layer Normalization](https://arxiv.org/abs/1607.06450)
 *
 * @doc {heading: 'Layers', subheading: 'Normalization', namespace: 'layers'}
 */
export function layerNormalization(args) {
    return new LayerNormalization(args);
}
// Padding Layers.
/**
 * Zero-padding layer for 2D input (e.g., image).
 *
 * This layer can add rows and columns of zeros
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, paddedRows, paddedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, paddedRows, paddedCols]`.
 *
 * @doc {heading: 'Layers', subheading: 'Padding', namespace: 'layers'}
 */
export function zeroPadding2d(args) {
    return new ZeroPadding2D(args);
}
// Pooling Layers.
/**
 * Average pooling operation for spatial data.
 *
 * Input shape: `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 *
 * `tf.avgPool1d` is an alias.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function averagePooling1d(args) {
    return new AveragePooling1D(args);
}
export function avgPool1d(args) {
    return averagePooling1d(args);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling1d(args) {
    return averagePooling1d(args);
}
/**
 * Average pooling operation for spatial data.
 *
 * Input shape:
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, rows, cols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, pooleRows, pooledCols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, pooleRows, pooledCols]`
 *
 * `tf.avgPool2d` is an alias.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function averagePooling2d(args) {
    return new AveragePooling2D(args);
}
export function avgPool2d(args) {
    return averagePooling2d(args);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling2d(args) {
    return averagePooling2d(args);
}
/**
 * Average pooling operation for 3D data.
 *
 * Input shape
 *   - If `dataFormat === channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, depths, rows, cols, channels]`
 *   - If `dataFormat === channelsFirst`:
 *      4D tensor with shape:
 *       `[batchSize, channels, depths, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, pooledDepths, pooledRows, pooledCols, channels]`
 *   - If `dataFormat=channelsFirst`:
 *       5D tensor with shape:
 *       `[batchSize, channels, pooledDepths, pooledRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function averagePooling3d(args) {
    return new AveragePooling3D(args);
}
export function avgPool3d(args) {
    return averagePooling3d(args);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling3d(args) {
    return averagePooling3d(args);
}
/**
 * Global average pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape:2D tensor with shape: `[batchSize, features]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalAveragePooling1d(args) {
    return new GlobalAveragePooling1D(args);
}
/**
 * Global average pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalAveragePooling2d(args) {
    return new GlobalAveragePooling2D(args);
}
/**
 * Global max pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape:2D tensor with shape: `[batchSize, features]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalMaxPooling1d(args) {
    return new GlobalMaxPooling1D(args);
}
/**
 * Global max pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalMaxPooling2d(args) {
    return new GlobalMaxPooling2D(args);
}
/**
 * Max pooling operation for temporal data.
 *
 * Input shape:  `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function maxPooling1d(args) {
    return new MaxPooling1D(args);
}
/**
 * Max pooling operation for spatial data.
 *
 * Input shape
 *   - If `dataFormat === CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, rows, cols, channels]`
 *   - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *       `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, pooleRows, pooledCols, channels]`
 *   - If `dataFormat=CHANNEL_FIRST`:
 *       4D tensor with shape:
 *       `[batchSize, channels, pooleRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function maxPooling2d(args) {
    return new MaxPooling2D(args);
}
/**
 * Max pooling operation for 3D data.
 *
 * Input shape
 *   - If `dataFormat === channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, depths, rows, cols, channels]`
 *   - If `dataFormat === channelsFirst`:
 *      5D tensor with shape:
 *       `[batchSize, channels, depths, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, pooledDepths, pooledRows, pooledCols, channels]`
 *   - If `dataFormat=channelsFirst`:
 *       5D tensor with shape:
 *       `[batchSize, channels, pooledDepths, pooledRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function maxPooling3d(args) {
    return new MaxPooling3D(args);
}
// Recurrent Layers.
/**
 * Gated Recurrent Unit - Cho et al. 2014.
 *
 * This is an `RNN` layer consisting of one `GRUCell`. However, unlike
 * the underlying `GRUCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.gru({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `GRUCell`'s number of units.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function gru(args) {
    return new GRU(args);
}
/**
 * Cell class for `GRU`.
 *
 * `GRUCell` is distinct from the `RNN` subclass `GRU` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `GRU` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.gruCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `GRUCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.gruCell({units: 4}),
 *   tf.layers.gruCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `gruCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `GRUCell`, use the
 * `tf.layers.gru`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function gruCell(args) {
    return new GRUCell(args);
}
/**
 * Long-Short Term Memory layer - Hochreiter 1997.
 *
 * This is an `RNN` layer consisting of one `LSTMCell`. However, unlike
 * the underlying `LSTMCell`, the `apply` method of `LSTM` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const lstm = tf.layers.lstm({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = lstm.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `LSTMCell`'s number of units.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function lstm(args) {
    return new LSTM(args);
}
/**
 * Cell class for `LSTM`.
 *
 * `LSTMCell` is distinct from the `RNN` subclass `LSTM` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `LSTM` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.lstmCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `LSTMCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.lstmCell({units: 4}),
 *   tf.layers.lstmCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `lstmCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `LSTMCell`, use the
 * `tf.layers.lstm`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function lstmCell(args) {
    return new LSTMCell(args);
}
/**
 * Fully-connected RNN where the output is to be fed back to input.
 *
 * This is an `RNN` layer consisting of one `SimpleRNNCell`. However, unlike
 * the underlying `SimpleRNNCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.simpleRNN({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `SimpleRNNCell`'s number of units.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function simpleRNN(args) {
    return new SimpleRNN(args);
}
/**
 * Cell class for `SimpleRNN`.
 *
 * `SimpleRNNCell` is distinct from the `RNN` subclass `SimpleRNN` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `SimpleRNN` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.simpleRNNCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `SimpleRNNCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.simpleRNNCell({units: 4}),
 *   tf.layers.simpleRNNCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `SimpleRNNCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `SimpleRNNCell`, use the
 * `tf.layers.simpleRNN`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function simpleRNNCell(args) {
    return new SimpleRNNCell(args);
}
/**
 * Convolutional LSTM layer - Xingjian Shi 2015.
 *
 * This is an `ConvRNN2D` layer consisting of one `ConvLSTM2DCell`. However,
 * unlike the underlying `ConvLSTM2DCell`, the `apply` method of `ConvLSTM2D`
 * operates on a sequence of inputs. The shape of the input (not including the
 * first, batch dimension) needs to be 4-D, with the first dimension being time
 * steps. For example:
 *
 * ```js
 * const filters = 3;
 * const kernelSize = 3;
 *
 * const batchSize = 4;
 * const sequenceLength = 2;
 * const size = 5;
 * const channels = 3;
 *
 * const inputShape = [batchSize, sequenceLength, size, size, channels];
 * const input = tf.ones(inputShape);
 *
 * const layer = tf.layers.convLstm2d({filters, kernelSize});
 *
 * const output = layer.apply(input);
 * ```
 */
/** @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'} */
export function convLstm2d(args) {
    return new ConvLSTM2D(args);
}
/**
 * Cell class for `ConvLSTM2D`.
 *
 * `ConvLSTM2DCell` is distinct from the `ConvRNN2D` subclass `ConvLSTM2D` in
 * that its `call` method takes the input data of only a single time step and
 * returns the cell's output at the time step, while `ConvLSTM2D` takes the
 * input data over a number of time steps. For example:
 *
 * ```js
 * const filters = 3;
 * const kernelSize = 3;
 *
 * const sequenceLength = 1;
 * const size = 5;
 * const channels = 3;
 *
 * const inputShape = [sequenceLength, size, size, channels];
 * const input = tf.ones(inputShape);
 *
 * const cell = tf.layers.convLstm2dCell({filters, kernelSize});
 *
 * cell.build(input.shape);
 *
 * const outputSize = size - kernelSize + 1;
 * const outShape = [sequenceLength, outputSize, outputSize, filters];
 *
 * const initialH = tf.zeros(outShape);
 * const initialC = tf.zeros(outShape);
 *
 * const [o, h, c] = cell.call([input, initialH, initialC], {});
 * ```
 */
/** @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'} */
export function convLstm2dCell(args) {
    return new ConvLSTM2DCell(args);
}
/**
 * Base class for recurrent layers.
 *
 * Input shape:
 *   3D tensor with shape `[batchSize, timeSteps, inputDim]`.
 *
 * Output shape:
 *   - if `returnState`, an Array of tensors (i.e., `tf.Tensor`s). The first
 *     tensor is the output. The remaining tensors are the states at the
 *     last time step, each with shape `[batchSize, units]`.
 *   - if `returnSequences`, the output will have shape
 *     `[batchSize, timeSteps, units]`.
 *   - else, the output will have shape `[batchSize, units]`.
 *
 * Masking:
 *   This layer supports masking for input data with a variable number
 *   of timesteps. To introduce masks to your data,
 *   use an embedding layer with the `mask_zero` parameter
 *   set to `True`.
 *
 * Notes on using statefulness in RNNs:
 *   You can set RNN layers to be 'stateful', which means that the states
 *   computed for the samples in one batch will be reused as initial states
 *   for the samples in the next batch. This assumes a one-to-one mapping
 *   between samples in different successive batches.
 *
 *   To enable statefulness:
 *     - specify `stateful: true` in the layer constructor.
 *     - specify a fixed batch size for your model, by passing
 *       if sequential model:
 *         `batchInputShape=[...]` to the first layer in your model.
 *       else for functional model with 1 or more Input layers:
 *         `batchShape=[...]` to all the first layers in your model.
 *       This is the expected shape of your inputs *including the batch size*.
 *       It should be a tuple of integers, e.g. `(32, 10, 100)`.
 *     - specify `shuffle=False` when calling fit().
 *
 *   To reset the states of your model, call `.resetStates()` on either
 *   a specific layer, or on your entire model.
 *
 * Note on specifying the initial state of RNNs
 *   You can specify the initial state of RNN layers symbolically by
 *   calling them with the option `initialState`. The value of
 *   `initialState` should be a tensor or list of tensors representing
 *   the initial state of the RNN layer.
 *
 *   You can specify the initial state of RNN layers numerically by
 *   calling `resetStates` with the keyword argument `states`. The value of
 *   `states` should be a numpy array or list of numpy arrays representing
 *   the initial state of the RNN layer.
 *
 * Note on passing external constants to RNNs
 *   You can pass "external" constants to the cell using the `constants`
 *   keyword argument of `RNN.call` method. This requires that the `cell.call`
 *   method accepts the same keyword argument `constants`. Such constants
 *   can be used to conditon the cell transformation on additional static inputs
 *   (not changing over time), a.k.a an attention mechanism.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function rnn(args) {
    return new RNN(args);
}
/**
 * Wrapper allowing a stack of RNN cells to behave as a single cell.
 *
 * Used to implement efficient stacked RNNs.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function stackedRNNCells(args) {
    return new StackedRNNCells(args);
}
// Wrapper Layers.
/** @doc {heading: 'Layers', subheading: 'Wrapper', namespace: 'layers'} */
export function bidirectional(args) {
    return new Bidirectional(args);
}
/**
 * This wrapper applies a layer to every temporal slice of an input.
 *
 * The input should be at least 3D,  and the dimension of the index `1` will be
 * considered to be the temporal dimension.
 *
 * Consider a batch of 32 samples, where each sample is a sequence of 10 vectors
 * of 16 dimensions. The batch input shape of the layer is then `[32,  10,
 * 16]`, and the `inputShape`, not including the sample dimension, is
 * `[10, 16]`.
 *
 * You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10
 * timesteps, independently:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.dense({units: 8}),
 *   inputShape: [10, 16],
 * }));
 *
 * // Now model.outputShape = [null, 10, 8].
 * // The output will then have shape `[32, 10, 8]`.
 *
 * // In subsequent layers, there is no need for `inputShape`:
 * model.add(tf.layers.timeDistributed({layer: tf.layers.dense({units: 32})}));
 * console.log(JSON.stringify(model.outputs[0].shape));
 * // Now model.outputShape = [null, 10, 32].
 * ```
 *
 * The output will then have shape `[32, 10, 32]`.
 *
 * `TimeDistributed` can be used with arbitrary layers, not just `Dense`, for
 * instance a `Conv2D` layer.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.conv2d({filters: 64, kernelSize: [3, 3]}),
 *   inputShape: [10, 299, 299, 3],
 * }));
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Wrapper', namespace: 'layers'}
 */
export function timeDistributed(args) {
    return new TimeDistributed(args);
}
// Aliases for pooling.
export const globalMaxPool1d = globalMaxPooling1d;
export const globalMaxPool2d = globalMaxPooling2d;
export const maxPool1d = maxPooling1d;
export const maxPool2d = maxPooling2d;
export { Layer, RNN, RNNCell, input /* alias for tf.input */ };
/**
 * Apply additive zero-centered Gaussian noise.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * This is useful to mitigate overfitting
 * (you could see it as a form of random data augmentation).
 * Gaussian Noise (GS) is a natural choice as corruption process
 * for real valued inputs.
 *
 * # Arguments
 *     stddev: float, standard deviation of the noise distribution.
 *
 * # Input shape
 *         Arbitrary. Use the keyword argument `input_shape`
 *         (tuple of integers, does not include the samples axis)
 *         when using this layer as the first layer in a model.
 *
 * # Output shape
 *         Same shape as input.
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
export function gaussianNoise(args) {
    return new GaussianNoise(args);
}
/**
 * Apply multiplicative 1-centered Gaussian noise.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
 *      http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
export function gaussianDropout(args) {
    return new GaussianDropout(args);
}
/**
 * Applies Alpha Dropout to the input.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
 * to their original values, in order to ensure the self-normalizing property
 * even after this dropout.
 * Alpha Dropout fits well to Scaled Exponential Linear Units
 * by randomly setting activations to the negative saturation value.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *   - `noise_shape`: A 1-D `Tensor` of type `int32`, representing the
 *     shape for randomly generated keep/drop flags.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
export function alphaDropout(args) {
    return new AlphaDropout(args);
}
/**
 * Masks a sequence by using a mask value to skip timesteps.
 *
 * If all features for a given sample timestep are equal to `mask_value`,
 * then the sample timestep will be masked (skipped) in all downstream layers
 * (as long as they support masking).
 *
 * If any downstream layer does not support masking yet receives such
 * an input mask, an exception will be raised.
 *
 * Arguments:
 *   - `maskValue`: Either None or mask value to skip.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * @doc {heading: 'Layers', subheading: 'Mask', namespace: 'layers'}
 */
export function masking(args) {
    return new Masking(args);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXhwb3J0c19sYXllcnMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvZXhwb3J0c19sYXllcnMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSCxPQUFPLEVBQUMsVUFBVSxFQUFpQixNQUFNLHNCQUFzQixDQUFDO0FBQ2hFLE9BQU8sRUFBQyxLQUFLLEVBQVksTUFBTSxtQkFBbUIsQ0FBQztBQUNuRCxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxHQUFHLEVBQWdCLFNBQVMsRUFBc0IsS0FBSyxFQUFrQixJQUFJLEVBQWlCLE9BQU8sRUFBb0IsZUFBZSxFQUEyQixNQUFNLCtCQUErQixDQUFDO0FBQ2pOLE9BQU8sRUFBQyxNQUFNLEVBQUUsTUFBTSxFQUFFLGVBQWUsRUFBRSxNQUFNLEVBQWlCLFVBQVUsRUFBdUIsZUFBZSxFQUEwQixZQUFZLEVBQXlCLGVBQWUsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQzlOLE9BQU8sRUFBQyxlQUFlLEVBQTJCLE1BQU0sa0NBQWtDLENBQUM7QUFDM0YsT0FBTyxFQUFDLFVBQVUsRUFBa0IsY0FBYyxFQUFxQixNQUFNLGtDQUFrQyxDQUFDO0FBQ2hILE9BQU8sRUFBQyxVQUFVLEVBQXVCLEtBQUssRUFBa0IsT0FBTyxFQUFvQixPQUFPLEVBQW9CLE9BQU8sRUFBZSxPQUFPLEVBQW9CLFlBQVksRUFBeUIsT0FBTyxFQUFvQixnQkFBZ0IsRUFBOEIsTUFBTSxlQUFlLENBQUM7QUFDM1MsT0FBTyxFQUFDLFNBQVMsRUFBcUIsTUFBTSxxQkFBcUIsQ0FBQztBQUNsRSxPQUFPLEVBQUMsR0FBRyxFQUFFLE9BQU8sRUFBRSxXQUFXLEVBQXdCLEdBQUcsRUFBZ0IsT0FBTyxFQUFFLE9BQU8sRUFBRSxRQUFRLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUM5SCxPQUFPLEVBQUMsWUFB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Q25ELE9BQU8sSUFBSSxhQUFhLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDakMsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXVCRztBQUNILE1BQU0sVUFBVSxlQUFlLENBQUMsSUFBeUI7SUFDdkQsT0FBTyxJQUFJLGVBQWUsQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUNuQyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQThCRztBQUNILE1BQU0sVUFBVSxZQUFZLENBQUMsSUFBc0I7SUFDakQsT0FBTyxJQUFJLFlBQVksQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUNoQyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FzQkc7QUFDSCxNQUFNLFVBQVUsT0FBTyxDQUFDLElBQWtCO0lBQ3hDLE9BQU8sSUFBSSxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDM0IsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7SW5wdXRMYXllciwgSW5wdXRMYXllckFyZ3N9IGZyb20gJy4vZW5naW5lL2lucHV0X2xheWVyJztcbmltcG9ydCB7TGF5ZXIsIExheWVyQXJnc30gZnJvbSAnLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtpbnB1dH0gZnJvbSAnLi9leHBvcnRzJztcbmltcG9ydCB7RUxVLCBFTFVMYXllckFyZ3MsIExlYWt5UmVMVSwgTGVha3lSZUxVTGF5ZXJBcmdzLCBQUmVMVSwgUFJlTFVMYXllckFyZ3MsIFJlTFUsIFJlTFVMYXllckFyZ3MsIFNvZnRtYXgsIFNvZnRtYXhMYXllckFyZ3MsIFRocmVzaG9sZGVkUmVMVSwgVGhyZXNob2xkZWRSZUxVTGF5ZXJBcmdzfSBmcm9tICcuL2xheWVycy9hZHZhbmNlZF9hY3RpdmF0aW9ucyc7XG5pbXBvcnQge0NvbnYxRCwgQ29udjJELCBDb252MkRUcmFuc3Bvc2UsIENvbnYzRCwgQ29udkxheWVyQXJncywgQ3JvcHBpbmcyRCwgQ3JvcHBpbmcyRExheWVyQXJncywgU2VwYXJhYmxlQ29udjJELCBTZXBhcmFibGVDb252TGF5ZXJBcmdzLCBVcFNhbXBsaW5nMkQsIFVwU2FtcGxpbmcyRExheWVyQXJncywgQ29udjNEVHJhbnNwb3NlfSBmcm9tICcuL2xheWVycy9jb252b2x1dGlvbmFsJztcbmltcG9ydCB7RGVwdGh3aXNlQ29udjJELCBEZXB0aHdpc2VDb252MkRMYXllckFyZ3N9IGZyb20gJy4vbGF5ZXJzL2NvbnZvbHV0aW9uYWxfZGVwdGh3aXNlJztcbmltcG9ydCB7Q29udkxTVE0yRCwgQ29udkxTVE0yREFyZ3MsIENvbnZMU1RNMkRDZWxsLCBDb252TFNUTTJEQ2VsbEFyZ3N9IGZyb20gJy4vbGF5ZXJzL2NvbnZvbHV0aW9uYWxfcmVjdXJyZW50JztcbmltcG9ydCB7QWN0aXZhdGlvbiwgQWN0aXZhdGlvbkxheWVyQXJncywgRGVuc2UsIERlbnNlTGF5ZXJBcmdzLCBEcm9wb3V0LCBEcm9wb3V0TGF5ZXJBcmdzLCBGbGF0dGVuLCBGbGF0dGVuTGF5ZXJBcmdzLCBNYXNraW5nLCBNYXNraW5nQXJncywgUGVybXV0ZSwgUGVybXV0ZUxheWVyQXJncywgUmVwZWF0VmVjdG9yLCBSZXBlYXRWZWN0b3JMYXllckFyZ3MsIFJlc2hhcGUsIFJlc2hhcGVMYXllckFyZ3MsIFNwYXRpYWxEcm9wb3V0MUQsIFNwYXRpYWxEcm9wb3V0MURMYXllckNvbmZpZ30gZnJvbSAnLi9sYXllcnMvY29yZSc7XG5pbXBvcnQge0VtYmVkZGluZywgRW1iZWRkaW5nTGF5ZXJBcmdzfSBmcm9tICcuL2xheWVycy9lbWJlZGRpbmdzJztcbmltcG9ydCB7QWRkLCBBdmVyYWdlLCBDb25jYXRlbmF0ZSwgQ29uY2F0ZW5hdGVMYXllckFyZ3MsIERvdCwgRG90TGF5ZXJBcmdzLCBNYXhpbXVtLCBNaW5pbXVtLCBNdWx0aXBseX0gZnJvbSAnLi9sYXllcnMvbWVyZ2UnO1xuaW1wb3J0IHtBbHBoYURyb3BvdXQsIEFscGhhRHJvcG91dEFyZ3MsIEdhdXNzaWFuRHJvcG91dCwgR2F1c3NpYW5Ecm9wb3V0QXJncywgR2F1c3NpYW5Ob2lzZSwgR2F1c3NpYW5Ob2lzZUFyZ3N9IGZyb20gJy4vbGF5ZXJzL25vaXNlJztcbmltcG9ydCB7QmF0Y2hOb3JtYWxpemF0aW9uLCBCYXRjaE5vcm1hbGl6YXRpb25MYXllckFyZ3MsIExheWVyTm9ybWFsaXphdGlvbiwgTGF5ZXJOb3JtYWxpemF0aW9uTGF5ZXJBcmdzfSBmcm9tICcuL2xheWVycy9ub3JtYWxpemF0aW9uJztcbmltcG9ydCB7WmVyb1BhZGRpbmcyRCwgWmVyb1BhZGRpbmcyRExheWVyQXJnc30gZnJvbSAnLi9sYXllcnMvcGFkZGluZyc7XG5pbXBvcnQge0F2ZXJhZ2VQb29saW5nMUQsIEF2ZXJhZ2VQb29saW5nMkQsIEF2ZXJhZ2VQb29saW5nM0QsIEdsb2JhbEF2ZXJhZ2VQb29saW5nMUQsIEdsb2JhbEF2ZXJhZ2VQb29saW5nMkQsIEdsb2JhbE1heFBvb2xpbmcxRCwgR2xvYmFsTWF4UG9vbGluZzJELCBHbG9iYWxQb29saW5nMkRMYXllckFyZ3MsIE1heFBvb2xpbmcxRCwgTWF4UG9vbGluZzJELCBNYXhQb29saW5nM0QsIFBvb2xpbmcxRExheWVyQXJncywgUG9vbGluZzJETGF5ZXJBcmdzLCBQb29saW5nM0RMYXllckFyZ3N9IGZyb20gJy4vbGF5ZXJzL3Bvb2xpbmcnO1xuaW1wb3J0IHtHUlUsIEdSVUNlbGwsIEdSVUNlbGxMYXllckFyZ3MsIEdSVUxheWVyQXJncywgTFNUTSwgTFNUTUNlbGwsIExTVE1DZWxsTGF5ZXJBcmdzLCBMU1RNTGF5ZXJBcmdzLCBSTk4sIFJOTkNlbGwsIFJOTkxheWVyQXJncywgU2ltcGxlUk5OLCBTaW1wbGVSTk5DZWxsLCBTaW1wbGVSTk5DZWxsTGF5ZXJBcmdzLCBTaW1wbGVSTk5MYXllckFyZ3MsIFN0YWNrZWRSTk5DZWxscywgU3RhY2tlZFJOTkNlbGxzQXJnc30gZnJvbSAnLi9sYXllcnMvcmVjdXJyZW50JztcbmltcG9ydCB7QmlkaXJlY3Rpb25hbCwgQmlkaXJlY3Rpb25hbExheWVyQXJncywgVGltZURpc3RyaWJ1dGVkLCBXcmFwcGVyTGF5ZXJBcmdzfSBmcm9tICcuL2xheWVycy93cmFwcGVycyc7XG5cbi8vIFRPRE8oY2Fpcyk6IEFkZCBkb2Mgc3RyaW5nIHRvIGFsbCB0aGUgcHVibGljIHN0YXRpYyBmdW5jdGlvbnMgaW4gdGhpc1xuLy8gICBjbGFzczsgaW5jbHVkZSBleGVjdHVhYmxlIEphdmFTY3JpcHQgY29kZSBzbmlwcGV0cyB3aGVyZSBhcHBsaWNhYmxlXG4vLyAgIChiLzc0MDc0NDU4KS5cblxuLy8gSW5wdXQgTGF5ZXIuXG4vKipcbiAqIEFuIGlucHV0IGxheWVyIGlzIGFuIGVudHJ5IHBvaW50IGludG8gYSBgdGYuTGF5ZXJzTW9kZWxgLlxuICpcbiAqIGBJbnB1dExheWVyYCBpcyBnZW5lcmF0ZWQgYXV0b21hdGljYWxseSBmb3IgYHRmLlNlcXVlbnRpYWxgYCBtb2RlbHMgYnlcbiAqIHNwZWNpZnlpbmcgdGhlIGBpbnB1dHNoYXBlYCBvciBgYmF0Y2hJbnB1dFNoYXBlYCBmb3IgdGhlIGZpcnN0IGxheWVyLiAgSXRcbiAqIHNob3VsZCBub3QgYmUgc3BlY2lmaWVkIGV4cGxpY2l0bHkuIEhvd2V2ZXIsIGl0IGNhbiBiZSB1c2VmdWwgc29tZXRpbWVzLFxuICogZS5nLiwgd2hlbiBjb25zdHJ1Y3RpbmcgYSBzZXF1ZW50aWFsIG1vZGVsIGZyb20gYSBzdWJzZXQgb2YgYW5vdGhlclxuICogc2VxdWVudGlhbCBtb2RlbCdzIGxheWVycy4gTGlrZSB0aGUgY29kZSBzbmlwcGV0IGJlbG93IHNob3dzLlxuICpcbiAqIGBgYGpzXG4gKiAvLyBEZWZpbmUgYSBtb2RlbCB3aGljaCBzaW1wbHkgYWRkcyB0d28gaW5wdXRzLlxuICogY29uc3QgbW9kZWwxID0gdGYuc2VxdWVudGlhbCgpO1xuICogbW9kZWwxLmFkZCh0Zi5sYXllcnMuZGVuc2Uoe2lucHV0U2hhcGU6IFs0XSwgdW5pdHM6IDMsIGFjdGl2YXRpb246ICdyZWx1J30pKTtcbiAqIG1vZGVsMS5hZGQodGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgYWN0aXZhdGlvbjogJ3NpZ21vaWQnfSkpO1xuICogbW9kZWwxLnN1bW1hcnkoKTtcbiAqIG1vZGVsMS5wcmVkaWN0KHRmLnplcm9zKFsxLCA0XSkpLnByaW50KCk7XG4gKlxuICogLy8gQ29uc3RydWN0IGFub3RoZXIgbW9kZWwsIHJldXNpbmcgdGhlIHNlY29uZCBsYXllciBvZiBgbW9kZWwxYCB3aGlsZVxuICogLy8gbm90IHVzaW5nIHRoZSBmaXJzdCBsYXllciBvZiBgbW9kZWwxYC4gTm90ZSB0aGF0IHlvdSBjYW5ub3QgYWRkIHRoZSBzZWNvbmRcbiAqIC8vIGxheWVyIG9mIGBtb2RlbGAgZGlyZWN0bHkgYXMgdGhlIGZpcnN0IGxheWVyIG9mIHRoZSBuZXcgc2VxdWVudGlhbCBtb2RlbCxcbiAqIC8vIGJlY2F1c2UgZG9pbmcgc28gd2lsbCBsZWFkIHRvIGFuIGVycm9yIHJlbGF0ZWQgdG8gdGhlIGZhY3QgdGhhdCB0aGUgbGF5ZXJcbiAqIC8vIGlzIG5vdCBhbiBpbnB1dCBsYXllci4gSW5zdGVhZCwgeW91IG5lZWQgdG8gY3JlYXRlIGFuIGBpbnB1dExheWVyYCBhbmQgYWRkXG4gKiAvLyBpdCB0byB0aGUgbmV3IHNlcXVlbnRpYWwgbW9kZWwgYmVmb3JlIGFkZGluZyB0aGUgcmV1c2VkIGxheWVyLlxuICogY29uc3QgbW9kZWwyID0gdGYuc2VxdWVudGlhbCgpO1xuICogLy8gVXNlIGFuIGlucHV0U2hhcGUgdGhhdCBtYXRjaGVzIHRoZSBpbnB1dCBzaGFwZSBvZiBgbW9kZWwxYCdzIHNlY29uZFxuICogLy8gbGF5ZXIuXG4gKiBtb2RlbDIuYWRkKHRmLmxheWVycy5pbnB1dExheWVyKHtpbnB1dFNoYXBlOiBbM119KSk7XG4gKiBtb2RlbDIuYWRkKG1vZGVsMS5sYXllcnNbMV0pO1xuICogbW9kZWwyLnN1bW1hcnkoKTtcbiAqIG1vZGVsMi5wcmVkaWN0KHRmLnplcm9zKFsxLCAzXSkpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ0lucHV0cycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpbnB1dExheWVyKGFyZ3M6IElucHV0TGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgSW5wdXRMYXllcihhcmdzKTtcbn1cblxuLy8gQWR2YW5jZWQgQWN0aXZhdGlvbiBMYXllcnMuXG5cbi8qKlxuICogRXhwb25ldGlhbCBMaW5lYXIgVW5pdCAoRUxVKS5cbiAqXG4gKiBJdCBmb2xsb3dzOlxuICogYGYoeCkgPSAgYWxwaGEgKiAoZXhwKHgpIC0gMS4pIGZvciB4IDwgMGAsXG4gKiBgZih4KSA9IHggZm9yIHggPj0gMGAuXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIEFyYml0cmFyeS4gVXNlIHRoZSBjb25maWd1cmF0aW9uIGBpbnB1dFNoYXBlYCB3aGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlXG4gKiAgIGZpcnN0IGxheWVyIGluIGEgbW9kZWwuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICBTYW1lIHNoYXBlIGFzIHRoZSBpbnB1dC5cbiAqXG4gKiBSZWZlcmVuY2VzOlxuICogICAtIFtGYXN0IGFuZCBBY2N1cmF0ZSBEZWVwIE5ldHdvcmsgTGVhcm5pbmcgYnkgRXhwb25lbnRpYWwgTGluZWFyIFVuaXRzXG4gKiAoRUxVcyldKGh0dHBzOi8vYXJ4aXYub3JnL2Ficy8xNTExLjA3Mjg5djEpXG4gKlxuICogQGRvYyB7XG4gKiAgIGhlYWRpbmc6ICdMYXllcnMnLFxuICogICBzdWJoZWFkaW5nOiAnQWR2YW5jZWQgQWN0aXZhdGlvbicsXG4gKiAgIG5hbWVzcGFjZTogJ2xheWVycydcbiAqIH1cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGVsdShhcmdzPzogRUxVTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgRUxVKGFyZ3MpO1xufVxuXG4vKipcbiAqIFJlY3RpZmllZCBMaW5lYXIgVW5pdCBhY3RpdmF0aW9uIGZ1bmN0aW9uLlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICBBcmJpdHJhcnkuIFVzZSB0aGUgY29uZmlnIGZpZWxkIGBpbnB1dFNoYXBlYCAoQXJyYXkgb2YgaW50ZWdlcnMsIGRvZXNcbiAqICAgbm90IGluY2x1ZGUgdGhlIHNhbXBsZSBheGlzKSB3aGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlIGZpcnN0IGxheWVyXG4gKiAgIGluIGEgbW9kZWwuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICBTYW1lIHNoYXBlIGFzIHRoZSBpbnB1dC5cbiAqXG4gKiBAZG9jIHtcbiAqICAgaGVhZGluZzogJ0xheWVycycsXG4gKiAgIHN1YmhlYWRpbmc6ICdBZHZhbmNlZCBBY3RpdmF0aW9uJyxcbiAqICAgbmFtZXNwYWNlOiAnbGF5ZXJzJ1xuICogfVxuICovXG5leHBvcnQgZnVuY3Rpb24gcmVMVShhcmdzPzogUmVMVUxheWVyQXJncykge1xuICByZXR1cm4gbmV3IFJlTFUoYXJncyk7XG59XG5cbi8qKlxuICogTGVha3kgdmVyc2lvbiBvZiBhIHJlY3RpZmllZCBsaW5lYXIgdW5pdC5cbiAqXG4gKiBJdCBhbGxvd3MgYSBzbWFsbCBncmFkaWVudCB3aGVuIHRoZSB1bml0IGlzIG5vdCBhY3RpdmU6XG4gKiBgZih4KSA9IGFscGhhICogeCBmb3IgeCA8IDAuYFxuICogYGYoeCkgPSB4IGZvciB4ID49IDAuYFxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICBBcmJpdHJhcnkuIFVzZSB0aGUgY29uZmlndXJhdGlvbiBgaW5wdXRTaGFwZWAgd2hlbiB1c2luZyB0aGlzIGxheWVyIGFzIHRoZVxuICogICBmaXJzdCBsYXllciBpbiBhIG1vZGVsLlxuICpcbiAqIE91dHB1dCBzaGFwZTpcbiAqICAgU2FtZSBzaGFwZSBhcyB0aGUgaW5wdXQuXG4gKlxuICogQGRvYyB7XG4gKiAgIGhlYWRpbmc6ICdMYXllcnMnLFxuICogICBzdWJoZWFkaW5nOiAnQWR2YW5jZWQgQWN0aXZhdGlvbicsXG4gKiAgIG5hbWVzcGFjZTogJ2xheWVycydcbiAqIH1cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGxlYWt5UmVMVShhcmdzPzogTGVha3lSZUxVTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgTGVha3lSZUxVKGFyZ3MpO1xufVxuXG4vKipcbiAqIFBhcmFtZXRlcml6ZWQgdmVyc2lvbiBvZiBhIGxlYWt5IHJlY3RpZmllZCBsaW5lYXIgdW5pdC5cbiAqXG4gKiBJdCBmb2xsb3dzXG4gKiBgZih4KSA9IGFscGhhICogeCBmb3IgeCA8IDAuYFxuICogYGYoeCkgPSB4IGZvciB4ID49IDAuYFxuICogd2hlcmVpbiBgYWxwaGFgIGlzIGEgdHJhaW5hYmxlIHdlaWdodC5cbiAqXG4gKiBJbnB1dCBzaGFwZTpcbiAqICAgQXJiaXRyYXJ5LiBVc2UgdGhlIGNvbmZpZ3VyYXRpb24gYGlucHV0U2hhcGVgIHdoZW4gdXNpbmcgdGhpcyBsYXllciBhcyB0aGVcbiAqICAgZmlyc3QgbGF5ZXIgaW4gYSBtb2RlbC5cbiAqXG4gKiBPdXRwdXQgc2hhcGU6XG4gKiAgIFNhbWUgc2hhcGUgYXMgdGhlIGlucHV0LlxuICpcbiAqIEBkb2Mge1xuICogICBoZWFkaW5nOiAnTGF5ZXJzJyxcbiAqICAgc3ViaGVhZGluZzogJ0FkdmFuY2VkIEFjdGl2YXRpb24nLFxuICogICBuYW1lc3BhY2U6ICdsYXllcnMnXG4gKiB9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVsdShhcmdzPzogUFJlTFVMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBQUmVMVShhcmdzKTtcbn1cblxuLyoqXG4gKiBTb2Z0bWF4IGFjdGl2YXRpb24gbGF5ZXIuXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIEFyYml0cmFyeS4gVXNlIHRoZSBjb25maWd1cmF0aW9uIGBpbnB1dFNoYXBlYCB3aGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlXG4gKiAgIGZpcnN0IGxheWVyIGluIGEgbW9kZWwuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICBTYW1lIHNoYXBlIGFzIHRoZSBpbnB1dC5cbiAqXG4gKiBAZG9jIHtcbiAqICAgaGVhZGluZzogJ0xheWVycycsXG4gKiAgIHN1YmhlYWRpbmc6ICdBZHZhbmNlZCBBY3RpdmF0aW9uJyxcbiAqICAgbmFtZXNwYWNlOiAnbGF5ZXJzJ1xuICogfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc29mdG1heChhcmdzPzogU29mdG1heExheWVyQXJncykge1xuICByZXR1cm4gbmV3IFNvZnRtYXgoYXJncyk7XG59XG5cbi8qKlxuICogVGhyZXNob2xkZWQgUmVjdGlmaWVkIExpbmVhciBVbml0LlxuICpcbiAqIEl0IGZvbGxvd3M6XG4gKiBgZih4KSA9IHggZm9yIHggPiB0aGV0YWAsXG4gKiBgZih4KSA9IDAgb3RoZXJ3aXNlYC5cbiAqXG4gKiBJbnB1dCBzaGFwZTpcbiAqICAgQXJiaXRyYXJ5LiBVc2UgdGhlIGNvbmZpZ3VyYXRpb24gYGlucHV0U2hhcGVgIHdoZW4gdXNpbmcgdGhpcyBsYXllciBhcyB0aGVcbiAqICAgZmlyc3QgbGF5ZXIgaW4gYSBtb2RlbC5cbiAqXG4gKiBPdXRwdXQgc2hhcGU6XG4gKiAgIFNhbWUgc2hhcGUgYXMgdGhlIGlucHV0LlxuICpcbiAqIFJlZmVyZW5jZXM6XG4gKiAgIC0gW1plcm8tQmlhcyBBdXRvZW5jb2RlcnMgYW5kIHRoZSBCZW5lZml0cyBvZiBDby1BZGFwdGluZ1xuICogRmVhdHVyZXNdKGh0dHA6Ly9hcnhpdi5vcmcvYWJzLzE0MDIuMzMzNylcbiAqXG4gKiBAZG9jIHtcbiAqICAgaGVhZGluZzogJ0xheWVycycsXG4gKiAgIHN1YmhlYWRpbmc6ICdBZHZhbmNlZCBBY3RpdmF0aW9uJyxcbiAqICAgbmFtZXNwYWNlOiAnbGF5ZXJzJ1xuICogfVxuICovXG5leHBvcnQgZnVuY3Rpb24gdGhyZXNob2xkZWRSZUxVKGFyZ3M/OiBUaHJlc2hvbGRlZFJlTFVMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBUaHJlc2hvbGRlZFJlTFUoYXJncyk7XG59XG5cbi8vIENvbnZvbHV0aW9uYWwgTGF5ZXJzLlxuXG4vKipcbiAqIDFEIGNvbnZvbHV0aW9uIGxheWVyIChlLmcuLCB0ZW1wb3JhbCBjb252b2x1dGlvbikuXG4gKlxuICogVGhpcyBsYXllciBjcmVhdGVzIGEgY29udm9sdXRpb24ga2VybmVsIHRoYXQgaXMgY29udm9sdmVkXG4gKiB3aXRoIHRoZSBsYXllciBpbnB1dCBvdmVyIGEgc2luZ2xlIHNwYXRpYWwgKG9yIHRlbXBvcmFsKSBkaW1lbnNpb25cbiAqIHRvIHByb2R1Y2UgYSB0ZW5zb3Igb2Ygb3V0cHV0cy5cbiAqXG4gKiBJZiBgdXNlX2JpYXNgIGlzIFRydWUsIGEgYmlhcyB2ZWN0b3IgaXMgY3JlYXRlZCBhbmQgYWRkZWQgdG8gdGhlIG91dHB1dHMuXG4gKlxuICogSWYgYGFjdGl2YXRpb25gIGlzIG5vdCBgbnVsbGAsIGl0IGlzIGFwcGxpZWQgdG8gdGhlIG91dHB1dHMgYXMgd2VsbC5cbiAqXG4gKiBXaGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlIGZpcnN0IGxheWVyIGluIGEgbW9kZWwsIHByb3ZpZGUgYW5cbiAqIGBpbnB1dFNoYXBlYCBhcmd1bWVudCBgQXJyYXlgIG9yIGBudWxsYC5cbiAqXG4gKiBGb3IgZXhhbXBsZSwgYGlucHV0U2hhcGVgIHdvdWxkIGJlOlxuICogLSBgWzEwLCAxMjhdYCBmb3Igc2VxdWVuY2VzIG9mIDEwIHZlY3RvcnMgb2YgMTI4LWRpbWVuc2lvbmFsIHZlY3RvcnNcbiAqIC0gYFtudWxsLCAxMjhdYCBmb3IgdmFyaWFibGUtbGVuZ3RoIHNlcXVlbmNlcyBvZiAxMjgtZGltZW5zaW9uYWwgdmVjdG9ycy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ0NvbnZvbHV0aW9uYWwnLCAgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbnYxZChhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgQ29udjFEKGFyZ3MpO1xufVxuXG4vKipcbiAqIDJEIGNvbnZvbHV0aW9uIGxheWVyIChlLmcuIHNwYXRpYWwgY29udm9sdXRpb24gb3ZlciBpbWFnZXMpLlxuICpcbiAqIFRoaXMgbGF5ZXIgY3JlYXRlcyBhIGNvbnZvbHV0aW9uIGtlcm5lbCB0aGF0IGlzIGNvbnZvbHZlZFxuICogd2l0aCB0aGUgbGF5ZXIgaW5wdXQgdG8gcHJvZHVjZSBhIHRlbnNvciBvZiBvdXRwdXRzLlxuICpcbiAqIElmIGB1c2VCaWFzYCBpcyBUcnVlLCBhIGJpYXMgdmVjdG9yIGlzIGNyZWF0ZWQgYW5kIGFkZGVkIHRvIHRoZSBvdXRwdXRzLlxuICpcbiAqIElmIGBhY3RpdmF0aW9uYCBpcyBub3QgYG51bGxgLCBpdCBpcyBhcHBsaWVkIHRvIHRoZSBvdXRwdXRzIGFzIHdlbGwuXG4gKlxuICogV2hlbiB1c2luZyB0aGlzIGxheWVyIGFzIHRoZSBmaXJzdCBsYXllciBpbiBhIG1vZGVsLFxuICogcHJvdmlkZSB0aGUga2V5d29yZCBhcmd1bWVudCBgaW5wdXRTaGFwZWBcbiAqIChBcnJheSBvZiBpbnRlZ2VycywgZG9lcyBub3QgaW5jbHVkZSB0aGUgc2FtcGxlIGF4aXMpLFxuICogZS5nLiBgaW5wdXRTaGFwZT1bMTI4LCAxMjgsIDNdYCBmb3IgMTI4eDEyOCBSR0IgcGljdHVyZXNcbiAqIGluIGBkYXRhRm9ybWF0PSdjaGFubmVsc0xhc3QnYC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ0NvbnZvbHV0aW9uYWwnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gY29udjJkKGFyZ3M6IENvbnZMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBDb252MkQoYXJncyk7XG59XG5cbi8qKlxuICogVHJhbnNwb3NlZCBjb252b2x1dGlvbmFsIGxheWVyIChzb21ldGltZXMgY2FsbGVkIERlY29udm9sdXRpb24pLlxuICpcbiAqIFRoZSBuZWVkIGZvciB0cmFuc3Bvc2VkIGNvbnZvbHV0aW9ucyBnZW5lcmFsbHkgYXJpc2VzXG4gKiBmcm9tIHRoZSBkZXNpcmUgdG8gdXNlIGEgdHJhbnNmb3JtYXRpb24gZ29pbmcgaW4gdGhlIG9wcG9zaXRlIGRpcmVjdGlvbiBvZlxuICogYSBub3JtYWwgY29udm9sdXRpb24sIGkuZS4sIGZyb20gc29tZXRoaW5nIHRoYXQgaGFzIHRoZSBzaGFwZSBvZiB0aGUgb3V0cHV0XG4gKiBvZiBzb21lIGNvbnZvbHV0aW9uIHRvIHNvbWV0aGluZyB0aGF0IGhhcyB0aGUgc2hhcGUgb2YgaXRzIGlucHV0IHdoaWxlXG4gKiBtYWludGFpbmluZyBhIGNvbm5lY3Rpdml0eSBwYXR0ZXJuIHRoYXQgaXMgY29tcGF0aWJsZSB3aXRoIHNhaWRcbiAqIGNvbnZvbHV0aW9uLlxuICpcbiAqIFdoZW4gdXNpbmcgdGhpcyBsYXllciBhcyB0aGUgZmlyc3QgbGF5ZXIgaW4gYSBtb2RlbCwgcHJvdmlkZSB0aGVcbiAqIGNvbmZpZ3VyYXRpb24gYGlucHV0U2hhcGVgIChgQXJyYXlgIG9mIGludGVnZXJzLCBkb2VzIG5vdCBpbmNsdWRlIHRoZVxuICogc2FtcGxlIGF4aXMpLCBlLmcuLCBgaW5wdXRTaGFwZTogWzEyOCwgMTI4LCAzXWAgZm9yIDEyOHgxMjggUkdCIHBpY3R1cmVzIGluXG4gKiBgZGF0YUZvcm1hdDogJ2NoYW5uZWxzTGFzdCdgLlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTpcbiAqICAgYFtiYXRjaCwgY2hhbm5lbHMsIHJvd3MsIGNvbHNdYCBpZiBgZGF0YUZvcm1hdGAgaXMgYCdjaGFubmVsc0ZpcnN0J2AuXG4gKiAgIG9yIDREIHRlbnNvciB3aXRoIHNoYXBlXG4gKiAgIGBbYmF0Y2gsIHJvd3MsIGNvbHMsIGNoYW5uZWxzXWAgaWYgYGRhdGFGb3JtYXRgIGlzIGAnY2hhbm5lbHNMYXN0YC5cbiAqXG4gKiBPdXRwdXQgc2hhcGU6XG4gKiAgIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICBgW2JhdGNoLCBmaWx0ZXJzLCBuZXdSb3dzLCBuZXdDb2xzXWAgaWYgYGRhdGFGb3JtYXRgIGlzXG4gKiBgJ2NoYW5uZWxzRmlyc3QnYC4gb3IgNEQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgIGBbYmF0Y2gsIG5ld1Jvd3MsIG5ld0NvbHMsIGZpbHRlcnNdYCBpZiBgZGF0YUZvcm1hdGAgaXMgYCdjaGFubmVsc0xhc3QnYC5cbiAqXG4gKiBSZWZlcmVuY2VzOlxuICogICAtIFtBIGd1aWRlIHRvIGNvbnZvbHV0aW9uIGFyaXRobWV0aWMgZm9yIGRlZXBcbiAqIGxlYXJuaW5nXShodHRwczovL2FyeGl2Lm9yZy9hYnMvMTYwMy4wNzI4NXYxKVxuICogICAtIFtEZWNvbnZvbHV0aW9uYWxcbiAqIE5ldHdvcmtzXShodHRwOi8vd3d3Lm1hdHRoZXd6ZWlsZXIuY29tL3B1YnMvY3ZwcjIwMTAvY3ZwcjIwMTAucGRmKVxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb25hbCcsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb252MmRUcmFuc3Bvc2UoYXJnczogQ29udkxheWVyQXJncykge1xuICByZXR1cm4gbmV3IENvbnYyRFRyYW5zcG9zZShhcmdzKTtcbn1cblxuLyoqXG4gKiAzRCBjb252b2x1dGlvbiBsYXllciAoZS5nLiBzcGF0aWFsIGNvbnZvbHV0aW9uIG92ZXIgdm9sdW1lcykuXG4gKlxuICogVGhpcyBsYXllciBjcmVhdGVzIGEgY29udm9sdXRpb24ga2VybmVsIHRoYXQgaXMgY29udm9sdmVkXG4gKiB3aXRoIHRoZSBsYXllciBpbnB1dCB0byBwcm9kdWNlIGEgdGVuc29yIG9mIG91dHB1dHMuXG4gKlxuICogSWYgYHVzZUJpYXNgIGlzIFRydWUsIGEgYmlhcyB2ZWN0b3IgaXMgY3JlYXRlZCBhbmQgYWRkZWQgdG8gdGhlIG91dHB1dHMuXG4gKlxuICogSWYgYGFjdGl2YXRpb25gIGlzIG5vdCBgbnVsbGAsIGl0IGlzIGFwcGxpZWQgdG8gdGhlIG91dHB1dHMgYXMgd2VsbC5cbiAqXG4gKiBXaGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlIGZpcnN0IGxheWVyIGluIGEgbW9kZWwsXG4gKiBwcm92aWRlIHRoZSBrZXl3b3JkIGFyZ3VtZW50IGBpbnB1dFNoYXBlYFxuICogKEFycmF5IG9mIGludGVnZXJzLCBkb2VzIG5vdCBpbmNsdWRlIHRoZSBzYW1wbGUgYXhpcyksXG4gKiBlLmcuIGBpbnB1dFNoYXBlPVsxMjgsIDEyOCwgMTI4LCAxXWAgZm9yIDEyOHgxMjh4MTI4IGdyYXlzY2FsZSB2b2x1bWVzXG4gKiBpbiBgZGF0YUZvcm1hdD0nY2hhbm5lbHNMYXN0J2AuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdDb252b2x1dGlvbmFsJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbnYzZChhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgQ29udjNEKGFyZ3MpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29udjNkVHJhbnNwb3NlKGFyZ3M6IENvbnZMYXllckFyZ3MpOiBMYXllciB7XG4gIHJldHVybiBuZXcgQ29udjNEVHJhbnNwb3NlKGFyZ3MpO1xufVxuXG4vKipcbiAqIERlcHRod2lzZSBzZXBhcmFibGUgMkQgY29udm9sdXRpb24uXG4gKlxuICogU2VwYXJhYmxlIGNvbnZvbHV0aW9uIGNvbnNpc3RzIG9mIGZpcnN0IHBlcmZvcm1pbmdcbiAqIGEgZGVwdGh3aXNlIHNwYXRpYWwgY29udm9sdXRpb25cbiAqICh3aGljaCBhY3RzIG9uIGVhY2ggaW5wdXQgY2hhbm5lbCBzZXBhcmF0ZWx5KVxuICogZm9sbG93ZWQgYnkgYSBwb2ludHdpc2UgY29udm9sdXRpb24gd2hpY2ggbWl4ZXMgdG9nZXRoZXIgdGhlIHJlc3VsdGluZ1xuICogb3V0cHV0IGNoYW5uZWxzLiBUaGUgYGRlcHRoTXVsdGlwbGllcmAgYXJndW1lbnQgY29udHJvbHMgaG93IG1hbnlcbiAqIG91dHB1dCBjaGFubmVscyBhcmUgZ2VuZXJhdGVkIHBlciBpbnB1dCBjaGFubmVsIGluIHRoZSBkZXB0aHdpc2Ugc3RlcC5cbiAqXG4gKiBJbnR1aXRpdmVseSwgc2VwYXJhYmxlIGNvbnZvbHV0aW9ucyBjYW4gYmUgdW5kZXJzdG9vZCBhc1xuICogYSB3YXkgdG8gZmFjdG9yaXplIGEgY29udm9sdXRpb24ga2VybmVsIGludG8gdHdvIHNtYWxsZXIga2VybmVscyxcbiAqIG9yIGFzIGFuIGV4dHJlbWUgdmVyc2lvbiBvZiBhbiBJbmNlcHRpb24gYmxvY2suXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgIGBbYmF0Y2gsIGNoYW5uZWxzLCByb3dzLCBjb2xzXWAgaWYgZGF0YV9mb3JtYXQ9J2NoYW5uZWxzRmlyc3QnXG4gKiAgIG9yIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgIGBbYmF0Y2gsIHJvd3MsIGNvbHMsIGNoYW5uZWxzXWAgaWYgZGF0YV9mb3JtYXQ9J2NoYW5uZWxzTGFzdCcuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTpcbiAqICAgICBgW2JhdGNoLCBmaWx0ZXJzLCBuZXdSb3dzLCBuZXdDb2xzXWAgaWYgZGF0YV9mb3JtYXQ9J2NoYW5uZWxzRmlyc3QnXG4gKiAgIG9yIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgIGBbYmF0Y2gsIG5ld1Jvd3MsIG5ld0NvbHMsIGZpbHRlcnNdYCBpZiBkYXRhX2Zvcm1hdD0nY2hhbm5lbHNMYXN0Jy5cbiAqICAgICBgcm93c2AgYW5kIGBjb2xzYCB2YWx1ZXMgbWlnaHQgaGF2ZSBjaGFuZ2VkIGR1ZSB0byBwYWRkaW5nLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb25hbCcsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzZXBhcmFibGVDb252MmQoYXJnczogU2VwYXJhYmxlQ29udkxheWVyQXJncykge1xuICByZXR1cm4gbmV3IFNlcGFyYWJsZUNvbnYyRChhcmdzKTtcbn1cblxuLyoqXG4gKiBDcm9wcGluZyBsYXllciBmb3IgMkQgaW5wdXQgKGUuZy4sIGltYWdlKS5cbiAqXG4gKiBUaGlzIGxheWVyIGNhbiBjcm9wIGFuIGlucHV0XG4gKiBhdCB0aGUgdG9wLCBib3R0b20sIGxlZnQgYW5kIHJpZ2h0IHNpZGUgb2YgYW4gaW1hZ2UgdGVuc29yLlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTpcbiAqICAgLSBJZiBgZGF0YUZvcm1hdGAgaXMgYFwiY2hhbm5lbHNMYXN0XCJgOlxuICogICAgIGBbYmF0Y2gsIHJvd3MsIGNvbHMsIGNoYW5uZWxzXWBcbiAqICAgLSBJZiBgZGF0YV9mb3JtYXRgIGlzIGBcImNoYW5uZWxzX2ZpcnN0XCJgOlxuICogICAgIGBbYmF0Y2gsIGNoYW5uZWxzLCByb3dzLCBjb2xzXWAuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICA0RCB3aXRoIHNoYXBlOlxuICogICAtIElmIGBkYXRhRm9ybWF0YCBpcyBgXCJjaGFubmVsc0xhc3RcImA6XG4gKiAgICAgYFtiYXRjaCwgY3JvcHBlZFJvd3MsIGNyb3BwZWRDb2xzLCBjaGFubmVsc11gXG4gKiAgICAtIElmIGBkYXRhRm9ybWF0YCBpcyBgXCJjaGFubmVsc0ZpcnN0XCJgOlxuICogICAgIGBbYmF0Y2gsIGNoYW5uZWxzLCBjcm9wcGVkUm93cywgY3JvcHBlZENvbHNdYC5cbiAqXG4gKiBFeGFtcGxlc1xuICogYGBganNcbiAqXG4gKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoKTtcbiAqIG1vZGVsLmFkZCh0Zi5sYXllcnMuY3JvcHBpbmcyRCh7Y3JvcHBpbmc6W1syLCAyXSwgWzIsIDJdXSxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpbnB1dFNoYXBlOiBbMTI4LCAxMjgsIDNdfSkpO1xuICogLy9ub3cgb3V0cHV0IHNoYXBlIGlzIFtiYXRjaCwgMTI0LCAxMjQsIDNdXG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ0NvbnZvbHV0aW9uYWwnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gY3JvcHBpbmcyRChhcmdzOiBDcm9wcGluZzJETGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgQ3JvcHBpbmcyRChhcmdzKTtcbn1cblxuLyoqXG4gKiBVcHNhbXBsaW5nIGxheWVyIGZvciAyRCBpbnB1dHMuXG4gKlxuICogUmVwZWF0cyB0aGUgcm93cyBhbmQgY29sdW1ucyBvZiB0aGUgZGF0YVxuICogYnkgc2l6ZVswXSBhbmQgc2l6ZVsxXSByZXNwZWN0aXZlbHkuXG4gKlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICAgNEQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgICAgLSBJZiBgZGF0YUZvcm1hdGAgaXMgYFwiY2hhbm5lbHNMYXN0XCJgOlxuICogICAgICAgICBgW2JhdGNoLCByb3dzLCBjb2xzLCBjaGFubmVsc11gXG4gKiAgICAgLSBJZiBgZGF0YUZvcm1hdGAgaXMgYFwiY2hhbm5lbHNGaXJzdFwiYDpcbiAqICAgICAgICBgW2JhdGNoLCBjaGFubmVscywgcm93cywgY29sc11gXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICAgIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgIC0gSWYgYGRhdGFGb3JtYXRgIGlzIGBcImNoYW5uZWxzTGFzdFwiYDpcbiAqICAgICAgICBgW2JhdGNoLCB1cHNhbXBsZWRSb3dzLCB1cHNhbXBsZWRDb2xzLCBjaGFubmVsc11gXG4gKiAgICAgLSBJZiBgZGF0YUZvcm1hdGAgaXMgYFwiY2hhbm5lbHNGaXJzdFwiYDpcbiAqICAgICAgICAgYFtiYXRjaCwgY2hhbm5lbHMsIHVwc2FtcGxlZFJvd3MsIHVwc2FtcGxlZENvbHNdYFxuICpcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ0NvbnZvbHV0aW9uYWwnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gdXBTYW1wbGluZzJkKGFyZ3M6IFVwU2FtcGxpbmcyRExheWVyQXJncykge1xuICByZXR1cm4gbmV3IFVwU2FtcGxpbmcyRChhcmdzKTtcbn1cblxuLy8gQ29udm9sdXRpb25hbChkZXB0aHdpc2UpIExheWVycy5cblxuLyoqXG4gKiBEZXB0aHdpc2Ugc2VwYXJhYmxlIDJEIGNvbnZvbHV0aW9uLlxuICpcbiAqIERlcHRod2lzZSBTZXBhcmFibGUgY29udm9sdXRpb25zIGNvbnNpc3RzIGluIHBlcmZvcm1pbmcganVzdCB0aGUgZmlyc3Qgc3RlcFxuICogaW4gYSBkZXB0aHdpc2Ugc3BhdGlhbCBjb252b2x1dGlvbiAod2hpY2ggYWN0cyBvbiBlYWNoIGlucHV0IGNoYW5uZWxcbiAqIHNlcGFyYXRlbHkpLiBUaGUgYGRlcHRoTXVsdHBsaWVyYCBhcmd1bWVudCBjb250cm9scyBob3cgbWFueSBvdXRwdXQgY2hhbm5lbHNcbiAqIGFyZSBnZW5lcmF0ZWQgcGVyIGlucHV0IGNoYW5uZWwgaW4gdGhlIGRlcHRod2lzZSBzdGVwLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb25hbCcsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBkZXB0aHdpc2VDb252MmQoYXJnczogRGVwdGh3aXNlQ29udjJETGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgRGVwdGh3aXNlQ29udjJEKGFyZ3MpO1xufVxuXG4vLyBCYXNpYyBMYXllcnMuXG5cbi8qKlxuICogQXBwbGllcyBhbiBhY3RpdmF0aW9uIGZ1bmN0aW9uIHRvIGFuIG91dHB1dC5cbiAqXG4gKiBUaGlzIGxheWVyIGFwcGxpZXMgZWxlbWVudC13aXNlIGFjdGl2YXRpb24gZnVuY3Rpb24uICBPdGhlciBsYXllcnMsIG5vdGFibHlcbiAqIGBkZW5zZWAgY2FuIGFsc28gYXBwbHkgYWN0aXZhdGlvbiBmdW5jdGlvbnMuICBVc2UgdGhpcyBpc29sYXRlZCBhY3RpdmF0aW9uXG4gKiBmdW5jdGlvbiB0byBleHRyYWN0IHRoZSB2YWx1ZXMgYmVmb3JlIGFuZCBhZnRlciB0aGVcbiAqIGFjdGl2YXRpb24uIEZvciBpbnN0YW5jZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW5wdXQgPSB0Zi5pbnB1dCh7c2hhcGU6IFs1XX0pO1xuICogY29uc3QgZGVuc2VMYXllciA9IHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDF9KTtcbiAqIGNvbnN0IGFjdGl2YXRpb25MYXllciA9IHRmLmxheWVycy5hY3RpdmF0aW9uKHthY3RpdmF0aW9uOiAncmVsdTYnfSk7XG4gKlxuICogLy8gT2J0YWluIHRoZSBvdXRwdXQgc3ltYm9saWMgdGVuc29ycyBieSBhcHBseWluZyB0aGUgbGF5ZXJzIGluIG9yZGVyLlxuICogY29uc3QgZGVuc2VPdXRwdXQgPSBkZW5zZUxheWVyLmFwcGx5KGlucHV0KTtcbiAqIGNvbnN0IGFjdGl2YXRpb25PdXRwdXQgPSBhY3RpdmF0aW9uTGF5ZXIuYXBwbHkoZGVuc2VPdXRwdXQpO1xuICpcbiAqIC8vIENyZWF0ZSB0aGUgbW9kZWwgYmFzZWQgb24gdGhlIGlucHV0cy5cbiAqIGNvbnN0IG1vZGVsID0gdGYubW9kZWwoe1xuICogICAgIGlucHV0czogaW5wdXQsXG4gKiAgICAgb3V0cHV0czogW2RlbnNlT3V0cHV0LCBhY3RpdmF0aW9uT3V0cHV0XVxuICogfSk7XG4gKlxuICogLy8gQ29sbGVjdCBib3RoIG91dHB1dHMgYW5kIHByaW50IHNlcGFyYXRlbHkuXG4gKiBjb25zdCBbZGVuc2VPdXQsIGFjdGl2YXRpb25PdXRdID0gbW9kZWwucHJlZGljdCh0Zi5yYW5kb21Ob3JtYWwoWzYsIDVdKSk7XG4gKiBkZW5zZU91dC5wcmludCgpO1xuICogYWN0aXZhdGlvbk91dC5wcmludCgpO1xuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdCYXNpYycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhY3RpdmF0aW9uKGFyZ3M6IEFjdGl2YXRpb25MYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBBY3RpdmF0aW9uKGFyZ3MpO1xufVxuXG4vKipcbiAqIENyZWF0ZXMgYSBkZW5zZSAoZnVsbHkgY29ubmVjdGVkKSBsYXllci5cbiAqXG4gKiBUaGlzIGxheWVyIGltcGxlbWVudHMgdGhlIG9wZXJhdGlvbjpcbiAqICAgYG91dHB1dCA9IGFjdGl2YXRpb24oZG90KGlucHV0LCBrZXJuZWwpICsgYmlhcylgXG4gKlxuICogYGFjdGl2YXRpb25gIGlzIHRoZSBlbGVtZW50LXdpc2UgYWN0aXZhdGlvbiBmdW5jdGlvblxuICogICBwYXNzZWQgYXMgdGhlIGBhY3RpdmF0aW9uYCBhcmd1bWVudC5cbiAqXG4gKiBga2VybmVsYCBpcyBhIHdlaWdodHMgbWF0cml4IGNyZWF0ZWQgYnkgdGhlIGxheWVyLlxuICpcbiAqIGBiaWFzYCBpcyBhIGJpYXMgdmVjdG9yIGNyZWF0ZWQgYnkgdGhlIGxheWVyIChvbmx5IGFwcGxpY2FibGUgaWYgYHVzZUJpYXNgXG4gKiBpcyBgdHJ1ZWApLlxuICpcbiAqICoqSW5wdXQgc2hhcGU6KipcbiAqXG4gKiAgIG5EIGB0Zi5UZW5zb3JgIHdpdGggc2hhcGU6IGAoYmF0Y2hTaXplLCAuLi4sIGlucHV0RGltKWAuXG4gKlxuICogICBUaGUgbW9zdCBjb21tb24gc2l0dWF0aW9uIHdvdWxkIGJlXG4gKiAgIGEgMkQgaW5wdXQgd2l0aCBzaGFwZSBgKGJhdGNoU2l6ZSwgaW5wdXREaW0pYC5cbiAqXG4gKiAqKk91dHB1dCBzaGFwZToqKlxuICpcbiAqICAgbkQgdGVuc29yIHdpdGggc2hhcGU6IGAoYmF0Y2hTaXplLCAuLi4sIHVuaXRzKWAuXG4gKlxuICogICBGb3IgaW5zdGFuY2UsIGZvciBhIDJEIGlucHV0IHdpdGggc2hhcGUgYChiYXRjaFNpemUsIGlucHV0RGltKWAsXG4gKiAgIHRoZSBvdXRwdXQgd291bGQgaGF2ZSBzaGFwZSBgKGJhdGNoU2l6ZSwgdW5pdHMpYC5cbiAqXG4gKiBOb3RlOiBpZiB0aGUgaW5wdXQgdG8gdGhlIGxheWVyIGhhcyBhIHJhbmsgZ3JlYXRlciB0aGFuIDIsIHRoZW4gaXQgaXNcbiAqIGZsYXR0ZW5lZCBwcmlvciB0byB0aGUgaW5pdGlhbCBkb3QgcHJvZHVjdCB3aXRoIHRoZSBrZXJuZWwuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdCYXNpYycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBkZW5zZShhcmdzOiBEZW5zZUxheWVyQXJncykge1xuICByZXR1cm4gbmV3IERlbnNlKGFyZ3MpO1xufVxuXG4vKipcbiAqIEFwcGxpZXNcbiAqIFtkcm9wb3V0XShodHRwOi8vd3d3LmNzLnRvcm9udG8uZWR1L35yc2FsYWtodS9wYXBlcnMvc3JpdmFzdGF2YTE0YS5wZGYpIHRvXG4gKiB0aGUgaW5wdXQuXG4gKlxuICogRHJvcG91dCBjb25zaXN0cyBpbiByYW5kb21seSBzZXR0aW5nIGEgZnJhY3Rpb24gYHJhdGVgIG9mIGlucHV0IHVuaXRzIHRvIDAgYXRcbiAqIGVhY2ggdXBkYXRlIGR1cmluZyB0cmFpbmluZyB0aW1lLCB3aGljaCBoZWxwcyBwcmV2ZW50IG92ZXJmaXR0aW5nLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQmFzaWMnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZHJvcG91dChhcmdzOiBEcm9wb3V0TGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgRHJvcG91dChhcmdzKTtcbn1cblxuLyoqXG4gKiBTcGF0aWFsIDFEIHZlcnNpb24gb2YgRHJvcG91dC5cbiAqXG4gKiBUaGlzIExheWVyIHR5cGUgcGVyZm9ybXMgdGhlIHNhbWUgZnVuY3Rpb24gYXMgdGhlIERyb3BvdXQgbGF5ZXIsIGJ1dCBpdCBkcm9wc1xuICogZW50aXJlIDFEIGZlYXR1cmUgbWFwcyBpbnN0ZWFkIG9mIGluZGl2aWR1YWwgZWxlbWVudHMuIEZvciBleGFtcGxlLCBpZiBhblxuICogaW5wdXQgZXhhbXBsZSBjb25zaXN0cyBvZiAzIHRpbWVzdGVwcyBhbmQgdGhlIGZlYXR1cmUgbWFwIGZvciBlYWNoIHRpbWVzdGVwXG4gKiBoYXMgYSBzaXplIG9mIDQsIGEgYHNwYXRpYWxEcm9wb3V0MWRgIGxheWVyIG1heSB6ZXJvIG91dCB0aGUgZmVhdHVyZSBtYXBzXG4gKiBvZiB0aGUgMXN0IHRpbWVzdGVwcyBhbmQgMm5kIHRpbWVzdGVwcyBjb21wbGV0ZWx5IHdoaWxlIHNwYXJpbmcgYWxsIGZlYXR1cmVcbiAqIGVsZW1lbnRzIG9mIHRoZSAzcmQgdGltZXN0ZXAuXG4gKlxuICogSWYgYWRqYWNlbnQgZnJhbWVzICh0aW1lc3RlcHMpIGFyZSBzdHJvbmdseSBjb3JyZWxhdGVkIChhcyBpcyBub3JtYWxseSB0aGVcbiAqIGNhc2UgaW4gZWFybHkgY29udm9sdXRpb24gbGF5ZXJzKSwgcmVndWxhciBkcm9wb3V0IHdpbGwgbm90IHJlZ3VsYXJpemUgdGhlXG4gKiBhY3RpdmF0aW9uIGFuZCB3aWxsIG90aGVyd2lzZSBqdXN0IHJlc3VsdCBpbiBtZXJlbHkgYW4gZWZmZWN0aXZlIGxlYXJuaW5nXG4gKiByYXRlIGRlY3JlYXNlLiBJbiB0aGlzIGNhc2UsIGBzcGF0aWFsRHJvcG91dDFkYCB3aWxsIGhlbHAgcHJvbW90ZVxuICogaW5kZXBlbmRlbmNlIGFtb25nIGZlYXR1cmUgbWFwcyBhbmQgc2hvdWxkIGJlIHVzZWQgaW5zdGVhZC5cbiAqXG4gKiAqKkFyZ3VtZW50czoqKlxuICogICByYXRlOiBBIGZsb2F0aW5nLXBvaW50IG51bWJlciA+PTAgYW5kIDw9MS4gRnJhY3Rpb24gb2YgdGhlIGlucHV0IGVsZW1lbnRzXG4gKiAgICAgdG8gZHJvcC5cbiAqXG4gKiAqKklucHV0IHNoYXBlOioqXG4gKiAgIDNEIHRlbnNvciB3aXRoIHNoYXBlIGAoc2FtcGxlcywgdGltZXN0ZXBzLCBjaGFubmVscylgLlxuICpcbiAqICoqT3V0cHV0IHNoYXBlOioqXG4gKiAgIFNhbWUgYXMgdGhlIGlucHV0IHNoYXBlLlxuICpcbiAqIFJlZmVyZW5jZXM6XG4gKiAgIC0gW0VmZmljaWVudCBPYmplY3QgTG9jYWxpemF0aW9uIFVzaW5nIENvbnZvbHV0aW9uYWxcbiAqICAgICAgTmV0d29ya3NdKGh0dHBzOi8vYXJ4aXYub3JnL2Ficy8xNDExLjQyODApXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdCYXNpYycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzcGF0aWFsRHJvcG91dDFkKGFyZ3M6IFNwYXRpYWxEcm9wb3V0MURMYXllckNvbmZpZykge1xuICByZXR1cm4gbmV3IFNwYXRpYWxEcm9wb3V0MUQoYXJncyk7XG59XG5cbi8qKlxuICogRmxhdHRlbnMgdGhlIGlucHV0LiBEb2VzIG5vdCBhZmZlY3QgdGhlIGJhdGNoIHNpemUuXG4gKlxuICogQSBgRmxhdHRlbmAgbGF5ZXIgZmxhdHRlbnMgZWFjaCBiYXRjaCBpbiBpdHMgaW5wdXRzIHRvIDFEIChtYWtpbmcgdGhlIG91dHB1dFxuICogMkQpLlxuICpcbiAqIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dCA9IHRmLmlucHV0KHtzaGFwZTogWzQsIDNdfSk7XG4gKiBjb25zdCBmbGF0dGVuTGF5ZXIgPSB0Zi5sYXllcnMuZmxhdHRlbigpO1xuICogLy8gSW5zcGVjdCB0aGUgaW5mZXJyZWQgb3V0cHV0IHNoYXBlIG9mIHRoZSBmbGF0dGVuIGxheWVyLCB3aGljaFxuICogLy8gZXF1YWxzIGBbbnVsbCwgMTJdYC4gVGhlIDJuZCBkaW1lbnNpb24gaXMgNCAqIDMsIGkuZS4sIHRoZSByZXN1bHQgb2YgdGhlXG4gKiAvLyBmbGF0dGVuaW5nLiAoVGhlIDFzdCBkaW1lbnNpb24gaXMgdGhlIHVuZGVybWluZWQgYmF0Y2ggc2l6ZS4pXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShmbGF0dGVuTGF5ZXIuYXBwbHkoaW5wdXQpLnNoYXBlKSk7XG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ0Jhc2ljJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGZsYXR0ZW4oYXJncz86IEZsYXR0ZW5MYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBGbGF0dGVuKGFyZ3MpO1xufVxuXG4vKipcbiAqIFJlcGVhdHMgdGhlIGlucHV0IG4gdGltZXMgaW4gYSBuZXcgZGltZW5zaW9uLlxuICpcbiAqIGBgYGpzXG4gKiAgY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKCk7XG4gKiAgbW9kZWwuYWRkKHRmLmxheWVycy5yZXBlYXRWZWN0b3Ioe246IDQsIGlucHV0U2hhcGU6IFsyXX0pKTtcbiAqICBjb25zdCB4ID0gdGYudGVuc29yMmQoW1sxMCwgMjBdXSk7XG4gKiAgLy8gVXNlIHRoZSBtb2RlbCB0byBkbyBpbmZlcmVuY2Ugb24gYSBkYXRhIHBvaW50IHRoZSBtb2RlbCBoYXNuJ3Qgc2VlXG4gKiAgbW9kZWwucHJlZGljdCh4KS5wcmludCgpO1xuICogIC8vIG91dHB1dCBzaGFwZSBpcyBub3cgW2JhdGNoLCAyLCA0XVxuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdCYXNpYycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiByZXBlYXRWZWN0b3IoYXJnczogUmVwZWF0VmVjdG9yTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgUmVwZWF0VmVjdG9yKGFyZ3MpO1xufVxuXG4vKipcbiAqIFJlc2hhcGVzIGFuIGlucHV0IHRvIGEgY2VydGFpbiBzaGFwZS5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW5wdXQgPSB0Zi5pbnB1dCh7c2hhcGU6IFs0LCAzXX0pO1xuICogY29uc3QgcmVzaGFwZUxheWVyID0gdGYubGF5ZXJzLnJlc2hhcGUoe3RhcmdldFNoYXBlOiBbMiwgNl19KTtcbiAqIC8vIEluc3BlY3QgdGhlIGluZmVycmVkIG91dHB1dCBzaGFwZSBvZiB0aGUgUmVzaGFwZSBsYXllciwgd2hpY2hcbiAqIC8vIGVxdWFscyBgW251bGwsIDIsIDZdYC4gKFRoZSAxc3QgZGltZW5zaW9uIGlzIHRoZSB1bmRlcm1pbmVkIGJhdGNoIHNpemUuKVxuICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkocmVzaGFwZUxheWVyLmFwcGx5KGlucHV0KS5zaGFwZSkpO1xuICogYGBgXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIEFyYml0cmFyeSwgYWx0aG91Z2ggYWxsIGRpbWVuc2lvbnMgaW4gdGhlIGlucHV0IHNoYXBlIG11c3QgYmUgZml4ZWQuXG4gKiAgIFVzZSB0aGUgY29uZmlndXJhdGlvbiBgaW5wdXRTaGFwZWAgd2hlbiB1c2luZyB0aGlzIGxheWVyIGFzIHRoZVxuICogICBmaXJzdCBsYXllciBpbiBhIG1vZGVsLlxuICpcbiAqXG4gKiBPdXRwdXQgc2hhcGU6XG4gKiAgIFtiYXRjaFNpemUsIHRhcmdldFNoYXBlWzBdLCB0YXJnZXRTaGFwZVsxXSwgLi4uLFxuICogICAgdGFyZ2V0U2hhcGVbdGFyZ2V0U2hhcGUubGVuZ3RoIC0gMV1dLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQmFzaWMnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gcmVzaGFwZShhcmdzOiBSZXNoYXBlTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgUmVzaGFwZShhcmdzKTtcbn1cblxuLyoqXG4gKiBQZXJtdXRlcyB0aGUgZGltZW5zaW9ucyBvZiB0aGUgaW5wdXQgYWNjb3JkaW5nIHRvIGEgZ2l2ZW4gcGF0dGVybi5cbiAqXG4gKiBVc2VmdWwgZm9yLCBlLmcuLCBjb25uZWN0aW5nIFJOTnMgYW5kIGNvbnZuZXRzIHRvZ2V0aGVyLlxuICpcbiAqIEV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICogbW9kZWwuYWRkKHRmLmxheWVycy5wZXJtdXRlKHtcbiAqICAgZGltczogWzIsIDFdLFxuICogICBpbnB1dFNoYXBlOiBbMTAsIDY0XVxuICogfSkpO1xuICogY29uc29sZS5sb2cobW9kZWwub3V0cHV0U2hhcGUpO1xuICogLy8gTm93IG1vZGVsJ3Mgb3V0cHV0IHNoYXBlIGlzIFtudWxsLCA2NCwgMTBdLCB3aGVyZSBudWxsIGlzIHRoZVxuICogLy8gdW5wZXJtdXRlZCBzYW1wbGUgKGJhdGNoKSBkaW1lbnNpb24uXG4gKiBgYGBcbiAqXG4gKiBJbnB1dCBzaGFwZTpcbiAqICAgQXJiaXRyYXJ5LiBVc2UgdGhlIGNvbmZpZ3VyYXRpb24gZmllbGQgYGlucHV0U2hhcGVgIHdoZW4gdXNpbmcgdGhpc1xuICogICBsYXllciBhcyB0aGUgZmlyc3QgbGF5ZXIgaW4gYSBtb2RlbC5cbiAqXG4gKiBPdXRwdXQgc2hhcGU6XG4gKiAgIFNhbWUgcmFuayBhcyB0aGUgaW5wdXQgc2hhcGUsIGJ1dCB3aXRoIHRoZSBkaW1lbnNpb25zIHJlLW9yZGVyZWQgKGkuZS4sXG4gKiAgIHBlcm11dGVkKSBhY2NvcmRpbmcgdG8gdGhlIGBkaW1zYCBjb25maWd1cmF0aW9uIG9mIHRoaXMgbGF5ZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdCYXNpYycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwZXJtdXRlKGFyZ3M6IFBlcm11dGVMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBQZXJtdXRlKGFyZ3MpO1xufVxuXG4vKipcbiAqIE1hcHMgcG9zaXRpdmUgaW50ZWdlcnMgKGluZGljZXMpIGludG8gZGVuc2UgdmVjdG9ycyBvZiBmaXhlZCBzaXplLlxuICogZWcuIFtbNF0sIFsyMF1dIC0+IFtbMC4yNSwgMC4xXSwgWzAuNiwgLTAuMl1dXG4gKlxuICogKipJbnB1dCBzaGFwZToqKiAyRCB0ZW5zb3Igd2l0aCBzaGFwZTogYFtiYXRjaFNpemUsIHNlcXVlbmNlTGVuZ3RoXWAuXG4gKlxuICogKipPdXRwdXQgc2hhcGU6KiogM0QgdGVuc29yIHdpdGggc2hhcGU6IGBbYmF0Y2hTaXplLCBzZXF1ZW5jZUxlbmd0aCxcbiAqIG91dHB1dERpbV1gLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQmFzaWMnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZW1iZWRkaW5nKGFyZ3M6IEVtYmVkZGluZ0xheWVyQXJncykge1xuICByZXR1cm4gbmV3IEVtYmVkZGluZyhhcmdzKTtcbn1cblxuLy8gTWVyZ2UgTGF5ZXJzLlxuXG4vKipcbiAqIExheWVyIHRoYXQgcGVyZm9ybXMgZWxlbWVudC13aXNlIGFkZGl0aW9uIG9uIGFuIGBBcnJheWAgb2YgaW5wdXRzLlxuICpcbiAqIEl0IHRha2VzIGFzIGlucHV0IGEgbGlzdCBvZiB0ZW5zb3JzLCBhbGwgb2YgdGhlIHNhbWUgc2hhcGUsIGFuZCByZXR1cm5zIGFcbiAqIHNpbmdsZSB0ZW5zb3IgKGFsc28gb2YgdGhlIHNhbWUgc2hhcGUpLiBUaGUgaW5wdXRzIGFyZSBzcGVjaWZpZWQgYXMgYW5cbiAqIGBBcnJheWAgd2hlbiB0aGUgYGFwcGx5YCBtZXRob2Qgb2YgdGhlIGBBZGRgIGxheWVyIGluc3RhbmNlIGlzIGNhbGxlZC4gRm9yXG4gKiBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IGFkZExheWVyID0gdGYubGF5ZXJzLmFkZCgpO1xuICogY29uc3Qgc3VtID0gYWRkTGF5ZXIuYXBwbHkoW2lucHV0MSwgaW5wdXQyXSk7XG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShzdW0uc2hhcGUpKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdNZXJnZScsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhZGQoYXJncz86IExheWVyQXJncykge1xuICByZXR1cm4gbmV3IEFkZChhcmdzKTtcbn1cblxuLyoqXG4gKiBMYXllciB0aGF0IHBlcmZvcm1zIGVsZW1lbnQtd2lzZSBhdmVyYWdpbmcgb24gYW4gYEFycmF5YCBvZiBpbnB1dHMuXG4gKlxuICogSXQgdGFrZXMgYXMgaW5wdXQgYSBsaXN0IG9mIHRlbnNvcnMsIGFsbCBvZiB0aGUgc2FtZSBzaGFwZSwgYW5kIHJldHVybnMgYVxuICogc2luZ2xlIHRlbnNvciAoYWxzbyBvZiB0aGUgc2FtZSBzaGFwZSkuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IGF2ZXJhZ2VMYXllciA9IHRmLmxheWVycy5hdmVyYWdlKCk7XG4gKiBjb25zdCBhdmVyYWdlID0gYXZlcmFnZUxheWVyLmFwcGx5KFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkoYXZlcmFnZS5zaGFwZSkpO1xuICogLy8gWW91IGdldCBbbnVsbCwgMiwgMl0sIHdpdGggdGhlIGZpcnN0IGRpbWVuc2lvbiBhcyB0aGUgdW5kZXRlcm1pbmVkIGJhdGNoXG4gKiAvLyBkaW1lbnNpb24uXG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ01lcmdlJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGF2ZXJhZ2UoYXJncz86IExheWVyQXJncykge1xuICByZXR1cm4gbmV3IEF2ZXJhZ2UoYXJncyk7XG59XG5cbi8qKlxuICogTGF5ZXIgdGhhdCBjb25jYXRlbmF0ZXMgYW4gYEFycmF5YCBvZiBpbnB1dHMuXG4gKlxuICogSXQgdGFrZXMgYSBsaXN0IG9mIHRlbnNvcnMsIGFsbCBvZiB0aGUgc2FtZSBzaGFwZSBleGNlcHQgZm9yIHRoZVxuICogY29uY2F0ZW5hdGlvbiBheGlzLCBhbmQgcmV0dXJucyBhIHNpbmdsZSB0ZW5zb3IsIHRoZSBjb25jYXRlbmF0aW9uXG4gKiBvZiBhbGwgaW5wdXRzLiBGb3IgZXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW5wdXQxID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IGlucHV0MiA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDNdfSk7XG4gKiBjb25zdCBjb25jYXRMYXllciA9IHRmLmxheWVycy5jb25jYXRlbmF0ZSgpO1xuICogY29uc3Qgb3V0cHV0ID0gY29uY2F0TGF5ZXIuYXBwbHkoW2lucHV0MSwgaW5wdXQyXSk7XG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShvdXRwdXQuc2hhcGUpKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDVdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLiBUaGUgbGFzdCBkaW1lbnNpb24gKDUpIGlzIHRoZSByZXN1bHQgb2YgY29uY2F0ZW5hdGluZyB0aGVcbiAqIC8vIGxhc3QgZGltZW5zaW9ucyBvZiB0aGUgaW5wdXRzICgyIGFuZCAzKS5cbiAqIGBgYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnTWVyZ2UnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gY29uY2F0ZW5hdGUoYXJncz86IENvbmNhdGVuYXRlTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgQ29uY2F0ZW5hdGUoYXJncyk7XG59XG5cbi8qKlxuICogTGF5ZXIgdGhhdCBjb21wdXRlcyB0aGUgZWxlbWVudC13aXNlIG1heGltdW0gYW4gYEFycmF5YCBvZiBpbnB1dHMuXG4gKlxuICogSXQgdGFrZXMgYXMgaW5wdXQgYSBsaXN0IG9mIHRlbnNvcnMsIGFsbCBvZiB0aGUgc2FtZSBzaGFwZSBhbmQgcmV0dXJucyBhXG4gKiBzaW5nbGUgdGVuc29yIChhbHNvIG9mIHRoZSBzYW1lIHNoYXBlKS4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGlucHV0MSA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDJdfSk7XG4gKiBjb25zdCBpbnB1dDIgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgbWF4TGF5ZXIgPSB0Zi5sYXllcnMubWF4aW11bSgpO1xuICogY29uc3QgbWF4ID0gbWF4TGF5ZXIuYXBwbHkoW2lucHV0MSwgaW5wdXQyXSk7XG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShtYXguc2hhcGUpKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdNZXJnZScsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBtYXhpbXVtKGFyZ3M/OiBMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBNYXhpbXVtKGFyZ3MpO1xufVxuXG4vKipcbiAqIExheWVyIHRoYXQgY29tcHV0ZXMgdGhlIGVsZW1lbnQtd2lzZSBtaW5pbXVtIG9mIGFuIGBBcnJheWAgb2YgaW5wdXRzLlxuICpcbiAqIEl0IHRha2VzIGFzIGlucHV0IGEgbGlzdCBvZiB0ZW5zb3JzLCBhbGwgb2YgdGhlIHNhbWUgc2hhcGUgYW5kIHJldHVybnMgYVxuICogc2luZ2xlIHRlbnNvciAoYWxzbyBvZiB0aGUgc2FtZSBzaGFwZSkuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IG1pbkxheWVyID0gdGYubGF5ZXJzLm1pbmltdW0oKTtcbiAqIGNvbnN0IG1pbiA9IG1pbkxheWVyLmFwcGx5KFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkobWluLnNoYXBlKSk7XG4gKiAvLyBZb3UgZ2V0IFtudWxsLCAyLCAyXSwgd2l0aCB0aGUgZmlyc3QgZGltZW5zaW9uIGFzIHRoZSB1bmRldGVybWluZWQgYmF0Y2hcbiAqIC8vIGRpbWVuc2lvbi5cbiAqIGBgYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnTWVyZ2UnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gbWluaW11bShhcmdzPzogTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgTWluaW11bShhcmdzKTtcbn1cblxuLyoqXG4gKiBMYXllciB0aGF0IG11bHRpcGxpZXMgKGVsZW1lbnQtd2lzZSkgYW4gYEFycmF5YCBvZiBpbnB1dHMuXG4gKlxuICogSXQgdGFrZXMgYXMgaW5wdXQgYW4gQXJyYXkgb2YgdGVuc29ycywgYWxsIG9mIHRoZSBzYW1lXG4gKiBzaGFwZSwgYW5kIHJldHVybnMgYSBzaW5nbGUgdGVuc29yIChhbHNvIG9mIHRoZSBzYW1lIHNoYXBlKS5cbiAqIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IGlucHV0MyA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDJdfSk7XG4gKiBjb25zdCBtdWx0aXBseUxheWVyID0gdGYubGF5ZXJzLm11bHRpcGx5KCk7XG4gKiBjb25zdCBwcm9kdWN0ID0gbXVsdGlwbHlMYXllci5hcHBseShbaW5wdXQxLCBpbnB1dDIsIGlucHV0M10pO1xuICogY29uc29sZS5sb2cocHJvZHVjdC5zaGFwZSk7XG4gKiAvLyBZb3UgZ2V0IFtudWxsLCAyLCAyXSwgd2l0aCB0aGUgZmlyc3QgZGltZW5zaW9uIGFzIHRoZSB1bmRldGVybWluZWQgYmF0Y2hcbiAqIC8vIGRpbWVuc2lvbi5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ01lcmdlJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG11bHRpcGx5KGFyZ3M/OiBMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBNdWx0aXBseShhcmdzKTtcbn1cblxuLyoqXG4gKiBMYXllciB0aGF0IGNvbXB1dGVzIGEgZG90IHByb2R1Y3QgYmV0d2VlbiBzYW1wbGVzIGluIHR3byB0ZW5zb3JzLlxuICpcbiAqIEUuZy4sIGlmIGFwcGxpZWQgdG8gYSBsaXN0IG9mIHR3byB0ZW5zb3JzIGBhYCBhbmQgYGJgIGJvdGggb2Ygc2hhcGVcbiAqIGBbYmF0Y2hTaXplLCBuXWAsIHRoZSBvdXRwdXQgd2lsbCBiZSBhIHRlbnNvciBvZiBzaGFwZSBgW2JhdGNoU2l6ZSwgMV1gLFxuICogd2hlcmUgZWFjaCBlbnRyeSBhdCBpbmRleCBgW2ksIDBdYCB3aWxsIGJlIHRoZSBkb3QgcHJvZHVjdCBiZXR3ZWVuXG4gKiBgYVtpLCA6XWAgYW5kIGBiW2ksIDpdYC5cbiAqXG4gKiBFeGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBkb3RMYXllciA9IHRmLmxheWVycy5kb3Qoe2F4ZXM6IC0xfSk7XG4gKiBjb25zdCB4MSA9IHRmLnRlbnNvcjJkKFtbMTAsIDIwXSwgWzMwLCA0MF1dKTtcbiAqIGNvbnN0IHgyID0gdGYudGVuc29yMmQoW1stMSwgLTJdLCBbLTMsIC00XV0pO1xuICpcbiAqIC8vIEludm9rZSB0aGUgbGF5ZXIncyBhcHBseSgpIG1ldGhvZCBpbiBlYWdlciAoaW1wZXJhdGl2ZSkgbW9kZS5cbiAqIGNvbnN0IHkgPSBkb3RMYXllci5hcHBseShbeDEsIHgyXSk7XG4gKiB5LnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ01lcmdlJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGRvdChhcmdzOiBEb3RMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBEb3QoYXJncyk7XG59XG5cbi8vIE5vcm1hbGl6YXRpb24gTGF5ZXJzLlxuXG4vKipcbiAqIEJhdGNoIG5vcm1hbGl6YXRpb24gbGF5ZXIgKElvZmZlIGFuZCBTemVnZWR5LCAyMDE0KS5cbiAqXG4gKiBOb3JtYWxpemUgdGhlIGFjdGl2YXRpb25zIG9mIHRoZSBwcmV2aW91cyBsYXllciBhdCBlYWNoIGJhdGNoLFxuICogaS5lLiBhcHBsaWVzIGEgdHJhbnNmb3JtYXRpb24gdGhhdCBtYWludGFpbnMgdGhlIG1lYW4gYWN0aXZhdGlvblxuICogY2xvc2UgdG8gMCBhbmQgdGhlIGFjdGl2YXRpb24gc3RhbmRhcmQgZGV2aWF0aW9uIGNsb3NlIHRvIDEuXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIEFyYml0cmFyeS4gVXNlIHRoZSBrZXl3b3JkIGFyZ3VtZW50IGBpbnB1dFNoYXBlYCAoQXJyYXkgb2YgaW50ZWdlcnMsIGRvZXNcbiAqICAgbm90IGluY2x1ZGUgdGhlIHNhbXBsZSBheGlzKSB3aGVuIGNhbGxpbmcgdGhlIGNvbnN0cnVjdG9yIG9mIHRoaXMgY2xhc3MsXG4gKiAgIGlmIHRoaXMgbGF5ZXIgaXMgdXNlZCBhcyBhIGZpcnN0IGxheWVyIGluIGEgbW9kZWwuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICBTYW1lIHNoYXBlIGFzIGlucHV0LlxuICpcbiAqIFJlZmVyZW5jZXM6XG4gKiAgIC0gW0JhdGNoIE5vcm1hbGl6YXRpb246IEFjY2VsZXJhdGluZyBEZWVwIE5ldHdvcmsgVHJhaW5pbmcgYnkgUmVkdWNpbmdcbiAqIEludGVybmFsIENvdmFyaWF0ZSBTaGlmdF0oaHR0cHM6Ly9hcnhpdi5vcmcvYWJzLzE1MDIuMDMxNjcpXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdOb3JtYWxpemF0aW9uJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGJhdGNoTm9ybWFsaXphdGlvbihhcmdzPzogQmF0Y2hOb3JtYWxpemF0aW9uTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgQmF0Y2hOb3JtYWxpemF0aW9uKGFyZ3MpO1xufVxuXG4vKipcbiAqIExheWVyLW5vcm1hbGl6YXRpb24gbGF5ZXIgKEJhIGV0IGFsLiwgMjAxNikuXG4gKlxuICogTm9ybWFsaXplcyB0aGUgYWN0aXZhdGlvbnMgb2YgdGhlIHByZXZpb3VzIGxheWVyIGZvciBlYWNoIGdpdmVuIGV4YW1wbGUgaW4gYVxuICogYmF0Y2ggaW5kZXBlbmRlbnRseSwgaW5zdGVhZCBvZiBhY3Jvc3MgYSBiYXRjaCBsaWtlIGluIGBiYXRjaE5vcm1hbGl6YXRpb25gLlxuICogSW4gb3RoZXIgd29yZHMsIHRoaXMgbGF5ZXIgYXBwbGllcyBhIHRyYW5zZm9ybWF0aW9uIHRoYXQgbWFpbnRhbmlzIHRoZSBtZWFuXG4gKiBhY3RpdmF0aW9uIHdpdGhpbiBlYWNoIGV4YW1wbGUgY2xvc2UgdG8wIGFuZCBhY3RpdmF0aW9uIHZhcmlhbmNlIGNsb3NlIHRvIDEuXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIEFyYml0cmFyeS4gVXNlIHRoZSBhcmd1bWVudCBgaW5wdXRTaGFwZWAgd2hlbiB1c2luZyB0aGlzIGxheWVyIGFzIHRoZSBmaXJzdFxuICogICBsYXllciBpbiBhIG1vZGVsLlxuICpcbiAqIE91dHB1dCBzaGFwZTpcbiAqICAgU2FtZSBhcyBpbnB1dC5cbiAqXG4gKiBSZWZlcmVuY2VzOlxuICogICAtIFtMYXllciBOb3JtYWxpemF0aW9uXShodHRwczovL2FyeGl2Lm9yZy9hYnMvMTYwNy4wNjQ1MClcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ05vcm1hbGl6YXRpb24nLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gbGF5ZXJOb3JtYWxpemF0aW9uKGFyZ3M/OiBMYXllck5vcm1hbGl6YXRpb25MYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBMYXllck5vcm1hbGl6YXRpb24oYXJncyk7XG59XG5cbi8vIFBhZGRpbmcgTGF5ZXJzLlxuXG4vKipcbiAqIFplcm8tcGFkZGluZyBsYXllciBmb3IgMkQgaW5wdXQgKGUuZy4sIGltYWdlKS5cbiAqXG4gKiBUaGlzIGxheWVyIGNhbiBhZGQgcm93cyBhbmQgY29sdW1ucyBvZiB6ZXJvc1xuICogYXQgdGhlIHRvcCwgYm90dG9tLCBsZWZ0IGFuZCByaWdodCBzaWRlIG9mIGFuIGltYWdlIHRlbnNvci5cbiAqXG4gKiBJbnB1dCBzaGFwZTpcbiAqICAgNEQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgIC0gSWYgYGRhdGFGb3JtYXRgIGlzIGBcImNoYW5uZWxzTGFzdFwiYDpcbiAqICAgICBgW2JhdGNoLCByb3dzLCBjb2xzLCBjaGFubmVsc11gXG4gKiAgIC0gSWYgYGRhdGFfZm9ybWF0YCBpcyBgXCJjaGFubmVsc19maXJzdFwiYDpcbiAqICAgICBgW2JhdGNoLCBjaGFubmVscywgcm93cywgY29sc11gLlxuICpcbiAqIE91dHB1dCBzaGFwZTpcbiAqICAgNEQgd2l0aCBzaGFwZTpcbiAqICAgLSBJZiBgZGF0YUZvcm1hdGAgaXMgYFwiY2hhbm5lbHNMYXN0XCJgOlxuICogICAgIGBbYmF0Y2gsIHBhZGRlZFJvd3MsIHBhZGRlZENvbHMsIGNoYW5uZWxzXWBcbiAqICAgIC0gSWYgYGRhdGFGb3JtYXRgIGlzIGBcImNoYW5uZWxzRmlyc3RcImA6XG4gKiAgICAgYFtiYXRjaCwgY2hhbm5lbHMsIHBhZGRlZFJvd3MsIHBhZGRlZENvbHNdYC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ1BhZGRpbmcnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gemVyb1BhZGRpbmcyZChhcmdzPzogWmVyb1BhZGRpbmcyRExheWVyQXJncykge1xuICByZXR1cm4gbmV3IFplcm9QYWRkaW5nMkQoYXJncyk7XG59XG5cbi8vIFBvb2xpbmcgTGF5ZXJzLlxuXG4vKipcbiAqIEF2ZXJhZ2UgcG9vbGluZyBvcGVyYXRpb24gZm9yIHNwYXRpYWwgZGF0YS5cbiAqXG4gKiBJbnB1dCBzaGFwZTogYFtiYXRjaFNpemUsIGluTGVuZ3RoLCBjaGFubmVsc11gXG4gKlxuICogT3V0cHV0IHNoYXBlOiBgW2JhdGNoU2l6ZSwgcG9vbGVkTGVuZ3RoLCBjaGFubmVsc11gXG4gKlxuICogYHRmLmF2Z1Bvb2wxZGAgaXMgYW4gYWxpYXMuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdQb29saW5nJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGF2ZXJhZ2VQb29saW5nMWQoYXJnczogUG9vbGluZzFETGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgQXZlcmFnZVBvb2xpbmcxRChhcmdzKTtcbn1cbmV4cG9ydCBmdW5jdGlvbiBhdmdQb29sMWQoYXJnczogUG9vbGluZzFETGF5ZXJBcmdzKSB7XG4gIHJldHVybiBhdmVyYWdlUG9vbGluZzFkKGFyZ3MpO1xufVxuLy8gRm9yIGJhY2t3YXJkcyBjb21wYXRpYmlsaXR5LlxuLy8gU2VlIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzE1MlxuZXhwb3J0IGZ1bmN0aW9uIGF2Z1Bvb2xpbmcxZChhcmdzOiBQb29saW5nMURMYXllckFyZ3MpIHtcbiAgcmV0dXJuIGF2ZXJhZ2VQb29saW5nMWQoYXJncyk7XG59XG5cbi8qKlxuICogQXZlcmFnZSBwb29saW5nIG9wZXJhdGlvbiBmb3Igc3BhdGlhbCBkYXRhLlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogIC0gSWYgYGRhdGFGb3JtYXQgPT09IENIQU5ORUxfTEFTVGA6XG4gKiAgICAgIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgICBgW2JhdGNoU2l6ZSwgcm93cywgY29scywgY2hhbm5lbHNdYFxuICogIC0gSWYgYGRhdGFGb3JtYXQgPT09IENIQU5ORUxfRklSU1RgOlxuICogICAgICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTpcbiAqICAgICAgYFtiYXRjaFNpemUsIGNoYW5uZWxzLCByb3dzLCBjb2xzXWBcbiAqXG4gKiBPdXRwdXQgc2hhcGVcbiAqICAtIElmIGBkYXRhRm9ybWF0ID09PSBDSEFOTkVMX0xBU1RgOlxuICogICAgICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTpcbiAqICAgICAgYFtiYXRjaFNpemUsIHBvb2xlUm93cywgcG9vbGVkQ29scywgY2hhbm5lbHNdYFxuICogIC0gSWYgYGRhdGFGb3JtYXQgPT09IENIQU5ORUxfRklSU1RgOlxuICogICAgICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTpcbiAqICAgICAgYFtiYXRjaFNpemUsIGNoYW5uZWxzLCBwb29sZVJvd3MsIHBvb2xlZENvbHNdYFxuICpcbiAqIGB0Zi5hdmdQb29sMmRgIGlzIGFuIGFsaWFzLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUG9vbGluZycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhdmVyYWdlUG9vbGluZzJkKGFyZ3M6IFBvb2xpbmcyRExheWVyQXJncykge1xuICByZXR1cm4gbmV3IEF2ZXJhZ2VQb29saW5nMkQoYXJncyk7XG59XG5leHBvcnQgZnVuY3Rpb24gYXZnUG9vbDJkKGFyZ3M6IFBvb2xpbmcyRExheWVyQXJncykge1xuICByZXR1cm4gYXZlcmFnZVBvb2xpbmcyZChhcmdzKTtcbn1cbi8vIEZvciBiYWNrd2FyZHMgY29tcGF0aWJpbGl0eS5cbi8vIFNlZSBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2lzc3Vlcy8xNTJcbmV4cG9ydCBmdW5jdGlvbiBhdmdQb29saW5nMmQoYXJnczogUG9vbGluZzJETGF5ZXJBcmdzKSB7XG4gIHJldHVybiBhdmVyYWdlUG9vbGluZzJkKGFyZ3MpO1xufVxuXG4vKipcbiAqIEF2ZXJhZ2UgcG9vbGluZyBvcGVyYXRpb24gZm9yIDNEIGRhdGEuXG4gKlxuICogSW5wdXQgc2hhcGVcbiAqICAgLSBJZiBgZGF0YUZvcm1hdCA9PT0gY2hhbm5lbHNMYXN0YDpcbiAqICAgICAgIDVEIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgICAgYFtiYXRjaFNpemUsIGRlcHRocywgcm93cywgY29scywgY2hhbm5lbHNdYFxuICogICAtIElmIGBkYXRhRm9ybWF0ID09PSBjaGFubmVsc0ZpcnN0YDpcbiAqICAgICAgNEQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgICAgICBgW2JhdGNoU2l6ZSwgY2hhbm5lbHMsIGRlcHRocywgcm93cywgY29sc11gXG4gKlxuICogT3V0cHV0IHNoYXBlXG4gKiAgIC0gSWYgYGRhdGFGb3JtYXQ9Y2hhbm5lbHNMYXN0YDpcbiAqICAgICAgIDVEIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgICAgYFtiYXRjaFNpemUsIHBvb2xlZERlcHRocywgcG9vbGVkUm93cywgcG9vbGVkQ29scywgY2hhbm5lbHNdYFxuICogICAtIElmIGBkYXRhRm9ybWF0PWNoYW5uZWxzRmlyc3RgOlxuICogICAgICAgNUQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgICAgICBgW2JhdGNoU2l6ZSwgY2hhbm5lbHMsIHBvb2xlZERlcHRocywgcG9vbGVkUm93cywgcG9vbGVkQ29sc11gXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdQb29saW5nJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGF2ZXJhZ2VQb29saW5nM2QoYXJnczogUG9vbGluZzNETGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgQXZlcmFnZVBvb2xpbmczRChhcmdzKTtcbn1cbmV4cG9ydCBmdW5jdGlvbiBhdmdQb29sM2QoYXJnczogUG9vbGluZzNETGF5ZXJBcmdzKSB7XG4gIHJldHVybiBhdmVyYWdlUG9vbGluZzNkKGFyZ3MpO1xufVxuLy8gRm9yIGJhY2t3YXJkcyBjb21wYXRpYmlsaXR5LlxuLy8gU2VlIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzE1MlxuZXhwb3J0IGZ1bmN0aW9uIGF2Z1Bvb2xpbmczZChhcmdzOiBQb29saW5nM0RMYXllckFyZ3MpIHtcbiAgcmV0dXJuIGF2ZXJhZ2VQb29saW5nM2QoYXJncyk7XG59XG5cbi8qKlxuICogR2xvYmFsIGF2ZXJhZ2UgcG9vbGluZyBvcGVyYXRpb24gZm9yIHRlbXBvcmFsIGRhdGEuXG4gKlxuICogSW5wdXQgU2hhcGU6IDNEIHRlbnNvciB3aXRoIHNoYXBlOiBgW2JhdGNoU2l6ZSwgc3RlcHMsIGZlYXR1cmVzXWAuXG4gKlxuICogT3V0cHV0IFNoYXBlOjJEIHRlbnNvciB3aXRoIHNoYXBlOiBgW2JhdGNoU2l6ZSwgZmVhdHVyZXNdYC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ1Bvb2xpbmcnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2xvYmFsQXZlcmFnZVBvb2xpbmcxZChhcmdzPzogTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgR2xvYmFsQXZlcmFnZVBvb2xpbmcxRChhcmdzKTtcbn1cblxuLyoqXG4gKiBHbG9iYWwgYXZlcmFnZSBwb29saW5nIG9wZXJhdGlvbiBmb3Igc3BhdGlhbCBkYXRhLlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICAtIElmIGBkYXRhRm9ybWF0YCBpcyBgQ0hBTk5FTF9MQVNUYDpcbiAqICAgICAgIDREIHRlbnNvciB3aXRoIHNoYXBlOiBgW2JhdGNoU2l6ZSwgcm93cywgY29scywgY2hhbm5lbHNdYC5cbiAqICAgLSBJZiBgZGF0YUZvcm1hdGAgaXMgYENIQU5ORUxfRklSU1RgOlxuICogICAgICAgNEQgdGVuc29yIHdpdGggc2hhcGU6IGBbYmF0Y2hTaXplLCBjaGFubmVscywgcm93cywgY29sc11gLlxuICpcbiAqIE91dHB1dCBzaGFwZTpcbiAqICAgMkQgdGVuc29yIHdpdGggc2hhcGU6IGBbYmF0Y2hTaXplLCBjaGFubmVsc11gLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUG9vbGluZycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnbG9iYWxBdmVyYWdlUG9vbGluZzJkKGFyZ3M6IEdsb2JhbFBvb2xpbmcyRExheWVyQXJncykge1xuICByZXR1cm4gbmV3IEdsb2JhbEF2ZXJhZ2VQb29saW5nMkQoYXJncyk7XG59XG5cbi8qKlxuICogR2xvYmFsIG1heCBwb29saW5nIG9wZXJhdGlvbiBmb3IgdGVtcG9yYWwgZGF0YS5cbiAqXG4gKiBJbnB1dCBTaGFwZTogM0QgdGVuc29yIHdpdGggc2hhcGU6IGBbYmF0Y2hTaXplLCBzdGVwcywgZmVhdHVyZXNdYC5cbiAqXG4gKiBPdXRwdXQgU2hhcGU6MkQgdGVuc29yIHdpdGggc2hhcGU6IGBbYmF0Y2hTaXplLCBmZWF0dXJlc11gLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUG9vbGluZycsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnbG9iYWxNYXhQb29saW5nMWQoYXJncz86IExheWVyQXJncykge1xuICByZXR1cm4gbmV3IEdsb2JhbE1heFBvb2xpbmcxRChhcmdzKTtcbn1cblxuLyoqXG4gKiBHbG9iYWwgbWF4IHBvb2xpbmcgb3BlcmF0aW9uIGZvciBzcGF0aWFsIGRhdGEuXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIC0gSWYgYGRhdGFGb3JtYXRgIGlzIGBDSEFOTkVMX0xBU1RgOlxuICogICAgICAgNEQgdGVuc29yIHdpdGggc2hhcGU6IGBbYmF0Y2hTaXplLCByb3dzLCBjb2xzLCBjaGFubmVsc11gLlxuICogICAtIElmIGBkYXRhRm9ybWF0YCBpcyBgQ0hBTk5FTF9GSVJTVGA6XG4gKiAgICAgICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTogYFtiYXRjaFNpemUsIGNoYW5uZWxzLCByb3dzLCBjb2xzXWAuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICAyRCB0ZW5zb3Igd2l0aCBzaGFwZTogYFtiYXRjaFNpemUsIGNoYW5uZWxzXWAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdQb29saW5nJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdsb2JhbE1heFBvb2xpbmcyZChhcmdzOiBHbG9iYWxQb29saW5nMkRMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBHbG9iYWxNYXhQb29saW5nMkQoYXJncyk7XG59XG5cbi8qKlxuICogTWF4IHBvb2xpbmcgb3BlcmF0aW9uIGZvciB0ZW1wb3JhbCBkYXRhLlxuICpcbiAqIElucHV0IHNoYXBlOiAgYFtiYXRjaFNpemUsIGluTGVuZ3RoLCBjaGFubmVsc11gXG4gKlxuICogT3V0cHV0IHNoYXBlOiBgW2JhdGNoU2l6ZSwgcG9vbGVkTGVuZ3RoLCBjaGFubmVsc11gXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdQb29saW5nJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2xpbmcxZChhcmdzOiBQb29saW5nMURMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBNYXhQb29saW5nMUQoYXJncyk7XG59XG5cbi8qKlxuICogTWF4IHBvb2xpbmcgb3BlcmF0aW9uIGZvciBzcGF0aWFsIGRhdGEuXG4gKlxuICogSW5wdXQgc2hhcGVcbiAqICAgLSBJZiBgZGF0YUZvcm1hdCA9PT0gQ0hBTk5FTF9MQVNUYDpcbiAqICAgICAgIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgICAgYFtiYXRjaFNpemUsIHJvd3MsIGNvbHMsIGNoYW5uZWxzXWBcbiAqICAgLSBJZiBgZGF0YUZvcm1hdCA9PT0gQ0hBTk5FTF9GSVJTVGA6XG4gKiAgICAgIDREIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgICAgYFtiYXRjaFNpemUsIGNoYW5uZWxzLCByb3dzLCBjb2xzXWBcbiAqXG4gKiBPdXRwdXQgc2hhcGVcbiAqICAgLSBJZiBgZGF0YUZvcm1hdD1DSEFOTkVMX0xBU1RgOlxuICogICAgICAgNEQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgICAgICBgW2JhdGNoU2l6ZSwgcG9vbGVSb3dzLCBwb29sZWRDb2xzLCBjaGFubmVsc11gXG4gKiAgIC0gSWYgYGRhdGFGb3JtYXQ9Q0hBTk5FTF9GSVJTVGA6XG4gKiAgICAgICA0RCB0ZW5zb3Igd2l0aCBzaGFwZTpcbiAqICAgICAgIGBbYmF0Y2hTaXplLCBjaGFubmVscywgcG9vbGVSb3dzLCBwb29sZWRDb2xzXWBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ1Bvb2xpbmcnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gbWF4UG9vbGluZzJkKGFyZ3M6IFBvb2xpbmcyRExheWVyQXJncykge1xuICByZXR1cm4gbmV3IE1heFBvb2xpbmcyRChhcmdzKTtcbn1cblxuLyoqXG4gKiBNYXggcG9vbGluZyBvcGVyYXRpb24gZm9yIDNEIGRhdGEuXG4gKlxuICogSW5wdXQgc2hhcGVcbiAqICAgLSBJZiBgZGF0YUZvcm1hdCA9PT0gY2hhbm5lbHNMYXN0YDpcbiAqICAgICAgIDVEIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgICAgYFtiYXRjaFNpemUsIGRlcHRocywgcm93cywgY29scywgY2hhbm5lbHNdYFxuICogICAtIElmIGBkYXRhRm9ybWF0ID09PSBjaGFubmVsc0ZpcnN0YDpcbiAqICAgICAgNUQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgICAgICBgW2JhdGNoU2l6ZSwgY2hhbm5lbHMsIGRlcHRocywgcm93cywgY29sc11gXG4gKlxuICogT3V0cHV0IHNoYXBlXG4gKiAgIC0gSWYgYGRhdGFGb3JtYXQ9Y2hhbm5lbHNMYXN0YDpcbiAqICAgICAgIDVEIHRlbnNvciB3aXRoIHNoYXBlOlxuICogICAgICAgYFtiYXRjaFNpemUsIHBvb2xlZERlcHRocywgcG9vbGVkUm93cywgcG9vbGVkQ29scywgY2hhbm5lbHNdYFxuICogICAtIElmIGBkYXRhRm9ybWF0PWNoYW5uZWxzRmlyc3RgOlxuICogICAgICAgNUQgdGVuc29yIHdpdGggc2hhcGU6XG4gKiAgICAgICBgW2JhdGNoU2l6ZSwgY2hhbm5lbHMsIHBvb2xlZERlcHRocywgcG9vbGVkUm93cywgcG9vbGVkQ29sc11gXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdQb29saW5nJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2xpbmczZChhcmdzOiBQb29saW5nM0RMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBNYXhQb29saW5nM0QoYXJncyk7XG59XG5cbi8vIFJlY3VycmVudCBMYXllcnMuXG5cbi8qKlxuICogR2F0ZWQgUmVjdXJyZW50IFVuaXQgLSBDaG8gZXQgYWwuIDIwMTQuXG4gKlxuICogVGhpcyBpcyBhbiBgUk5OYCBsYXllciBjb25zaXN0aW5nIG9mIG9uZSBgR1JVQ2VsbGAuIEhvd2V2ZXIsIHVubGlrZVxuICogdGhlIHVuZGVybHlpbmcgYEdSVUNlbGxgLCB0aGUgYGFwcGx5YCBtZXRob2Qgb2YgYFNpbXBsZVJOTmAgb3BlcmF0ZXNcbiAqIG9uIGEgc2VxdWVuY2Ugb2YgaW5wdXRzLiBUaGUgc2hhcGUgb2YgdGhlIGlucHV0IChub3QgaW5jbHVkaW5nIHRoZSBmaXJzdCxcbiAqIGJhdGNoIGRpbWVuc2lvbikgbmVlZHMgdG8gYmUgYXQgbGVhc3QgMi1ELCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYmVpbmdcbiAqIHRpbWUgc3RlcHMuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBybm4gPSB0Zi5sYXllcnMuZ3J1KHt1bml0czogOCwgcmV0dXJuU2VxdWVuY2VzOiB0cnVlfSk7XG4gKlxuICogLy8gQ3JlYXRlIGFuIGlucHV0IHdpdGggMTAgdGltZSBzdGVwcy5cbiAqIGNvbnN0IGlucHV0ID0gdGYuaW5wdXQoe3NoYXBlOiBbMTAsIDIwXX0pO1xuICogY29uc3Qgb3V0cHV0ID0gcm5uLmFwcGx5KGlucHV0KTtcbiAqXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShvdXRwdXQuc2hhcGUpKTtcbiAqIC8vIFtudWxsLCAxMCwgOF06IDFzdCBkaW1lbnNpb24gaXMgdW5rbm93biBiYXRjaCBzaXplOyAybmQgZGltZW5zaW9uIGlzIHRoZVxuICogLy8gc2FtZSBhcyB0aGUgc2VxdWVuY2UgbGVuZ3RoIG9mIGBpbnB1dGAsIGR1ZSB0byBgcmV0dXJuU2VxdWVuY2VzYDogYHRydWVgO1xuICogLy8gM3JkIGRpbWVuc2lvbiBpcyB0aGUgYEdSVUNlbGxgJ3MgbnVtYmVyIG9mIHVuaXRzLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUmVjdXJyZW50JywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdydShhcmdzOiBHUlVMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBHUlUoYXJncyk7XG59XG5cbi8qKlxuICogQ2VsbCBjbGFzcyBmb3IgYEdSVWAuXG4gKlxuICogYEdSVUNlbGxgIGlzIGRpc3RpbmN0IGZyb20gdGhlIGBSTk5gIHN1YmNsYXNzIGBHUlVgIGluIHRoYXQgaXRzXG4gKiBgYXBwbHlgIG1ldGhvZCB0YWtlcyB0aGUgaW5wdXQgZGF0YSBvZiBvbmx5IGEgc2luZ2xlIHRpbWUgc3RlcCBhbmQgcmV0dXJuc1xuICogdGhlIGNlbGwncyBvdXRwdXQgYXQgdGhlIHRpbWUgc3RlcCwgd2hpbGUgYEdSVWAgdGFrZXMgdGhlIGlucHV0IGRhdGFcbiAqIG92ZXIgYSBudW1iZXIgb2YgdGltZSBzdGVwcy4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGNlbGwgPSB0Zi5sYXllcnMuZ3J1Q2VsbCh7dW5pdHM6IDJ9KTtcbiAqIGNvbnN0IGlucHV0ID0gdGYuaW5wdXQoe3NoYXBlOiBbMTBdfSk7XG4gKiBjb25zdCBvdXRwdXQgPSBjZWxsLmFwcGx5KGlucHV0KTtcbiAqXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShvdXRwdXQuc2hhcGUpKTtcbiAqIC8vIFtudWxsLCAxMF06IFRoaXMgaXMgdGhlIGNlbGwncyBvdXRwdXQgYXQgYSBzaW5nbGUgdGltZSBzdGVwLiBUaGUgMXN0XG4gKiAvLyBkaW1lbnNpb24gaXMgdGhlIHVua25vd24gYmF0Y2ggc2l6ZS5cbiAqIGBgYFxuICpcbiAqIEluc3RhbmNlKHMpIG9mIGBHUlVDZWxsYCBjYW4gYmUgdXNlZCB0byBjb25zdHJ1Y3QgYFJOTmAgbGF5ZXJzLiBUaGVcbiAqIG1vc3QgdHlwaWNhbCB1c2Ugb2YgdGhpcyB3b3JrZmxvdyBpcyB0byBjb21iaW5lIGEgbnVtYmVyIG9mIGNlbGxzIGludG8gYVxuICogc3RhY2tlZCBSTk4gY2VsbCAoaS5lLiwgYFN0YWNrZWRSTk5DZWxsYCBpbnRlcm5hbGx5KSBhbmQgdXNlIGl0IHRvIGNyZWF0ZSBhblxuICogUk5OLiBGb3IgZXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgY2VsbHMgPSBbXG4gKiAgIHRmLmxheWVycy5ncnVDZWxsKHt1bml0czogNH0pLFxuICogICB0Zi5sYXllcnMuZ3J1Q2VsbCh7dW5pdHM6IDh9KSxcbiAqIF07XG4gKiBjb25zdCBybm4gPSB0Zi5sYXllcnMucm5uKHtjZWxsOiBjZWxscywgcmV0dXJuU2VxdWVuY2VzOiB0cnVlfSk7XG4gKlxuICogLy8gQ3JlYXRlIGFuIGlucHV0IHdpdGggMTAgdGltZSBzdGVwcyBhbmQgYSBsZW5ndGgtMjAgdmVjdG9yIGF0IGVhY2ggc3RlcC5cbiAqIGNvbnN0IGlucHV0ID0gdGYuaW5wdXQoe3NoYXBlOiBbMTAsIDIwXX0pO1xuICogY29uc3Qgb3V0cHV0ID0gcm5uLmFwcGx5KGlucHV0KTtcbiAqXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShvdXRwdXQuc2hhcGUpKTtcbiAqIC8vIFtudWxsLCAxMCwgOF06IDFzdCBkaW1lbnNpb24gaXMgdW5rbm93biBiYXRjaCBzaXplOyAybmQgZGltZW5zaW9uIGlzIHRoZVxuICogLy8gc2FtZSBhcyB0aGUgc2VxdWVuY2UgbGVuZ3RoIG9mIGBpbnB1dGAsIGR1ZSB0byBgcmV0dXJuU2VxdWVuY2VzYDogYHRydWVgO1xuICogLy8gM3JkIGRpbWVuc2lvbiBpcyB0aGUgbGFzdCBgZ3J1Q2VsbGAncyBudW1iZXIgb2YgdW5pdHMuXG4gKiBgYGBcbiAqXG4gKiBUbyBjcmVhdGUgYW4gYFJOTmAgY29uc2lzdGluZyBvZiBvbmx5ICpvbmUqIGBHUlVDZWxsYCwgdXNlIHRoZVxuICogYHRmLmxheWVycy5ncnVgLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUmVjdXJyZW50JywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdydUNlbGwoYXJnczogR1JVQ2VsbExheWVyQXJncykge1xuICByZXR1cm4gbmV3IEdSVUNlbGwoYXJncyk7XG59XG5cbi8qKlxuICogTG9uZy1TaG9ydCBUZXJtIE1lbW9yeSBsYXllciAtIEhvY2hyZWl0ZXIgMTk5Ny5cbiAqXG4gKiBUaGlzIGlzIGFuIGBSTk5gIGxheWVyIGNvbnNpc3Rpbmcgb2Ygb25lIGBMU1RNQ2VsbGAuIEhvd2V2ZXIsIHVubGlrZVxuICogdGhlIHVuZGVybHlpbmcgYExTVE1DZWxsYCwgdGhlIGBhcHBseWAgbWV0aG9kIG9mIGBMU1RNYCBvcGVyYXRlc1xuICogb24gYSBzZXF1ZW5jZSBvZiBpbnB1dHMuIFRoZSBzaGFwZSBvZiB0aGUgaW5wdXQgKG5vdCBpbmNsdWRpbmcgdGhlIGZpcnN0LFxuICogYmF0Y2ggZGltZW5zaW9uKSBuZWVkcyB0byBiZSBhdCBsZWFzdCAyLUQsIHdpdGggdGhlIGZpcnN0IGRpbWVuc2lvbiBiZWluZ1xuICogdGltZSBzdGVwcy4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGxzdG0gPSB0Zi5sYXllcnMubHN0bSh7dW5pdHM6IDgsIHJldHVyblNlcXVlbmNlczogdHJ1ZX0pO1xuICpcbiAqIC8vIENyZWF0ZSBhbiBpbnB1dCB3aXRoIDEwIHRpbWUgc3RlcHMuXG4gKiBjb25zdCBpbnB1dCA9IHRmLmlucHV0KHtzaGFwZTogWzEwLCAyMF19KTtcbiAqIGNvbnN0IG91dHB1dCA9IGxzdG0uYXBwbHkoaW5wdXQpO1xuICpcbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dHB1dC5zaGFwZSkpO1xuICogLy8gW251bGwsIDEwLCA4XTogMXN0IGRpbWVuc2lvbiBpcyB1bmtub3duIGJhdGNoIHNpemU7IDJuZCBkaW1lbnNpb24gaXMgdGhlXG4gKiAvLyBzYW1lIGFzIHRoZSBzZXF1ZW5jZSBsZW5ndGggb2YgYGlucHV0YCwgZHVlIHRvIGByZXR1cm5TZXF1ZW5jZXNgOiBgdHJ1ZWA7XG4gKiAvLyAzcmQgZGltZW5zaW9uIGlzIHRoZSBgTFNUTUNlbGxgJ3MgbnVtYmVyIG9mIHVuaXRzLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUmVjdXJyZW50JywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGxzdG0oYXJnczogTFNUTUxheWVyQXJncykge1xuICByZXR1cm4gbmV3IExTVE0oYXJncyk7XG59XG5cbi8qKlxuICogQ2VsbCBjbGFzcyBmb3IgYExTVE1gLlxuICpcbiAqIGBMU1RNQ2VsbGAgaXMgZGlzdGluY3QgZnJvbSB0aGUgYFJOTmAgc3ViY2xhc3MgYExTVE1gIGluIHRoYXQgaXRzXG4gKiBgYXBwbHlgIG1ldGhvZCB0YWtlcyB0aGUgaW5wdXQgZGF0YSBvZiBvbmx5IGEgc2luZ2xlIHRpbWUgc3RlcCBhbmQgcmV0dXJuc1xuICogdGhlIGNlbGwncyBvdXRwdXQgYXQgdGhlIHRpbWUgc3RlcCwgd2hpbGUgYExTVE1gIHRha2VzIHRoZSBpbnB1dCBkYXRhXG4gKiBvdmVyIGEgbnVtYmVyIG9mIHRpbWUgc3RlcHMuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBjZWxsID0gdGYubGF5ZXJzLmxzdG1DZWxsKHt1bml0czogMn0pO1xuICogY29uc3QgaW5wdXQgPSB0Zi5pbnB1dCh7c2hhcGU6IFsxMF19KTtcbiAqIGNvbnN0IG91dHB1dCA9IGNlbGwuYXBwbHkoaW5wdXQpO1xuICpcbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dHB1dC5zaGFwZSkpO1xuICogLy8gW251bGwsIDEwXTogVGhpcyBpcyB0aGUgY2VsbCdzIG91dHB1dCBhdCBhIHNpbmdsZSB0aW1lIHN0ZXAuIFRoZSAxc3RcbiAqIC8vIGRpbWVuc2lvbiBpcyB0aGUgdW5rbm93biBiYXRjaCBzaXplLlxuICogYGBgXG4gKlxuICogSW5zdGFuY2Uocykgb2YgYExTVE1DZWxsYCBjYW4gYmUgdXNlZCB0byBjb25zdHJ1Y3QgYFJOTmAgbGF5ZXJzLiBUaGVcbiAqIG1vc3QgdHlwaWNhbCB1c2Ugb2YgdGhpcyB3b3JrZmxvdyBpcyB0byBjb21iaW5lIGEgbnVtYmVyIG9mIGNlbGxzIGludG8gYVxuICogc3RhY2tlZCBSTk4gY2VsbCAoaS5lLiwgYFN0YWNrZWRSTk5DZWxsYCBpbnRlcm5hbGx5KSBhbmQgdXNlIGl0IHRvIGNyZWF0ZSBhblxuICogUk5OLiBGb3IgZXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgY2VsbHMgPSBbXG4gKiAgIHRmLmxheWVycy5sc3RtQ2VsbCh7dW5pdHM6IDR9KSxcbiAqICAgdGYubGF5ZXJzLmxzdG1DZWxsKHt1bml0czogOH0pLFxuICogXTtcbiAqIGNvbnN0IHJubiA9IHRmLmxheWVycy5ybm4oe2NlbGw6IGNlbGxzLCByZXR1cm5TZXF1ZW5jZXM6IHRydWV9KTtcbiAqXG4gKiAvLyBDcmVhdGUgYW4gaW5wdXQgd2l0aCAxMCB0aW1lIHN0ZXBzIGFuZCBhIGxlbmd0aC0yMCB2ZWN0b3IgYXQgZWFjaCBzdGVwLlxuICogY29uc3QgaW5wdXQgPSB0Zi5pbnB1dCh7c2hhcGU6IFsxMCwgMjBdfSk7XG4gKiBjb25zdCBvdXRwdXQgPSBybm4uYXBwbHkoaW5wdXQpO1xuICpcbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dHB1dC5zaGFwZSkpO1xuICogLy8gW251bGwsIDEwLCA4XTogMXN0IGRpbWVuc2lvbiBpcyB1bmtub3duIGJhdGNoIHNpemU7IDJuZCBkaW1lbnNpb24gaXMgdGhlXG4gKiAvLyBzYW1lIGFzIHRoZSBzZXF1ZW5jZSBsZW5ndGggb2YgYGlucHV0YCwgZHVlIHRvIGByZXR1cm5TZXF1ZW5jZXNgOiBgdHJ1ZWA7XG4gKiAvLyAzcmQgZGltZW5zaW9uIGlzIHRoZSBsYXN0IGBsc3RtQ2VsbGAncyBudW1iZXIgb2YgdW5pdHMuXG4gKiBgYGBcbiAqXG4gKiBUbyBjcmVhdGUgYW4gYFJOTmAgY29uc2lzdGluZyBvZiBvbmx5ICpvbmUqIGBMU1RNQ2VsbGAsIHVzZSB0aGVcbiAqIGB0Zi5sYXllcnMubHN0bWAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdSZWN1cnJlbnQnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gbHN0bUNlbGwoYXJnczogTFNUTUNlbGxMYXllckFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBMU1RNQ2VsbChhcmdzKTtcbn1cblxuLyoqXG4gKiBGdWxseS1jb25uZWN0ZWQgUk5OIHdoZXJlIHRoZSBvdXRwdXQgaXMgdG8gYmUgZmVkIGJhY2sgdG8gaW5wdXQuXG4gKlxuICogVGhpcyBpcyBhbiBgUk5OYCBsYXllciBjb25zaXN0aW5nIG9mIG9uZSBgU2ltcGxlUk5OQ2VsbGAuIEhvd2V2ZXIsIHVubGlrZVxuICogdGhlIHVuZGVybHlpbmcgYFNpbXBsZVJOTkNlbGxgLCB0aGUgYGFwcGx5YCBtZXRob2Qgb2YgYFNpbXBsZVJOTmAgb3BlcmF0ZXNcbiAqIG9uIGEgc2VxdWVuY2Ugb2YgaW5wdXRzLiBUaGUgc2hhcGUgb2YgdGhlIGlucHV0IChub3QgaW5jbHVkaW5nIHRoZSBmaXJzdCxcbiAqIGJhdGNoIGRpbWVuc2lvbikgbmVlZHMgdG8gYmUgYXQgbGVhc3QgMi1ELCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYmVpbmdcbiAqIHRpbWUgc3RlcHMuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBybm4gPSB0Zi5sYXllcnMuc2ltcGxlUk5OKHt1bml0czogOCwgcmV0dXJuU2VxdWVuY2VzOiB0cnVlfSk7XG4gKlxuICogLy8gQ3JlYXRlIGFuIGlucHV0IHdpdGggMTAgdGltZSBzdGVwcy5cbiAqIGNvbnN0IGlucHV0ID0gdGYuaW5wdXQoe3NoYXBlOiBbMTAsIDIwXX0pO1xuICogY29uc3Qgb3V0cHV0ID0gcm5uLmFwcGx5KGlucHV0KTtcbiAqXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShvdXRwdXQuc2hhcGUpKTtcbiAqIC8vIFtudWxsLCAxMCwgOF06IDFzdCBkaW1lbnNpb24gaXMgdW5rbm93biBiYXRjaCBzaXplOyAybmQgZGltZW5zaW9uIGlzIHRoZVxuICogLy8gc2FtZSBhcyB0aGUgc2VxdWVuY2UgbGVuZ3RoIG9mIGBpbnB1dGAsIGR1ZSB0byBgcmV0dXJuU2VxdWVuY2VzYDogYHRydWVgO1xuICogLy8gM3JkIGRpbWVuc2lvbiBpcyB0aGUgYFNpbXBsZVJOTkNlbGxgJ3MgbnVtYmVyIG9mIHVuaXRzLlxuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdSZWN1cnJlbnQnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc2ltcGxlUk5OKGFyZ3M6IFNpbXBsZVJOTkxheWVyQXJncykge1xuICByZXR1cm4gbmV3IFNpbXBsZVJOTihhcmdzKTtcbn1cblxuLyoqXG4gKiBDZWxsIGNsYXNzIGZvciBgU2ltcGxlUk5OYC5cbiAqXG4gKiBgU2ltcGxlUk5OQ2VsbGAgaXMgZGlzdGluY3QgZnJvbSB0aGUgYFJOTmAgc3ViY2xhc3MgYFNpbXBsZVJOTmAgaW4gdGhhdCBpdHNcbiAqIGBhcHBseWAgbWV0aG9kIHRha2VzIHRoZSBpbnB1dCBkYXRhIG9mIG9ubHkgYSBzaW5nbGUgdGltZSBzdGVwIGFuZCByZXR1cm5zXG4gKiB0aGUgY2VsbCdzIG91dHB1dCBhdCB0aGUgdGltZSBzdGVwLCB3aGlsZSBgU2ltcGxlUk5OYCB0YWtlcyB0aGUgaW5wdXQgZGF0YVxuICogb3ZlciBhIG51bWJlciBvZiB0aW1lIHN0ZXBzLiBGb3IgZXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgY2VsbCA9IHRmLmxheWVycy5zaW1wbGVSTk5DZWxsKHt1bml0czogMn0pO1xuICogY29uc3QgaW5wdXQgPSB0Zi5pbnB1dCh7c2hhcGU6IFsxMF19KTtcbiAqIGNvbnN0IG91dHB1dCA9IGNlbGwuYXBwbHkoaW5wdXQpO1xuICpcbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dHB1dC5zaGFwZSkpO1xuICogLy8gW251bGwsIDEwXTogVGhpcyBpcyB0aGUgY2VsbCdzIG91dHB1dCBhdCBhIHNpbmdsZSB0aW1lIHN0ZXAuIFRoZSAxc3RcbiAqIC8vIGRpbWVuc2lvbiBpcyB0aGUgdW5rbm93biBiYXRjaCBzaXplLlxuICogYGBgXG4gKlxuICogSW5zdGFuY2Uocykgb2YgYFNpbXBsZVJOTkNlbGxgIGNhbiBiZSB1c2VkIHRvIGNvbnN0cnVjdCBgUk5OYCBsYXllcnMuIFRoZVxuICogbW9zdCB0eXBpY2FsIHVzZSBvZiB0aGlzIHdvcmtmbG93IGlzIHRvIGNvbWJpbmUgYSBudW1iZXIgb2YgY2VsbHMgaW50byBhXG4gKiBzdGFja2VkIFJOTiBjZWxsIChpLmUuLCBgU3RhY2tlZFJOTkNlbGxgIGludGVybmFsbHkpIGFuZCB1c2UgaXQgdG8gY3JlYXRlIGFuXG4gKiBSTk4uIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBjZWxscyA9IFtcbiAqICAgdGYubGF5ZXJzLnNpbXBsZVJOTkNlbGwoe3VuaXRzOiA0fSksXG4gKiAgIHRmLmxheWVycy5zaW1wbGVSTk5DZWxsKHt1bml0czogOH0pLFxuICogXTtcbiAqIGNvbnN0IHJubiA9IHRmLmxheWVycy5ybm4oe2NlbGw6IGNlbGxzLCByZXR1cm5TZXF1ZW5jZXM6IHRydWV9KTtcbiAqXG4gKiAvLyBDcmVhdGUgYW4gaW5wdXQgd2l0aCAxMCB0aW1lIHN0ZXBzIGFuZCBhIGxlbmd0aC0yMCB2ZWN0b3IgYXQgZWFjaCBzdGVwLlxuICogY29uc3QgaW5wdXQgPSB0Zi5pbnB1dCh7c2hhcGU6IFsxMCwgMjBdfSk7XG4gKiBjb25zdCBvdXRwdXQgPSBybm4uYXBwbHkoaW5wdXQpO1xuICpcbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dHB1dC5zaGFwZSkpO1xuICogLy8gW251bGwsIDEwLCA4XTogMXN0IGRpbWVuc2lvbiBpcyB1bmtub3duIGJhdGNoIHNpemU7IDJuZCBkaW1lbnNpb24gaXMgdGhlXG4gKiAvLyBzYW1lIGFzIHRoZSBzZXF1ZW5jZSBsZW5ndGggb2YgYGlucHV0YCwgZHVlIHRvIGByZXR1cm5TZXF1ZW5jZXNgOiBgdHJ1ZWA7XG4gKiAvLyAzcmQgZGltZW5zaW9uIGlzIHRoZSBsYXN0IGBTaW1wbGVSTk5DZWxsYCdzIG51bWJlciBvZiB1bml0cy5cbiAqIGBgYFxuICpcbiAqIFRvIGNyZWF0ZSBhbiBgUk5OYCBjb25zaXN0aW5nIG9mIG9ubHkgKm9uZSogYFNpbXBsZVJOTkNlbGxgLCB1c2UgdGhlXG4gKiBgdGYubGF5ZXJzLnNpbXBsZVJOTmAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdSZWN1cnJlbnQnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc2ltcGxlUk5OQ2VsbChhcmdzOiBTaW1wbGVSTk5DZWxsTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgU2ltcGxlUk5OQ2VsbChhcmdzKTtcbn1cblxuLyoqXG4gKiBDb252b2x1dGlvbmFsIExTVE0gbGF5ZXIgLSBYaW5namlhbiBTaGkgMjAxNS5cbiAqXG4gKiBUaGlzIGlzIGFuIGBDb252Uk5OMkRgIGxheWVyIGNvbnNpc3Rpbmcgb2Ygb25lIGBDb252TFNUTTJEQ2VsbGAuIEhvd2V2ZXIsXG4gKiB1bmxpa2UgdGhlIHVuZGVybHlpbmcgYENvbnZMU1RNMkRDZWxsYCwgdGhlIGBhcHBseWAgbWV0aG9kIG9mIGBDb252TFNUTTJEYFxuICogb3BlcmF0ZXMgb24gYSBzZXF1ZW5jZSBvZiBpbnB1dHMuIFRoZSBzaGFwZSBvZiB0aGUgaW5wdXQgKG5vdCBpbmNsdWRpbmcgdGhlXG4gKiBmaXJzdCwgYmF0Y2ggZGltZW5zaW9uKSBuZWVkcyB0byBiZSA0LUQsIHdpdGggdGhlIGZpcnN0IGRpbWVuc2lvbiBiZWluZyB0aW1lXG4gKiBzdGVwcy4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGZpbHRlcnMgPSAzO1xuICogY29uc3Qga2VybmVsU2l6ZSA9IDM7XG4gKlxuICogY29uc3QgYmF0Y2hTaXplID0gNDtcbiAqIGNvbnN0IHNlcXVlbmNlTGVuZ3RoID0gMjtcbiAqIGNvbnN0IHNpemUgPSA1O1xuICogY29uc3QgY2hhbm5lbHMgPSAzO1xuICpcbiAqIGNvbnN0IGlucHV0U2hhcGUgPSBbYmF0Y2hTaXplLCBzZXF1ZW5jZUxlbmd0aCwgc2l6ZSwgc2l6ZSwgY2hhbm5lbHNdO1xuICogY29uc3QgaW5wdXQgPSB0Zi5vbmVzKGlucHV0U2hhcGUpO1xuICpcbiAqIGNvbnN0IGxheWVyID0gdGYubGF5ZXJzLmNvbnZMc3RtMmQoe2ZpbHRlcnMsIGtlcm5lbFNpemV9KTtcbiAqXG4gKiBjb25zdCBvdXRwdXQgPSBsYXllci5hcHBseShpbnB1dCk7XG4gKiBgYGBcbiAqL1xuLyoqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUmVjdXJyZW50JywgbmFtZXNwYWNlOiAnbGF5ZXJzJ30gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb252THN0bTJkKGFyZ3M6IENvbnZMU1RNMkRBcmdzKSB7XG4gIHJldHVybiBuZXcgQ29udkxTVE0yRChhcmdzKTtcbn1cblxuLyoqXG4gKiBDZWxsIGNsYXNzIGZvciBgQ29udkxTVE0yRGAuXG4gKlxuICogYENvbnZMU1RNMkRDZWxsYCBpcyBkaXN0aW5jdCBmcm9tIHRoZSBgQ29udlJOTjJEYCBzdWJjbGFzcyBgQ29udkxTVE0yRGAgaW5cbiAqIHRoYXQgaXRzIGBjYWxsYCBtZXRob2QgdGFrZXMgdGhlIGlucHV0IGRhdGEgb2Ygb25seSBhIHNpbmdsZSB0aW1lIHN0ZXAgYW5kXG4gKiByZXR1cm5zIHRoZSBjZWxsJ3Mgb3V0cHV0IGF0IHRoZSB0aW1lIHN0ZXAsIHdoaWxlIGBDb252TFNUTTJEYCB0YWtlcyB0aGVcbiAqIGlucHV0IGRhdGEgb3ZlciBhIG51bWJlciBvZiB0aW1lIHN0ZXBzLiBGb3IgZXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgZmlsdGVycyA9IDM7XG4gKiBjb25zdCBrZXJuZWxTaXplID0gMztcbiAqXG4gKiBjb25zdCBzZXF1ZW5jZUxlbmd0aCA9IDE7XG4gKiBjb25zdCBzaXplID0gNTtcbiAqIGNvbnN0IGNoYW5uZWxzID0gMztcbiAqXG4gKiBjb25zdCBpbnB1dFNoYXBlID0gW3NlcXVlbmNlTGVuZ3RoLCBzaXplLCBzaXplLCBjaGFubmVsc107XG4gKiBjb25zdCBpbnB1dCA9IHRmLm9uZXMoaW5wdXRTaGFwZSk7XG4gKlxuICogY29uc3QgY2VsbCA9IHRmLmxheWVycy5jb252THN0bTJkQ2VsbCh7ZmlsdGVycywga2VybmVsU2l6ZX0pO1xuICpcbiAqIGNlbGwuYnVpbGQoaW5wdXQuc2hhcGUpO1xuICpcbiAqIGNvbnN0IG91dHB1dFNpemUgPSBzaXplIC0ga2VybmVsU2l6ZSArIDE7XG4gKiBjb25zdCBvdXRTaGFwZSA9IFtzZXF1ZW5jZUxlbmd0aCwgb3V0cHV0U2l6ZSwgb3V0cHV0U2l6ZSwgZmlsdGVyc107XG4gKlxuICogY29uc3QgaW5pdGlhbEggPSB0Zi56ZXJvcyhvdXRTaGFwZSk7XG4gKiBjb25zdCBpbml0aWFsQyA9IHRmLnplcm9zKG91dFNoYXBlKTtcbiAqXG4gKiBjb25zdCBbbywgaCwgY10gPSBjZWxsLmNhbGwoW2lucHV0LCBpbml0aWFsSCwgaW5pdGlhbENdLCB7fSk7XG4gKiBgYGBcbiAqL1xuLyoqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnUmVjdXJyZW50JywgbmFtZXNwYWNlOiAnbGF5ZXJzJ30gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb252THN0bTJkQ2VsbChhcmdzOiBDb252TFNUTTJEQ2VsbEFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBDb252TFNUTTJEQ2VsbChhcmdzKTtcbn1cblxuLyoqXG4gKiBCYXNlIGNsYXNzIGZvciByZWN1cnJlbnQgbGF5ZXJzLlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICAzRCB0ZW5zb3Igd2l0aCBzaGFwZSBgW2JhdGNoU2l6ZSwgdGltZVN0ZXBzLCBpbnB1dERpbV1gLlxuICpcbiAqIE91dHB1dCBzaGFwZTpcbiAqICAgLSBpZiBgcmV0dXJuU3RhdGVgLCBhbiBBcnJheSBvZiB0ZW5zb3JzIChpLmUuLCBgdGYuVGVuc29yYHMpLiBUaGUgZmlyc3RcbiAqICAgICB0ZW5zb3IgaXMgdGhlIG91dHB1dC4gVGhlIHJlbWFpbmluZyB0ZW5zb3JzIGFyZSB0aGUgc3RhdGVzIGF0IHRoZVxuICogICAgIGxhc3QgdGltZSBzdGVwLCBlYWNoIHdpdGggc2hhcGUgYFtiYXRjaFNpemUsIHVuaXRzXWAuXG4gKiAgIC0gaWYgYHJldHVyblNlcXVlbmNlc2AsIHRoZSBvdXRwdXQgd2lsbCBoYXZlIHNoYXBlXG4gKiAgICAgYFtiYXRjaFNpemUsIHRpbWVTdGVwcywgdW5pdHNdYC5cbiAqICAgLSBlbHNlLCB0aGUgb3V0cHV0IHdpbGwgaGF2ZSBzaGFwZSBgW2JhdGNoU2l6ZSwgdW5pdHNdYC5cbiAqXG4gKiBNYXNraW5nOlxuICogICBUaGlzIGxheWVyIHN1cHBvcnRzIG1hc2tpbmcgZm9yIGlucHV0IGRhdGEgd2l0aCBhIHZhcmlhYmxlIG51bWJlclxuICogICBvZiB0aW1lc3RlcHMuIFRvIGludHJvZHVjZSBtYXNrcyB0byB5b3VyIGRhdGEsXG4gKiAgIHVzZSBhbiBlbWJlZGRpbmcgbGF5ZXIgd2l0aCB0aGUgYG1hc2tfemVyb2AgcGFyYW1ldGVyXG4gKiAgIHNldCB0byBgVHJ1ZWAuXG4gKlxuICogTm90ZXMgb24gdXNpbmcgc3RhdGVmdWxuZXNzIGluIFJOTnM6XG4gKiAgIFlvdSBjYW4gc2V0IFJOTiBsYXllcnMgdG8gYmUgJ3N0YXRlZnVsJywgd2hpY2ggbWVhbnMgdGhhdCB0aGUgc3RhdGVzXG4gKiAgIGNvbXB1dGVkIGZvciB0aGUgc2FtcGxlcyBpbiBvbmUgYmF0Y2ggd2lsbCBiZSByZXVzZWQgYXMgaW5pdGlhbCBzdGF0ZXNcbiAqICAgZm9yIHRoZSBzYW1wbGVzIGluIHRoZSBuZXh0IGJhdGNoLiBUaGlzIGFzc3VtZXMgYSBvbmUtdG8tb25lIG1hcHBpbmdcbiAqICAgYmV0d2VlbiBzYW1wbGVzIGluIGRpZmZlcmVudCBzdWNjZXNzaXZlIGJhdGNoZXMuXG4gKlxuICogICBUbyBlbmFibGUgc3RhdGVmdWxuZXNzOlxuICogICAgIC0gc3BlY2lmeSBgc3RhdGVmdWw6IHRydWVgIGluIHRoZSBsYXllciBjb25zdHJ1Y3Rvci5cbiAqICAgICAtIHNwZWNpZnkgYSBmaXhlZCBiYXRjaCBzaXplIGZvciB5b3VyIG1vZGVsLCBieSBwYXNzaW5nXG4gKiAgICAgICBpZiBzZXF1ZW50aWFsIG1vZGVsOlxuICogICAgICAgICBgYmF0Y2hJbnB1dFNoYXBlPVsuLi5dYCB0byB0aGUgZmlyc3QgbGF5ZXIgaW4geW91ciBtb2RlbC5cbiAqICAgICAgIGVsc2UgZm9yIGZ1bmN0aW9uYWwgbW9kZWwgd2l0aCAxIG9yIG1vcmUgSW5wdXQgbGF5ZXJzOlxuICogICAgICAgICBgYmF0Y2hTaGFwZT1bLi4uXWAgdG8gYWxsIHRoZSBmaXJzdCBsYXllcnMgaW4geW91ciBtb2RlbC5cbiAqICAgICAgIFRoaXMgaXMgdGhlIGV4cGVjdGVkIHNoYXBlIG9mIHlvdXIgaW5wdXRzICppbmNsdWRpbmcgdGhlIGJhdGNoIHNpemUqLlxuICogICAgICAgSXQgc2hvdWxkIGJlIGEgdHVwbGUgb2YgaW50ZWdlcnMsIGUuZy4gYCgzMiwgMTAsIDEwMClgLlxuICogICAgIC0gc3BlY2lmeSBgc2h1ZmZsZT1GYWxzZWAgd2hlbiBjYWxsaW5nIGZpdCgpLlxuICpcbiAqICAgVG8gcmVzZXQgdGhlIHN0YXRlcyBvZiB5b3VyIG1vZGVsLCBjYWxsIGAucmVzZXRTdGF0ZXMoKWAgb24gZWl0aGVyXG4gKiAgIGEgc3BlY2lmaWMgbGF5ZXIsIG9yIG9uIHlvdXIgZW50aXJlIG1vZGVsLlxuICpcbiAqIE5vdGUgb24gc3BlY2lmeWluZyB0aGUgaW5pdGlhbCBzdGF0ZSBvZiBSTk5zXG4gKiAgIFlvdSBjYW4gc3BlY2lmeSB0aGUgaW5pdGlhbCBzdGF0ZSBvZiBSTk4gbGF5ZXJzIHN5bWJvbGljYWxseSBieVxuICogICBjYWxsaW5nIHRoZW0gd2l0aCB0aGUgb3B0aW9uIGBpbml0aWFsU3RhdGVgLiBUaGUgdmFsdWUgb2ZcbiAqICAgYGluaXRpYWxTdGF0ZWAgc2hvdWxkIGJlIGEgdGVuc29yIG9yIGxpc3Qgb2YgdGVuc29ycyByZXByZXNlbnRpbmdcbiAqICAgdGhlIGluaXRpYWwgc3RhdGUgb2YgdGhlIFJOTiBsYXllci5cbiAqXG4gKiAgIFlvdSBjYW4gc3BlY2lmeSB0aGUgaW5pdGlhbCBzdGF0ZSBvZiBSTk4gbGF5ZXJzIG51bWVyaWNhbGx5IGJ5XG4gKiAgIGNhbGxpbmcgYHJlc2V0U3RhdGVzYCB3aXRoIHRoZSBrZXl3b3JkIGFyZ3VtZW50IGBzdGF0ZXNgLiBUaGUgdmFsdWUgb2ZcbiAqICAgYHN0YXRlc2Agc2hvdWxkIGJlIGEgbnVtcHkgYXJyYXkgb3IgbGlzdCBvZiBudW1weSBhcnJheXMgcmVwcmVzZW50aW5nXG4gKiAgIHRoZSBpbml0aWFsIHN0YXRlIG9mIHRoZSBSTk4gbGF5ZXIuXG4gKlxuICogTm90ZSBvbiBwYXNzaW5nIGV4dGVybmFsIGNvbnN0YW50cyB0byBSTk5zXG4gKiAgIFlvdSBjYW4gcGFzcyBcImV4dGVybmFsXCIgY29uc3RhbnRzIHRvIHRoZSBjZWxsIHVzaW5nIHRoZSBgY29uc3RhbnRzYFxuICogICBrZXl3b3JkIGFyZ3VtZW50IG9mIGBSTk4uY2FsbGAgbWV0aG9kLiBUaGlzIHJlcXVpcmVzIHRoYXQgdGhlIGBjZWxsLmNhbGxgXG4gKiAgIG1ldGhvZCBhY2NlcHRzIHRoZSBzYW1lIGtleXdvcmQgYXJndW1lbnQgYGNvbnN0YW50c2AuIFN1Y2ggY29uc3RhbnRzXG4gKiAgIGNhbiBiZSB1c2VkIHRvIGNvbmRpdG9uIHRoZSBjZWxsIHRyYW5zZm9ybWF0aW9uIG9uIGFkZGl0aW9uYWwgc3RhdGljIGlucHV0c1xuICogICAobm90IGNoYW5naW5nIG92ZXIgdGltZSksIGEuay5hIGFuIGF0dGVudGlvbiBtZWNoYW5pc20uXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdSZWN1cnJlbnQnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gcm5uKGFyZ3M6IFJOTkxheWVyQXJncykge1xuICByZXR1cm4gbmV3IFJOTihhcmdzKTtcbn1cblxuLyoqXG4gKiBXcmFwcGVyIGFsbG93aW5nIGEgc3RhY2sgb2YgUk5OIGNlbGxzIHRvIGJlaGF2ZSBhcyBhIHNpbmdsZSBjZWxsLlxuICpcbiAqIFVzZWQgdG8gaW1wbGVtZW50IGVmZmljaWVudCBzdGFja2VkIFJOTnMuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdSZWN1cnJlbnQnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc3RhY2tlZFJOTkNlbGxzKGFyZ3M6IFN0YWNrZWRSTk5DZWxsc0FyZ3Mpe1xuICByZXR1cm4gbmV3IFN0YWNrZWRSTk5DZWxscyhhcmdzKTtcbn1cblxuLy8gV3JhcHBlciBMYXllcnMuXG5cbi8qKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ1dyYXBwZXInLCBuYW1lc3BhY2U6ICdsYXllcnMnfSAqL1xuZXhwb3J0IGZ1bmN0aW9uIGJpZGlyZWN0aW9uYWwoYXJnczogQmlkaXJlY3Rpb25hbExheWVyQXJncykge1xuICByZXR1cm4gbmV3IEJpZGlyZWN0aW9uYWwoYXJncyk7XG59XG5cbi8qKlxuICogVGhpcyB3cmFwcGVyIGFwcGxpZXMgYSBsYXllciB0byBldmVyeSB0ZW1wb3JhbCBzbGljZSBvZiBhbiBpbnB1dC5cbiAqXG4gKiBUaGUgaW5wdXQgc2hvdWxkIGJlIGF0IGxlYXN0IDNELCAgYW5kIHRoZSBkaW1lbnNpb24gb2YgdGhlIGluZGV4IGAxYCB3aWxsIGJlXG4gKiBjb25zaWRlcmVkIHRvIGJlIHRoZSB0ZW1wb3JhbCBkaW1lbnNpb24uXG4gKlxuICogQ29uc2lkZXIgYSBiYXRjaCBvZiAzMiBzYW1wbGVzLCB3aGVyZSBlYWNoIHNhbXBsZSBpcyBhIHNlcXVlbmNlIG9mIDEwIHZlY3RvcnNcbiAqIG9mIDE2IGRpbWVuc2lvbnMuIFRoZSBiYXRjaCBpbnB1dCBzaGFwZSBvZiB0aGUgbGF5ZXIgaXMgdGhlbiBgWzMyLCAgMTAsXG4gKiAxNl1gLCBhbmQgdGhlIGBpbnB1dFNoYXBlYCwgbm90IGluY2x1ZGluZyB0aGUgc2FtcGxlIGRpbWVuc2lvbiwgaXNcbiAqIGBbMTAsIDE2XWAuXG4gKlxuICogWW91IGNhbiB0aGVuIHVzZSBgVGltZURpc3RyaWJ1dGVkYCB0byBhcHBseSBhIGBEZW5zZWAgbGF5ZXIgdG8gZWFjaCBvZiB0aGUgMTBcbiAqIHRpbWVzdGVwcywgaW5kZXBlbmRlbnRseTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKCk7XG4gKiBtb2RlbC5hZGQodGYubGF5ZXJzLnRpbWVEaXN0cmlidXRlZCh7XG4gKiAgIGxheWVyOiB0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiA4fSksXG4gKiAgIGlucHV0U2hhcGU6IFsxMCwgMTZdLFxuICogfSkpO1xuICpcbiAqIC8vIE5vdyBtb2RlbC5vdXRwdXRTaGFwZSA9IFtudWxsLCAxMCwgOF0uXG4gKiAvLyBUaGUgb3V0cHV0IHdpbGwgdGhlbiBoYXZlIHNoYXBlIGBbMzIsIDEwLCA4XWAuXG4gKlxuICogLy8gSW4gc3Vic2VxdWVudCBsYXllcnMsIHRoZXJlIGlzIG5vIG5lZWQgZm9yIGBpbnB1dFNoYXBlYDpcbiAqIG1vZGVsLmFkZCh0Zi5sYXllcnMudGltZURpc3RyaWJ1dGVkKHtsYXllcjogdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMzJ9KX0pKTtcbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG1vZGVsLm91dHB1dHNbMF0uc2hhcGUpKTtcbiAqIC8vIE5vdyBtb2RlbC5vdXRwdXRTaGFwZSA9IFtudWxsLCAxMCwgMzJdLlxuICogYGBgXG4gKlxuICogVGhlIG91dHB1dCB3aWxsIHRoZW4gaGF2ZSBzaGFwZSBgWzMyLCAxMCwgMzJdYC5cbiAqXG4gKiBgVGltZURpc3RyaWJ1dGVkYCBjYW4gYmUgdXNlZCB3aXRoIGFyYml0cmFyeSBsYXllcnMsIG5vdCBqdXN0IGBEZW5zZWAsIGZvclxuICogaW5zdGFuY2UgYSBgQ29udjJEYCBsYXllci5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKCk7XG4gKiBtb2RlbC5hZGQodGYubGF5ZXJzLnRpbWVEaXN0cmlidXRlZCh7XG4gKiAgIGxheWVyOiB0Zi5sYXllcnMuY29udjJkKHtmaWx0ZXJzOiA2NCwga2VybmVsU2l6ZTogWzMsIDNdfSksXG4gKiAgIGlucHV0U2hhcGU6IFsxMCwgMjk5LCAyOTksIDNdLFxuICogfSkpO1xuICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkobW9kZWwub3V0cHV0c1swXS5zaGFwZSkpO1xuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdXcmFwcGVyJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHRpbWVEaXN0cmlidXRlZChhcmdzOiBXcmFwcGVyTGF5ZXJBcmdzKSB7XG4gIHJldHVybiBuZXcgVGltZURpc3RyaWJ1dGVkKGFyZ3MpO1xufVxuXG4vLyBBbGlhc2VzIGZvciBwb29saW5nLlxuZXhwb3J0IGNvbnN0IGdsb2JhbE1heFBvb2wxZCA9IGdsb2JhbE1heFBvb2xpbmcxZDtcbmV4cG9ydCBjb25zdCBnbG9iYWxNYXhQb29sMmQgPSBnbG9iYWxNYXhQb29saW5nMmQ7XG5leHBvcnQgY29uc3QgbWF4UG9vbDFkID0gbWF4UG9vbGluZzFkO1xuZXhwb3J0IGNvbnN0IG1heFBvb2wyZCA9IG1heFBvb2xpbmcyZDtcblxuZXhwb3J0IHtMYXllciwgUk5OLCBSTk5DZWxsLCBpbnB1dCAvKiBhbGlhcyBmb3IgdGYuaW5wdXQgKi99O1xuXG4vKipcbiAqIEFwcGx5IGFkZGl0aXZlIHplcm8tY2VudGVyZWQgR2F1c3NpYW4gbm9pc2UuXG4gKlxuICogQXMgaXQgaXMgYSByZWd1bGFyaXphdGlvbiBsYXllciwgaXQgaXMgb25seSBhY3RpdmUgYXQgdHJhaW5pbmcgdGltZS5cbiAqXG4gKiBUaGlzIGlzIHVzZWZ1bCB0byBtaXRpZ2F0ZSBvdmVyZml0dGluZ1xuICogKHlvdSBjb3VsZCBzZWUgaXQgYXMgYSBmb3JtIG9mIHJhbmRvbSBkYXRhIGF1Z21lbnRhdGlvbikuXG4gKiBHYXVzc2lhbiBOb2lzZSAoR1MpIGlzIGEgbmF0dXJhbCBjaG9pY2UgYXMgY29ycnVwdGlvbiBwcm9jZXNzXG4gKiBmb3IgcmVhbCB2YWx1ZWQgaW5wdXRzLlxuICpcbiAqICMgQXJndW1lbnRzXG4gKiAgICAgc3RkZGV2OiBmbG9hdCwgc3RhbmRhcmQgZGV2aWF0aW9uIG9mIHRoZSBub2lzZSBkaXN0cmlidXRpb24uXG4gKlxuICogIyBJbnB1dCBzaGFwZVxuICogICAgICAgICBBcmJpdHJhcnkuIFVzZSB0aGUga2V5d29yZCBhcmd1bWVudCBgaW5wdXRfc2hhcGVgXG4gKiAgICAgICAgICh0dXBsZSBvZiBpbnRlZ2VycywgZG9lcyBub3QgaW5jbHVkZSB0aGUgc2FtcGxlcyBheGlzKVxuICogICAgICAgICB3aGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlIGZpcnN0IGxheWVyIGluIGEgbW9kZWwuXG4gKlxuICogIyBPdXRwdXQgc2hhcGVcbiAqICAgICAgICAgU2FtZSBzaGFwZSBhcyBpbnB1dC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ05vaXNlJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdhdXNzaWFuTm9pc2UoYXJnczogR2F1c3NpYW5Ob2lzZUFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBHYXVzc2lhbk5vaXNlKGFyZ3MpO1xufVxuXG4vKipcbiAqIEFwcGx5IG11bHRpcGxpY2F0aXZlIDEtY2VudGVyZWQgR2F1c3NpYW4gbm9pc2UuXG4gKlxuICogQXMgaXQgaXMgYSByZWd1bGFyaXphdGlvbiBsYXllciwgaXQgaXMgb25seSBhY3RpdmUgYXQgdHJhaW5pbmcgdGltZS5cbiAqXG4gKiBBcmd1bWVudHM6XG4gKiAgIC0gYHJhdGVgOiBmbG9hdCwgZHJvcCBwcm9iYWJpbGl0eSAoYXMgd2l0aCBgRHJvcG91dGApLlxuICogICAgIFRoZSBtdWx0aXBsaWNhdGl2ZSBub2lzZSB3aWxsIGhhdmVcbiAqICAgICBzdGFuZGFyZCBkZXZpYXRpb24gYHNxcnQocmF0ZSAvICgxIC0gcmF0ZSkpYC5cbiAqXG4gKiBJbnB1dCBzaGFwZTpcbiAqICAgQXJiaXRyYXJ5LiBVc2UgdGhlIGtleXdvcmQgYXJndW1lbnQgYGlucHV0U2hhcGVgXG4gKiAgICh0dXBsZSBvZiBpbnRlZ2VycywgZG9lcyBub3QgaW5jbHVkZSB0aGUgc2FtcGxlcyBheGlzKVxuICogICB3aGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlIGZpcnN0IGxheWVyIGluIGEgbW9kZWwuXG4gKlxuICogT3V0cHV0IHNoYXBlOlxuICogICBTYW1lIHNoYXBlIGFzIGlucHV0LlxuICpcbiAqIFJlZmVyZW5jZXM6XG4gKiAgIC0gW0Ryb3BvdXQ6IEEgU2ltcGxlIFdheSB0byBQcmV2ZW50IE5ldXJhbCBOZXR3b3JrcyBmcm9tIE92ZXJmaXR0aW5nXShcbiAqICAgICAgaHR0cDovL3d3dy5jcy50b3JvbnRvLmVkdS9+cnNhbGFraHUvcGFwZXJzL3NyaXZhc3RhdmExNGEucGRmKVxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnTm9pc2UnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2F1c3NpYW5Ecm9wb3V0KGFyZ3M6IEdhdXNzaWFuRHJvcG91dEFyZ3MpIHtcbiAgcmV0dXJuIG5ldyBHYXVzc2lhbkRyb3BvdXQoYXJncyk7XG59XG5cbi8qKlxuICogQXBwbGllcyBBbHBoYSBEcm9wb3V0IHRvIHRoZSBpbnB1dC5cbiAqXG4gKiBBcyBpdCBpcyBhIHJlZ3VsYXJpemF0aW9uIGxheWVyLCBpdCBpcyBvbmx5IGFjdGl2ZSBhdCB0cmFpbmluZyB0aW1lLlxuICpcbiAqIEFscGhhIERyb3BvdXQgaXMgYSBgRHJvcG91dGAgdGhhdCBrZWVwcyBtZWFuIGFuZCB2YXJpYW5jZSBvZiBpbnB1dHNcbiAqIHRvIHRoZWlyIG9yaWdpbmFsIHZhbHVlcywgaW4gb3JkZXIgdG8gZW5zdXJlIHRoZSBzZWxmLW5vcm1hbGl6aW5nIHByb3BlcnR5XG4gKiBldmVuIGFmdGVyIHRoaXMgZHJvcG91dC5cbiAqIEFscGhhIERyb3BvdXQgZml0cyB3ZWxsIHRvIFNjYWxlZCBFeHBvbmVudGlhbCBMaW5lYXIgVW5pdHNcbiAqIGJ5IHJhbmRvbWx5IHNldHRpbmcgYWN0aXZhdGlvbnMgdG8gdGhlIG5lZ2F0aXZlIHNhdHVyYXRpb24gdmFsdWUuXG4gKlxuICogQXJndW1lbnRzOlxuICogICAtIGByYXRlYDogZmxvYXQsIGRyb3AgcHJvYmFiaWxpdHkgKGFzIHdpdGggYERyb3BvdXRgKS5cbiAqICAgICBUaGUgbXVsdGlwbGljYXRpdmUgbm9pc2Ugd2lsbCBoYXZlXG4gKiAgICAgc3RhbmRhcmQgZGV2aWF0aW9uIGBzcXJ0KHJhdGUgLyAoMSAtIHJhdGUpKWAuXG4gKiAgIC0gYG5vaXNlX3NoYXBlYDogQSAxLUQgYFRlbnNvcmAgb2YgdHlwZSBgaW50MzJgLCByZXByZXNlbnRpbmcgdGhlXG4gKiAgICAgc2hhcGUgZm9yIHJhbmRvbWx5IGdlbmVyYXRlZCBrZWVwL2Ryb3AgZmxhZ3MuXG4gKlxuICogSW5wdXQgc2hhcGU6XG4gKiAgIEFyYml0cmFyeS4gVXNlIHRoZSBrZXl3b3JkIGFyZ3VtZW50IGBpbnB1dFNoYXBlYFxuICogICAodHVwbGUgb2YgaW50ZWdlcnMsIGRvZXMgbm90IGluY2x1ZGUgdGhlIHNhbXBsZXMgYXhpcylcbiAqICAgd2hlbiB1c2luZyB0aGlzIGxheWVyIGFzIHRoZSBmaXJzdCBsYXllciBpbiBhIG1vZGVsLlxuICpcbiAqIE91dHB1dCBzaGFwZTpcbiAqICAgU2FtZSBzaGFwZSBhcyBpbnB1dC5cbiAqXG4gKiBSZWZlcmVuY2VzOlxuICogICAtIFtTZWxmLU5vcm1hbGl6aW5nIE5ldXJhbCBOZXR3b3Jrc10oaHR0cHM6Ly9hcnhpdi5vcmcvYWJzLzE3MDYuMDI1MTUpXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdOb2lzZScsIG5hbWVzcGFjZTogJ2xheWVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhbHBoYURyb3BvdXQoYXJnczogQWxwaGFEcm9wb3V0QXJncykge1xuICByZXR1cm4gbmV3IEFscGhhRHJvcG91dChhcmdzKTtcbn1cblxuLyoqXG4gKiBNYXNrcyBhIHNlcXVlbmNlIGJ5IHVzaW5nIGEgbWFzayB2YWx1ZSB0byBza2lwIHRpbWVzdGVwcy5cbiAqXG4gKiBJZiBhbGwgZmVhdHVyZXMgZm9yIGEgZ2l2ZW4gc2FtcGxlIHRpbWVzdGVwIGFyZSBlcXVhbCB0byBgbWFza192YWx1ZWAsXG4gKiB0aGVuIHRoZSBzYW1wbGUgdGltZXN0ZXAgd2lsbCBiZSBtYXNrZWQgKHNraXBwZWQpIGluIGFsbCBkb3duc3RyZWFtIGxheWVyc1xuICogKGFzIGxvbmcgYXMgdGhleSBzdXBwb3J0IG1hc2tpbmcpLlxuICpcbiAqIElmIGFueSBkb3duc3RyZWFtIGxheWVyIGRvZXMgbm90IHN1cHBvcnQgbWFza2luZyB5ZXQgcmVjZWl2ZXMgc3VjaFxuICogYW4gaW5wdXQgbWFzaywgYW4gZXhjZXB0aW9uIHdpbGwgYmUgcmFpc2VkLlxuICpcbiAqIEFyZ3VtZW50czpcbiAqICAgLSBgbWFza1ZhbHVlYDogRWl0aGVyIE5vbmUgb3IgbWFzayB2YWx1ZSB0byBza2lwLlxuICpcbiAqIElucHV0IHNoYXBlOlxuICogICBBcmJpdHJhcnkuIFVzZSB0aGUga2V5d29yZCBhcmd1bWVudCBgaW5wdXRTaGFwZWBcbiAqICAgKHR1cGxlIG9mIGludGVnZXJzLCBkb2VzIG5vdCBpbmNsdWRlIHRoZSBzYW1wbGVzIGF4aXMpXG4gKiAgIHdoZW4gdXNpbmcgdGhpcyBsYXllciBhcyB0aGUgZmlyc3QgbGF5ZXIgaW4gYSBtb2RlbC5cbiAqXG4gKiBPdXRwdXQgc2hhcGU6XG4gKiAgIFNhbWUgc2hhcGUgYXMgaW5wdXQuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0xheWVycycsIHN1YmhlYWRpbmc6ICdNYXNrJywgbmFtZXNwYWNlOiAnbGF5ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1hc2tpbmcoYXJncz86IE1hc2tpbmdBcmdzKSB7XG4gIHJldHVybiBuZXcgTWFza2luZyhhcmdzKTtcbn1cbiJdfQ==
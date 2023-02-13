/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original Source: engine/training.py */
import * as tfc from '@tensorflow/tfjs-core';
import { io, Optimizer, scalar, serialization, Tensor, tensor1d, util } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { nameScope } from '../common';
import { NotImplementedError, RuntimeError, ValueError } from '../errors';
import { deserialize } from '../layers/serialization';
import * as losses from '../losses';
import * as Metrics from '../metrics';
import * as optimizers from '../optimizers';
import { checkUserDefinedMetadata } from '../user_defined_metadata';
import { count, pyListRepeat, singletonOrArray, toCamelCase, toSnakeCase, unique } from '../utils/generic_utils';
import { printSummary } from '../utils/layer_utils';
import { range } from '../utils/math_utils';
import { convertPythonicToTs } from '../utils/serialization_utils';
import { version } from '../version';
import { Container } from './container';
import { execute, FeedDict } from './executor';
import { evaluateDataset, fitDataset } from './training_dataset';
import { checkBatchSize, disposeNewTensors, ensureTensorsRank2OrHigher, fitTensors, makeBatches, sliceArrays, sliceArraysByIndices } from './training_tensors';
import { computeWeightedLoss, standardizeClassWeights, standardizeWeights } from './training_utils';
/**
 * Helper function for polymorphic input data: 1. singleton Tensor.
 */
export function isDataTensor(x) {
    return x instanceof Tensor;
}
/**
 * Helper function for polymorphic input data: 2. Array of Tensor.
 */
export function isDataArray(x) {
    return Array.isArray(x);
}
/**
 * Helper function for polymorphic input data: 3. "dict" of Tensor.
 */
export function isDataDict(x) {
    return !isDataTensor(x) && !isDataArray(x);
}
/**
 * Normalizes inputs and targets provided by users.
 * @param data User-provided input data (polymorphic).
 * @param names An Array of expected Tensor names.
 * @param shapes Optional Array of expected Tensor shapes.
 * @param checkBatchAxis Whether to check that the batch axis of the arrays
 *   match  the expected value found in `shapes`.
 * @param exceptionPrefix String prefix used for exception formatting.
 * @returns List of standardized input Tensors (one Tensor per model input).
 * @throws ValueError: in case of improperly formatted user data.
 */
export function standardizeInputData(data, names, shapes, checkBatchAxis = true, exceptionPrefix = '') {
    if (names == null || names.length === 0) {
        // Check for the case where the model expected no data, but some data got
        // sent.
        if (data != null) {
            let gotUnexpectedData = false;
            if (isDataArray(data) && data.length > 0) {
                gotUnexpectedData = true;
            }
            else if (isDataDict(data)) {
                for (const key in data) {
                    if (data.hasOwnProperty(key)) {
                        gotUnexpectedData = true;
                        break;
                    }
                }
            }
            else {
                // `data` is a singleton Tensor in this case.
                gotUnexpectedData = true;
            }
            if (gotUnexpectedData) {
                throw new ValueError(`Error when checking model ${exceptionPrefix} expected no data, ` +
                    `but got ${data}`);
            }
        }
        return [];
    }
    if (data == null) {
        return names.map(name => null);
    }
    let arrays;
    if (isDataDict(data)) {
        data = data;
        arrays = [];
        for (const name of names) {
            if (data[name] == null) {
                throw new ValueError(`No data provided for "${name}". Need data for each key in: ` +
                    `${names}`);
            }
            arrays.push(data[name]);
        }
    }
    else if (isDataArray(data)) {
        data = data;
        if (data.length !== names.length) {
            throw new ValueError(`Error when checking model ${exceptionPrefix}: the Array of ` +
                `Tensors that you are passing to your model is not the size the ` +
                `model expected. Expected to see ${names.length} Tensor(s), but ` +
                `instead got the following list of Tensor(s): ${data}`);
        }
        arrays = data;
    }
    else {
        data = data;
        if (names.length > 1) {
            throw new ValueError(`The model ${exceptionPrefix} expects ${names.length} Tensor(s), ` +
                `but only received one Tensor. Found: Tensor with shape ${data.shape}`);
        }
        arrays = [data];
    }
    arrays = ensureTensorsRank2OrHigher(arrays);
    // Check shape compatibility.
    if (shapes != null) {
        for (let i = 0; i < names.length; ++i) {
            if (shapes[i] == null) {
                continue;
            }
            const array = arrays[i];
            if (array.shape.length !== shapes[i].length) {
                throw new ValueError(`Error when checking ${exceptionPrefix}: expected ${names[i]} ` +
                    `to have ${shapes[i].length} dimension(s). but got array with ` +
                    `shape ${array.shape}`);
            }
            for (let j = 0; j < shapes[i].length; ++j) {
                if (j === 0 && !checkBatchAxis) {
                    // Skip the first (batch) axis.
                    continue;
                }
                const dim = array.shape[j];
                const refDim = shapes[i][j];
                if (refDim != null && refDim >= 0 && dim !== refDim) {
                    throw new ValueError(`${exceptionPrefix} expected a batch of elements where each ` +
                        `example has shape [${shapes[i].slice(1, shapes[i].length)}] ` +
                        `(i.e.,tensor shape [*,${shapes[i].slice(1, shapes[i].length)}])` +
                        ` but the ${exceptionPrefix} received an input with ${array.shape[0]}` +
                        ` examples, each with shape [${array.shape.slice(1, array.shape.length)}]` +
                        ` (tensor shape [${array.shape}])`);
                }
            }
        }
    }
    return arrays;
}
/**
 * User input validation for Tensors.
 * @param inputs `Array` of `tf.Tensor`s for inputs.
 * @param targets `Array` of `tf.Tensor`s for targets.
 * @param weights Optional `Array` of `tf.Tensor`s for sample weights.
 * @throws ValueError: in case of incorrectly formatted data.
 */
export function checkArrayLengths(inputs, targets, weights) {
    const setX = unique(inputs.map(input => input.shape[0]));
    setX.sort();
    const setY = unique(targets.map(target => target.shape[0]));
    setY.sort();
    // TODO(cais): Check `weights` as well.
    if (setX.length > 1) {
        throw new ValueError(`All input Tensors (x) should have the same number of samples. ` +
            `Got array shapes: ` +
            `${JSON.stringify(inputs.map(input => input.shape))}`);
    }
    if (setY.length > 1) {
        throw new ValueError(`All target Tensors (y) should have the same number of samples. ` +
            `Got array shapes: ` +
            `${JSON.stringify(targets.map(target => target.shape))}`);
    }
    if (setX.length > 0 && setY.length > 0 && !util.arraysEqual(setX, setY)) {
        throw new ValueError(`Input Tensors should have the same number of samples as target ` +
            `Tensors. Found ${setX[0]} input sample(s) and ${setY[0]} target ` +
            `sample(s).`);
    }
}
/**
 * Validation on the compatibility of targes and loss functions.
 *
 * This helps prevent users from using loss functions incorrectly.
 *
 * @param targets `Array` of `tf.Tensor`s of targets.
 * @param lossFns `Array` of loss functions.
 * @param outputShapes `Array` of shapes of model outputs.
 */
function checkLossAndTargetCompatibility(targets, lossFns, outputShapes) {
    // TODO(cais): Dedicated test coverage?
    const keyLosses = [
        losses.meanSquaredError, losses.binaryCrossentropy,
        losses.categoricalCrossentropy
    ];
    for (let i = 0; i < targets.length; ++i) {
        const y = targets[i];
        const loss = lossFns[i];
        const shape = outputShapes[i];
        if (loss == null) {
            continue;
        }
        if (loss === losses.categoricalCrossentropy) {
            if (y.shape[y.shape.length - 1] === 1) {
                throw new ValueError(`You are passing a target array of shape ${y.shape} while using ` +
                    `a loss 'categorical_crossentropy'. 'categorical_crossentropy'` +
                    `expects targets to be binary matrices (1s and 0s) of shape ` +
                    `[samples, classes].`);
                // TODO(cais): Example code in error message.
            }
        }
        if (keyLosses.indexOf(loss) !== -1) {
            const slicedYShape = y.shape.slice(1);
            const slicedShape = shape.slice(1);
            for (let j = 0; j < slicedYShape.length; ++j) {
                const targetDim = slicedYShape[j];
                const outDim = slicedShape[j];
                if (outDim != null && targetDim !== outDim) {
                    throw new ValueError(`A target Tensor with shape ${y.shape} was passed for an ` +
                        `output of shape ${shape}, while using a loss function that ` +
                        `expects targets to have the same shape as the output.`);
                }
            }
        }
    }
}
/**
 * Check inputs provided by the user.
 *
 * Porting Note: This corresponds to _standardize_input_data() in Python
 *   Keras. Because of the strong typing in TF.js, we do not need to convert
 *   the data. Specifically:
 *   1) in PyKeras, `data` can be `DataFrame` instances from pandas, for
 *      example. We don't need to worry about that here because there is no
 *      widely popular javascript/typesdcript equivalent of pandas (so far).
 *      If one becomes available in the future, we can add support.
 *   2) in PyKeras, inputs can be Python dict. But here we are stipulating
 * that the data is either a single `tf.Tensor` or an Array of `tf.Tensor`s. We
 * may add support for `Object` data inputs in the future when the need
 * arises.
 *
 * Instead, we perform basic checks for number of parameters and shapes.
 *
 * @param data: The input data.
 * @param names: Name for the inputs, from the model.
 * @param shapes: Expected shapes for the input data, from the model.
 * @param checkBatchAxis: Whether the size along the batch axis (i.e., the
 *   first dimension) will be checked for matching.
 * @param exceptionPrefix: Execption prefix message, used in generating error
 *   messages.
 * @throws ValueError: on incorrect number of inputs or mismatches in shapes.
 */
function checkInputData(data, names, shapes, checkBatchAxis = true, exceptionPrefix = '') {
    let arrays;
    if (Array.isArray(data)) {
        if (data.length !== names.length) {
            throw new ValueError(`Error when checking model ${exceptionPrefix}: the Array of ` +
                `Tensors that you are passing to your model is not the size the ` +
                `the model expected. Expected to see ${names.length} Tensor(s),` +
                ` but instead got ${data.length} Tensors(s).`);
        }
        arrays = data;
    }
    else {
        if (names.length > 1) {
            throw new ValueError(`The model expects ${names.length} ${exceptionPrefix} Tensors, ` +
                `but only received one Tensor. Found: array with shape ` +
                `${JSON.stringify(data.shape)}.`);
        }
        arrays = [data];
    }
    if (shapes != null) {
        for (let i = 0; i < names.length; ++i) {
            if (shapes[i] == null) {
                continue;
            }
            const array = arrays[i];
            if (array.shape.length !== shapes[i].length) {
                throw new ValueError(`Error when checking ${exceptionPrefix}: expected ${names[i]} ` +
                    `to have ${shapes[i].length} dimension(s), but got array with ` +
                    `shape ${JSON.stringify(array.shape)}`);
            }
            for (let j = 0; j < shapes[i].length; ++j) {
                if (j === 0 && !checkBatchAxis) {
                    continue;
                }
                const dim = array.shape[j];
                const refDim = shapes[i][j];
                if (refDim != null) {
                    if (refDim !== dim) {
                        throw new ValueError(`Error when checking ${exceptionPrefix}: expected ` +
                            `${names[i]} to have shape ${JSON.stringify(shapes[i])} but ` +
                            `got array with shape ${JSON.stringify(array.shape)}.`);
                    }
                }
            }
        }
    }
}
/**
 * Maps metric functions to model outputs.
 * @param metrics An shortcut strings name, metric function, `Array` or dict
 *   (`Object`) of metric functions.
 * @param outputNames An `Array` of the names of model outputs.
 * @returns An `Array` (one entry per model output) of `Array` of metric
 *   functions. For instance, if the model has 2 outputs, and for the first
 *   output we want to compute `binaryAccuracy` and `binaryCrossentropy`,
 *   and just `binaryAccuracy` for the second output, the `Array` would look
 *   like:
 *     `[[binaryAccuracy, binaryCrossentropy],  [binaryAccuracy]]`
 * @throws TypeError: incompatible metrics format.
 */
export function collectMetrics(metrics, outputNames) {
    if (metrics == null || Array.isArray(metrics) && metrics.length === 0) {
        return outputNames.map(name => []);
    }
    let wrappedMetrics;
    if (typeof metrics === 'string' || typeof metrics === 'function') {
        wrappedMetrics = [metrics];
    }
    else if (Array.isArray(metrics) || typeof metrics === 'object') {
        wrappedMetrics = metrics;
    }
    else {
        throw new TypeError('Type of metrics argument not understood. Expected an string,' +
            `function, Array, or Object, found: ${metrics}`);
    }
    if (Array.isArray(wrappedMetrics)) {
        // We then apply all metrics to all outputs.
        return outputNames.map(name => wrappedMetrics);
    }
    else {
        // In this case, metrics is a dict.
        const nestedMetrics = [];
        for (const name of outputNames) {
            let outputMetrics = wrappedMetrics.hasOwnProperty(name) ? wrappedMetrics[name] : [];
            if (!Array.isArray(outputMetrics)) {
                outputMetrics = [outputMetrics];
            }
            nestedMetrics.push(outputMetrics);
        }
        return nestedMetrics;
    }
}
const LAYERS_MODEL_FORMAT_NAME = 'layers-model';
/**
 * A `tf.LayersModel` is a directed, acyclic graph of `tf.Layer`s plus methods
 * for training, evaluation, prediction and saving.
 *
 * `tf.LayersModel` is the basic unit of training, inference and evaluation in
 * TensorFlow.js. To create a `tf.LayersModel`, use `tf.LayersModel`.
 *
 * See also:
 *   `tf.Sequential`, `tf.loadLayersModel`.
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
export class LayersModel extends Container {
    constructor(args) {
        super(args);
        this.isTraining = false;
    }
    /**
     * Print a text summary of the model's layers.
     *
     * The summary includes
     * - Name and type of all layers that comprise the model.
     * - Output shape(s) of the layers
     * - Number of weight parameters of each layer
     * - If the model has non-sequential-like topology, the inputs each layer
     *   receives
     * - The total number of trainable and non-trainable parameters of the model.
     *
     * ```js
     * const input1 = tf.input({shape: [10]});
     * const input2 = tf.input({shape: [20]});
     * const dense1 = tf.layers.dense({units: 4}).apply(input1);
     * const dense2 = tf.layers.dense({units: 8}).apply(input2);
     * const concat = tf.layers.concatenate().apply([dense1, dense2]);
     * const output =
     *     tf.layers.dense({units: 3, activation: 'softmax'}).apply(concat);
     *
     * const model = tf.model({inputs: [input1, input2], outputs: output});
     * model.summary();
     * ```
     *
     * @param lineLength Custom line length, in number of characters.
     * @param positions Custom widths of each of the columns, as either
     *   fractions of `lineLength` (e.g., `[0.5, 0.75, 1]`) or absolute number
     *   of characters (e.g., `[30, 50, 65]`). Each number corresponds to
     *   right-most (i.e., ending) position of a column.
     * @param printFn Custom print function. Can be used to replace the default
     *   `console.log`. For example, you can use `x => {}` to mute the printed
     *   messages in the console.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    summary(lineLength, positions, printFn = console.log) {
        if (!this.built) {
            throw new ValueError(`This model has never been called, thus its weights have not been ` +
                `created yet. So no summary can be displayed. Build the model ` +
                `first (e.g., by calling it on some test data).`);
        }
        printSummary(this, lineLength, positions, printFn);
    }
    /**
     * Configures and prepares the model for training and evaluation.  Compiling
     * outfits the model with an optimizer, loss, and/or metrics.  Calling `fit`
     * or `evaluate` on an un-compiled model will throw an error.
     *
     * @param args a `ModelCompileArgs` specifying the loss, optimizer, and
     * metrics to be used for fitting and evaluating this model.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    compile(args) {
        if (args.loss == null) {
            args.loss = [];
        }
        this.loss = args.loss;
        if (typeof args.optimizer === 'string') {
            this.optimizer_ = optimizers.getOptimizer(args.optimizer);
            this.isOptimizerOwned = true;
        }
        else {
            if (!(args.optimizer instanceof Optimizer)) {
                throw new ValueError(`User-defined optimizer must be an instance of tf.Optimizer.`);
            }
            this.optimizer_ = args.optimizer;
            this.isOptimizerOwned = false;
        }
        // TODO(cais): Add lossWeights.
        // TODO(cais): Add sampleWeightMode.
        // Prepare loss functions.
        let lossFunctions = [];
        if (!Array.isArray(args.loss) && typeof args.loss !== 'string' &&
            typeof args.loss !== 'function') {
            args.loss = args.loss;
            for (const name in args.loss) {
                if (this.outputNames.indexOf(name) === -1) {
                    throw new ValueError(`Unknown entry in loss dictionary: "${name}". ` +
                        `Only expected the following keys: ${this.outputNames}`);
                }
            }
            for (const name of this.outputNames) {
                if (args.loss[name] == null) {
                    console.warn(`Output "${name}" is missing from loss dictionary. We assume ` +
                        `this was done on purpose, and we will not be expecting data ` +
                        `to be passed to ${name} during training`);
                }
                lossFunctions.push(losses.get(args.loss[name]));
            }
        }
        else if (Array.isArray(args.loss)) {
            if (args.loss.length !== this.outputs.length) {
                throw new ValueError(`When passing an Array as loss, it should have one entry per ` +
                    `model output. The model has ${this.outputs.length} output(s), ` +
                    `but you passed loss=${args.loss}.`);
            }
            const theLosses = args.loss;
            lossFunctions = theLosses.map(l => losses.get(l));
        }
        else {
            const lossFunction = losses.get(args.loss);
            this.outputs.forEach(_ => {
                lossFunctions.push(lossFunction);
            });
        }
        this.lossFunctions = lossFunctions;
        this.feedOutputNames = [];
        this.feedOutputShapes = [];
        this.feedLossFns = [];
        for (let i = 0; i < this.outputs.length; ++i) {
            // TODO(cais): Logic for skipping target(s).
            const shape = this.internalOutputShapes[i];
            const name = this.outputNames[i];
            this.feedOutputNames.push(name);
            this.feedOutputShapes.push(shape);
            this.feedLossFns.push(this.lossFunctions[i]);
        }
        // TODO(cais): Add logic for output masks.
        // TODO(cais): Add logic for sample weights.
        const skipTargetIndices = [];
        // Prepare metrics.
        this.metrics = args.metrics;
        // TODO(cais): Add weightedMetrics.
        this.metricsNames = ['loss'];
        this.metricsTensors = [];
        // Compute total loss.
        // Porting Note: In PyKeras, metrics_tensors are symbolic tensor objects.
        //   Here, metricsTensors are TypeScript functions. This difference is due
        //   to the difference in symbolic/imperative property of the backends.
        nameScope('loss', () => {
            for (let i = 0; i < this.outputs.length; ++i) {
                if (skipTargetIndices.indexOf(i) !== -1) {
                    continue;
                }
                // TODO(cais): Add weightedLoss, sampleWeight and mask.
                //   The following line should be weightedLoss
                const weightedLoss = this.lossFunctions[i];
                if (this.outputs.length > 1) {
                    this.metricsTensors.push([weightedLoss, i]);
                    this.metricsNames.push(this.outputNames[i] + '_loss');
                }
            }
            // Porting Note: Due to the imperative nature of the backend, we calculate
            //   the regularizer penalties in the totalLossFunction, instead of here.
        });
        const nestedMetrics = collectMetrics(args.metrics, this.outputNames);
        // TODO(cais): Add nestedWeightedMetrics.
        /**
         * Helper function used in loop below.
         */
        const appendMetric = (outputIndex, metricName, metricTensor) => {
            if (this.outputNames.length > 1) {
                metricName = this.outputNames[outputIndex] + '_' + metricName;
            }
            this.metricsNames.push(metricName);
            this.metricsTensors.push([metricTensor, outputIndex]);
        };
        nameScope('metric', () => {
            for (let i = 0; i < this.outputs.length; ++i) {
                if (skipTargetIndices.indexOf(i) !== -1) {
                    continue;
                }
                const outputMetrics = nestedMetrics[i];
                // TODO(cais): Add weights and outputWeightedMetrics.
                // TODO(cais): Add optional arg `weights` to the following function.
                const handleMetrics = (metrics) => {
                    const metricNamePrefix = '';
                    let metricName;
                    let accFn;
                    let weightedMetricFn;
                    //  TODO(cais): Use 'weights_' for weighted metrics.
                    for (const metric of metrics) {
                        if (typeof metric === 'string' &&
                            ['accuracy', 'acc', 'crossentropy', 'ce'].indexOf(metric) !==
                                -1) {
                            const outputShape = this.internalOutputShapes[i];
                            if (outputShape[outputShape.length - 1] === 1 ||
                                this.lossFunctions[i] === losses.binaryCrossentropy) {
                                // case: binary accuracy/crossentropy.
                                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                    accFn = Metrics.binaryAccuracy;
                                }
                                else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                    accFn = Metrics.binaryCrossentropy;
                                }
                            }
                            else if (this.lossFunctions[i] ===
                                losses.sparseCategoricalCrossentropy) {
                                // case: categorical accuracy / crossentropy with sparse
                                // targets.
                                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                    accFn = Metrics.sparseCategoricalAccuracy;
                                }
                                else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                    accFn = Metrics.sparseCategoricalCrossentropy;
                                }
                            }
                            else {
                                // case: categorical accuracy / crossentropy.
                                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                    accFn = Metrics.categoricalAccuracy;
                                }
                                else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                    accFn = Metrics.categoricalCrossentropy;
                                }
                            }
                            let suffix;
                            if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                suffix = 'acc';
                            }
                            else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                suffix = 'ce';
                            }
                            // TODO(cais): Add weighting actually.
                            weightedMetricFn = accFn;
                            metricName = metricNamePrefix + suffix;
                        }
                        else {
                            const metricFn = Metrics.get(metric);
                            // TODO(cais): Add weighting actually.
                            weightedMetricFn = metricFn;
                            metricName =
                                metricNamePrefix + Metrics.getLossOrMetricName(metric);
                        }
                        // TODO(cais): Add weighting and masking to metricResult.
                        let metricResult;
                        nameScope(metricName, () => {
                            metricResult = weightedMetricFn;
                        });
                        appendMetric(i, metricName, metricResult);
                    }
                };
                handleMetrics(outputMetrics);
                // TODO(cais): Call handleMetrics with weights.
            }
        });
        // Porting Notes: Given the imperative backend of tfjs-core,
        //   there is no need for constructing the symbolic graph and placeholders.
        this.collectedTrainableWeights = this.trainableWeights;
    }
    /**
     * Check trainable weights count consistency.
     *
     * This will raise a warning if `this.trainableWeights` and
     * `this.collectedTrainableWeights` are inconsistent (i.e., have different
     * numbers of parameters).
     * Inconsistency will typically arise when one modifies `model.trainable`
     * without calling `model.compile()` again.
     */
    checkTrainableWeightsConsistency() {
        if (this.collectedTrainableWeights == null) {
            return;
        }
        if (this.trainableWeights.length !==
            this.collectedTrainableWeights.length) {
            console.warn('Discrepancy between trainableweights and collected trainable ' +
                'weights. Did you set `model.trainable` without calling ' +
                '`model.compile()` afterwards?');
        }
    }
    /**
     * Returns the loss value & metrics values for the model in test mode.
     *
     * Loss and metrics are specified during `compile()`, which needs to happen
     * before calls to `evaluate()`.
     *
     * Computation is done in batches.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const result = model.evaluate(
     *     tf.ones([8, 10]), tf.ones([8, 1]), {batchSize: 4});
     * result.print();
     * ```
     *
     * @param x `tf.Tensor` of test data, or an `Array` of `tf.Tensor`s if the
     * model has multiple inputs.
     * @param y `tf.Tensor` of target data, or an `Array` of `tf.Tensor`s if the
     * model has multiple outputs.
     * @param args A `ModelEvaluateArgs`, containing optional fields.
     *
     * @return `Scalar` test loss (if the model has a single output and no
     *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
     *   and/or metrics). The attribute `model.metricsNames`
     *   will give you the display labels for the scalar outputs.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    evaluate(x, y, args = {}) {
        const batchSize = args.batchSize == null ? 32 : args.batchSize;
        checkBatchSize(batchSize);
        // TODO(cais): Standardize `config.sampleWeights` as well.
        // Validate user data.
        const checkBatchAxis = true;
        const standardizedOuts = this.standardizeUserDataXY(x, y, checkBatchAxis, batchSize);
        try {
            // TODO(cais): If uses `useLearningPhase`, set the corresponding element
            // of the input to 0.
            const ins = standardizedOuts[0].concat(standardizedOuts[1]);
            this.makeTestFunction();
            const f = this.testFunction;
            const testOuts = this.testLoop(f, ins, batchSize, args.verbose, args.steps);
            return singletonOrArray(testOuts);
        }
        finally {
            disposeNewTensors(standardizedOuts[0], x);
            disposeNewTensors(standardizedOuts[1], y);
        }
    }
    // TODO(cais): Add code snippet below once real dataset objects are
    //   available.
    /**
     * Evaluate model using a dataset object.
     *
     * Note: Unlike `evaluate()`, this method is asynchronous (`async`);
     *
     * @param dataset A dataset object. Its `iterator()` method is expected
     *   to generate a dataset iterator object, the `next()` method of which
     *   is expected to produce data batches for evaluation. The return value
     *   of the `next()` call ought to contain a boolean `done` field and a
     *   `value` field. The `value` field is expected to be an array of two
     *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
     *   case is for models with exactly one input and one output (e.g..
     *   a sequential model). The latter case is for models with multiple
     *   inputs and/or multiple outputs. Of the two items in the array, the
     *   first is the input feature(s) and the second is the output target(s).
     * @param args A configuration object for the dataset-based evaluation.
     * @returns Loss and metric values as an Array of `Scalar` objects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async evaluateDataset(dataset, args) {
        this.makeTestFunction();
        return evaluateDataset(this, dataset, args);
    }
    /**
     * Get number of samples provided for training, evaluation or prediction.
     *
     * @param ins Input `tf.Tensor`.
     * @param batchSize Integer batch size, optional.
     * @param steps Total number of steps (batches of samples) before
     * declaring loop finished. Optional.
     * @param stepsName The public API's parameter name for `steps`.
     * @returns Number of samples provided.
     */
    checkNumSamples(ins, batchSize, steps, stepsName = 'steps') {
        let numSamples;
        if (steps != null) {
            numSamples = null;
            if (batchSize != null) {
                throw new ValueError(`If ${stepsName} is set, batchSize must be null or undefined.` +
                    `Got batchSize = ${batchSize}`);
            }
        }
        else if (ins != null) {
            if (Array.isArray(ins)) {
                numSamples = ins[0].shape[0];
            }
            else {
                numSamples = ins.shape[0];
            }
        }
        else {
            throw new ValueError(`Either the input data should have a defined shape, or ` +
                `${stepsName} shoud be specified.`);
        }
        return numSamples;
    }
    /**
     * Execute internal tensors of the model with input data feed.
     * @param inputs Input data feed. Must match the inputs of the model.
     * @param outputs Names of the output tensors to be fetched. Must match
     *   names of the SymbolicTensors that belong to the graph.
     * @returns Fetched values for `outputs`.
     */
    execute(inputs, outputs) {
        if (Array.isArray(outputs) && outputs.length === 0) {
            throw new ValueError('`outputs` is an empty Array, which is not allowed.');
        }
        const outputsIsArray = Array.isArray(outputs);
        const outputNames = (outputsIsArray ? outputs : [outputs]);
        const outputSymbolicTensors = this.retrieveSymbolicTensors(outputNames);
        // Format the input into a FeedDict.
        const feedDict = new FeedDict();
        if (inputs instanceof Tensor) {
            inputs = [inputs];
        }
        if (Array.isArray(inputs)) {
            if (inputs.length !== this.inputs.length) {
                throw new ValueError(`The number of inputs provided (${inputs.length}) ` +
                    `does not match the number of inputs of this model ` +
                    `(${this.inputs.length}).`);
            }
            for (let i = 0; i < this.inputs.length; ++i) {
                feedDict.add(this.inputs[i], inputs[i]);
            }
        }
        else {
            for (const input of this.inputs) {
                const tensorValue = inputs[input.name];
                if (tensorValue == null) {
                    throw new ValueError(`No value is provided for the model's input ${input.name}`);
                }
                feedDict.add(input, tensorValue);
            }
        }
        // Run execution.
        const executeOutputs = execute(outputSymbolicTensors, feedDict);
        return outputsIsArray ? executeOutputs : executeOutputs[0];
    }
    /**
     * Retrieve the model's internal symbolic tensors from symbolic-tensor names.
     */
    retrieveSymbolicTensors(symbolicTensorNames) {
        const outputSymbolicTensors = pyListRepeat(null, symbolicTensorNames.length);
        let outputsRemaining = symbolicTensorNames.length;
        for (const layer of this.layers) {
            const layerOutputs = Array.isArray(layer.output) ? layer.output : [layer.output];
            const layerOutputNames = layerOutputs.map(output => output.name);
            for (let i = 0; i < symbolicTensorNames.length; ++i) {
                const index = layerOutputNames.indexOf(symbolicTensorNames[i]);
                if (index !== -1) {
                    outputSymbolicTensors[i] = layerOutputs[index];
                    outputsRemaining--;
                }
                if (outputsRemaining === 0) {
                    break;
                }
            }
            if (outputsRemaining === 0) {
                break;
            }
        }
        if (outputsRemaining > 0) {
            const remainingNames = [];
            outputSymbolicTensors.forEach((tensor, i) => {
                if (tensor == null) {
                    remainingNames.push(symbolicTensorNames[i]);
                }
            });
            throw new ValueError(`Cannot find SymbolicTensors for output name(s): ` +
                `${JSON.stringify(remainingNames)}`);
        }
        return outputSymbolicTensors;
    }
    /**
     * Helper method to loop over some data in batches.
     *
     * Porting Note: Not using the functional approach in the Python equivalent
     *   due to the imperative backend.
     * Porting Note: Does not support step mode currently.
     *
     * @param ins: input data
     * @param batchSize: integer batch size.
     * @param verbose: verbosity model
     * @returns: Predictions as `tf.Tensor` (if a single output) or an `Array` of
     *   `tf.Tensor` (if multipe outputs).
     */
    predictLoop(ins, batchSize = 32, verbose = false) {
        return tfc.tidy(() => {
            const numSamples = this.checkNumSamples(ins);
            if (verbose) {
                throw new NotImplementedError('Verbose predictLoop() is not implemented yet.');
            }
            // Sample-based predictions.
            // Porting Note: Tensor currently does not support sliced assignments as
            //   in numpy, e.g., x[1:3] = y. Therefore we use concatenation while
            //   iterating over the batches.
            const batches = makeBatches(numSamples, batchSize);
            const outsBatches = this.outputs.map(output => []);
            // TODO(cais): Can the scope() be pushed down inside the for loop?
            for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
                const batchOuts = tfc.tidy(() => {
                    const batchStart = batches[batchIndex][0];
                    const batchEnd = batches[batchIndex][1];
                    // TODO(cais): Take care of the case of the last element is a flag for
                    //   training/test.
                    const insBatch = sliceArrays(ins, batchStart, batchEnd);
                    // Construct the feeds for execute();
                    const feeds = [];
                    if (Array.isArray(insBatch)) {
                        for (let i = 0; i < insBatch.length; ++i) {
                            feeds.push({ key: this.inputs[i], value: insBatch[i] });
                        }
                    }
                    else {
                        feeds.push({ key: this.inputs[0], value: insBatch });
                    }
                    const feedDict = new FeedDict(feeds);
                    return execute(this.outputs, feedDict);
                });
                batchOuts.forEach((batchOut, i) => outsBatches[i].push(batchOut));
            }
            return singletonOrArray(outsBatches.map(batches => tfc.concat(batches, 0)));
        });
    }
    /**
     * Generates output predictions for the input samples.
     *
     * Computation is done in batches.
     *
     * Note: the "step" mode of predict() is currently not supported.
     *   This is because the TensorFlow.js core backend is imperative only.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.predict(tf.ones([8, 10]), {batchSize: 4}).print();
     * ```
     *
     * @param x The input data, as a Tensor, or an `Array` of `tf.Tensor`s if
     *   the model has multiple inputs.
     * @param args A `ModelPredictArgs` object containing optional fields.
     *
     * @return Prediction results as a `tf.Tensor`(s).
     *
     * @exception ValueError In case of mismatch between the provided input data
     *   and the model's expectations, or in case a stateful model receives a
     *   number of samples that is not a multiple of the batch size.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predict(x, args = {}) {
        const xsRank2OrHigher = ensureTensorsRank2OrHigher(x);
        checkInputData(xsRank2OrHigher, this.inputNames, this.feedInputShapes, false);
        try {
            // TODO(cais): Take care of stateful models.
            //   if (this.stateful) ...
            // TODO(cais): Take care of the learning_phase boolean flag.
            //   if (this.useLearningPhase) ...
            const batchSize = args.batchSize == null ? 32 : args.batchSize;
            checkBatchSize(batchSize);
            return this.predictLoop(xsRank2OrHigher, batchSize);
        }
        finally {
            disposeNewTensors(xsRank2OrHigher, x);
        }
    }
    /**
     * Returns predictions for a single batch of samples.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.predictOnBatch(tf.ones([8, 10])).print();
     * ```
     * @param x: Input samples, as a Tensor (for models with exactly one
     *   input) or an array of Tensors (for models with more than one input).
     * @return Tensor(s) of predictions
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predictOnBatch(x) {
        checkInputData(x, this.inputNames, this.feedInputShapes, true);
        // TODO(cais): Take care of the learning_phase boolean flag.
        //   if (this.useLearningPhase) ...
        const batchSize = (Array.isArray(x) ? x[0] : x).shape[0];
        return this.predictLoop(x, batchSize);
    }
    standardizeUserDataXY(x, y, checkBatchAxis = true, batchSize) {
        // TODO(cais): Add sampleWeight, classWeight
        if (this.optimizer_ == null) {
            throw new RuntimeError('You must compile a model before training/testing. Use ' +
                'LayersModel.compile(modelCompileArgs).');
        }
        const outputShapes = [];
        for (let i = 0; i < this.feedOutputShapes.length; ++i) {
            const outputShape = this.feedOutputShapes[i];
            const lossFn = this.feedLossFns[i];
            if (lossFn === losses.sparseCategoricalCrossentropy) {
                outputShapes.push(outputShape.slice(0, outputShape.length - 1).concat([1]));
            }
            else {
                // Porting Note: Because of strong typing `lossFn` must be a function.
                outputShapes.push(outputShape);
            }
        }
        x = standardizeInputData(x, this.feedInputNames, this.feedInputShapes, false, 'input');
        y = standardizeInputData(y, this.feedOutputNames, outputShapes, false, 'target');
        // TODO(cais): Standardize sampleWeights & classWeights.
        checkArrayLengths(x, y, null);
        // TODO(cais): Check sampleWeights as well.
        checkLossAndTargetCompatibility(y, this.feedLossFns, this.feedOutputShapes);
        if (this.stateful && batchSize != null && batchSize > 0) {
            if (x[0].shape[0] % batchSize !== 0) {
                throw new ValueError(`In a stateful network, you should only pass inputs with a ` +
                    `number of samples that is divisible by the batch size ` +
                    `${batchSize}. Found: ${x[0].shape[0]} sample(s).`);
            }
        }
        return [x, y];
    }
    async standardizeUserData(x, y, sampleWeight, classWeight, checkBatchAxis = true, batchSize) {
        const [standardXs, standardYs] = this.standardizeUserDataXY(x, y, checkBatchAxis, batchSize);
        // TODO(cais): Handle sampleWeights.
        if (sampleWeight != null) {
            throw new Error('sample weight is not supported yet.');
        }
        let standardSampleWeights = null;
        if (classWeight != null) {
            const classWeights = standardizeClassWeights(classWeight, this.outputNames);
            standardSampleWeights = [];
            for (let i = 0; i < classWeights.length; ++i) {
                standardSampleWeights.push(await standardizeWeights(standardYs[i], null, classWeights[i]));
            }
        }
        // TODO(cais): Deal with the case of model.stateful == true.
        return [standardXs, standardYs, standardSampleWeights];
    }
    /**
     * Loop over some test data in batches.
     * @param f A Function returning a list of tensors.
     * @param ins Array of tensors to be fed to `f`.
     * @param batchSize Integer batch size or `null` / `undefined`.
     * @param verbose verbosity mode.
     * @param steps Total number of steps (batches of samples) before
     * declaring test finished. Ignored with the default value of `null` /
     * `undefined`.
     * @returns Array of Scalars.
     */
    testLoop(f, ins, batchSize, verbose = 0, steps) {
        return tfc.tidy(() => {
            const numSamples = this.checkNumSamples(ins, batchSize, steps, 'steps');
            const outs = [];
            if (verbose > 0) {
                throw new NotImplementedError('Verbose mode is not implemented yet.');
            }
            // TODO(cais): Use `indicesForConversionToDense' to prevent slow down.
            if (steps != null) {
                throw new NotImplementedError('steps mode in testLoop() is not implemented yet');
            }
            else {
                const batches = makeBatches(numSamples, batchSize);
                const indexArray = tensor1d(range(0, numSamples));
                for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
                    const batchStart = batches[batchIndex][0];
                    const batchEnd = batches[batchIndex][1];
                    const batchIds = K.sliceAlongFirstAxis(indexArray, batchStart, batchEnd - batchStart);
                    // TODO(cais): In ins, train flag can be a number, instead of an
                    //   Tensor? Do we need to handle this in tfjs-layers?
                    const insBatch = sliceArraysByIndices(ins, batchIds);
                    const batchOuts = f(insBatch);
                    if (batchIndex === 0) {
                        for (let i = 0; i < batchOuts.length; ++i) {
                            outs.push(scalar(0));
                        }
                    }
                    for (let i = 0; i < batchOuts.length; ++i) {
                        const batchOut = batchOuts[i];
                        outs[i] =
                            tfc.add(outs[i], tfc.mul(batchEnd - batchStart, batchOut));
                    }
                }
                for (let i = 0; i < outs.length; ++i) {
                    outs[i] = tfc.div(outs[i], numSamples);
                }
            }
            return outs;
        });
    }
    getDedupedMetricsNames() {
        const outLabels = this.metricsNames;
        // Rename duplicated metrics names (can happen with an output layer
        // shared among multiple dataflows).
        const dedupedOutLabels = [];
        for (let i = 0; i < outLabels.length; ++i) {
            const label = outLabels[i];
            let newLabel = label;
            if (count(outLabels, label) > 1) {
                const dupIndex = count(outLabels.slice(0, i), label);
                newLabel += `_${dupIndex}`;
            }
            dedupedOutLabels.push(newLabel);
        }
        return dedupedOutLabels;
    }
    /**
     * Creates a function that performs the following actions:
     *
     * 1. computes the losses
     * 2. sums them to get the total loss
     * 3. call the optimizer computes the gradients of the LayersModel's
     *    trainable weights w.r.t. the total loss and update the variables
     * 4. calculates the metrics
     * 5. returns the values of the losses and metrics.
     */
    makeTrainFunction() {
        return (data) => {
            const lossValues = [];
            const inputs = data.slice(0, this.inputs.length);
            const targets = data.slice(this.inputs.length, this.inputs.length + this.outputs.length);
            const sampleWeights = data.slice(this.inputs.length + this.outputs.length, this.inputs.length + this.outputs.length * 2);
            const metricsValues = [];
            // Create a function that computes the total loss based on the
            // inputs. This function is used for obtaining gradients through
            // backprop.
            const totalLossFunction = () => {
                const feeds = [];
                for (let i = 0; i < this.inputs.length; ++i) {
                    feeds.push({ key: this.inputs[i], value: inputs[i] });
                }
                const feedDict = new FeedDict(feeds);
                const outputs = execute(this.outputs, feedDict, { 'training': true });
                // TODO(cais): Take care of the case of multiple outputs from a
                //   single layer?
                let totalLoss;
                for (let i = 0; i < this.lossFunctions.length; ++i) {
                    const lossFunction = this.lossFunctions[i];
                    let loss = lossFunction(targets[i], outputs[i]);
                    if (sampleWeights[i] != null) {
                        loss = computeWeightedLoss(loss, sampleWeights[i]);
                    }
                    // TODO(cais): push Scalar instead.
                    const meanLoss = tfc.mean(loss);
                    // TODO(cais): Use a scope() instead, to avoid ownership.
                    lossValues.push(meanLoss);
                    if (i === 0) {
                        totalLoss = loss;
                    }
                    else {
                        totalLoss = tfc.add(totalLoss, loss);
                    }
                }
                // Compute the metrics.
                // TODO(cais): These should probably be calculated outside
                //   totalLossFunction to benefit speed?
                for (let i = 0; i < this.metricsTensors.length; ++i) {
                    let weightedMetric;
                    if (this.outputs.length > 1 && i < this.outputs.length) {
                        weightedMetric = lossValues[i];
                    }
                    else {
                        const metric = this.metricsTensors[i][0];
                        const outputIndex = this.metricsTensors[i][1];
                        weightedMetric =
                            tfc.mean(metric(targets[outputIndex], outputs[outputIndex]));
                    }
                    tfc.keep(weightedMetric);
                    // TODO(cais): Use a scope() instead, to avoid ownership.
                    metricsValues.push(weightedMetric);
                }
                totalLoss = tfc.mean(totalLoss);
                // Add regularizer penalties.
                this.calculateLosses().forEach(regularizerLoss => {
                    totalLoss = tfc.add(totalLoss, regularizerLoss);
                });
                return totalLoss;
            };
            const variables = this.collectedTrainableWeights.map(param => param.read());
            const returnCost = true;
            const totalLossValue = this.optimizer_.minimize(totalLossFunction, returnCost, variables);
            return [totalLossValue].concat(metricsValues);
        };
    }
    /**
     * Create a function which, when invoked with an array of `tf.Tensor`s as a
     * batch of inputs, returns the prespecified loss and metrics of the model
     * under the batch of input data.
     */
    makeTestFunction() {
        this.testFunction = (data) => {
            return tfc.tidy(() => {
                const valOutputs = [];
                let totalLoss;
                const inputs = data.slice(0, this.inputs.length);
                const targets = data.slice(this.inputs.length, this.inputs.length + this.outputs.length);
                const feeds = [];
                for (let i = 0; i < this.inputs.length; ++i) {
                    feeds.push({ key: this.inputs[i], value: inputs[i] });
                }
                const feedDict = new FeedDict(feeds);
                const outputs = execute(this.outputs, feedDict);
                // Compute total loss.
                for (let i = 0; i < this.lossFunctions.length; ++i) {
                    const lossFunction = this.lossFunctions[i];
                    // TODO(cais): Add sample weighting and replace the simple
                    // averaging.
                    const loss = tfc.mean(lossFunction(targets[i], outputs[i]));
                    if (i === 0) {
                        totalLoss = loss;
                    }
                    else {
                        totalLoss = tfc.add(totalLoss, loss);
                    }
                    valOutputs.push(totalLoss);
                }
                // Compute the metrics.
                for (let i = 0; i < this.metricsTensors.length; ++i) {
                    const metric = this.metricsTensors[i][0];
                    const outputIndex = this.metricsTensors[i][1];
                    // TODO(cais): Replace K.mean() with a proper weighting function.
                    const meanMetric = tfc.mean(metric(targets[outputIndex], outputs[outputIndex]));
                    valOutputs.push(meanMetric);
                }
                return valOutputs;
            });
        };
    }
    /**
     * Trains the model for a fixed number of epochs (iterations on a
     * dataset).
     *
     * ```js
     * const model = tf.sequential({
     *     layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * for (let i = 1; i < 5 ; ++i) {
     *   const h = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
     *       batchSize: 4,
     *       epochs: 3
     *   });
     *   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
     * }
     * ```
     *
     * @param x `tf.Tensor` of training data, or an array of `tf.Tensor`s if the
     * model has multiple inputs. If all inputs in the model are named, you
     * can also pass a dictionary mapping input names to `tf.Tensor`s.
     * @param y `tf.Tensor` of target (label) data, or an array of `tf.Tensor`s if
     * the model has multiple outputs. If all outputs in the model are named,
     * you can also pass a dictionary mapping output names to `tf.Tensor`s.
     * @param args A `ModelFitArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @exception ValueError In case of mismatch between the provided input
     * data and what the model expects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async fit(x, y, args = {}) {
        return fitTensors(this, x, y, args);
    }
    // TODO(cais): Add code snippet below when it's possible to instantiate
    //   actual dataset objects.
    /**
     * Trains the model using a dataset object.
     *
     * @param dataset A dataset object. Its `iterator()` method is expected
     *   to generate a dataset iterator object, the `next()` method of which
     *   is expected to produce data batches for training. The return value
     *   of the `next()` call ought to contain a boolean `done` field and a
     *   `value` field. The `value` field is expected to be an array of two
     *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
     *   case is for models with exactly one input and one output (e.g..
     *   a sequential model). The latter case is for models with multiple
     *   inputs and/or multiple outputs.
     *   Of the two items in the array, the first is the input feature(s) and
     *   the second is the output target(s).
     * @param args A `ModelFitDatasetArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async fitDataset(dataset, args) {
        return fitDataset(this, dataset, args);
    }
    /**
     * Runs a single gradient update on a single batch of data.
     *
     * This method differs from `fit()` and `fitDataset()` in the following
     * regards:
     *   - It operates on exactly one batch of data.
     *   - It returns only the loss and matric values, instead of
     *     returning the batch-by-batch loss and metric values.
     *   - It doesn't support fine-grained options such as verbosity and
     *     callbacks.
     *
     * @param x Input data. It could be one of the following:
     *   - A `tf.Tensor`, or an Array of `tf.Tensor`s (in case the model has
     *     multiple inputs).
     *   - An Object mapping input names to corresponding `tf.Tensor` (if the
     *     model has named inputs).
     * @param y Target darta. It could be either a `tf.Tensor` a multiple
     *   `tf.Tensor`s. It should be consistent with `x`.
     * @returns Training loss or losses (in case the model has
     *   multiple outputs), along with metrics (if any), as numbers.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async trainOnBatch(x, y) {
        // TODO(cais): Support sampleWeight and classWeight.
        // TODO(cais): Support Dataset objects.
        const standardizeOut = await this.standardizeUserData(x, y);
        const inputs = standardizeOut[0];
        const targets = standardizeOut[1];
        const trainFunction = this.makeTrainFunction();
        const losses = trainFunction(inputs.concat(targets));
        const lossValues = [];
        for (const loss of losses) {
            const v = await loss.data();
            lossValues.push(v[0]);
        }
        tfc.dispose(losses);
        disposeNewTensors(standardizeOut[0], x);
        disposeNewTensors(standardizeOut[1], y);
        return singletonOrArray(lossValues);
    }
    /**
     * Extract weight values of the model.
     *
     * @param config: An instance of `io.SaveConfig`, which specifies
     * model-saving options such as whether only trainable weights are to be
     * saved.
     * @returns A `NamedTensorMap` mapping original weight names (i.e.,
     *   non-uniqueified weight names) to their values.
     */
    getNamedWeights(config) {
        const namedWeights = [];
        const trainableOnly = config != null && config.trainableOnly;
        const weights = trainableOnly ? this.trainableWeights : this.weights;
        const weightValues = this.getWeights(trainableOnly);
        for (let i = 0; i < weights.length; ++i) {
            if (trainableOnly && !weights[i].trainable) {
                // Optionally skip non-trainable weights.
                continue;
            }
            namedWeights.push({ name: weights[i].originalName, tensor: weightValues[i] });
        }
        return namedWeights;
    }
    /**
     * Setter used for force stopping of LayersModel.fit() (i.e., training).
     *
     * Example:
     *
     * ```js
     * const input = tf.input({shape: [10]});
     * const output = tf.layers.dense({units: 1}).apply(input);
     * const model = tf.model({inputs: [input], outputs: [output]});
     * model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
     * const xs = tf.ones([8, 10]);
     * const ys = tf.zeros([8, 1]);
     *
     * const history = await model.fit(xs, ys, {
     *   epochs: 10,
     *   callbacks: {
     *     onEpochEnd: async (epoch, logs) => {
     *       if (epoch === 2) {
     *         model.stopTraining = true;
     *       }
     *     }
     *   }
     * });
     *
     * // There should be only 3 values in the loss array, instead of 10
     * values,
     * // due to the stopping after 3 epochs.
     * console.log(history.history.loss);
     * ```
     */
    set stopTraining(stop) {
        this.stopTraining_ = stop;
    }
    get stopTraining() {
        return this.stopTraining_;
    }
    get optimizer() {
        return this.optimizer_;
    }
    set optimizer(optimizer) {
        if (this.optimizer_ !== optimizer) {
            this.optimizer_ = optimizer;
            this.isOptimizerOwned = false;
        }
    }
    dispose() {
        const result = super.dispose();
        if (result.refCountAfterDispose === 0 && this.optimizer != null &&
            this.isOptimizerOwned) {
            const numTensorsBeforeOptmizerDisposal = tfc.memory().numTensors;
            this.optimizer_.dispose();
            result.numDisposedVariables +=
                numTensorsBeforeOptmizerDisposal - tfc.memory().numTensors;
        }
        return result;
    }
    getLossIdentifiers() {
        let lossNames;
        if (typeof this.loss === 'string') {
            lossNames = toSnakeCase(this.loss);
        }
        else if (Array.isArray(this.loss)) {
            for (const loss of this.loss) {
                if (typeof loss !== 'string') {
                    throw new Error('Serialization of non-string loss is not supported.');
                }
            }
            lossNames = this.loss.map(name => toSnakeCase(name));
        }
        else {
            const outputNames = Object.keys(this.loss);
            lossNames = {};
            const losses = this.loss;
            for (const outputName of outputNames) {
                if (typeof losses[outputName] === 'string') {
                    lossNames[outputName] =
                        toSnakeCase(losses[outputName]);
                }
                else {
                    throw new Error('Serialization of non-string loss is not supported.');
                }
            }
        }
        return lossNames;
    }
    getMetricIdentifiers() {
        if (typeof this.metrics === 'string' ||
            typeof this.metrics === 'function') {
            return [toSnakeCase(Metrics.getLossOrMetricName(this.metrics))];
        }
        else if (Array.isArray(this.metrics)) {
            return this.metrics.map(metric => toSnakeCase(Metrics.getLossOrMetricName(metric)));
        }
        else {
            const metricsIdentifiers = {};
            for (const key in this.metrics) {
                metricsIdentifiers[key] =
                    toSnakeCase(Metrics.getLossOrMetricName(this.metrics[key]));
            }
            return metricsIdentifiers;
        }
    }
    getTrainingConfig() {
        return {
            loss: this.getLossIdentifiers(),
            metrics: this.getMetricIdentifiers(),
            optimizer_config: {
                class_name: this.optimizer.getClassName(),
                config: this.optimizer.getConfig()
            }
        };
        // TODO(cais): Add weight_metrics when they are supported.
        // TODO(cais): Add sample_weight_mode when it's supported.
        // TODO(cais): Add loss_weights when it's supported.
    }
    loadTrainingConfig(trainingConfig) {
        if (trainingConfig.weighted_metrics != null) {
            throw new Error('Loading weight_metrics is not supported yet.');
        }
        if (trainingConfig.loss_weights != null) {
            throw new Error('Loading loss_weights is not supported yet.');
        }
        if (trainingConfig.sample_weight_mode != null) {
            throw new Error('Loading sample_weight_mode is not supported yet.');
        }
        const tsConfig = convertPythonicToTs(trainingConfig.optimizer_config);
        const optimizer = deserialize(tsConfig);
        let loss;
        if (typeof trainingConfig.loss === 'string') {
            loss = toCamelCase(trainingConfig.loss);
        }
        else if (Array.isArray(trainingConfig.loss)) {
            loss = trainingConfig.loss.map(lossEntry => toCamelCase(lossEntry));
        }
        else if (trainingConfig.loss != null) {
            loss = {};
            for (const key in trainingConfig.loss) {
                loss[key] = toCamelCase(trainingConfig.loss[key]);
            }
        }
        let metrics;
        if (Array.isArray(trainingConfig.metrics)) {
            metrics = trainingConfig.metrics.map(metric => toCamelCase(metric));
        }
        else if (trainingConfig.metrics != null) {
            metrics = {};
            for (const key in trainingConfig.metrics) {
                metrics[key] = toCamelCase(trainingConfig.metrics[key]);
            }
        }
        this.compile({ loss, metrics, optimizer });
    }
    /**
     * Save the configuration and/or weights of the LayersModel.
     *
     * An `IOHandler` is an object that has a `save` method of the proper
     * signature defined. The `save` method manages the storing or
     * transmission of serialized data ("artifacts") that represent the
     * model's topology and weights onto or via a specific medium, such as
     * file downloads, local storage, IndexedDB in the web browser and HTTP
     * requests to a server. TensorFlow.js provides `IOHandler`
     * implementations for a number of frequently used saving mediums, such as
     * `tf.io.browserDownloads` and `tf.io.browserLocalStorage`. See `tf.io`
     * for more details.
     *
     * This method also allows you to refer to certain types of `IOHandler`s
     * as URL-like string shortcuts, such as 'localstorage://' and
     * 'indexeddb://'.
     *
     * Example 1: Save `model`'s topology and weights to browser [local
     * storage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage);
     * then load it back.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * console.log('Prediction from original model:');
     * model.predict(tf.ones([1, 3])).print();
     *
     * const saveResults = await model.save('localstorage://my-model-1');
     *
     * const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
     * console.log('Prediction from loaded model:');
     * loadedModel.predict(tf.ones([1, 3])).print();
     * ```
     *
     * Example 2. Saving `model`'s topology and weights to browser
     * [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API);
     * then load it back.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * console.log('Prediction from original model:');
     * model.predict(tf.ones([1, 3])).print();
     *
     * const saveResults = await model.save('indexeddb://my-model-1');
     *
     * const loadedModel = await tf.loadLayersModel('indexeddb://my-model-1');
     * console.log('Prediction from loaded model:');
     * loadedModel.predict(tf.ones([1, 3])).print();
     * ```
     *
     * Example 3. Saving `model`'s topology and weights as two files
     * (`my-model-1.json` and `my-model-1.weights.bin`) downloaded from
     * browser.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * const saveResults = await model.save('downloads://my-model-1');
     * ```
     *
     * Example 4. Send  `model`'s topology and weights to an HTTP server.
     * See the documentation of `tf.io.http` for more details
     * including specifying request parameters and implementation of the
     * server.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * const saveResults = await model.save('http://my-server/model/upload');
     * ```
     *
     * @param handlerOrURL An instance of `IOHandler` or a URL-like,
     * scheme-based string shortcut for `IOHandler`.
     * @param config Options for saving the model.
     * @returns A `Promise` of `SaveResult`, which summarizes the result of
     * the saving, such as byte sizes of the saved artifacts for the model's
     *   topology and weight values.
     *
     * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
     */
    async save(handlerOrURL, config) {
        if (typeof handlerOrURL === 'string') {
            const handlers = io.getSaveHandlers(handlerOrURL);
            if (handlers.length === 0) {
                throw new ValueError(`Cannot find any save handlers for URL '${handlerOrURL}'`);
            }
            else if (handlers.length > 1) {
                throw new ValueError(`Found more than one (${handlers.length}) save handlers for ` +
                    `URL '${handlerOrURL}'`);
            }
            handlerOrURL = handlers[0];
        }
        if (handlerOrURL.save == null) {
            throw new ValueError('LayersModel.save() cannot proceed because the IOHandler ' +
                'provided does not have the `save` attribute defined.');
        }
        const weightDataAndSpecs = await io.encodeWeights(this.getNamedWeights(config));
        const returnString = false;
        const unusedArg = null;
        const modelConfig = this.toJSON(unusedArg, returnString);
        const modelArtifacts = {
            modelTopology: modelConfig,
            format: LAYERS_MODEL_FORMAT_NAME,
            generatedBy: `TensorFlow.js tfjs-layers v${version}`,
            convertedBy: null,
        };
        const includeOptimizer = config == null ? false : config.includeOptimizer;
        if (includeOptimizer && this.optimizer != null) {
            modelArtifacts.trainingConfig = this.getTrainingConfig();
            const weightType = 'optimizer';
            const { data: optimizerWeightData, specs: optimizerWeightSpecs } = await io.encodeWeights(await this.optimizer.getWeights(), weightType);
            weightDataAndSpecs.specs.push(...optimizerWeightSpecs);
            weightDataAndSpecs.data = io.concatenateArrayBuffers([weightDataAndSpecs.data, optimizerWeightData]);
        }
        if (this.userDefinedMetadata != null) {
            // Check serialized size of user-defined metadata.
            const checkSize = true;
            checkUserDefinedMetadata(this.userDefinedMetadata, this.name, checkSize);
            modelArtifacts.userDefinedMetadata = this.userDefinedMetadata;
        }
        modelArtifacts.weightData = weightDataAndSpecs.data;
        modelArtifacts.weightSpecs = weightDataAndSpecs.specs;
        return handlerOrURL.save(modelArtifacts);
    }
    /**
     * Set user-defined metadata.
     *
     * The set metadata will be serialized together with the topology
     * and weights of the model during `save()` calls.
     *
     * @param setUserDefinedMetadata
     */
    setUserDefinedMetadata(userDefinedMetadata) {
        checkUserDefinedMetadata(userDefinedMetadata, this.name);
        this.userDefinedMetadata = userDefinedMetadata;
    }
    /**
     * Get user-defined metadata.
     *
     * The metadata is supplied via one of the two routes:
     *   1. By calling `setUserDefinedMetadata()`.
     *   2. Loaded during model loading (if the model is constructed
     *      via `tf.loadLayersModel()`.)
     *
     * If no user-defined metadata is available from either of the
     * two routes, this function will return `undefined`.
     */
    getUserDefinedMetadata() {
        return this.userDefinedMetadata;
    }
}
// The class name is 'Model' rather than 'LayersModel' for backwards
// compatibility since this class name shows up in the serialization format.
/** @nocollapse */
LayersModel.className = 'Model';
serialization.registerClass(LayersModel);
/**
 * A `tf.Functional` is an alias to `tf.LayersModel`.
 *
 * See also:
 *   `tf.LayersModel`, `tf.Sequential`, `tf.loadLayersModel`.
 */
/** @doc {heading: 'Models', subheading: 'Classes'} */
export class Functional extends LayersModel {
}
Functional.className = 'Functional';
serialization.registerClass(Functional);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhaW5pbmcuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvZW5naW5lL3RyYWluaW5nLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUgseUNBQXlDO0FBRXpDLE9BQU8sS0FBSyxHQUFHLE1BQU0sdUJBQXVCLENBQUM7QUFDN0MsT0FBTyxFQUFDLEVBQUUsRUFBMEQsU0FBUyxFQUFVLE1BQU0sRUFBRSxhQUFhLEVBQUUsTUFBTSxFQUFZLFFBQVEsRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUU3SyxPQUFPLEtBQUssQ0FBQyxNQUFNLHlCQUF5QixDQUFDO0FBRTdDLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDcEMsT0FBTyxFQUFDLG1CQUFtQixFQUFFLFlBQVksRUFBRSxVQUFVLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFLeEUsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLHlCQUF5QixDQUFDO0FBQ3BELE9BQU8sS0FBSyxNQUFNLE1BQU0sV0FBVyxDQUFDO0FBQ3BDLE9BQU8sS0FBSyxPQUFPLE1BQU0sWUFBWSxDQUFDO0FBQ3RDLE9BQU8sS0FBSyxVQUFVLE1BQU0sZUFBZSxDQUFDO0FBRTVDLE9BQU8sRUFBQyx3QkFBd0IsRUFBQyxNQUFNLDBCQUEwQixDQUFDO0FBQ2xFLE9BQU8sRUFBQyxLQUFLLEVBQUUsWUFBWSxFQUFFLGdCQUFnQixFQUFFLFdBQVcsRUFBRSxXQUFXLEVBQUUsTUFBTSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDL0csT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLHNCQUFzQixDQUFDO0FBQ2xELE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUMxQyxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSw4QkFBOEIsQ0FBQztBQUVqRSxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRW5DLE9BQU8sRUFBQyxTQUFTLEVBQWdCLE1BQU0sYUFBYSxDQUFDO0FBRXJELE9BQU8sRUFBQyxPQUFPLEVBQUUsUUFBUSxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRTdDLE9BQU8sRUFBQyxlQUFlLEVBQUUsVUFBVSxFQUFnRCxNQUFNLG9CQUFvQixDQUFDO0FBQzlHLE9BQU8sRUFBQyxjQUFjLEVBQUUsaUJBQWlCLEVBQUUsMEJBQTBCLEVBQUUsVUFBVSxFQUFFLFdBQVcsRUFBZ0IsV0FBVyxFQUFFLG9CQUFvQixFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDM0ssT0FBTyxFQUE4QixtQkFBbUIsRUFBRSx1QkFBdUIsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGtCQUFrQixDQUFDO0FBRS9IOztHQUVHO0FBQ0gsTUFBTSxVQUFVLFlBQVksQ0FBQyxDQUMrQjtJQUMxRCxPQUFPLENBQUMsWUFBWSxNQUFNLENBQUM7QUFDN0IsQ0FBQztBQUVEOztHQUVHO0FBQ0gsTUFBTSxVQUFVLFdBQVcsQ0FBQyxDQUM2QjtJQUN2RCxPQUFPLEtBQUssQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDMUIsQ0FBQztBQUVEOztHQUVHO0FBQ0gsTUFBTSxVQUFVLFVBQVUsQ0FBQyxDQUM2QjtJQUN0RCxPQUFPLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQzdDLENBQUM7QUFFRDs7Ozs7Ozs7OztHQVVHO0FBQ0gsTUFBTSxVQUFVLG9CQUFvQixDQUNoQyxJQUFtRCxFQUFFLEtBQWUsRUFDcEUsTUFBZ0IsRUFBRSxjQUFjLEdBQUcsSUFBSSxFQUFFLGVBQWUsR0FBRyxFQUFFO0lBQy9ELElBQUksS0FBSyxJQUFJLElBQUksSUFBSSxLQUFLLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtRQUN2Qyx5RUFBeUU7UUFDekUsUUFBUTtRQUNSLElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtZQUNoQixJQUFJLGlCQUFpQixHQUFHLEtBQUssQ0FBQztZQUM5QixJQUFJLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSyxJQUFpQixDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUU7Z0JBQ3RELGlCQUFpQixHQUFHLElBQUksQ0FBQzthQUMxQjtpQkFBTSxJQUFJLFVBQVUsQ0FBQyxJQUFJLENBQUMsRUFBRTtnQkFDM0IsS0FBSyxNQUFNLEdBQUcsSUFBSSxJQUFJLEVBQUU7b0JBQ3RCLElBQUksSUFBSSxDQUFDLGNBQWMsQ0FBQyxHQUFHLENBQUMsRUFBRTt3QkFDNUIsaUJBQWlCLEdBQUcsSUFBSSxDQUFDO3dCQUN6QixNQUFNO3FCQUNQO2lCQUNGO2FBQ0Y7aUJBQU07Z0JBQ0wsNkNBQTZDO2dCQUM3QyxpQkFBaUIsR0FBRyxJQUFJLENBQUM7YUFDMUI7WUFDRCxJQUFJLGlCQUFpQixFQUFFO2dCQUNyQixNQUFNLElBQUksVUFBVSxDQUNoQiw2QkFBNkIsZUFBZSxxQkFBcUI7b0JBQ2pFLFdBQVcsSUFBSSxFQUFFLENBQUMsQ0FBQzthQUN4QjtTQUNGO1FBQ0QsT0FBTyxFQUFFLENBQUM7S0FDWDtJQUNELElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtRQUNoQixPQUFPLEtBQUssQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQztLQUNoQztJQUVELElBQUksTUFBZ0IsQ0FBQztJQUNyQixJQUFJLFVBQVUsQ0FBQyxJQUFJLENBQUMsRUFBRTtRQUNwQixJQUFJLEdBQUcsSUFBcUMsQ0FBQztRQUM3QyxNQUFNLEdBQUcsRUFBRSxDQUFDO1FBQ1osS0FBSyxNQUFNLElBQUksSUFBSSxLQUFLLEVBQUU7WUFDeEIsSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxFQUFFO2dCQUN0QixNQUFNLElBQUksVUFBVSxDQUNoQix5QkFBeUIsSUFBSSxnQ0FBZ0M7b0JBQzdELEdBQUcsS0FBSyxFQUFFLENBQUMsQ0FBQzthQUNqQjtZQUNELE1BQU0sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7U0FDekI7S0FDRjtTQUFNLElBQUksV0FBVyxDQUFDLElBQUksQ0FBQyxFQUFFO1FBQzVCLElBQUksR0FBRyxJQUFnQixDQUFDO1FBQ3hCLElBQUksSUFBSSxDQUFDLE1BQU0sS0FBSyxLQUFLLENBQUMsTUFBTSxFQUFFO1lBQ2hDLE1BQU0sSUFBSSxVQUFVLENBQ2hCLDZCQUE2QixlQUFlLGlCQUFpQjtnQkFDN0QsaUVBQWlFO2dCQUNqRSxtQ0FBbUMsS0FBSyxDQUFDLE1BQU0sa0JBQWtCO2dCQUNqRSxnREFBZ0QsSUFBSSxFQUFFLENBQUMsQ0FBQztTQUM3RDtRQUNELE1BQU0sR0FBRyxJQUFJLENBQUM7S0FDZjtTQUFNO1FBQ0wsSUFBSSxHQUFHLElBQWMsQ0FBQztRQUN0QixJQUFJLEtBQUssQ0FBQyxNQUFNLEdBQUcsQ0FBQyxFQUFFO1lBQ3BCLE1BQU0sSUFBSSxVQUFVLENBQ2hCLGFBQWEsZUFBZSxZQUFZLEtBQUssQ0FBQyxNQUFNLGNBQWM7Z0JBQ2xFLDBEQUNJLElBQUksQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO1NBQ3ZCO1FBQ0QsTUFBTSxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUM7S0FDakI7SUFFRCxNQUFNLEdBQUcsMEJBQTBCLENBQUMsTUFBTSxDQUFDLENBQUM7SUFFNUMsNkJBQTZCO0lBQzdCLElBQUksTUFBTSxJQUFJLElBQUksRUFBRTtRQUNsQixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsS0FBSyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtZQUNyQyxJQUFJLE1BQU0sQ0FBQyxDQUFDLENBQUMsSUFBSSxJQUFJLEVBQUU7Z0JBQ3JCLFNBQVM7YUFDVjtZQUNELE1BQU0sS0FBSyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN4QixJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsTUFBTSxLQUFLLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLEVBQUU7Z0JBQzNDLE1BQU0sSUFBSSxVQUFVLENBQ2hCLHVCQUF1QixlQUFlLGNBQWMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHO29CQUMvRCxXQUFXLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLG9DQUFvQztvQkFDL0QsU0FBUyxLQUFLLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQzthQUM3QjtZQUNELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO2dCQUN6QyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxjQUFjLEVBQUU7b0JBQzlCLCtCQUErQjtvQkFDL0IsU0FBUztpQkFDVjtnQkFDRCxNQUFNLEdBQUcsR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMzQixNQUFNLE1BQU0sR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQzVCLElBQUksTUFBTSxJQUFJLElBQUksSUFBSSxNQUFNLElBQUksQ0FBQyxJQUFJLEdBQUcsS0FBSyxNQUFNLEVBQUU7b0JBQ25ELE1BQU0sSUFBSSxVQUFVLENBQ2hCLEdBQUcsZUFBZSwyQ0FBMkM7d0JBQzdELHNCQUFzQixNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLElBQUk7d0JBQzlELHlCQUNJLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsSUFBSTt3QkFDNUMsWUFBWSxlQUFlLDJCQUN2QixLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFO3dCQUNwQiwrQkFDSSxLQUFLLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsR0FBRzt3QkFDL0MsbUJBQW1CLEtBQUssQ0FBQyxLQUFLLElBQUksQ0FBQyxDQUFDO2lCQUN6QzthQUNGO1NBQ0Y7S0FDRjtJQUNELE9BQU8sTUFBTSxDQUFDO0FBQ2hCLENBQUM7QUFFRDs7Ozs7O0dBTUc7QUFDSCxNQUFNLFVBQVUsaUJBQWlCLENBQzdCLE1BQWdCLEVBQUUsT0FBaUIsRUFBRSxPQUFrQjtJQUN6RCxNQUFNLElBQUksR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pELElBQUksQ0FBQyxJQUFJLEVBQUUsQ0FBQztJQUNaLE1BQU0sSUFBSSxHQUFHLE1BQU0sQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDNUQsSUFBSSxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ1osdUNBQXVDO0lBQ3ZDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUU7UUFDbkIsTUFBTSxJQUFJLFVBQVUsQ0FDaEIsZ0VBQWdFO1lBQ2hFLG9CQUFvQjtZQUNwQixHQUFHLElBQUksQ0FBQyxTQUFTLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUM1RDtJQUNELElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUU7UUFDbkIsTUFBTSxJQUFJLFVBQVUsQ0FDaEIsaUVBQWlFO1lBQ2pFLG9CQUFvQjtZQUNwQixHQUFHLElBQUksQ0FBQyxTQUFTLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUMvRDtJQUNELElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRTtRQUN2RSxNQUFNLElBQUksVUFBVSxDQUNoQixpRUFBaUU7WUFDakUsa0JBQWtCLElBQUksQ0FBQyxDQUFDLENBQUMsd0JBQXdCLElBQUksQ0FBQyxDQUFDLENBQUMsVUFBVTtZQUNsRSxZQUFZLENBQUMsQ0FBQztLQUNuQjtBQUNILENBQUM7QUFFRDs7Ozs7Ozs7R0FRRztBQUNILFNBQVMsK0JBQStCLENBQ3BDLE9BQWlCLEVBQUUsT0FBeUIsRUFBRSxZQUFxQjtJQUNyRSx1Q0FBdUM7SUFDdkMsTUFBTSxTQUFTLEdBQUc7UUFDaEIsTUFBTSxDQUFDLGdCQUFnQixFQUFFLE1BQU0sQ0FBQyxrQkFBa0I7UUFDbEQsTUFBTSxDQUFDLHVCQUF1QjtLQUMvQixDQUFDO0lBQ0YsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDdkMsTUFBTSxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3JCLE1BQU0sSUFBSSxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN4QixNQUFNLEtBQUssR0FBRyxZQUFZLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDOUIsSUFBSSxJQUFJLElBQUksSUFBSSxFQUFFO1lBQ2hCLFNBQVM7U0FDVjtRQUNELElBQUksSUFBSSxLQUFLLE1BQU0sQ0FBQyx1QkFBdUIsRUFBRTtZQUMzQyxJQUFJLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxFQUFFO2dCQUNyQyxNQUFNLElBQUksVUFBVSxDQUNoQiwyQ0FBMkMsQ0FBQyxDQUFDLEtBQUssZUFBZTtvQkFDakUsK0RBQStEO29CQUMvRCw2REFBNkQ7b0JBQzdELHFCQUFxQixDQUFDLENBQUM7Z0JBQzNCLDZDQUE2QzthQUM5QztTQUNGO1FBQ0QsSUFBSSxTQUFTLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFO1lBQ2xDLE1BQU0sWUFBWSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3RDLE1BQU0sV0FBVyxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDbkMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFlBQVksQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7Z0JBQzVDLE1BQU0sU0FBUyxHQUFHLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDbEMsTUFBTSxNQUFNLEdBQUcsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUM5QixJQUFJLE1BQU0sSUFBSSxJQUFJLElBQUksU0FBUyxLQUFLLE1BQU0sRUFBRTtvQkFDMUMsTUFBTSxJQUFJLFVBQVUsQ0FDaEIsOEJBQThCLENBQUMsQ0FBQyxLQUFLLHFCQUFxQjt3QkFDMUQsbUJBQW1CLEtBQUsscUNBQXFDO3dCQUM3RCx1REFBdUQsQ0FBQyxDQUFDO2lCQUM5RDthQUNGO1NBQ0Y7S0FDRjtBQUNILENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXlCRztBQUNILFNBQVMsY0FBYyxDQUNuQixJQUFxQixFQUFFLEtBQWUsRUFBRSxNQUFnQixFQUN4RCxjQUFjLEdBQUcsSUFBSSxFQUFFLGVBQWUsR0FBRyxFQUFFO0lBQzdDLElBQUksTUFBZ0IsQ0FBQztJQUNyQixJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLEVBQUU7UUFDdkIsSUFBSSxJQUFJLENBQUMsTUFBTSxLQUFLLEtBQUssQ0FBQyxNQUFNLEVBQUU7WUFDaEMsTUFBTSxJQUFJLFVBQVUsQ0FDaEIsNkJBQTZCLGVBQWUsaUJBQWlCO2dCQUM3RCxpRUFBaUU7Z0JBQ2pFLHVDQUF1QyxLQUFLLENBQUMsTUFBTSxhQUFhO2dCQUNoRSxvQkFBb0IsSUFBSSxDQUFDLE1BQU0sY0FBYyxDQUFDLENBQUM7U0FDcEQ7UUFDRCxNQUFNLEdBQUcsSUFBSSxDQUFDO0tBQ2Y7U0FBTTtRQUNMLElBQUksS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUU7WUFDcEIsTUFBTSxJQUFJLFVBQVUsQ0FDaEIscUJBQXFCLEtBQUssQ0FBQyxNQUFNLElBQUksZUFBZSxZQUFZO2dCQUNoRSx3REFBd0Q7Z0JBQ3hELEdBQUcsSUFBSSxDQUFDLFNBQVMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1NBQ3ZDO1FBQ0QsTUFBTSxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUM7S0FDakI7SUFFRCxJQUFJLE1BQU0sSUFBSSxJQUFJLEVBQUU7UUFDbEIsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEtBQUssQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDckMsSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksSUFBSSxFQUFFO2dCQUNyQixTQUFTO2FBQ1Y7WUFDRCxNQUFNLEtBQUssR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDeEIsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLE1BQU0sS0FBSyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxFQUFFO2dCQUMzQyxNQUFNLElBQUksVUFBVSxDQUNoQix1QkFBdUIsZUFBZSxjQUFjLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRztvQkFDL0QsV0FBVyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxvQ0FBb0M7b0JBQy9ELFNBQVMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDO2FBQzdDO1lBQ0QsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7Z0JBQ3pDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGNBQWMsRUFBRTtvQkFDOUIsU0FBUztpQkFDVjtnQkFDRCxNQUFNLEdBQUcsR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMzQixNQUFNLE1BQU0sR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQzVCLElBQUksTUFBTSxJQUFJLElBQUksRUFBRTtvQkFDbEIsSUFBSSxNQUFNLEtBQUssR0FBRyxFQUFFO3dCQUNsQixNQUFNLElBQUksVUFBVSxDQUNoQix1QkFBdUIsZUFBZSxhQUFhOzRCQUNuRCxHQUFHLEtBQUssQ0FBQyxDQUFDLENBQUMsa0JBQWtCLElBQUksQ0FBQyxTQUFTLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU87NEJBQzdELHdCQUF3QixJQUFJLENBQUMsU0FBUyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLENBQUM7cUJBQzdEO2lCQUNGO2FBQ0Y7U0FDRjtLQUNGO0FBQ0gsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7R0FZRztBQUNILE1BQU0sVUFBVSxjQUFjLENBQzFCLE9BQytDLEVBQy9DLFdBQXFCO0lBQ3ZCLElBQUksT0FBTyxJQUFJLElBQUksSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxJQUFJLE9BQU8sQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ3JFLE9BQU8sV0FBVyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDO0tBQ3BDO0lBRUQsSUFBSSxjQUMrQyxDQUFDO0lBQ3BELElBQUksT0FBTyxPQUFPLEtBQUssUUFBUSxJQUFJLE9BQU8sT0FBTyxLQUFLLFVBQVUsRUFBRTtRQUNoRSxjQUFjLEdBQUcsQ0FBQyxPQUFPLENBQUMsQ0FBQztLQUM1QjtTQUFNLElBQUksS0FBSyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsSUFBSSxPQUFPLE9BQU8sS0FBSyxRQUFRLEVBQUU7UUFDaEUsY0FBYyxHQUFHLE9BQzBELENBQUM7S0FDN0U7U0FBTTtRQUNMLE1BQU0sSUFBSSxTQUFTLENBQ2YsOERBQThEO1lBQzlELHNDQUFzQyxPQUFPLEVBQUUsQ0FBQyxDQUFDO0tBQ3REO0lBRUQsSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLGNBQWMsQ0FBQyxFQUFFO1FBQ2pDLDRDQUE0QztRQUM1QyxPQUFPLFdBQVcsQ0FBQyxHQUFHLENBQ2xCLElBQUksQ0FBQyxFQUFFLENBQUMsY0FBOEMsQ0FBQyxDQUFDO0tBQzdEO1NBQU07UUFDTCxtQ0FBbUM7UUFDbkMsTUFBTSxhQUFhLEdBQXdDLEVBQUUsQ0FBQztRQUM5RCxLQUFLLE1BQU0sSUFBSSxJQUFJLFdBQVcsRUFBRTtZQUM5QixJQUFJLGFBQWEsR0FDYixjQUFjLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztZQUNwRSxJQUFJLENBQUMsS0FBSyxDQUFDLE9BQU8sQ0FBQyxhQUFhLENBQUMsRUFBRTtnQkFDakMsYUFBYSxHQUFHLENBQUMsYUFBYSxDQUFDLENBQUM7YUFDakM7WUFDRCxhQUFhLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1NBQ25DO1FBQ0QsT0FBTyxhQUFhLENBQUM7S0FDdEI7QUFDSCxDQUFDO0FBMkRELE1BQU0sd0JBQXdCLEdBQUcsY0FBYyxDQUFDO0FBRWhEOzs7Ozs7Ozs7OztHQVdHO0FBQ0gsTUFBTSxPQUFPLFdBQVksU0FBUSxTQUFTO0lBNEN4QyxZQUFZLElBQW1CO1FBQzdCLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNaLElBQUksQ0FBQyxVQUFVLEdBQUcsS0FBSyxDQUFDO0lBQzFCLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQWtDRztJQUNILE9BQU8sQ0FDSCxVQUFtQixFQUFFLFNBQW9CLEVBQ3pDLFVBRW9ELE9BQU8sQ0FBQyxHQUFHO1FBQ2pFLElBQUksQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFO1lBQ2YsTUFBTSxJQUFJLFVBQVUsQ0FDaEIsbUVBQW1FO2dCQUNuRSwrREFBK0Q7Z0JBQy9ELGdEQUFnRCxDQUFDLENBQUM7U0FDdkQ7UUFDRCxZQUFZLENBQUMsSUFBSSxFQUFFLFVBQVUsRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDckQsQ0FBQztJQUVEOzs7Ozs7Ozs7T0FTRztJQUNILE9BQU8sQ0FBQyxJQUFzQjtRQUM1QixJQUFJLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxFQUFFO1lBQ3JCLElBQUksQ0FBQyxJQUFJLEdBQUcsRUFBRSxDQUFDO1NBQ2hCO1FBQ0QsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDO1FBRXRCLElBQUksT0FBTyxJQUFJLENBQUMsU0FBUyxLQUFLLFFBQVEsRUFBRTtZQUN0QyxJQUFJLENBQUMsVUFBVSxHQUFHLFVBQVUsQ0FBQyxZQUFZLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1lBQzFELElBQUksQ0FBQyxnQkFBZ0IsR0FBRyxJQUFJLENBQUM7U0FDOUI7YUFBTTtZQUNMLElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLFlBQVksU0FBUyxDQUFDLEVBQUU7Z0JBQzFDLE1BQU0sSUFBSSxVQUFVLENBQ2hCLDZEQUE2RCxDQUFDLENBQUM7YUFDcEU7WUFDRCxJQUFJLENBQUMsVUFBVSxHQUFHLElBQUksQ0FBQyxTQUFTLENBQUM7WUFDakMsSUFBSSxDQUFDLGdCQUFnQixHQUFHLEtBQUssQ0FBQztTQUMvQjtRQUVELCtCQUErQjtRQUMvQixvQ0FBb0M7UUFFcEMsMEJBQTBCO1FBQzFCLElBQUksYUFBYSxHQUFxQixFQUFFLENBQUM7UUFDekMsSUFBSSxDQUFDLEtBQUssQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLE9BQU8sSUFBSSxDQUFDLElBQUksS0FBSyxRQUFRO1lBQzFELE9BQU8sSUFBSSxDQUFDLElBQUksS0FBSyxVQUFVLEVBQUU7WUFDbkMsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUMsSUFBc0MsQ0FBQztZQUN4RCxLQUFLLE1BQU0sSUFBSSxJQUFJLElBQUksQ0FBQyxJQUFJLEVBQUU7Z0JBQzVCLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUU7b0JBQ3pDLE1BQU0sSUFBSSxVQUFVLENBQ2hCLHNDQUFzQyxJQUFJLEtBQUs7d0JBQy9DLHFDQUFxQyxJQUFJLENBQUMsV0FBVyxFQUFFLENBQUMsQ0FBQztpQkFDOUQ7YUFDRjtZQUNELEtBQUssTUFBTSxJQUFJLElBQUksSUFBSSxDQUFDLFdBQVcsRUFBRTtnQkFDbkMsSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksRUFBRTtvQkFDM0IsT0FBTyxDQUFDLElBQUksQ0FDUixXQUFXLElBQUksK0NBQStDO3dCQUM5RCw4REFBOEQ7d0JBQzlELG1CQUFtQixJQUFJLGtCQUFrQixDQUFDLENBQUM7aUJBQ2hEO2dCQUNELGFBQWEsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUNqRDtTQUNGO2FBQU0sSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRTtZQUNuQyxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsTUFBTSxLQUFLLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxFQUFFO2dCQUM1QyxNQUFNLElBQUksVUFBVSxDQUNoQiw4REFBOEQ7b0JBQzlELCtCQUErQixJQUFJLENBQUMsT0FBTyxDQUFDLE1BQU0sY0FBYztvQkFDaEUsdUJBQXVCLElBQUksQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO2FBQzFDO1lBQ0QsTUFBTSxTQUFTLEdBQUcsSUFBSSxDQUFDLElBQW9DLENBQUM7WUFDNUQsYUFBYSxHQUFHLFNBQVMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDbkQ7YUFBTTtZQUNMLE1BQU0sWUFBWSxHQUFHLE1BQU0sQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1lBQzNDLElBQUksQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFO2dCQUN2QixhQUFhLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDO1lBQ25DLENBQUMsQ0FBQyxDQUFDO1NBQ0o7UUFFRCxJQUFJLENBQUMsYUFBYSxHQUFHLGFBQWEsQ0FBQztRQUVuQyxJQUFJLENBQUMsZUFBZSxHQUFHLEVBQUUsQ0FBQztRQUMxQixJQUFJLENBQUMsZ0JBQWdCLEdBQUcsRUFBRSxDQUFDO1FBQzNCLElBQUksQ0FBQyxXQUFXLEdBQUcsRUFBRSxDQUFDO1FBQ3RCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtZQUM1Qyw0Q0FBNEM7WUFDNUMsTUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLG9CQUFvQixDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzNDLE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDakMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDaEMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUNsQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDOUM7UUFFRCwwQ0FBMEM7UUFDMUMsNENBQTRDO1FBQzVDLE1BQU0saUJBQWlCLEdBQWEsRUFBRSxDQUFDO1FBRXZDLG1CQUFtQjtRQUNuQixJQUFJLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUM7UUFDNUIsbUNBQW1DO1FBQ25DLElBQUksQ0FBQyxZQUFZLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUM3QixJQUFJLENBQUMsY0FBYyxHQUFHLEVBQUUsQ0FBQztRQUV6QixzQkFBc0I7UUFDdEIseUVBQXlFO1FBQ3pFLDBFQUEwRTtRQUMxRSx1RUFBdUU7UUFDdkUsU0FBUyxDQUFDLE1BQU0sRUFBRSxHQUFHLEVBQUU7WUFDckIsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO2dCQUM1QyxJQUFJLGlCQUFpQixDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRTtvQkFDdkMsU0FBUztpQkFDVjtnQkFDRCx1REFBdUQ7Z0JBQ3ZELDhDQUE4QztnQkFDOUMsTUFBTSxZQUFZLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDM0MsSUFBSSxJQUFJLENBQUMsT0FBTyxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUU7b0JBQzNCLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLENBQUMsWUFBWSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQzVDLElBQUksQ0FBQyxZQUFZLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUM7aUJBQ3ZEO2FBQ0Y7WUFFRCwwRUFBMEU7WUFDMUUseUVBQXlFO1FBQzNFLENBQUMsQ0FBQyxDQUFDO1FBRUgsTUFBTSxhQUFhLEdBQUcsY0FBYyxDQUFDLElBQUksQ0FBQyxPQUFPLEVBQUUsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQ3JFLHlDQUF5QztRQUV6Qzs7V0FFRztRQUNILE1BQU0sWUFBWSxHQUNkLENBQUMsV0FBbUIsRUFBRSxVQUFrQixFQUN2QyxZQUE0QixFQUFFLEVBQUU7WUFDL0IsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUU7Z0JBQy9CLFVBQVUsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLFdBQVcsQ0FBQyxHQUFHLEdBQUcsR0FBRyxVQUFVLENBQUM7YUFDL0Q7WUFDRCxJQUFJLENBQUMsWUFBWSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQztZQUNuQyxJQUFJLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFDLFlBQVksRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDO1FBQ3hELENBQUMsQ0FBQztRQUVOLFNBQVMsQ0FBQyxRQUFRLEVBQUUsR0FBRyxFQUFFO1lBQ3ZCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtnQkFDNUMsSUFBSSxpQkFBaUIsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUU7b0JBQ3ZDLFNBQVM7aUJBQ1Y7Z0JBQ0QsTUFBTSxhQUFhLEdBQUcsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUN2QyxxREFBcUQ7Z0JBRXJELG9FQUFvRTtnQkFDcEUsTUFBTSxhQUFhLEdBQUcsQ0FBQyxPQUFxQyxFQUFFLEVBQUU7b0JBQzlELE1BQU0sZ0JBQWdCLEdBQUcsRUFBRSxDQUFDO29CQUM1QixJQUFJLFVBQWtCLENBQUM7b0JBQ3ZCLElBQUksS0FBcUIsQ0FBQztvQkFDMUIsSUFBSSxnQkFBZ0MsQ0FBQztvQkFDckMsb0RBQW9EO29CQUVwRCxLQUFLLE1BQU0sTUFBTSxJQUFJLE9BQU8sRUFBRTt3QkFDNUIsSUFBSSxPQUFPLE1BQU0sS0FBSyxRQUFROzRCQUMxQixDQUFDLFVBQVUsRUFBRSxLQUFLLEVBQUUsY0FBYyxFQUFFLElBQUksQ0FBQyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUM7Z0NBQ3JELENBQUMsQ0FBQyxFQUFFOzRCQUNWLE1BQU0sV0FBVyxHQUFHLElBQUksQ0FBQyxvQkFBb0IsQ0FBQyxDQUFDLENBQUMsQ0FBQzs0QkFFakQsSUFBSSxXQUFXLENBQUMsV0FBVyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDO2dDQUN6QyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxLQUFLLE1BQU0sQ0FBQyxrQkFBa0IsRUFBRTtnQ0FDdkQsc0NBQXNDO2dDQUN0QyxJQUFJLENBQUMsVUFBVSxFQUFFLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRTtvQ0FDOUMsS0FBSyxHQUFHLE9BQU8sQ0FBQyxjQUFjLENBQUM7aUNBQ2hDO3FDQUFNLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFO29DQUN4RCxLQUFLLEdBQUcsT0FBTyxDQUFDLGtCQUFrQixDQUFDO2lDQUNwQzs2QkFDRjtpQ0FBTSxJQUNILElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDO2dDQUNyQixNQUFNLENBQUMsNkJBQTZCLEVBQUU7Z0NBQ3hDLHdEQUF3RDtnQ0FDeEQsV0FBVztnQ0FDWCxJQUFJLENBQUMsVUFBVSxFQUFFLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRTtvQ0FDOUMsS0FBSyxHQUFHLE9BQU8sQ0FBQyx5QkFBeUIsQ0FBQztpQ0FDM0M7cUNBQU0sSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUU7b0NBQ3hELEtBQUssR0FBRyxPQUFPLENBQUMsNkJBQTZCLENBQUM7aUNBQy9DOzZCQUNGO2lDQUFNO2dDQUNMLDZDQUE2QztnQ0FDN0MsSUFBSSxDQUFDLFVBQVUsRUFBRSxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUU7b0NBQzlDLEtBQUssR0FBRyxPQUFPLENBQUMsbUJBQW1CLENBQUM7aUNBQ3JDO3FDQUFNLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFO29DQUN4RCxLQUFLLEdBQUcsT0FBTyxDQUFDLHVCQUF1QixDQUFDO2lDQUN6Qzs2QkFDRjs0QkFDRCxJQUFJLE1BQWMsQ0FBQzs0QkFDbkIsSUFBSSxDQUFDLFVBQVUsRUFBRSxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUU7Z0NBQzlDLE1BQU0sR0FBRyxLQUFLLENBQUM7NkJBQ2hCO2lDQUFNLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFO2dDQUN4RCxNQUFNLEdBQUcsSUFBSSxDQUFDOzZCQUNmOzRCQUNELHNDQUFzQzs0QkFDdEMsZ0JBQWdCLEdBQUcsS0FBSyxDQUFDOzRCQUN6QixVQUFVLEdBQUcsZ0JBQWdCLEdBQUcsTUFBTSxDQUFDO3lCQUN4Qzs2QkFBTTs0QkFDTCxNQUFNLFFBQVEsR0FBRyxPQUFPLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDOzRCQUNyQyxzQ0FBc0M7NEJBQ3RDLGdCQUFnQixHQUFHLFFBQVEsQ0FBQzs0QkFDNUIsVUFBVTtnQ0FDTixnQkFBZ0IsR0FBRyxPQUFPLENBQUMsbUJBQW1CLENBQUMsTUFBTSxDQUFDLENBQUM7eUJBQzVEO3dCQUVELHlEQUF5RDt3QkFDekQsSUFBSSxZQUE0QixDQUFDO3dCQUNqQyxTQUFTLENBQUMsVUFBVSxFQUFFLEdBQUcsRUFBRTs0QkFDekIsWUFBWSxHQUFHLGdCQUFnQixDQUFDO3dCQUNsQyxDQUFDLENBQUMsQ0FBQzt3QkFDSCxZQUFZLENBQUMsQ0FBQyxFQUFFLFVBQVUsRUFBRSxZQUFZLENBQUMsQ0FBQztxQkFDM0M7Z0JBQ0gsQ0FBQyxDQUFDO2dCQUVGLGFBQWEsQ0FBQyxhQUFhLENBQUMsQ0FBQztnQkFDN0IsK0NBQStDO2FBQ2hEO1FBQ0gsQ0FBQyxDQUFDLENBQUM7UUFFSCw0REFBNEQ7UUFDNUQsMkVBQTJFO1FBQzNFLElBQUksQ0FBQyx5QkFBeUIsR0FBRyxJQUFJLENBQUMsZ0JBQWdCLENBQUM7SUFDekQsQ0FBQztJQUVEOzs7Ozs7OztPQVFHO0lBQ08sZ0NBQWdDO1FBQ3hDLElBQUksSUFBSSxDQUFDLHlCQUF5QixJQUFJLElBQUksRUFBRTtZQUMxQyxPQUFPO1NBQ1I7UUFDRCxJQUFJLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxNQUFNO1lBQzVCLElBQUksQ0FBQyx5QkFBeUIsQ0FBQyxNQUFNLEVBQUU7WUFDekMsT0FBTyxDQUFDLElBQUksQ0FDUiwrREFBK0Q7Z0JBQy9ELHlEQUF5RDtnQkFDekQsK0JBQStCLENBQUMsQ0FBQztTQUN0QztJQUNILENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O09BOEJHO0lBQ0gsUUFBUSxDQUNKLENBQWtCLEVBQUUsQ0FBa0IsRUFDdEMsT0FBMEIsRUFBRTtRQUM5QixNQUFNLFNBQVMsR0FBRyxJQUFJLENBQUMsU0FBUyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDO1FBQy9ELGNBQWMsQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUUxQiwwREFBMEQ7UUFDMUQsc0JBQXNCO1FBQ3RCLE1BQU0sY0FBYyxHQUFHLElBQUksQ0FBQztRQUM1QixNQUFNLGdCQUFnQixHQUNsQixJQUFJLENBQUMscUJBQXFCLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxjQUFjLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDaEUsSUFBSTtZQUNGLHdFQUF3RTtZQUN4RSxxQkFBcUI7WUFDckIsTUFBTSxHQUFHLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDNUQsSUFBSSxDQUFDLGdCQUFnQixFQUFFLENBQUM7WUFDeEIsTUFBTSxDQUFDLEdBQUcsSUFBSSxDQUFDLFlBQVksQ0FBQztZQUM1QixNQUFNLFFBQVEsR0FDVixJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsU0FBUyxFQUFFLElBQUksQ0FBQyxPQUFPLEVBQUUsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQy9ELE9BQU8sZ0JBQWdCLENBQUMsUUFBUSxDQUFDLENBQUM7U0FDbkM7Z0JBQVM7WUFDUixpQkFBaUIsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztZQUMxQyxpQkFBaUIsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztTQUMzQztJQUNILENBQUM7SUFFRCxtRUFBbUU7SUFDbkUsZUFBZTtJQUNmOzs7Ozs7Ozs7Ozs7Ozs7Ozs7O09BbUJHO0lBQ0gsS0FBSyxDQUFDLGVBQWUsQ0FBQyxPQUFvQixFQUFFLElBQStCO1FBRXpFLElBQUksQ0FBQyxnQkFBZ0IsRUFBRSxDQUFDO1FBQ3hCLE9BQU8sZUFBZSxDQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDOUMsQ0FBQztJQUVEOzs7Ozs7Ozs7T0FTRztJQUNLLGVBQWUsQ0FDbkIsR0FBb0IsRUFBRSxTQUFrQixFQUFFLEtBQWMsRUFDeEQsU0FBUyxHQUFHLE9BQU87UUFDckIsSUFBSSxVQUFrQixDQUFDO1FBQ3ZCLElBQUksS0FBSyxJQUFJLElBQUksRUFBRTtZQUNqQixVQUFVLEdBQUcsSUFBSSxDQUFDO1lBQ2xCLElBQUksU0FBUyxJQUFJLElBQUksRUFBRTtnQkFDckIsTUFBTSxJQUFJLFVBQVUsQ0FDaEIsTUFBTSxTQUFTLCtDQUErQztvQkFDOUQsbUJBQW1CLFNBQVMsRUFBRSxDQUFDLENBQUM7YUFDckM7U0FDRjthQUFNLElBQUksR0FBRyxJQUFJLElBQUksRUFBRTtZQUN0QixJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLEVBQUU7Z0JBQ3RCLFVBQVUsR0FBRyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzlCO2lCQUFNO2dCQUNMLFVBQVUsR0FBRyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNCO1NBQ0Y7YUFBTTtZQUNMLE1BQU0sSUFBSSxVQUFVLENBQ2hCLHdEQUF3RDtnQkFDeEQsR0FBRyxTQUFTLHNCQUFzQixDQUFDLENBQUM7U0FDekM7UUFDRCxPQUFPLFVBQVUsQ0FBQztJQUNwQixDQUFDO0lBRUQ7Ozs7OztPQU1HO0lBQ0gsT0FBTyxDQUFDLE1BQXNDLEVBQUUsT0FBd0I7UUFFdEUsSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxJQUFJLE9BQU8sQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1lBQ2xELE1BQU0sSUFBSSxVQUFVLENBQ2hCLG9EQUFvRCxDQUFDLENBQUM7U0FDM0Q7UUFFRCxNQUFNLGNBQWMsR0FBRyxLQUFLLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQzlDLE1BQU0sV0FBVyxHQUNiLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxPQUFtQixDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQWlCLENBQUMsQ0FBQyxDQUFDO1FBQ2pFLE1BQU0scUJBQXFCLEdBQUcsSUFBSSxDQUFDLHVCQUF1QixDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRXhFLG9DQUFvQztRQUNwQyxNQUFNLFFBQVEsR0FBRyxJQUFJLFFBQVEsRUFBRSxDQUFDO1FBQ2hDLElBQUksTUFBTSxZQUFZLE1BQU0sRUFBRTtZQUM1QixNQUFNLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQztTQUNuQjtRQUNELElBQUksS0FBSyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsRUFBRTtZQUN6QixJQUFJLE1BQU0sQ0FBQyxNQUFNLEtBQUssSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLEVBQUU7Z0JBQ3hDLE1BQU0sSUFBSSxVQUFVLENBQ2hCLGtDQUFrQyxNQUFNLENBQUMsTUFBTSxJQUFJO29CQUNuRCxvREFBb0Q7b0JBQ3BELElBQUksSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLElBQUksQ0FBQyxDQUFDO2FBQ2pDO1lBQ0QsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO2dCQUMzQyxRQUFRLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDekM7U0FDRjthQUFNO1lBQ0wsS0FBSyxNQUFNLEtBQUssSUFBSSxJQUFJLENBQUMsTUFBTSxFQUFFO2dCQUMvQixNQUFNLFdBQVcsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO2dCQUN2QyxJQUFJLFdBQVcsSUFBSSxJQUFJLEVBQUU7b0JBQ3ZCLE1BQU0sSUFBSSxVQUFVLENBQ2hCLDhDQUE4QyxLQUFLLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQztpQkFDakU7Z0JBQ0QsUUFBUSxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsV0FBVyxDQUFDLENBQUM7YUFDbEM7U0FDRjtRQUVELGlCQUFpQjtRQUNqQixNQUFNLGNBQWMsR0FBRyxPQUFPLENBQUMscUJBQXFCLEVBQUUsUUFBUSxDQUFhLENBQUM7UUFDNUUsT0FBTyxjQUFjLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdELENBQUM7SUFFRDs7T0FFRztJQUNLLHVCQUF1QixDQUFDLG1CQUE2QjtRQUUzRCxNQUFNLHFCQUFxQixHQUN2QixZQUFZLENBQUMsSUFBSSxFQUFFLG1CQUFtQixDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ25ELElBQUksZ0JBQWdCLEdBQUcsbUJBQW1CLENBQUMsTUFBTSxDQUFDO1FBQ2xELEtBQUssTUFBTSxLQUFLLElBQUksSUFBSSxDQUFDLE1BQU0sRUFBRTtZQUMvQixNQUFNLFlBQVksR0FDZCxLQUFLLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUM7WUFDaEUsTUFBTSxnQkFBZ0IsR0FBRyxZQUFZLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxDQUFDO1lBQ2pFLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxtQkFBbUIsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7Z0JBQ25ELE1BQU0sS0FBSyxHQUFHLGdCQUFnQixDQUFDLE9BQU8sQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMvRCxJQUFJLEtBQUssS0FBSyxDQUFDLENBQUMsRUFBRTtvQkFDaEIscUJBQXFCLENBQUMsQ0FBQyxDQUFDLEdBQUcsWUFBWSxDQUFDLEtBQUssQ0FBQyxDQUFDO29CQUMvQyxnQkFBZ0IsRUFBRSxDQUFDO2lCQUNwQjtnQkFDRCxJQUFJLGdCQUFnQixLQUFLLENBQUMsRUFBRTtvQkFDMUIsTUFBTTtpQkFDUDthQUNGO1lBQ0QsSUFBSSxnQkFBZ0IsS0FBSyxDQUFDLEVBQUU7Z0JBQzFCLE1BQU07YUFDUDtTQUNGO1FBRUQsSUFBSSxnQkFBZ0IsR0FBRyxDQUFDLEVBQUU7WUFDeEIsTUFBTSxjQUFjLEdBQWEsRUFBRSxDQUFDO1lBQ3BDLHFCQUFxQixDQUFDLE9BQU8sQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRTtnQkFDMUMsSUFBSSxNQUFNLElBQUksSUFBSSxFQUFFO29CQUNsQixjQUFjLENBQUMsSUFBSSxDQUFDLG1CQUFtQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7aUJBQzdDO1lBQ0gsQ0FBQyxDQUFDLENBQUM7WUFDSCxNQUFNLElBQUksVUFBVSxDQUNoQixrREFBa0Q7Z0JBQ2xELEdBQUcsSUFBSSxDQUFDLFNBQVMsQ0FBQyxjQUFjLENBQUMsRUFBRSxDQUFDLENBQUM7U0FDMUM7UUFDRCxPQUFPLHFCQUFxQixDQUFDO0lBQy9CLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7O09BWUc7SUFDSyxXQUFXLENBQUMsR0FBb0IsRUFBRSxTQUFTLEdBQUcsRUFBRSxFQUFFLE9BQU8sR0FBRyxLQUFLO1FBRXZFLE9BQU8sR0FBRyxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDbkIsTUFBTSxVQUFVLEdBQUcsSUFBSSxDQUFDLGVBQWUsQ0FBQyxHQUFHLENBQUMsQ0FBQztZQUM3QyxJQUFJLE9BQU8sRUFBRTtnQkFDWCxNQUFNLElBQUksbUJBQW1CLENBQ3pCLCtDQUErQyxDQUFDLENBQUM7YUFDdEQ7WUFFRCw0QkFBNEI7WUFDNUIsd0VBQXdFO1lBQ3hFLHFFQUFxRTtZQUNyRSxnQ0FBZ0M7WUFFaEMsTUFBTSxPQUFPLEdBQUcsV0FBVyxDQUFDLFVBQVUsRUFBRSxTQUFTLENBQUMsQ0FBQztZQUNuRCxNQUFNLFdBQVcsR0FBZSxJQUFJLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1lBRS9ELGtFQUFrRTtZQUNsRSxLQUFLLElBQUksVUFBVSxHQUFHLENBQUMsRUFBRSxVQUFVLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxFQUFFLFVBQVUsRUFBRTtnQkFDbEUsTUFBTSxTQUFTLEdBQUcsR0FBRyxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7b0JBQzlCLE1BQU0sVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDMUMsTUFBTSxRQUFRLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUN4QyxzRUFBc0U7b0JBQ3RFLG1CQUFtQjtvQkFDbkIsTUFBTSxRQUFRLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxVQUFVLEVBQUUsUUFBUSxDQUFDLENBQUM7b0JBRXhELHFDQUFxQztvQkFDckMsTUFBTSxLQUFLLEdBQUcsRUFBRSxDQUFDO29CQUNqQixJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLEVBQUU7d0JBQzNCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxRQUFRLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFOzRCQUN4QyxLQUFLLENBQUMsSUFBSSxDQUFDLEVBQUMsR0FBRyxFQUFFLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBQyxDQUFDLENBQUM7eUJBQ3ZEO3FCQUNGO3lCQUFNO3dCQUNMLEtBQUssQ0FBQyxJQUFJLENBQUMsRUFBQyxHQUFHLEVBQUUsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxLQUFLLEVBQUUsUUFBUSxFQUFDLENBQUMsQ0FBQztxQkFDcEQ7b0JBQ0QsTUFBTSxRQUFRLEdBQUcsSUFBSSxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUM7b0JBQ3JDLE9BQU8sT0FBTyxDQUFDLElBQUksQ0FBQyxPQUFPLEVBQUUsUUFBUSxDQUFhLENBQUM7Z0JBQ3JELENBQUMsQ0FBQyxDQUFDO2dCQUNILFNBQVMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxRQUFRLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7YUFDbkU7WUFDRCxPQUFPLGdCQUFnQixDQUNuQixXQUFXLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxPQUFPLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzFELENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQztJQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQTBCRztJQUNILE9BQU8sQ0FBQyxDQUFrQixFQUFFLE9BQXlCLEVBQUU7UUFDckQsTUFBTSxlQUFlLEdBQUcsMEJBQTBCLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEQsY0FBYyxDQUNWLGVBQWUsRUFBRSxJQUFJLENBQUMsVUFBVSxFQUFFLElBQUksQ0FBQyxlQUFlLEVBQUUsS0FBSyxDQUFDLENBQUM7UUFDbkUsSUFBSTtZQUNGLDRDQUE0QztZQUM1QywyQkFBMkI7WUFDM0IsNERBQTREO1lBQzVELG1DQUFtQztZQUNuQyxNQUFNLFNBQVMsR0FBRyxJQUFJLENBQUMsU0FBUyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDO1lBQy9ELGNBQWMsQ0FBQyxTQUFTLENBQUMsQ0FBQztZQUMxQixPQUFPLElBQUksQ0FBQyxXQUFXLENBQUMsZUFBZSxFQUFFLFNBQVMsQ0FBQyxDQUFDO1NBQ3JEO2dCQUFTO1lBQ1IsaUJBQWlCLENBQUMsZUFBZSxFQUFFLENBQUMsQ0FBQyxDQUFDO1NBQ3ZDO0lBQ0gsQ0FBQztJQUVEOzs7Ozs7Ozs7Ozs7OztPQWNHO0lBQ0gsY0FBYyxDQUFDLENBQWtCO1FBQy9CLGNBQWMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLFVBQVUsRUFBRSxJQUFJLENBQUMsZUFBZSxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQy9ELDREQUE0RDtRQUM1RCxtQ0FBbUM7UUFDbkMsTUFBTSxTQUFTLEdBQUcsQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN6RCxPQUFPLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQ3hDLENBQUM7SUFFUyxxQkFBcUIsQ0FDM0IsQ0FBZ0QsRUFDaEQsQ0FBZ0QsRUFBRSxjQUFjLEdBQUcsSUFBSSxFQUN2RSxTQUFrQjtRQUNwQiw0Q0FBNEM7UUFDNUMsSUFBSSxJQUFJLENBQUMsVUFBVSxJQUFJLElBQUksRUFBRTtZQUMzQixNQUFNLElBQUksWUFBWSxDQUNsQix3REFBd0Q7Z0JBQ3hELHdDQUF3QyxDQUFDLENBQUM7U0FDL0M7UUFDRCxNQUFNLFlBQVksR0FBWSxFQUFFLENBQUM7UUFDakMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDckQsTUFBTSxXQUFXLEdBQUcsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzdDLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDbkMsSUFBSSxNQUFNLEtBQUssTUFBTSxDQUFDLDZCQUE2QixFQUFFO2dCQUNuRCxZQUFZLENBQUMsSUFBSSxDQUNiLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQy9EO2lCQUFNO2dCQUNMLHNFQUFzRTtnQkFDdEUsWUFBWSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQzthQUNoQztTQUNGO1FBQ0QsQ0FBQyxHQUFHLG9CQUFvQixDQUNwQixDQUFDLEVBQUUsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsZUFBZSxFQUFFLEtBQUssRUFBRSxPQUFPLENBQUMsQ0FBQztRQUNsRSxDQUFDLEdBQUcsb0JBQW9CLENBQ3BCLENBQUMsRUFBRSxJQUFJLENBQUMsZUFBZSxFQUFFLFlBQVksRUFBRSxLQUFLLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFDNUQsd0RBQXdEO1FBQ3hELGlCQUFpQixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDOUIsMkNBQTJDO1FBQzNDLCtCQUErQixDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDO1FBQzVFLElBQUksSUFBSSxDQUFDLFFBQVEsSUFBSSxTQUFTLElBQUksSUFBSSxJQUFJLFNBQVMsR0FBRyxDQUFDLEVBQUU7WUFDdkQsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLFNBQVMsS0FBSyxDQUFDLEVBQUU7Z0JBQ25DLE1BQU0sSUFBSSxVQUFVLENBQ2hCLDREQUE0RDtvQkFDNUQsd0RBQXdEO29CQUN4RCxHQUFHLFNBQVMsWUFBWSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxhQUFhLENBQUMsQ0FBQzthQUN6RDtTQUNGO1FBQ0QsT0FBTyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNoQixDQUFDO0lBRVMsS0FBSyxDQUFDLG1CQUFtQixDQUMvQixDQUFnRCxFQUNoRCxDQUFnRCxFQUNoRCxZQUE2RCxFQUM3RCxXQUFzRCxFQUN0RCxjQUFjLEdBQUcsSUFBSSxFQUNyQixTQUFrQjtRQUNwQixNQUFNLENBQUMsVUFBVSxFQUFFLFVBQVUsQ0FBQyxHQUMxQixJQUFJLENBQUMscUJBQXFCLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxjQUFjLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDaEUsb0NBQW9DO1FBQ3BDLElBQUksWUFBWSxJQUFJLElBQUksRUFBRTtZQUN4QixNQUFNLElBQUksS0FBSyxDQUFDLHFDQUFxQyxDQUFDLENBQUM7U0FDeEQ7UUFFRCxJQUFJLHFCQUFxQixHQUFhLElBQUksQ0FBQztRQUMzQyxJQUFJLFdBQVcsSUFBSSxJQUFJLEVBQUU7WUFDdkIsTUFBTSxZQUFZLEdBQ2QsdUJBQXVCLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztZQUMzRCxxQkFBcUIsR0FBRyxFQUFFLENBQUM7WUFDM0IsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFlBQVksQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7Z0JBQzVDLHFCQUFxQixDQUFDLElBQUksQ0FDdEIsTUFBTSxrQkFBa0IsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxFQUFFLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDckU7U0FDRjtRQUVELDREQUE0RDtRQUM1RCxPQUFPLENBQUMsVUFBVSxFQUFFLFVBQVUsRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO0lBQ3pELENBQUM7SUFFRDs7Ozs7Ozs7OztPQVVHO0lBQ0ssUUFBUSxDQUNaLENBQStCLEVBQUUsR0FBYSxFQUFFLFNBQWtCLEVBQ2xFLE9BQU8sR0FBRyxDQUFDLEVBQUUsS0FBYztRQUM3QixPQUFPLEdBQUcsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO1lBQ25CLE1BQU0sVUFBVSxHQUFHLElBQUksQ0FBQyxlQUFlLENBQUMsR0FBRyxFQUFFLFNBQVMsRUFBRSxLQUFLLEVBQUUsT0FBTyxDQUFDLENBQUM7WUFDeEUsTUFBTSxJQUFJLEdBQWEsRUFBRSxDQUFDO1lBQzFCLElBQUksT0FBTyxHQUFHLENBQUMsRUFBRTtnQkFDZixNQUFNLElBQUksbUJBQW1CLENBQUMsc0NBQXNDLENBQUMsQ0FBQzthQUN2RTtZQUNELHNFQUFzRTtZQUN0RSxJQUFJLEtBQUssSUFBSSxJQUFJLEVBQUU7Z0JBQ2pCLE1BQU0sSUFBSSxtQkFBbUIsQ0FDekIsaURBQWlELENBQUMsQ0FBQzthQUN4RDtpQkFBTTtnQkFDTCxNQUFNLE9BQU8sR0FBRyxXQUFXLENBQUMsVUFBVSxFQUFFLFNBQVMsQ0FBQyxDQUFDO2dCQUNuRCxNQUFNLFVBQVUsR0FBRyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQyxDQUFDO2dCQUNsRCxLQUFLLElBQUksVUFBVSxHQUFHLENBQUMsRUFBRSxVQUFVLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxFQUFFLFVBQVUsRUFBRTtvQkFDbEUsTUFBTSxVQUFVLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUMxQyxNQUFNLFFBQVEsR0FBRyxPQUFPLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQ3hDLE1BQU0sUUFBUSxHQUNWLENBQUMsQ0FBQyxtQkFBbUIsQ0FDakIsVUFBVSxFQUFFLFVBQVUsRUFBRSxRQUFRLEdBQUcsVUFBVSxDQUFhLENBQUM7b0JBQ25FLGdFQUFnRTtvQkFDaEUsc0RBQXNEO29CQUN0RCxNQUFNLFFBQVEsR0FBRyxvQkFBb0IsQ0FBQyxHQUFHLEVBQUUsUUFBUSxDQUFhLENBQUM7b0JBQ2pFLE1BQU0sU0FBUyxHQUFHLENBQUMsQ0FBQyxRQUFRLENBQUMsQ0FBQztvQkFDOUIsSUFBSSxVQUFVLEtBQUssQ0FBQyxFQUFFO3dCQUNwQixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsU0FBUyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTs0QkFDekMsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQzt5QkFDdEI7cUJBQ0Y7b0JBQ0QsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFNBQVMsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7d0JBQ3pDLE1BQU0sUUFBUSxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQzt3QkFDOUIsSUFBSSxDQUFDLENBQUMsQ0FBQzs0QkFDSCxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLFFBQVEsR0FBRyxVQUFVLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQztxQkFDaEU7aUJBQ0Y7Z0JBQ0QsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7b0JBQ3BDLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQztpQkFDeEM7YUFDRjtZQUNELE9BQU8sSUFBSSxDQUFDO1FBQ2QsQ0FBQyxDQUFDLENBQUM7SUFDTCxDQUFDO0lBRVMsc0JBQXNCO1FBQzlCLE1BQU0sU0FBUyxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUM7UUFDcEMsbUVBQW1FO1FBQ25FLG9DQUFvQztRQUNwQyxNQUFNLGdCQUFnQixHQUFHLEVBQUUsQ0FBQztRQUM1QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsU0FBUyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtZQUN6QyxNQUFNLEtBQUssR0FBRyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDM0IsSUFBSSxRQUFRLEdBQUcsS0FBSyxDQUFDO1lBQ3JCLElBQUksS0FBSyxDQUFDLFNBQVMsRUFBRSxLQUFLLENBQUMsR0FBRyxDQUFDLEVBQUU7Z0JBQy9CLE1BQU0sUUFBUSxHQUFHLEtBQUssQ0FBQyxTQUFTLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxLQUFLLENBQUMsQ0FBQztnQkFDckQsUUFBUSxJQUFJLElBQUksUUFBUSxFQUFFLENBQUM7YUFDNUI7WUFDRCxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUM7U0FDakM7UUFDRCxPQUFPLGdCQUFnQixDQUFDO0lBQzFCLENBQUM7SUFFRDs7Ozs7Ozs7O09BU0c7SUFDTyxpQkFBaUI7UUFDekIsT0FBTyxDQUFDLElBQWMsRUFBRSxFQUFFO1lBQ3hCLE1BQU0sVUFBVSxHQUFhLEVBQUUsQ0FBQztZQUVoQyxNQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQ2pELE1BQU0sT0FBTyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQ3RCLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUM7WUFDbEUsTUFBTSxhQUFhLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FDNUIsSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxNQUFNLEVBQ3hDLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBRWxELE1BQU0sYUFBYSxHQUFhLEVBQUUsQ0FBQztZQUVuQyw4REFBOEQ7WUFDOUQsZ0VBQWdFO1lBQ2hFLFlBQVk7WUFDWixNQUFNLGlCQUFpQixHQUFHLEdBQUcsRUFBRTtnQkFDN0IsTUFBTSxLQUFLLEdBQUcsRUFBRSxDQUFDO2dCQUNqQixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7b0JBQzNDLEtBQUssQ0FBQyxJQUFJLENBQUMsRUFBQyxHQUFHLEVBQUUsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUMsQ0FBQztpQkFDckQ7Z0JBQ0QsTUFBTSxRQUFRLEdBQUcsSUFBSSxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUM7Z0JBQ3JDLE1BQU0sT0FBTyxHQUNULE9BQU8sQ0FBQyxJQUFJLENBQUMsT0FBTyxFQUFFLFFBQVEsRUFBRSxFQUFDLFVBQVUsRUFBRSxJQUFJLEVBQUMsQ0FBYSxDQUFDO2dCQUNwRSwrREFBK0Q7Z0JBQy9ELGtCQUFrQjtnQkFFbEIsSUFBSSxTQUFpQixDQUFDO2dCQUN0QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7b0JBQ2xELE1BQU0sWUFBWSxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQzNDLElBQUksSUFBSSxHQUFHLFlBQVksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQ2hELElBQUksYUFBYSxDQUFDLENBQUMsQ0FBQyxJQUFJLElBQUksRUFBRTt3QkFDNUIsSUFBSSxHQUFHLG1CQUFtQixDQUFDLElBQUksRUFBRSxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztxQkFDcEQ7b0JBRUQsbUNBQW1DO29CQUNuQyxNQUFNLFFBQVEsR0FBVyxHQUFHLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO29CQUN4Qyx5REFBeUQ7b0JBQ3pELFVBQVUsQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUM7b0JBQzFCLElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRTt3QkFDWCxTQUFTLEdBQUcsSUFBSSxDQUFDO3FCQUNsQjt5QkFBTTt3QkFDTCxTQUFTLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxTQUFTLEVBQUUsSUFBSSxDQUFDLENBQUM7cUJBQ3RDO2lCQUNGO2dCQUVELHVCQUF1QjtnQkFDdkIsMERBQTBEO2dCQUMxRCx3Q0FBd0M7Z0JBQ3hDLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsY0FBYyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtvQkFDbkQsSUFBSSxjQUFzQixDQUFDO29CQUUzQixJQUFJLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxDQUFDLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxNQUFNLEVBQUU7d0JBQ3RELGNBQWMsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUM7cUJBQ2hDO3lCQUFNO3dCQUNMLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7d0JBQ3pDLE1BQU0sV0FBVyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7d0JBQzlDLGNBQWM7NEJBQ1YsR0FBRyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUM7cUJBQ2xFO29CQUVELEdBQUcsQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLENBQUM7b0JBQ3pCLHlEQUF5RDtvQkFDekQsYUFBYSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQztpQkFDcEM7Z0JBRUQsU0FBUyxHQUFHLEdBQUcsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7Z0JBRWhDLDZCQUE2QjtnQkFDN0IsSUFBSSxDQUFDLGVBQWUsRUFBRSxDQUFDLE9BQU8sQ0FBQyxlQUFlLENBQUMsRUFBRTtvQkFDL0MsU0FBUyxHQUFHLEdBQUcsQ0FBQyxHQUFHLENBQUMsU0FBUyxFQUFFLGVBQWUsQ0FBQyxDQUFDO2dCQUNsRCxDQUFDLENBQUMsQ0FBQztnQkFFSCxPQUFPLFNBQW1CLENBQUM7WUFDN0IsQ0FBQyxDQUFDO1lBRUYsTUFBTSxTQUFTLEdBQUcsSUFBSSxDQUFDLHlCQUF5QixDQUFDLEdBQUcsQ0FDaEQsS0FBSyxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsSUFBSSxFQUFrQixDQUFDLENBQUM7WUFDM0MsTUFBTSxVQUFVLEdBQUcsSUFBSSxDQUFDO1lBQ3hCLE1BQU0sY0FBYyxHQUNoQixJQUFJLENBQUMsVUFBVSxDQUFDLFFBQVEsQ0FBQyxpQkFBaUIsRUFBRSxVQUFVLEVBQUUsU0FBUyxDQUFDLENBQUM7WUFFdkUsT0FBTyxDQUFDLGNBQWMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUNoRCxDQUFDLENBQUM7SUFDSixDQUFDO0lBRUQ7Ozs7T0FJRztJQUNLLGdCQUFnQjtRQUN0QixJQUFJLENBQUMsWUFBWSxHQUFHLENBQUMsSUFBYyxFQUFFLEVBQUU7WUFDckMsT0FBTyxHQUFHLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtnQkFDbkIsTUFBTSxVQUFVLEdBQWEsRUFBRSxDQUFDO2dCQUNoQyxJQUFJLFNBQWlCLENBQUM7Z0JBQ3RCLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUM7Z0JBQ2pELE1BQU0sT0FBTyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQ3RCLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUM7Z0JBQ2xFLE1BQU0sS0FBSyxHQUFHLEVBQUUsQ0FBQztnQkFDakIsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO29CQUMzQyxLQUFLLENBQUMsSUFBSSxDQUFDLEVBQUMsR0FBRyxFQUFFLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBQyxDQUFDLENBQUM7aUJBQ3JEO2dCQUNELE1BQU0sUUFBUSxHQUFHLElBQUksUUFBUSxDQUFDLEtBQUssQ0FBQyxDQUFDO2dCQUNyQyxNQUFNLE9BQU8sR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLE9BQU8sRUFBRSxRQUFRLENBQWEsQ0FBQztnQkFDNUQsc0JBQXNCO2dCQUN0QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7b0JBQ2xELE1BQU0sWUFBWSxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQzNDLDBEQUEwRDtvQkFDMUQsYUFBYTtvQkFDYixNQUFNLElBQUksR0FBVyxHQUFHLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDcEUsSUFBSSxDQUFDLEtBQUssQ0FBQyxFQUFFO3dCQUNYLFNBQVMsR0FBRyxJQUFJLENBQUM7cUJBQ2xCO3lCQUFNO3dCQUNMLFNBQVMsR0FBRyxHQUFHLENBQUMsR0FBRyxDQUFDLFNBQVMsRUFBRSxJQUFJLENBQUMsQ0FBQztxQkFDdEM7b0JBQ0QsVUFBVSxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQztpQkFDNUI7Z0JBQ0QsdUJBQXVCO2dCQUN2QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLGNBQWMsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7b0JBQ25ELE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQ3pDLE1BQU0sV0FBVyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQzlDLGlFQUFpRTtvQkFDakUsTUFBTSxVQUFVLEdBQ1osR0FBRyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQ2pFLFVBQVUsQ0FBQyxJQUFJLENBQUMsVUFBb0IsQ0FBQyxDQUFDO2lCQUN2QztnQkFDRCxPQUFPLFVBQVUsQ0FBQztZQUNwQixDQUFDLENBQUMsQ0FBQztRQUNMLENBQUMsQ0FBQztJQUNKLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O09BaUNHO0lBQ0gsS0FBSyxDQUFDLEdBQUcsQ0FDTCxDQUFnRCxFQUNoRCxDQUFnRCxFQUNoRCxPQUFxQixFQUFFO1FBQ3pCLE9BQU8sVUFBVSxDQUFDLElBQUksRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3RDLENBQUM7SUFFRCx1RUFBdUU7SUFDdkUsNEJBQTRCO0lBQzVCOzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQW9CRztJQUNILEtBQUssQ0FBQyxVQUFVLENBQUksT0FBbUIsRUFBRSxJQUE0QjtRQUVuRSxPQUFPLFVBQVUsQ0FBQyxJQUFJLEVBQUUsT0FBTyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3pDLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQXNCRztJQUNILEtBQUssQ0FBQyxZQUFZLENBQ2QsQ0FBZ0QsRUFDaEQsQ0FDNkI7UUFDL0Isb0RBQW9EO1FBQ3BELHVDQUF1QztRQUN2QyxNQUFNLGNBQWMsR0FBRyxNQUFNLElBQUksQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDNUQsTUFBTSxNQUFNLEdBQUcsY0FBYyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLE1BQU0sT0FBTyxHQUFHLGNBQWMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNsQyxNQUFNLGFBQWEsR0FBRyxJQUFJLENBQUMsaUJBQWlCLEVBQUUsQ0FBQztRQUMvQyxNQUFNLE1BQU0sR0FBRyxhQUFhLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3JELE1BQU0sVUFBVSxHQUFhLEVBQUUsQ0FBQztRQUNoQyxLQUFLLE1BQU0sSUFBSSxJQUFJLE1BQU0sRUFBRTtZQUN6QixNQUFNLENBQUMsR0FBRyxNQUFNLElBQUksQ0FBQyxJQUFJLEVBQUUsQ0FBQztZQUM1QixVQUFVLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ3ZCO1FBQ0QsR0FBRyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUNwQixpQkFBaUIsQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEMsaUJBQWlCLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3hDLE9BQU8sZ0JBQWdCLENBQUMsVUFBVSxDQUFDLENBQUM7SUFDdEMsQ0FBQztJQUVEOzs7Ozs7OztPQVFHO0lBQ08sZUFBZSxDQUFDLE1BQXNCO1FBQzlDLE1BQU0sWUFBWSxHQUFrQixFQUFFLENBQUM7UUFFdkMsTUFBTSxhQUFhLEdBQUcsTUFBTSxJQUFJLElBQUksSUFBSSxNQUFNLENBQUMsYUFBYSxDQUFDO1FBQzdELE1BQU0sT0FBTyxHQUFHLGFBQWEsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDO1FBQ3JFLE1BQU0sWUFBWSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDcEQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDdkMsSUFBSSxhQUFhLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUyxFQUFFO2dCQUMxQyx5Q0FBeUM7Z0JBQ3pDLFNBQVM7YUFDVjtZQUNELFlBQVksQ0FBQyxJQUFJLENBQ2IsRUFBQyxJQUFJLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxNQUFNLEVBQUUsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUMsQ0FBQztTQUMvRDtRQUNELE9BQU8sWUFBWSxDQUFDO0lBQ3RCLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7T0E2Qkc7SUFDSCxJQUFJLFlBQVksQ0FBQyxJQUFhO1FBQzVCLElBQUksQ0FBQyxhQUFhLEdBQUcsSUFBSSxDQUFDO0lBQzVCLENBQUM7SUFFRCxJQUFJLFlBQVk7UUFDZCxPQUFPLElBQUksQ0FBQyxhQUFhLENBQUM7SUFDNUIsQ0FBQztJQUVELElBQUksU0FBUztRQUNYLE9BQU8sSUFBSSxDQUFDLFVBQVUsQ0FBQztJQUN6QixDQUFDO0lBRUQsSUFBSSxTQUFTLENBQUMsU0FBb0I7UUFDaEMsSUFBSSxJQUFJLENBQUMsVUFBVSxLQUFLLFNBQVMsRUFBRTtZQUNqQyxJQUFJLENBQUMsVUFBVSxHQUFHLFNBQVMsQ0FBQztZQUM1QixJQUFJLENBQUMsZ0JBQWdCLEdBQUcsS0FBSyxDQUFDO1NBQy9CO0lBQ0gsQ0FBQztJQUVELE9BQU87UUFDTCxNQUFNLE1BQU0sR0FBRyxLQUFLLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDL0IsSUFBSSxNQUFNLENBQUMsb0JBQW9CLEtBQUssQ0FBQyxJQUFJLElBQUksQ0FBQyxTQUFTLElBQUksSUFBSTtZQUMzRCxJQUFJLENBQUMsZ0JBQWdCLEVBQUU7WUFDekIsTUFBTSxnQ0FBZ0MsR0FBRyxHQUFHLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDO1lBQ2pFLElBQUksQ0FBQyxVQUFVLENBQUMsT0FBTyxFQUFFLENBQUM7WUFDMUIsTUFBTSxDQUFDLG9CQUFvQjtnQkFDdkIsZ0NBQWdDLEdBQUcsR0FBRyxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQztTQUNoRTtRQUNELE9BQU8sTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFFTyxrQkFBa0I7UUFFeEIsSUFBSSxTQUNzQyxDQUFDO1FBQzNDLElBQUksT0FBTyxJQUFJLENBQUMsSUFBSSxLQUFLLFFBQVEsRUFBRTtZQUNqQyxTQUFTLEdBQUcsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQW1CLENBQUM7U0FDdEQ7YUFBTSxJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFO1lBQ25DLEtBQUssTUFBTSxJQUFJLElBQUksSUFBSSxDQUFDLElBQUksRUFBRTtnQkFDNUIsSUFBSSxPQUFPLElBQUksS0FBSyxRQUFRLEVBQUU7b0JBQzVCLE1BQU0sSUFBSSxLQUFLLENBQUMsb0RBQW9ELENBQUMsQ0FBQztpQkFDdkU7YUFDRjtZQUNELFNBQVMsR0FBSSxJQUFJLENBQUMsSUFBaUIsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLENBQzdDLENBQUM7U0FDdEI7YUFBTTtZQUNMLE1BQU0sV0FBVyxHQUFHLE1BQU0sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1lBQzNDLFNBQVMsR0FBRyxFQUE0QyxDQUFDO1lBQ3pELE1BQU0sTUFBTSxHQUNSLElBQUksQ0FBQyxJQUF1RCxDQUFDO1lBQ2pFLEtBQUssTUFBTSxVQUFVLElBQUksV0FBVyxFQUFFO2dCQUNwQyxJQUFJLE9BQU8sTUFBTSxDQUFDLFVBQVUsQ0FBQyxLQUFLLFFBQVEsRUFBRTtvQkFDMUMsU0FBUyxDQUFDLFVBQVUsQ0FBQzt3QkFDakIsV0FBVyxDQUFDLE1BQU0sQ0FBQyxVQUFVLENBQVcsQ0FBbUIsQ0FBQztpQkFDakU7cUJBQU07b0JBQ0wsTUFBTSxJQUFJLEtBQUssQ0FBQyxvREFBb0QsQ0FBQyxDQUFDO2lCQUN2RTthQUNGO1NBQ0Y7UUFDRCxPQUFPLFNBQVMsQ0FBQztJQUNuQixDQUFDO0lBRU8sb0JBQW9CO1FBRTFCLElBQUksT0FBTyxJQUFJLENBQUMsT0FBTyxLQUFLLFFBQVE7WUFDaEMsT0FBTyxJQUFJLENBQUMsT0FBTyxLQUFLLFVBQVUsRUFBRTtZQUN0QyxPQUFPLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ2pFO2FBQU0sSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsRUFBRTtZQUN0QyxPQUFPLElBQUksQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUNuQixNQUFNLENBQUMsRUFBRSxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsbUJBQW1CLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ2pFO2FBQU07WUFDTCxNQUFNLGtCQUFrQixHQUF1QyxFQUFFLENBQUM7WUFDbEUsS0FBSyxNQUFNLEdBQUcsSUFBSSxJQUFJLENBQUMsT0FBTyxFQUFFO2dCQUM5QixrQkFBa0IsQ0FBQyxHQUFHLENBQUM7b0JBQ25CLFdBQVcsQ0FBQyxPQUFPLENBQUMsbUJBQW1CLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDakU7WUFDRCxPQUFPLGtCQUFrQixDQUFDO1NBQzNCO0lBQ0gsQ0FBQztJQUVTLGlCQUFpQjtRQUN6QixPQUFPO1lBQ0wsSUFBSSxFQUFFLElBQUksQ0FBQyxrQkFBa0IsRUFBRTtZQUMvQixPQUFPLEVBQUUsSUFBSSxDQUFDLG9CQUFvQixFQUFFO1lBQ3BDLGdCQUFnQixFQUFFO2dCQUNoQixVQUFVLEVBQUUsSUFBSSxDQUFDLFNBQVMsQ0FBQyxZQUFZLEVBQUU7Z0JBQ3pDLE1BQU0sRUFBRSxJQUFJLENBQUMsU0FBUyxDQUFDLFNBQVMsRUFBRTthQUNUO1NBQzVCLENBQUM7UUFDRiwwREFBMEQ7UUFDMUQsMERBQTBEO1FBQzFELG9EQUFvRDtJQUN0RCxDQUFDO0lBRUQsa0JBQWtCLENBQUMsY0FBOEI7UUFDL0MsSUFBSSxjQUFjLENBQUMsZ0JBQWdCLElBQUksSUFBSSxFQUFFO1lBQzNDLE1BQU0sSUFBSSxLQUFLLENBQUMsOENBQThDLENBQUMsQ0FBQztTQUNqRTtRQUNELElBQUksY0FBYyxDQUFDLFlBQVksSUFBSSxJQUFJLEVBQUU7WUFDdkMsTUFBTSxJQUFJLEtBQUssQ0FBQyw0Q0FBNEMsQ0FBQyxDQUFDO1NBQy9EO1FBQ0QsSUFBSSxjQUFjLENBQUMsa0JBQWtCLElBQUksSUFBSSxFQUFFO1lBQzdDLE1BQU0sSUFBSSxLQUFLLENBQUMsa0RBQWtELENBQUMsQ0FBQztTQUNyRTtRQUVELE1BQU0sUUFBUSxHQUFHLG1CQUFtQixDQUFDLGNBQWMsQ0FBQyxnQkFBZ0IsQ0FDeEMsQ0FBQztRQUM3QixNQUFNLFNBQVMsR0FBRyxXQUFXLENBQUMsUUFBUSxDQUFjLENBQUM7UUFFckQsSUFBSSxJQUFJLENBQUM7UUFDVCxJQUFJLE9BQU8sY0FBYyxDQUFDLElBQUksS0FBSyxRQUFRLEVBQUU7WUFDM0MsSUFBSSxHQUFHLFdBQVcsQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLENBQUM7U0FDekM7YUFBTSxJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxFQUFFO1lBQzdDLElBQUksR0FBRyxjQUFjLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxTQUFTLENBQUMsRUFBRSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO1NBQ3JFO2FBQU0sSUFBSSxjQUFjLENBQUMsSUFBSSxJQUFJLElBQUksRUFBRTtZQUN0QyxJQUFJLEdBQUcsRUFBNEMsQ0FBQztZQUNwRCxLQUFLLE1BQU0sR0FBRyxJQUFJLGNBQWMsQ0FBQyxJQUFJLEVBQUU7Z0JBQ3JDLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxXQUFXLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBbUIsQ0FBQzthQUNyRTtTQUNGO1FBRUQsSUFBSSxPQUFPLENBQUM7UUFDWixJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsY0FBYyxDQUFDLE9BQU8sQ0FBQyxFQUFFO1lBQ3pDLE9BQU8sR0FBRyxjQUFjLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDO1NBQ3JFO2FBQU0sSUFBSSxjQUFjLENBQUMsT0FBTyxJQUFJLElBQUksRUFBRTtZQUN6QyxPQUFPLEdBQUcsRUFBK0MsQ0FBQztZQUMxRCxLQUFLLE1BQU0sR0FBRyxJQUFJLGNBQWMsQ0FBQyxPQUFPLEVBQUU7Z0JBQ3hDLE9BQU8sQ0FBQyxHQUFHLENBQUMsR0FBRyxXQUFXLENBQUMsY0FBYyxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO2FBQ3pEO1NBQ0Y7UUFFRCxJQUFJLENBQUMsT0FBTyxDQUFDLEVBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxTQUFTLEVBQUMsQ0FBQyxDQUFDO0lBQzNDLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7T0FnRkc7SUFDSCxLQUFLLENBQUMsSUFBSSxDQUFDLFlBQWlDLEVBQUUsTUFBc0I7UUFFbEUsSUFBSSxPQUFPLFlBQVksS0FBSyxRQUFRLEVBQUU7WUFDcEMsTUFBTSxRQUFRLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxZQUFZLENBQUMsQ0FBQztZQUNsRCxJQUFJLFFBQVEsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO2dCQUN6QixNQUFNLElBQUksVUFBVSxDQUNoQiwwQ0FBMEMsWUFBWSxHQUFHLENBQUMsQ0FBQzthQUNoRTtpQkFBTSxJQUFJLFFBQVEsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxFQUFFO2dCQUM5QixNQUFNLElBQUksVUFBVSxDQUNoQix3QkFBd0IsUUFBUSxDQUFDLE1BQU0sc0JBQXNCO29CQUM3RCxRQUFRLFlBQVksR0FBRyxDQUFDLENBQUM7YUFDOUI7WUFDRCxZQUFZLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQzVCO1FBQ0QsSUFBSSxZQUFZLENBQUMsSUFBSSxJQUFJLElBQUksRUFBRTtZQUM3QixNQUFNLElBQUksVUFBVSxDQUNoQiwwREFBMEQ7Z0JBQzFELHNEQUFzRCxDQUFDLENBQUM7U0FDN0Q7UUFFRCxNQUFNLGtCQUFrQixHQUNwQixNQUFNLEVBQUUsQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDO1FBRXpELE1BQU0sWUFBWSxHQUFHLEtBQUssQ0FBQztRQUMzQixNQUFNLFNBQVMsR0FBTyxJQUFJLENBQUM7UUFDM0IsTUFBTSxXQUFXLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxTQUFTLEVBQUUsWUFBWSxDQUFDLENBQUM7UUFDekQsTUFBTSxjQUFjLEdBQXNCO1lBQ3hDLGFBQWEsRUFBRSxXQUFXO1lBQzFCLE1BQU0sRUFBRSx3QkFBd0I7WUFDaEMsV0FBVyxFQUFFLDhCQUE4QixPQUFPLEVBQUU7WUFDcEQsV0FBVyxFQUFFLElBQUk7U0FDbEIsQ0FBQztRQUVGLE1BQU0sZ0JBQWdCLEdBQUcsTUFBTSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsZ0JBQWdCLENBQUM7UUFDMUUsSUFBSSxnQkFBZ0IsSUFBSSxJQUFJLENBQUMsU0FBUyxJQUFJLElBQUksRUFBRTtZQUM5QyxjQUFjLENBQUMsY0FBYyxHQUFHLElBQUksQ0FBQyxpQkFBaUIsRUFBRSxDQUFDO1lBQ3pELE1BQU0sVUFBVSxHQUFHLFdBQVcsQ0FBQztZQUMvQixNQUFNLEVBQUMsSUFBSSxFQUFFLG1CQUFtQixFQUFFLEtBQUssRUFBRSxvQkFBb0IsRUFBQyxHQUMxRCxNQUFNLEVBQUUsQ0FBQyxhQUFhLENBQUMsTUFBTSxJQUFJLENBQUMsU0FBUyxDQUFDLFVBQVUsRUFBRSxFQUFFLFVBQVUsQ0FBQyxDQUFDO1lBQzFFLGtCQUFrQixDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsR0FBRyxvQkFBb0IsQ0FBQyxDQUFDO1lBQ3ZELGtCQUFrQixDQUFDLElBQUksR0FBRyxFQUFFLENBQUMsdUJBQXVCLENBQ2hELENBQUMsa0JBQWtCLENBQUMsSUFBSSxFQUFFLG1CQUFtQixDQUFDLENBQUMsQ0FBQztTQUNyRDtRQUVELElBQUksSUFBSSxDQUFDLG1CQUFtQixJQUFJLElBQUksRUFBRTtZQUNwQyxrREFBa0Q7WUFDbEQsTUFBTSxTQUFTLEdBQUcsSUFBSSxDQUFDO1lBQ3ZCLHdCQUF3QixDQUFDLElBQUksQ0FBQyxtQkFBbUIsRUFBRSxJQUFJLENBQUMsSUFBSSxFQUFFLFNBQVMsQ0FBQyxDQUFDO1lBQ3pFLGNBQWMsQ0FBQyxtQkFBbUIsR0FBRyxJQUFJLENBQUMsbUJBQW1CLENBQUM7U0FDL0Q7UUFFRCxjQUFjLENBQUMsVUFBVSxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQztRQUNwRCxjQUFjLENBQUMsV0FBVyxHQUFHLGtCQUFrQixDQUFDLEtBQUssQ0FBQztRQUN0RCxPQUFPLFlBQVksQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLENBQUM7SUFDM0MsQ0FBQztJQUVEOzs7Ozs7O09BT0c7SUFDSCxzQkFBc0IsQ0FBQyxtQkFBdUI7UUFDNUMsd0JBQXdCLENBQUMsbUJBQW1CLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3pELElBQUksQ0FBQyxtQkFBbUIsR0FBRyxtQkFBbUIsQ0FBQztJQUNqRCxDQUFDO0lBRUQ7Ozs7Ozs7Ozs7T0FVRztJQUNILHNCQUFzQjtRQUNwQixPQUFPLElBQUksQ0FBQyxtQkFBbUIsQ0FBQztJQUNsQyxDQUFDOztBQTc0Q0Qsb0VBQW9FO0FBQ3BFLDRFQUE0RTtBQUM1RSxrQkFBa0I7QUFDWCxxQkFBUyxHQUFHLE9BQU8sQ0FBQztBQTQ0QzdCLGFBQWEsQ0FBQyxhQUFhLENBQUMsV0FBVyxDQUFDLENBQUM7QUFFekM7Ozs7O0dBS0c7QUFDSCxzREFBc0Q7QUFDdEQsTUFBTSxPQUFPLFVBQVcsU0FBUSxXQUFXOztBQUNsQyxvQkFBUyxHQUFHLFlBQVksQ0FBQztBQUVsQyxhQUFhLENBQUMsYUFBYSxDQUFDLFVBQVUsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZVxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XG4gKiBodHRwczovL29wZW5zb3VyY2Uub3JnL2xpY2Vuc2VzL01JVC5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyogT3JpZ2luYWwgU291cmNlOiBlbmdpbmUvdHJhaW5pbmcucHkgKi9cblxuaW1wb3J0ICogYXMgdGZjIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2lvLCBNb2RlbFByZWRpY3RDb25maWcgYXMgTW9kZWxQcmVkaWN0QXJncywgTmFtZWRUZW5zb3JNYXAsIE9wdGltaXplciwgU2NhbGFyLCBzY2FsYXIsIHNlcmlhbGl6YXRpb24sIFRlbnNvciwgVGVuc29yMUQsIHRlbnNvcjFkLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQgKiBhcyBLIGZyb20gJy4uL2JhY2tlbmQvdGZqc19iYWNrZW5kJztcbmltcG9ydCB7SGlzdG9yeSwgTW9kZWxMb2dnaW5nVmVyYm9zaXR5fSBmcm9tICcuLi9iYXNlX2NhbGxiYWNrcyc7XG5pbXBvcnQge25hbWVTY29wZX0gZnJvbSAnLi4vY29tbW9uJztcbmltcG9ydCB7Tm90SW1wbGVtZW50ZWRFcnJvciwgUnVudGltZUVycm9yLCBWYWx1ZUVycm9yfSBmcm9tICcuLi9lcnJvcnMnO1xuaW1wb3J0IHtTaGFwZX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge0xvc3NJZGVudGlmaWVyfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvbG9zc19jb25maWcnO1xuaW1wb3J0IHtPcHRpbWl6ZXJTZXJpYWxpemF0aW9ufSBmcm9tICcuLi9rZXJhc19mb3JtYXQvb3B0aW1pemVyX2NvbmZpZyc7XG5pbXBvcnQge01ldHJpY3NJZGVudGlmaWVyLCBUcmFpbmluZ0NvbmZpZ30gZnJvbSAnLi4va2VyYXNfZm9ybWF0L3RyYWluaW5nX2NvbmZpZyc7XG5pbXBvcnQge2Rlc2VyaWFsaXplfSBmcm9tICcuLi9sYXllcnMvc2VyaWFsaXphdGlvbic7XG5pbXBvcnQgKiBhcyBsb3NzZXMgZnJvbSAnLi4vbG9zc2VzJztcbmltcG9ydCAqIGFzIE1ldHJpY3MgZnJvbSAnLi4vbWV0cmljcyc7XG5pbXBvcnQgKiBhcyBvcHRpbWl6ZXJzIGZyb20gJy4uL29wdGltaXplcnMnO1xuaW1wb3J0IHtMb3NzT3JNZXRyaWNGbiwgTmFtZWRUZW5zb3J9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7Y2hlY2tVc2VyRGVmaW5lZE1ldGFkYXRhfSBmcm9tICcuLi91c2VyX2RlZmluZWRfbWV0YWRhdGEnO1xuaW1wb3J0IHtjb3VudCwgcHlMaXN0UmVwZWF0LCBzaW5nbGV0b25PckFycmF5LCB0b0NhbWVsQ2FzZSwgdG9TbmFrZUNhc2UsIHVuaXF1ZX0gZnJvbSAnLi4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5pbXBvcnQge3ByaW50U3VtbWFyeX0gZnJvbSAnLi4vdXRpbHMvbGF5ZXJfdXRpbHMnO1xuaW1wb3J0IHtyYW5nZX0gZnJvbSAnLi4vdXRpbHMvbWF0aF91dGlscyc7XG5pbXBvcnQge2NvbnZlcnRQeXRob25pY1RvVHN9IGZyb20gJy4uL3V0aWxzL3NlcmlhbGl6YXRpb25fdXRpbHMnO1xuaW1wb3J0IHtMYXllclZhcmlhYmxlfSBmcm9tICcuLi92YXJpYWJsZXMnO1xuaW1wb3J0IHt2ZXJzaW9ufSBmcm9tICcuLi92ZXJzaW9uJztcblxuaW1wb3J0IHtDb250YWluZXIsIENvbnRhaW5lckFyZ3N9IGZyb20gJy4vY29udGFpbmVyJztcbmltcG9ydCB7RGF0YXNldH0gZnJvbSAnLi9kYXRhc2V0X3N0dWInO1xuaW1wb3J0IHtleGVjdXRlLCBGZWVkRGljdH0gZnJvbSAnLi9leGVjdXRvcic7XG5pbXBvcnQge0Rpc3Bvc2VSZXN1bHQsIFN5bWJvbGljVGVuc29yfSBmcm9tICcuL3RvcG9sb2d5JztcbmltcG9ydCB7ZXZhbHVhdGVEYXRhc2V0LCBmaXREYXRhc2V0LCBNb2RlbEV2YWx1YXRlRGF0YXNldEFyZ3MsIE1vZGVsRml0RGF0YXNldEFyZ3N9IGZyb20gJy4vdHJhaW5pbmdfZGF0YXNldCc7XG5pbXBvcnQge2NoZWNrQmF0Y2hTaXplLCBkaXNwb3NlTmV3VGVuc29ycywgZW5zdXJlVGVuc29yc1JhbmsyT3JIaWdoZXIsIGZpdFRlbnNvcnMsIG1ha2VCYXRjaGVzLCBNb2RlbEZpdEFyZ3MsIHNsaWNlQXJyYXlzLCBzbGljZUFycmF5c0J5SW5kaWNlc30gZnJvbSAnLi90cmFpbmluZ190ZW5zb3JzJztcbmltcG9ydCB7Q2xhc3NXZWlnaHQsIENsYXNzV2VpZ2h0TWFwLCBjb21wdXRlV2VpZ2h0ZWRMb3NzLCBzdGFuZGFyZGl6ZUNsYXNzV2VpZ2h0cywgc3RhbmRhcmRpemVXZWlnaHRzfSBmcm9tICcuL3RyYWluaW5nX3V0aWxzJztcblxuLyoqXG4gKiBIZWxwZXIgZnVuY3Rpb24gZm9yIHBvbHltb3JwaGljIGlucHV0IGRhdGE6IDEuIHNpbmdsZXRvbiBUZW5zb3IuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpc0RhdGFUZW5zb3IoeDogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9fFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7W2lucHV0TmFtZTogc3RyaW5nXTogVGVuc29yW119KTogYm9vbGVhbiB7XG4gIHJldHVybiB4IGluc3RhbmNlb2YgVGVuc29yO1xufVxuXG4vKipcbiAqIEhlbHBlciBmdW5jdGlvbiBmb3IgcG9seW1vcnBoaWMgaW5wdXQgZGF0YTogMi4gQXJyYXkgb2YgVGVuc29yLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXNEYXRhQXJyYXkoeDogVGVuc29yfFRlbnNvcltdfFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9KTogYm9vbGVhbiB7XG4gIHJldHVybiBBcnJheS5pc0FycmF5KHgpO1xufVxuXG4vKipcbiAqIEhlbHBlciBmdW5jdGlvbiBmb3IgcG9seW1vcnBoaWMgaW5wdXQgZGF0YTogMy4gXCJkaWN0XCIgb2YgVGVuc29yLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXNEYXRhRGljdCh4OiBUZW5zb3J8VGVuc29yW118XG4gICAgICAgICAgICAgICAgICAgICAgICAgICB7W2lucHV0TmFtZTogc3RyaW5nXTogVGVuc29yfSk6IGJvb2xlYW4ge1xuICByZXR1cm4gIWlzRGF0YVRlbnNvcih4KSAmJiAhaXNEYXRhQXJyYXkoeCk7XG59XG5cbi8qKlxuICogTm9ybWFsaXplcyBpbnB1dHMgYW5kIHRhcmdldHMgcHJvdmlkZWQgYnkgdXNlcnMuXG4gKiBAcGFyYW0gZGF0YSBVc2VyLXByb3ZpZGVkIGlucHV0IGRhdGEgKHBvbHltb3JwaGljKS5cbiAqIEBwYXJhbSBuYW1lcyBBbiBBcnJheSBvZiBleHBlY3RlZCBUZW5zb3IgbmFtZXMuXG4gKiBAcGFyYW0gc2hhcGVzIE9wdGlvbmFsIEFycmF5IG9mIGV4cGVjdGVkIFRlbnNvciBzaGFwZXMuXG4gKiBAcGFyYW0gY2hlY2tCYXRjaEF4aXMgV2hldGhlciB0byBjaGVjayB0aGF0IHRoZSBiYXRjaCBheGlzIG9mIHRoZSBhcnJheXNcbiAqICAgbWF0Y2ggIHRoZSBleHBlY3RlZCB2YWx1ZSBmb3VuZCBpbiBgc2hhcGVzYC5cbiAqIEBwYXJhbSBleGNlcHRpb25QcmVmaXggU3RyaW5nIHByZWZpeCB1c2VkIGZvciBleGNlcHRpb24gZm9ybWF0dGluZy5cbiAqIEByZXR1cm5zIExpc3Qgb2Ygc3RhbmRhcmRpemVkIGlucHV0IFRlbnNvcnMgKG9uZSBUZW5zb3IgcGVyIG1vZGVsIGlucHV0KS5cbiAqIEB0aHJvd3MgVmFsdWVFcnJvcjogaW4gY2FzZSBvZiBpbXByb3Blcmx5IGZvcm1hdHRlZCB1c2VyIGRhdGEuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzdGFuZGFyZGl6ZUlucHV0RGF0YShcbiAgICBkYXRhOiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sIG5hbWVzOiBzdHJpbmdbXSxcbiAgICBzaGFwZXM/OiBTaGFwZVtdLCBjaGVja0JhdGNoQXhpcyA9IHRydWUsIGV4Y2VwdGlvblByZWZpeCA9ICcnKTogVGVuc29yW10ge1xuICBpZiAobmFtZXMgPT0gbnVsbCB8fCBuYW1lcy5sZW5ndGggPT09IDApIHtcbiAgICAvLyBDaGVjayBmb3IgdGhlIGNhc2Ugd2hlcmUgdGhlIG1vZGVsIGV4cGVjdGVkIG5vIGRhdGEsIGJ1dCBzb21lIGRhdGEgZ290XG4gICAgLy8gc2VudC5cbiAgICBpZiAoZGF0YSAhPSBudWxsKSB7XG4gICAgICBsZXQgZ290VW5leHBlY3RlZERhdGEgPSBmYWxzZTtcbiAgICAgIGlmIChpc0RhdGFBcnJheShkYXRhKSAmJiAoZGF0YSBhcyBUZW5zb3JbXSkubGVuZ3RoID4gMCkge1xuICAgICAgICBnb3RVbmV4cGVjdGVkRGF0YSA9IHRydWU7XG4gICAgICB9IGVsc2UgaWYgKGlzRGF0YURpY3QoZGF0YSkpIHtcbiAgICAgICAgZm9yIChjb25zdCBrZXkgaW4gZGF0YSkge1xuICAgICAgICAgIGlmIChkYXRhLmhhc093blByb3BlcnR5KGtleSkpIHtcbiAgICAgICAgICAgIGdvdFVuZXhwZWN0ZWREYXRhID0gdHJ1ZTtcbiAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgLy8gYGRhdGFgIGlzIGEgc2luZ2xldG9uIFRlbnNvciBpbiB0aGlzIGNhc2UuXG4gICAgICAgIGdvdFVuZXhwZWN0ZWREYXRhID0gdHJ1ZTtcbiAgICAgIH1cbiAgICAgIGlmIChnb3RVbmV4cGVjdGVkRGF0YSkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBFcnJvciB3aGVuIGNoZWNraW5nIG1vZGVsICR7ZXhjZXB0aW9uUHJlZml4fSBleHBlY3RlZCBubyBkYXRhLCBgICtcbiAgICAgICAgICAgIGBidXQgZ290ICR7ZGF0YX1gKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFtdO1xuICB9XG4gIGlmIChkYXRhID09IG51bGwpIHtcbiAgICByZXR1cm4gbmFtZXMubWFwKG5hbWUgPT4gbnVsbCk7XG4gIH1cblxuICBsZXQgYXJyYXlzOiBUZW5zb3JbXTtcbiAgaWYgKGlzRGF0YURpY3QoZGF0YSkpIHtcbiAgICBkYXRhID0gZGF0YSBhcyB7W2lucHV0TmFtZTogc3RyaW5nXTogVGVuc29yfTtcbiAgICBhcnJheXMgPSBbXTtcbiAgICBmb3IgKGNvbnN0IG5hbWUgb2YgbmFtZXMpIHtcbiAgICAgIGlmIChkYXRhW25hbWVdID09IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgTm8gZGF0YSBwcm92aWRlZCBmb3IgXCIke25hbWV9XCIuIE5lZWQgZGF0YSBmb3IgZWFjaCBrZXkgaW46IGAgK1xuICAgICAgICAgICAgYCR7bmFtZXN9YCk7XG4gICAgICB9XG4gICAgICBhcnJheXMucHVzaChkYXRhW25hbWVdKTtcbiAgICB9XG4gIH0gZWxzZSBpZiAoaXNEYXRhQXJyYXkoZGF0YSkpIHtcbiAgICBkYXRhID0gZGF0YSBhcyBUZW5zb3JbXTtcbiAgICBpZiAoZGF0YS5sZW5ndGggIT09IG5hbWVzLmxlbmd0aCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYEVycm9yIHdoZW4gY2hlY2tpbmcgbW9kZWwgJHtleGNlcHRpb25QcmVmaXh9OiB0aGUgQXJyYXkgb2YgYCArXG4gICAgICAgICAgYFRlbnNvcnMgdGhhdCB5b3UgYXJlIHBhc3NpbmcgdG8geW91ciBtb2RlbCBpcyBub3QgdGhlIHNpemUgdGhlIGAgK1xuICAgICAgICAgIGBtb2RlbCBleHBlY3RlZC4gRXhwZWN0ZWQgdG8gc2VlICR7bmFtZXMubGVuZ3RofSBUZW5zb3IocyksIGJ1dCBgICtcbiAgICAgICAgICBgaW5zdGVhZCBnb3QgdGhlIGZvbGxvd2luZyBsaXN0IG9mIFRlbnNvcihzKTogJHtkYXRhfWApO1xuICAgIH1cbiAgICBhcnJheXMgPSBkYXRhO1xuICB9IGVsc2Uge1xuICAgIGRhdGEgPSBkYXRhIGFzIFRlbnNvcjtcbiAgICBpZiAobmFtZXMubGVuZ3RoID4gMSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYFRoZSBtb2RlbCAke2V4Y2VwdGlvblByZWZpeH0gZXhwZWN0cyAke25hbWVzLmxlbmd0aH0gVGVuc29yKHMpLCBgICtcbiAgICAgICAgICBgYnV0IG9ubHkgcmVjZWl2ZWQgb25lIFRlbnNvci4gRm91bmQ6IFRlbnNvciB3aXRoIHNoYXBlICR7XG4gICAgICAgICAgICAgIGRhdGEuc2hhcGV9YCk7XG4gICAgfVxuICAgIGFycmF5cyA9IFtkYXRhXTtcbiAgfVxuXG4gIGFycmF5cyA9IGVuc3VyZVRlbnNvcnNSYW5rMk9ySGlnaGVyKGFycmF5cyk7XG5cbiAgLy8gQ2hlY2sgc2hhcGUgY29tcGF0aWJpbGl0eS5cbiAgaWYgKHNoYXBlcyAhPSBudWxsKSB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuYW1lcy5sZW5ndGg7ICsraSkge1xuICAgICAgaWYgKHNoYXBlc1tpXSA9PSBudWxsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgY29uc3QgYXJyYXkgPSBhcnJheXNbaV07XG4gICAgICBpZiAoYXJyYXkuc2hhcGUubGVuZ3RoICE9PSBzaGFwZXNbaV0ubGVuZ3RoKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY2hlY2tpbmcgJHtleGNlcHRpb25QcmVmaXh9OiBleHBlY3RlZCAke25hbWVzW2ldfSBgICtcbiAgICAgICAgICAgIGB0byBoYXZlICR7c2hhcGVzW2ldLmxlbmd0aH0gZGltZW5zaW9uKHMpLiBidXQgZ290IGFycmF5IHdpdGggYCArXG4gICAgICAgICAgICBgc2hhcGUgJHthcnJheS5zaGFwZX1gKTtcbiAgICAgIH1cbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgc2hhcGVzW2ldLmxlbmd0aDsgKytqKSB7XG4gICAgICAgIGlmIChqID09PSAwICYmICFjaGVja0JhdGNoQXhpcykge1xuICAgICAgICAgIC8vIFNraXAgdGhlIGZpcnN0IChiYXRjaCkgYXhpcy5cbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBkaW0gPSBhcnJheS5zaGFwZVtqXTtcbiAgICAgICAgY29uc3QgcmVmRGltID0gc2hhcGVzW2ldW2pdO1xuICAgICAgICBpZiAocmVmRGltICE9IG51bGwgJiYgcmVmRGltID49IDAgJiYgZGltICE9PSByZWZEaW0pIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgYCR7ZXhjZXB0aW9uUHJlZml4fSBleHBlY3RlZCBhIGJhdGNoIG9mIGVsZW1lbnRzIHdoZXJlIGVhY2ggYCArXG4gICAgICAgICAgICAgIGBleGFtcGxlIGhhcyBzaGFwZSBbJHtzaGFwZXNbaV0uc2xpY2UoMSwgc2hhcGVzW2ldLmxlbmd0aCl9XSBgICtcbiAgICAgICAgICAgICAgYChpLmUuLHRlbnNvciBzaGFwZSBbKiwke1xuICAgICAgICAgICAgICAgICAgc2hhcGVzW2ldLnNsaWNlKDEsIHNoYXBlc1tpXS5sZW5ndGgpfV0pYCArXG4gICAgICAgICAgICAgIGAgYnV0IHRoZSAke2V4Y2VwdGlvblByZWZpeH0gcmVjZWl2ZWQgYW4gaW5wdXQgd2l0aCAke1xuICAgICAgICAgICAgICAgICAgYXJyYXkuc2hhcGVbMF19YCArXG4gICAgICAgICAgICAgIGAgZXhhbXBsZXMsIGVhY2ggd2l0aCBzaGFwZSBbJHtcbiAgICAgICAgICAgICAgICAgIGFycmF5LnNoYXBlLnNsaWNlKDEsIGFycmF5LnNoYXBlLmxlbmd0aCl9XWAgK1xuICAgICAgICAgICAgICBgICh0ZW5zb3Igc2hhcGUgWyR7YXJyYXkuc2hhcGV9XSlgKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxuICByZXR1cm4gYXJyYXlzO1xufVxuXG4vKipcbiAqIFVzZXIgaW5wdXQgdmFsaWRhdGlvbiBmb3IgVGVuc29ycy5cbiAqIEBwYXJhbSBpbnB1dHMgYEFycmF5YCBvZiBgdGYuVGVuc29yYHMgZm9yIGlucHV0cy5cbiAqIEBwYXJhbSB0YXJnZXRzIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzIGZvciB0YXJnZXRzLlxuICogQHBhcmFtIHdlaWdodHMgT3B0aW9uYWwgYEFycmF5YCBvZiBgdGYuVGVuc29yYHMgZm9yIHNhbXBsZSB3ZWlnaHRzLlxuICogQHRocm93cyBWYWx1ZUVycm9yOiBpbiBjYXNlIG9mIGluY29ycmVjdGx5IGZvcm1hdHRlZCBkYXRhLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tBcnJheUxlbmd0aHMoXG4gICAgaW5wdXRzOiBUZW5zb3JbXSwgdGFyZ2V0czogVGVuc29yW10sIHdlaWdodHM/OiBUZW5zb3JbXSkge1xuICBjb25zdCBzZXRYID0gdW5pcXVlKGlucHV0cy5tYXAoaW5wdXQgPT4gaW5wdXQuc2hhcGVbMF0pKTtcbiAgc2V0WC5zb3J0KCk7XG4gIGNvbnN0IHNldFkgPSB1bmlxdWUodGFyZ2V0cy5tYXAodGFyZ2V0ID0+IHRhcmdldC5zaGFwZVswXSkpO1xuICBzZXRZLnNvcnQoKTtcbiAgLy8gVE9ETyhjYWlzKTogQ2hlY2sgYHdlaWdodHNgIGFzIHdlbGwuXG4gIGlmIChzZXRYLmxlbmd0aCA+IDEpIHtcbiAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgYEFsbCBpbnB1dCBUZW5zb3JzICh4KSBzaG91bGQgaGF2ZSB0aGUgc2FtZSBudW1iZXIgb2Ygc2FtcGxlcy4gYCArXG4gICAgICAgIGBHb3QgYXJyYXkgc2hhcGVzOiBgICtcbiAgICAgICAgYCR7SlNPTi5zdHJpbmdpZnkoaW5wdXRzLm1hcChpbnB1dCA9PiBpbnB1dC5zaGFwZSkpfWApO1xuICB9XG4gIGlmIChzZXRZLmxlbmd0aCA+IDEpIHtcbiAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgYEFsbCB0YXJnZXQgVGVuc29ycyAoeSkgc2hvdWxkIGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mIHNhbXBsZXMuIGAgK1xuICAgICAgICBgR290IGFycmF5IHNoYXBlczogYCArXG4gICAgICAgIGAke0pTT04uc3RyaW5naWZ5KHRhcmdldHMubWFwKHRhcmdldCA9PiB0YXJnZXQuc2hhcGUpKX1gKTtcbiAgfVxuICBpZiAoc2V0WC5sZW5ndGggPiAwICYmIHNldFkubGVuZ3RoID4gMCAmJiAhdXRpbC5hcnJheXNFcXVhbChzZXRYLCBzZXRZKSkge1xuICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICBgSW5wdXQgVGVuc29ycyBzaG91bGQgaGF2ZSB0aGUgc2FtZSBudW1iZXIgb2Ygc2FtcGxlcyBhcyB0YXJnZXQgYCArXG4gICAgICAgIGBUZW5zb3JzLiBGb3VuZCAke3NldFhbMF19IGlucHV0IHNhbXBsZShzKSBhbmQgJHtzZXRZWzBdfSB0YXJnZXQgYCArXG4gICAgICAgIGBzYW1wbGUocykuYCk7XG4gIH1cbn1cblxuLyoqXG4gKiBWYWxpZGF0aW9uIG9uIHRoZSBjb21wYXRpYmlsaXR5IG9mIHRhcmdlcyBhbmQgbG9zcyBmdW5jdGlvbnMuXG4gKlxuICogVGhpcyBoZWxwcyBwcmV2ZW50IHVzZXJzIGZyb20gdXNpbmcgbG9zcyBmdW5jdGlvbnMgaW5jb3JyZWN0bHkuXG4gKlxuICogQHBhcmFtIHRhcmdldHMgYEFycmF5YCBvZiBgdGYuVGVuc29yYHMgb2YgdGFyZ2V0cy5cbiAqIEBwYXJhbSBsb3NzRm5zIGBBcnJheWAgb2YgbG9zcyBmdW5jdGlvbnMuXG4gKiBAcGFyYW0gb3V0cHV0U2hhcGVzIGBBcnJheWAgb2Ygc2hhcGVzIG9mIG1vZGVsIG91dHB1dHMuXG4gKi9cbmZ1bmN0aW9uIGNoZWNrTG9zc0FuZFRhcmdldENvbXBhdGliaWxpdHkoXG4gICAgdGFyZ2V0czogVGVuc29yW10sIGxvc3NGbnM6IExvc3NPck1ldHJpY0ZuW10sIG91dHB1dFNoYXBlczogU2hhcGVbXSkge1xuICAvLyBUT0RPKGNhaXMpOiBEZWRpY2F0ZWQgdGVzdCBjb3ZlcmFnZT9cbiAgY29uc3Qga2V5TG9zc2VzID0gW1xuICAgIGxvc3Nlcy5tZWFuU3F1YXJlZEVycm9yLCBsb3NzZXMuYmluYXJ5Q3Jvc3NlbnRyb3B5LFxuICAgIGxvc3Nlcy5jYXRlZ29yaWNhbENyb3NzZW50cm9weVxuICBdO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IHRhcmdldHMubGVuZ3RoOyArK2kpIHtcbiAgICBjb25zdCB5ID0gdGFyZ2V0c1tpXTtcbiAgICBjb25zdCBsb3NzID0gbG9zc0Zuc1tpXTtcbiAgICBjb25zdCBzaGFwZSA9IG91dHB1dFNoYXBlc1tpXTtcbiAgICBpZiAobG9zcyA9PSBudWxsKSB7XG4gICAgICBjb250aW51ZTtcbiAgICB9XG4gICAgaWYgKGxvc3MgPT09IGxvc3Nlcy5jYXRlZ29yaWNhbENyb3NzZW50cm9weSkge1xuICAgICAgaWYgKHkuc2hhcGVbeS5zaGFwZS5sZW5ndGggLSAxXSA9PT0gMSkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBZb3UgYXJlIHBhc3NpbmcgYSB0YXJnZXQgYXJyYXkgb2Ygc2hhcGUgJHt5LnNoYXBlfSB3aGlsZSB1c2luZyBgICtcbiAgICAgICAgICAgIGBhIGxvc3MgJ2NhdGVnb3JpY2FsX2Nyb3NzZW50cm9weScuICdjYXRlZ29yaWNhbF9jcm9zc2VudHJvcHknYCArXG4gICAgICAgICAgICBgZXhwZWN0cyB0YXJnZXRzIHRvIGJlIGJpbmFyeSBtYXRyaWNlcyAoMXMgYW5kIDBzKSBvZiBzaGFwZSBgICtcbiAgICAgICAgICAgIGBbc2FtcGxlcywgY2xhc3Nlc10uYCk7XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IEV4YW1wbGUgY29kZSBpbiBlcnJvciBtZXNzYWdlLlxuICAgICAgfVxuICAgIH1cbiAgICBpZiAoa2V5TG9zc2VzLmluZGV4T2YobG9zcykgIT09IC0xKSB7XG4gICAgICBjb25zdCBzbGljZWRZU2hhcGUgPSB5LnNoYXBlLnNsaWNlKDEpO1xuICAgICAgY29uc3Qgc2xpY2VkU2hhcGUgPSBzaGFwZS5zbGljZSgxKTtcbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgc2xpY2VkWVNoYXBlLmxlbmd0aDsgKytqKSB7XG4gICAgICAgIGNvbnN0IHRhcmdldERpbSA9IHNsaWNlZFlTaGFwZVtqXTtcbiAgICAgICAgY29uc3Qgb3V0RGltID0gc2xpY2VkU2hhcGVbal07XG4gICAgICAgIGlmIChvdXREaW0gIT0gbnVsbCAmJiB0YXJnZXREaW0gIT09IG91dERpbSkge1xuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICBgQSB0YXJnZXQgVGVuc29yIHdpdGggc2hhcGUgJHt5LnNoYXBlfSB3YXMgcGFzc2VkIGZvciBhbiBgICtcbiAgICAgICAgICAgICAgYG91dHB1dCBvZiBzaGFwZSAke3NoYXBlfSwgd2hpbGUgdXNpbmcgYSBsb3NzIGZ1bmN0aW9uIHRoYXQgYCArXG4gICAgICAgICAgICAgIGBleHBlY3RzIHRhcmdldHMgdG8gaGF2ZSB0aGUgc2FtZSBzaGFwZSBhcyB0aGUgb3V0cHV0LmApO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG59XG5cbi8qKlxuICogQ2hlY2sgaW5wdXRzIHByb3ZpZGVkIGJ5IHRoZSB1c2VyLlxuICpcbiAqIFBvcnRpbmcgTm90ZTogVGhpcyBjb3JyZXNwb25kcyB0byBfc3RhbmRhcmRpemVfaW5wdXRfZGF0YSgpIGluIFB5dGhvblxuICogICBLZXJhcy4gQmVjYXVzZSBvZiB0aGUgc3Ryb25nIHR5cGluZyBpbiBURi5qcywgd2UgZG8gbm90IG5lZWQgdG8gY29udmVydFxuICogICB0aGUgZGF0YS4gU3BlY2lmaWNhbGx5OlxuICogICAxKSBpbiBQeUtlcmFzLCBgZGF0YWAgY2FuIGJlIGBEYXRhRnJhbWVgIGluc3RhbmNlcyBmcm9tIHBhbmRhcywgZm9yXG4gKiAgICAgIGV4YW1wbGUuIFdlIGRvbid0IG5lZWQgdG8gd29ycnkgYWJvdXQgdGhhdCBoZXJlIGJlY2F1c2UgdGhlcmUgaXMgbm9cbiAqICAgICAgd2lkZWx5IHBvcHVsYXIgamF2YXNjcmlwdC90eXBlc2RjcmlwdCBlcXVpdmFsZW50IG9mIHBhbmRhcyAoc28gZmFyKS5cbiAqICAgICAgSWYgb25lIGJlY29tZXMgYXZhaWxhYmxlIGluIHRoZSBmdXR1cmUsIHdlIGNhbiBhZGQgc3VwcG9ydC5cbiAqICAgMikgaW4gUHlLZXJhcywgaW5wdXRzIGNhbiBiZSBQeXRob24gZGljdC4gQnV0IGhlcmUgd2UgYXJlIHN0aXB1bGF0aW5nXG4gKiB0aGF0IHRoZSBkYXRhIGlzIGVpdGhlciBhIHNpbmdsZSBgdGYuVGVuc29yYCBvciBhbiBBcnJheSBvZiBgdGYuVGVuc29yYHMuIFdlXG4gKiBtYXkgYWRkIHN1cHBvcnQgZm9yIGBPYmplY3RgIGRhdGEgaW5wdXRzIGluIHRoZSBmdXR1cmUgd2hlbiB0aGUgbmVlZFxuICogYXJpc2VzLlxuICpcbiAqIEluc3RlYWQsIHdlIHBlcmZvcm0gYmFzaWMgY2hlY2tzIGZvciBudW1iZXIgb2YgcGFyYW1ldGVycyBhbmQgc2hhcGVzLlxuICpcbiAqIEBwYXJhbSBkYXRhOiBUaGUgaW5wdXQgZGF0YS5cbiAqIEBwYXJhbSBuYW1lczogTmFtZSBmb3IgdGhlIGlucHV0cywgZnJvbSB0aGUgbW9kZWwuXG4gKiBAcGFyYW0gc2hhcGVzOiBFeHBlY3RlZCBzaGFwZXMgZm9yIHRoZSBpbnB1dCBkYXRhLCBmcm9tIHRoZSBtb2RlbC5cbiAqIEBwYXJhbSBjaGVja0JhdGNoQXhpczogV2hldGhlciB0aGUgc2l6ZSBhbG9uZyB0aGUgYmF0Y2ggYXhpcyAoaS5lLiwgdGhlXG4gKiAgIGZpcnN0IGRpbWVuc2lvbikgd2lsbCBiZSBjaGVja2VkIGZvciBtYXRjaGluZy5cbiAqIEBwYXJhbSBleGNlcHRpb25QcmVmaXg6IEV4ZWNwdGlvbiBwcmVmaXggbWVzc2FnZSwgdXNlZCBpbiBnZW5lcmF0aW5nIGVycm9yXG4gKiAgIG1lc3NhZ2VzLlxuICogQHRocm93cyBWYWx1ZUVycm9yOiBvbiBpbmNvcnJlY3QgbnVtYmVyIG9mIGlucHV0cyBvciBtaXNtYXRjaGVzIGluIHNoYXBlcy5cbiAqL1xuZnVuY3Rpb24gY2hlY2tJbnB1dERhdGEoXG4gICAgZGF0YTogVGVuc29yfFRlbnNvcltdLCBuYW1lczogc3RyaW5nW10sIHNoYXBlcz86IFNoYXBlW10sXG4gICAgY2hlY2tCYXRjaEF4aXMgPSB0cnVlLCBleGNlcHRpb25QcmVmaXggPSAnJykge1xuICBsZXQgYXJyYXlzOiBUZW5zb3JbXTtcbiAgaWYgKEFycmF5LmlzQXJyYXkoZGF0YSkpIHtcbiAgICBpZiAoZGF0YS5sZW5ndGggIT09IG5hbWVzLmxlbmd0aCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYEVycm9yIHdoZW4gY2hlY2tpbmcgbW9kZWwgJHtleGNlcHRpb25QcmVmaXh9OiB0aGUgQXJyYXkgb2YgYCArXG4gICAgICAgICAgYFRlbnNvcnMgdGhhdCB5b3UgYXJlIHBhc3NpbmcgdG8geW91ciBtb2RlbCBpcyBub3QgdGhlIHNpemUgdGhlIGAgK1xuICAgICAgICAgIGB0aGUgbW9kZWwgZXhwZWN0ZWQuIEV4cGVjdGVkIHRvIHNlZSAke25hbWVzLmxlbmd0aH0gVGVuc29yKHMpLGAgK1xuICAgICAgICAgIGAgYnV0IGluc3RlYWQgZ290ICR7ZGF0YS5sZW5ndGh9IFRlbnNvcnMocykuYCk7XG4gICAgfVxuICAgIGFycmF5cyA9IGRhdGE7XG4gIH0gZWxzZSB7XG4gICAgaWYgKG5hbWVzLmxlbmd0aCA+IDEpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBUaGUgbW9kZWwgZXhwZWN0cyAke25hbWVzLmxlbmd0aH0gJHtleGNlcHRpb25QcmVmaXh9IFRlbnNvcnMsIGAgK1xuICAgICAgICAgIGBidXQgb25seSByZWNlaXZlZCBvbmUgVGVuc29yLiBGb3VuZDogYXJyYXkgd2l0aCBzaGFwZSBgICtcbiAgICAgICAgICBgJHtKU09OLnN0cmluZ2lmeShkYXRhLnNoYXBlKX0uYCk7XG4gICAgfVxuICAgIGFycmF5cyA9IFtkYXRhXTtcbiAgfVxuXG4gIGlmIChzaGFwZXMgIT0gbnVsbCkge1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbmFtZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGlmIChzaGFwZXNbaV0gPT0gbnVsbCkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IGFycmF5ID0gYXJyYXlzW2ldO1xuICAgICAgaWYgKGFycmF5LnNoYXBlLmxlbmd0aCAhPT0gc2hhcGVzW2ldLmxlbmd0aCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBFcnJvciB3aGVuIGNoZWNraW5nICR7ZXhjZXB0aW9uUHJlZml4fTogZXhwZWN0ZWQgJHtuYW1lc1tpXX0gYCArXG4gICAgICAgICAgICBgdG8gaGF2ZSAke3NoYXBlc1tpXS5sZW5ndGh9IGRpbWVuc2lvbihzKSwgYnV0IGdvdCBhcnJheSB3aXRoIGAgK1xuICAgICAgICAgICAgYHNoYXBlICR7SlNPTi5zdHJpbmdpZnkoYXJyYXkuc2hhcGUpfWApO1xuICAgICAgfVxuICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCBzaGFwZXNbaV0ubGVuZ3RoOyArK2opIHtcbiAgICAgICAgaWYgKGogPT09IDAgJiYgIWNoZWNrQmF0Y2hBeGlzKSB7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgZGltID0gYXJyYXkuc2hhcGVbal07XG4gICAgICAgIGNvbnN0IHJlZkRpbSA9IHNoYXBlc1tpXVtqXTtcbiAgICAgICAgaWYgKHJlZkRpbSAhPSBudWxsKSB7XG4gICAgICAgICAgaWYgKHJlZkRpbSAhPT0gZGltKSB7XG4gICAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgICBgRXJyb3Igd2hlbiBjaGVja2luZyAke2V4Y2VwdGlvblByZWZpeH06IGV4cGVjdGVkIGAgK1xuICAgICAgICAgICAgICAgIGAke25hbWVzW2ldfSB0byBoYXZlIHNoYXBlICR7SlNPTi5zdHJpbmdpZnkoc2hhcGVzW2ldKX0gYnV0IGAgK1xuICAgICAgICAgICAgICAgIGBnb3QgYXJyYXkgd2l0aCBzaGFwZSAke0pTT04uc3RyaW5naWZ5KGFycmF5LnNoYXBlKX0uYCk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG59XG5cbi8qKlxuICogTWFwcyBtZXRyaWMgZnVuY3Rpb25zIHRvIG1vZGVsIG91dHB1dHMuXG4gKiBAcGFyYW0gbWV0cmljcyBBbiBzaG9ydGN1dCBzdHJpbmdzIG5hbWUsIG1ldHJpYyBmdW5jdGlvbiwgYEFycmF5YCBvciBkaWN0XG4gKiAgIChgT2JqZWN0YCkgb2YgbWV0cmljIGZ1bmN0aW9ucy5cbiAqIEBwYXJhbSBvdXRwdXROYW1lcyBBbiBgQXJyYXlgIG9mIHRoZSBuYW1lcyBvZiBtb2RlbCBvdXRwdXRzLlxuICogQHJldHVybnMgQW4gYEFycmF5YCAob25lIGVudHJ5IHBlciBtb2RlbCBvdXRwdXQpIG9mIGBBcnJheWAgb2YgbWV0cmljXG4gKiAgIGZ1bmN0aW9ucy4gRm9yIGluc3RhbmNlLCBpZiB0aGUgbW9kZWwgaGFzIDIgb3V0cHV0cywgYW5kIGZvciB0aGUgZmlyc3RcbiAqICAgb3V0cHV0IHdlIHdhbnQgdG8gY29tcHV0ZSBgYmluYXJ5QWNjdXJhY3lgIGFuZCBgYmluYXJ5Q3Jvc3NlbnRyb3B5YCxcbiAqICAgYW5kIGp1c3QgYGJpbmFyeUFjY3VyYWN5YCBmb3IgdGhlIHNlY29uZCBvdXRwdXQsIHRoZSBgQXJyYXlgIHdvdWxkIGxvb2tcbiAqICAgbGlrZTpcbiAqICAgICBgW1tiaW5hcnlBY2N1cmFjeSwgYmluYXJ5Q3Jvc3NlbnRyb3B5XSwgIFtiaW5hcnlBY2N1cmFjeV1dYFxuICogQHRocm93cyBUeXBlRXJyb3I6IGluY29tcGF0aWJsZSBtZXRyaWNzIGZvcm1hdC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbGxlY3RNZXRyaWNzKFxuICAgIG1ldHJpY3M6IHN0cmluZ3xMb3NzT3JNZXRyaWNGbnxBcnJheTxzdHJpbmd8TG9zc09yTWV0cmljRm4+fFxuICAgIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogc3RyaW5nIHwgTG9zc09yTWV0cmljRm59LFxuICAgIG91dHB1dE5hbWVzOiBzdHJpbmdbXSk6IEFycmF5PEFycmF5PHN0cmluZ3xMb3NzT3JNZXRyaWNGbj4+IHtcbiAgaWYgKG1ldHJpY3MgPT0gbnVsbCB8fCBBcnJheS5pc0FycmF5KG1ldHJpY3MpICYmIG1ldHJpY3MubGVuZ3RoID09PSAwKSB7XG4gICAgcmV0dXJuIG91dHB1dE5hbWVzLm1hcChuYW1lID0+IFtdKTtcbiAgfVxuXG4gIGxldCB3cmFwcGVkTWV0cmljczogQXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPnxcbiAgICAgIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogc3RyaW5nIHwgTG9zc09yTWV0cmljRm59O1xuICBpZiAodHlwZW9mIG1ldHJpY3MgPT09ICdzdHJpbmcnIHx8IHR5cGVvZiBtZXRyaWNzID09PSAnZnVuY3Rpb24nKSB7XG4gICAgd3JhcHBlZE1ldHJpY3MgPSBbbWV0cmljc107XG4gIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheShtZXRyaWNzKSB8fCB0eXBlb2YgbWV0cmljcyA9PT0gJ29iamVjdCcpIHtcbiAgICB3cmFwcGVkTWV0cmljcyA9IG1ldHJpY3MgYXMgQXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPnxcbiAgICAgICAge1tvdXRwdXROYW1lOiBzdHJpbmddOiBzdHJpbmd9IHwge1tvdXRwdXROYW1lOiBzdHJpbmddOiBMb3NzT3JNZXRyaWNGbn07XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IFR5cGVFcnJvcihcbiAgICAgICAgJ1R5cGUgb2YgbWV0cmljcyBhcmd1bWVudCBub3QgdW5kZXJzdG9vZC4gRXhwZWN0ZWQgYW4gc3RyaW5nLCcgK1xuICAgICAgICBgZnVuY3Rpb24sIEFycmF5LCBvciBPYmplY3QsIGZvdW5kOiAke21ldHJpY3N9YCk7XG4gIH1cblxuICBpZiAoQXJyYXkuaXNBcnJheSh3cmFwcGVkTWV0cmljcykpIHtcbiAgICAvLyBXZSB0aGVuIGFwcGx5IGFsbCBtZXRyaWNzIHRvIGFsbCBvdXRwdXRzLlxuICAgIHJldHVybiBvdXRwdXROYW1lcy5tYXAoXG4gICAgICAgIG5hbWUgPT4gd3JhcHBlZE1ldHJpY3MgYXMgQXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPik7XG4gIH0gZWxzZSB7XG4gICAgLy8gSW4gdGhpcyBjYXNlLCBtZXRyaWNzIGlzIGEgZGljdC5cbiAgICBjb25zdCBuZXN0ZWRNZXRyaWNzOiBBcnJheTxBcnJheTxzdHJpbmd8TG9zc09yTWV0cmljRm4+PiA9IFtdO1xuICAgIGZvciAoY29uc3QgbmFtZSBvZiBvdXRwdXROYW1lcykge1xuICAgICAgbGV0IG91dHB1dE1ldHJpY3M6IHN0cmluZ3xMb3NzT3JNZXRyaWNGbnxBcnJheTxzdHJpbmd8TG9zc09yTWV0cmljRm4+ID1cbiAgICAgICAgICB3cmFwcGVkTWV0cmljcy5oYXNPd25Qcm9wZXJ0eShuYW1lKSA/IHdyYXBwZWRNZXRyaWNzW25hbWVdIDogW107XG4gICAgICBpZiAoIUFycmF5LmlzQXJyYXkob3V0cHV0TWV0cmljcykpIHtcbiAgICAgICAgb3V0cHV0TWV0cmljcyA9IFtvdXRwdXRNZXRyaWNzXTtcbiAgICAgIH1cbiAgICAgIG5lc3RlZE1ldHJpY3MucHVzaChvdXRwdXRNZXRyaWNzKTtcbiAgICB9XG4gICAgcmV0dXJuIG5lc3RlZE1ldHJpY3M7XG4gIH1cbn1cblxuZXhwb3J0IGludGVyZmFjZSBNb2RlbEV2YWx1YXRlQXJncyB7XG4gIC8qKlxuICAgKiBCYXRjaCBzaXplIChJbnRlZ2VyKS4gSWYgdW5zcGVjaWZpZWQsIGl0IHdpbGwgZGVmYXVsdCB0byAzMi5cbiAgICovXG4gIGJhdGNoU2l6ZT86IG51bWJlcjtcblxuICAvKipcbiAgICogVmVyYm9zaXR5IG1vZGUuXG4gICAqL1xuICB2ZXJib3NlPzogTW9kZWxMb2dnaW5nVmVyYm9zaXR5O1xuXG4gIC8qKlxuICAgKiBUZW5zb3Igb2Ygd2VpZ2h0cyB0byB3ZWlnaHQgdGhlIGNvbnRyaWJ1dGlvbiBvZiBkaWZmZXJlbnQgc2FtcGxlcyB0byB0aGVcbiAgICogbG9zcyBhbmQgbWV0cmljcy5cbiAgICovXG4gIHNhbXBsZVdlaWdodD86IFRlbnNvcjtcblxuICAvKipcbiAgICogaW50ZWdlcjogdG90YWwgbnVtYmVyIG9mIHN0ZXBzIChiYXRjaGVzIG9mIHNhbXBsZXMpXG4gICAqIGJlZm9yZSBkZWNsYXJpbmcgdGhlIGV2YWx1YXRpb24gcm91bmQgZmluaXNoZWQuIElnbm9yZWQgd2l0aCB0aGUgZGVmYXVsdFxuICAgKiB2YWx1ZSBvZiBgdW5kZWZpbmVkYC5cbiAgICovXG4gIHN0ZXBzPzogbnVtYmVyO1xufVxuXG4vKipcbiAqIENvbmZpZ3VyYXRpb24gZm9yIGNhbGxzIHRvIGBMYXllcnNNb2RlbC5jb21waWxlKClgLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIE1vZGVsQ29tcGlsZUFyZ3Mge1xuICAvKipcbiAgICogQW4gaW5zdGFuY2Ugb2YgYHRmLnRyYWluLk9wdGltaXplcmAgb3IgYSBzdHJpbmcgbmFtZSBmb3IgYW4gT3B0aW1pemVyLlxuICAgKi9cbiAgb3B0aW1pemVyOiBzdHJpbmd8T3B0aW1pemVyO1xuXG4gIC8qKlxuICAgKiBPYmplY3QgZnVuY3Rpb24ocykgb3IgbmFtZShzKSBvZiBvYmplY3QgZnVuY3Rpb24ocykuXG4gICAqIElmIHRoZSBtb2RlbCBoYXMgbXVsdGlwbGUgb3V0cHV0cywgeW91IGNhbiB1c2UgYSBkaWZmZXJlbnQgbG9zc1xuICAgKiBvbiBlYWNoIG91dHB1dCBieSBwYXNzaW5nIGEgZGljdGlvbmFyeSBvciBhbiBBcnJheSBvZiBsb3NzZXMuXG4gICAqIFRoZSBsb3NzIHZhbHVlIHRoYXQgd2lsbCBiZSBtaW5pbWl6ZWQgYnkgdGhlIG1vZGVsIHdpbGwgdGhlbiBiZSB0aGUgc3VtXG4gICAqIG9mIGFsbCBpbmRpdmlkdWFsIGxvc3Nlcy5cbiAgICovXG4gIGxvc3M6IHN0cmluZ3xzdHJpbmdbXXx7W291dHB1dE5hbWU6IHN0cmluZ106IHN0cmluZ318TG9zc09yTWV0cmljRm58XG4gICAgICBMb3NzT3JNZXRyaWNGbltdfHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTG9zc09yTWV0cmljRm59O1xuXG4gIC8qKlxuICAgKiBMaXN0IG9mIG1ldHJpY3MgdG8gYmUgZXZhbHVhdGVkIGJ5IHRoZSBtb2RlbCBkdXJpbmcgdHJhaW5pbmcgYW5kIHRlc3RpbmcuXG4gICAqIFR5cGljYWxseSB5b3Ugd2lsbCB1c2UgYG1ldHJpY3M9WydhY2N1cmFjeSddYC5cbiAgICogVG8gc3BlY2lmeSBkaWZmZXJlbnQgbWV0cmljcyBmb3IgZGlmZmVyZW50IG91dHB1dHMgb2YgYSBtdWx0aS1vdXRwdXRcbiAgICogbW9kZWwsIHlvdSBjb3VsZCBhbHNvIHBhc3MgYSBkaWN0aW9uYXJ5LlxuICAgKi9cbiAgbWV0cmljcz86IHN0cmluZ3xMb3NzT3JNZXRyaWNGbnxBcnJheTxzdHJpbmd8TG9zc09yTWV0cmljRm4+fFxuICAgICAge1tvdXRwdXROYW1lOiBzdHJpbmddOiBzdHJpbmcgfCBMb3NzT3JNZXRyaWNGbn07XG5cbiAgLy8gVE9ETyhjYWlzKTogQWRkIGxvc3NXZWlnaHRzLCBzYW1wbGVXZWlnaHRNb2RlLCB3ZWlnaHRlZE1ldHJpY3MsIGFuZFxuICAvLyAgIHRhcmdldFRlbnNvcnMuXG59XG5cbmNvbnN0IExBWUVSU19NT0RFTF9GT1JNQVRfTkFNRSA9ICdsYXllcnMtbW9kZWwnO1xuXG4vKipcbiAqIEEgYHRmLkxheWVyc01vZGVsYCBpcyBhIGRpcmVjdGVkLCBhY3ljbGljIGdyYXBoIG9mIGB0Zi5MYXllcmBzIHBsdXMgbWV0aG9kc1xuICogZm9yIHRyYWluaW5nLCBldmFsdWF0aW9uLCBwcmVkaWN0aW9uIGFuZCBzYXZpbmcuXG4gKlxuICogYHRmLkxheWVyc01vZGVsYCBpcyB0aGUgYmFzaWMgdW5pdCBvZiB0cmFpbmluZywgaW5mZXJlbmNlIGFuZCBldmFsdWF0aW9uIGluXG4gKiBUZW5zb3JGbG93LmpzLiBUbyBjcmVhdGUgYSBgdGYuTGF5ZXJzTW9kZWxgLCB1c2UgYHRmLkxheWVyc01vZGVsYC5cbiAqXG4gKiBTZWUgYWxzbzpcbiAqICAgYHRmLlNlcXVlbnRpYWxgLCBgdGYubG9hZExheWVyc01vZGVsYC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICovXG5leHBvcnQgY2xhc3MgTGF5ZXJzTW9kZWwgZXh0ZW5kcyBDb250YWluZXIgaW1wbGVtZW50cyB0ZmMuSW5mZXJlbmNlTW9kZWwge1xuICAvLyBUaGUgY2xhc3MgbmFtZSBpcyAnTW9kZWwnIHJhdGhlciB0aGFuICdMYXllcnNNb2RlbCcgZm9yIGJhY2t3YXJkc1xuICAvLyBjb21wYXRpYmlsaXR5IHNpbmNlIHRoaXMgY2xhc3MgbmFtZSBzaG93cyB1cCBpbiB0aGUgc2VyaWFsaXphdGlvbiBmb3JtYXQuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ01vZGVsJztcbiAgcHJvdGVjdGVkIG9wdGltaXplcl86IE9wdGltaXplcjtcbiAgLy8gV2hldGhlciB0aGUgbW9kZWwgaW5zdGFuY2Ugb3ducyB0aGUgb3B0aW1pemVyOiBgdHJ1ZWAgaWYgYW5kIG9ubHkgaWZcbiAgLy8gYG9wdGltaXplcmAgaXMgY3JlYXRlZCBmcm9tIGEgc3RyaW5nIHBhcmFtZXRlciBkdXJpbmcgYGNvbXBpbGUoKWAgY2FsbC5cbiAgcHJvdGVjdGVkIGlzT3B0aW1pemVyT3duZWQ6IGJvb2xlYW47XG5cbiAgbG9zczogc3RyaW5nfHN0cmluZ1tdfHtbb3V0cHV0TmFtZTogc3RyaW5nXTogc3RyaW5nfXxMb3NzT3JNZXRyaWNGbnxcbiAgICAgIExvc3NPck1ldHJpY0ZuW118e1tvdXRwdXROYW1lOiBzdHJpbmddOiBMb3NzT3JNZXRyaWNGbn07XG4gIGxvc3NGdW5jdGlvbnM6IExvc3NPck1ldHJpY0ZuW107XG5cbiAgLy8gVE9ETyhjYWlzKTogVGhlc2UgcHJpdmF0ZSB2YXJpYWJsZXMgc2hvdWxkIHByb2JhYmx5IG5vdCBoYXZlIHRoZSBzdHJpbmdcbiAgLy8gICAnZmVlZCcgaW4gdGhlaXIgbmFtZXMsIGJlY2F1c2Ugd2UgYXJlIG5vdCBkZWFsaW5nIHdpdGggYSBzeW1ib2xpY1xuICAvLyAgIGJhY2tlbmQuXG4gIHByaXZhdGUgZmVlZE91dHB1dFNoYXBlczogU2hhcGVbXTtcbiAgcHJpdmF0ZSBmZWVkTG9zc0ZuczogTG9zc09yTWV0cmljRm5bXTtcbiAgcHJpdmF0ZSBjb2xsZWN0ZWRUcmFpbmFibGVXZWlnaHRzOiBMYXllclZhcmlhYmxlW107XG4gIHByaXZhdGUgdGVzdEZ1bmN0aW9uOiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdO1xuICBoaXN0b3J5OiBIaXN0b3J5O1xuXG4gIC8vIEEgcHVibGljIHByb3BlcnR5IHRoYXQgY2FuIGJlIHNldCBieSBDYWxsYmFja3MgdG8gb3JkZXIgZWFybHkgc3RvcHBpbmdcbiAgLy8gZHVyaW5nIGBmaXQoKWAgY2FsbHMuXG4gIHByb3RlY3RlZCBzdG9wVHJhaW5pbmdfOiBib29sZWFuO1xuICBwcm90ZWN0ZWQgaXNUcmFpbmluZzogYm9vbGVhbjtcblxuICBtZXRyaWNzOiBzdHJpbmd8TG9zc09yTWV0cmljRm58QXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPnxcbiAgICAgIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogc3RyaW5nIHwgTG9zc09yTWV0cmljRm59O1xuICBtZXRyaWNzTmFtZXM6IHN0cmluZ1tdO1xuICAvLyBQb3J0aW5nIE5vdGU6IGBtZXRyaWNzX3RlbnNvcnNgIGluIFB5S2VyYXMgaXMgYSBzeW1ib2xpYyB0ZW5zb3IuIEJ1dCBnaXZlblxuICAvLyAgIHRoZSBpbXBlcmF0aXZlIG5hdHVyZSBvZiB0ZmpzLWNvcmUsIGBtZXRyaWNzVGVuc29yc2AgaXMgYVxuICAvLyAgIFR5cGVTY3JpcHQgZnVuY3Rpb24gaGVyZS5cbiAgLy8gICBBbHNvIG5vdGUgdGhhdCBkdWUgdG8gdGhlIGltcGVyYXRpdmUgbmF0dXJlIG9mIHRmanMtY29yZSwgYG1ldHJpY3NUZW5zb3JgXG4gIC8vICAgaGVyZSBuZWVkcyBhbiBvdXRwdXQgaW5kZXggdG8ga2VlcCB0cmFjayBvZiB3aGljaCBvdXRwdXQgb2YgdGhlXG4gIC8vICAgTGF5ZXJzTW9kZWwgYSBtZXRyaWMgYmVsb25ncyB0by4gVGhpcyBpcyB1bmxpa2UgYG1ldHJpY3NfdGVuc29yc2AgaW5cbiAgLy8gICBQeUtlcmFzLCB3aGljaCBpcyBhIGBsaXN0YCBvZiBzeW1ib2xpYyB0ZW5zb3JzLCBlYWNoIG9mIHdoaWNoIGhhc1xuICAvLyAgIGltcGxpY2l0IFwia25vd2xlZGdlXCIgb2YgdGhlIG91dHB1dHMgaXQgZGVwZW5kcyBvbi5cbiAgbWV0cmljc1RlbnNvcnM6IEFycmF5PFtMb3NzT3JNZXRyaWNGbiwgbnVtYmVyXT47XG5cbiAgLy8gVXNlciBkZWZpbmQgbWV0YWRhdGEgKGlmIGFueSkuXG4gIHByaXZhdGUgdXNlckRlZmluZWRNZXRhZGF0YToge307XG5cbiAgY29uc3RydWN0b3IoYXJnczogQ29udGFpbmVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMuaXNUcmFpbmluZyA9IGZhbHNlO1xuICB9XG5cbiAgLyoqXG4gICAqIFByaW50IGEgdGV4dCBzdW1tYXJ5IG9mIHRoZSBtb2RlbCdzIGxheWVycy5cbiAgICpcbiAgICogVGhlIHN1bW1hcnkgaW5jbHVkZXNcbiAgICogLSBOYW1lIGFuZCB0eXBlIG9mIGFsbCBsYXllcnMgdGhhdCBjb21wcmlzZSB0aGUgbW9kZWwuXG4gICAqIC0gT3V0cHV0IHNoYXBlKHMpIG9mIHRoZSBsYXllcnNcbiAgICogLSBOdW1iZXIgb2Ygd2VpZ2h0IHBhcmFtZXRlcnMgb2YgZWFjaCBsYXllclxuICAgKiAtIElmIHRoZSBtb2RlbCBoYXMgbm9uLXNlcXVlbnRpYWwtbGlrZSB0b3BvbG9neSwgdGhlIGlucHV0cyBlYWNoIGxheWVyXG4gICAqICAgcmVjZWl2ZXNcbiAgICogLSBUaGUgdG90YWwgbnVtYmVyIG9mIHRyYWluYWJsZSBhbmQgbm9uLXRyYWluYWJsZSBwYXJhbWV0ZXJzIG9mIHRoZSBtb2RlbC5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgaW5wdXQxID0gdGYuaW5wdXQoe3NoYXBlOiBbMTBdfSk7XG4gICAqIGNvbnN0IGlucHV0MiA9IHRmLmlucHV0KHtzaGFwZTogWzIwXX0pO1xuICAgKiBjb25zdCBkZW5zZTEgPSB0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiA0fSkuYXBwbHkoaW5wdXQxKTtcbiAgICogY29uc3QgZGVuc2UyID0gdGYubGF5ZXJzLmRlbnNlKHt1bml0czogOH0pLmFwcGx5KGlucHV0Mik7XG4gICAqIGNvbnN0IGNvbmNhdCA9IHRmLmxheWVycy5jb25jYXRlbmF0ZSgpLmFwcGx5KFtkZW5zZTEsIGRlbnNlMl0pO1xuICAgKiBjb25zdCBvdXRwdXQgPVxuICAgKiAgICAgdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMywgYWN0aXZhdGlvbjogJ3NvZnRtYXgnfSkuYXBwbHkoY29uY2F0KTtcbiAgICpcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5tb2RlbCh7aW5wdXRzOiBbaW5wdXQxLCBpbnB1dDJdLCBvdXRwdXRzOiBvdXRwdXR9KTtcbiAgICogbW9kZWwuc3VtbWFyeSgpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIGxpbmVMZW5ndGggQ3VzdG9tIGxpbmUgbGVuZ3RoLCBpbiBudW1iZXIgb2YgY2hhcmFjdGVycy5cbiAgICogQHBhcmFtIHBvc2l0aW9ucyBDdXN0b20gd2lkdGhzIG9mIGVhY2ggb2YgdGhlIGNvbHVtbnMsIGFzIGVpdGhlclxuICAgKiAgIGZyYWN0aW9ucyBvZiBgbGluZUxlbmd0aGAgKGUuZy4sIGBbMC41LCAwLjc1LCAxXWApIG9yIGFic29sdXRlIG51bWJlclxuICAgKiAgIG9mIGNoYXJhY3RlcnMgKGUuZy4sIGBbMzAsIDUwLCA2NV1gKS4gRWFjaCBudW1iZXIgY29ycmVzcG9uZHMgdG9cbiAgICogICByaWdodC1tb3N0IChpLmUuLCBlbmRpbmcpIHBvc2l0aW9uIG9mIGEgY29sdW1uLlxuICAgKiBAcGFyYW0gcHJpbnRGbiBDdXN0b20gcHJpbnQgZnVuY3Rpb24uIENhbiBiZSB1c2VkIHRvIHJlcGxhY2UgdGhlIGRlZmF1bHRcbiAgICogICBgY29uc29sZS5sb2dgLiBGb3IgZXhhbXBsZSwgeW91IGNhbiB1c2UgYHggPT4ge31gIHRvIG11dGUgdGhlIHByaW50ZWRcbiAgICogICBtZXNzYWdlcyBpbiB0aGUgY29uc29sZS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHN1bW1hcnkoXG4gICAgICBsaW5lTGVuZ3RoPzogbnVtYmVyLCBwb3NpdGlvbnM/OiBudW1iZXJbXSxcbiAgICAgIHByaW50Rm46XG4gICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgKG1lc3NhZ2U/OiBhbnksIC4uLm9wdGlvbmFsUGFyYW1zOiBhbnlbXSkgPT4gdm9pZCA9IGNvbnNvbGUubG9nKSB7XG4gICAgaWYgKCF0aGlzLmJ1aWx0KSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgVGhpcyBtb2RlbCBoYXMgbmV2ZXIgYmVlbiBjYWxsZWQsIHRodXMgaXRzIHdlaWdodHMgaGF2ZSBub3QgYmVlbiBgICtcbiAgICAgICAgICBgY3JlYXRlZCB5ZXQuIFNvIG5vIHN1bW1hcnkgY2FuIGJlIGRpc3BsYXllZC4gQnVpbGQgdGhlIG1vZGVsIGAgK1xuICAgICAgICAgIGBmaXJzdCAoZS5nLiwgYnkgY2FsbGluZyBpdCBvbiBzb21lIHRlc3QgZGF0YSkuYCk7XG4gICAgfVxuICAgIHByaW50U3VtbWFyeSh0aGlzLCBsaW5lTGVuZ3RoLCBwb3NpdGlvbnMsIHByaW50Rm4pO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbmZpZ3VyZXMgYW5kIHByZXBhcmVzIHRoZSBtb2RlbCBmb3IgdHJhaW5pbmcgYW5kIGV2YWx1YXRpb24uICBDb21waWxpbmdcbiAgICogb3V0Zml0cyB0aGUgbW9kZWwgd2l0aCBhbiBvcHRpbWl6ZXIsIGxvc3MsIGFuZC9vciBtZXRyaWNzLiAgQ2FsbGluZyBgZml0YFxuICAgKiBvciBgZXZhbHVhdGVgIG9uIGFuIHVuLWNvbXBpbGVkIG1vZGVsIHdpbGwgdGhyb3cgYW4gZXJyb3IuXG4gICAqXG4gICAqIEBwYXJhbSBhcmdzIGEgYE1vZGVsQ29tcGlsZUFyZ3NgIHNwZWNpZnlpbmcgdGhlIGxvc3MsIG9wdGltaXplciwgYW5kXG4gICAqIG1ldHJpY3MgdG8gYmUgdXNlZCBmb3IgZml0dGluZyBhbmQgZXZhbHVhdGluZyB0aGlzIG1vZGVsLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgY29tcGlsZShhcmdzOiBNb2RlbENvbXBpbGVBcmdzKTogdm9pZCB7XG4gICAgaWYgKGFyZ3MubG9zcyA9PSBudWxsKSB7XG4gICAgICBhcmdzLmxvc3MgPSBbXTtcbiAgICB9XG4gICAgdGhpcy5sb3NzID0gYXJncy5sb3NzO1xuXG4gICAgaWYgKHR5cGVvZiBhcmdzLm9wdGltaXplciA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHRoaXMub3B0aW1pemVyXyA9IG9wdGltaXplcnMuZ2V0T3B0aW1pemVyKGFyZ3Mub3B0aW1pemVyKTtcbiAgICAgIHRoaXMuaXNPcHRpbWl6ZXJPd25lZCA9IHRydWU7XG4gICAgfSBlbHNlIHtcbiAgICAgIGlmICghKGFyZ3Mub3B0aW1pemVyIGluc3RhbmNlb2YgT3B0aW1pemVyKSkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBVc2VyLWRlZmluZWQgb3B0aW1pemVyIG11c3QgYmUgYW4gaW5zdGFuY2Ugb2YgdGYuT3B0aW1pemVyLmApO1xuICAgICAgfVxuICAgICAgdGhpcy5vcHRpbWl6ZXJfID0gYXJncy5vcHRpbWl6ZXI7XG4gICAgICB0aGlzLmlzT3B0aW1pemVyT3duZWQgPSBmYWxzZTtcbiAgICB9XG5cbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgbG9zc1dlaWdodHMuXG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIHNhbXBsZVdlaWdodE1vZGUuXG5cbiAgICAvLyBQcmVwYXJlIGxvc3MgZnVuY3Rpb25zLlxuICAgIGxldCBsb3NzRnVuY3Rpb25zOiBMb3NzT3JNZXRyaWNGbltdID0gW107XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KGFyZ3MubG9zcykgJiYgdHlwZW9mIGFyZ3MubG9zcyAhPT0gJ3N0cmluZycgJiZcbiAgICAgICAgdHlwZW9mIGFyZ3MubG9zcyAhPT0gJ2Z1bmN0aW9uJykge1xuICAgICAgYXJncy5sb3NzID0gYXJncy5sb3NzIGFzIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogc3RyaW5nfTtcbiAgICAgIGZvciAoY29uc3QgbmFtZSBpbiBhcmdzLmxvc3MpIHtcbiAgICAgICAgaWYgKHRoaXMub3V0cHV0TmFtZXMuaW5kZXhPZihuYW1lKSA9PT0gLTEpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgYFVua25vd24gZW50cnkgaW4gbG9zcyBkaWN0aW9uYXJ5OiBcIiR7bmFtZX1cIi4gYCArXG4gICAgICAgICAgICAgIGBPbmx5IGV4cGVjdGVkIHRoZSBmb2xsb3dpbmcga2V5czogJHt0aGlzLm91dHB1dE5hbWVzfWApO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBmb3IgKGNvbnN0IG5hbWUgb2YgdGhpcy5vdXRwdXROYW1lcykge1xuICAgICAgICBpZiAoYXJncy5sb3NzW25hbWVdID09IG51bGwpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICAgIGBPdXRwdXQgXCIke25hbWV9XCIgaXMgbWlzc2luZyBmcm9tIGxvc3MgZGljdGlvbmFyeS4gV2UgYXNzdW1lIGAgK1xuICAgICAgICAgICAgICBgdGhpcyB3YXMgZG9uZSBvbiBwdXJwb3NlLCBhbmQgd2Ugd2lsbCBub3QgYmUgZXhwZWN0aW5nIGRhdGEgYCArXG4gICAgICAgICAgICAgIGB0byBiZSBwYXNzZWQgdG8gJHtuYW1lfSBkdXJpbmcgdHJhaW5pbmdgKTtcbiAgICAgICAgfVxuICAgICAgICBsb3NzRnVuY3Rpb25zLnB1c2gobG9zc2VzLmdldChhcmdzLmxvc3NbbmFtZV0pKTtcbiAgICAgIH1cbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkoYXJncy5sb3NzKSkge1xuICAgICAgaWYgKGFyZ3MubG9zcy5sZW5ndGggIT09IHRoaXMub3V0cHV0cy5sZW5ndGgpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgV2hlbiBwYXNzaW5nIGFuIEFycmF5IGFzIGxvc3MsIGl0IHNob3VsZCBoYXZlIG9uZSBlbnRyeSBwZXIgYCArXG4gICAgICAgICAgICBgbW9kZWwgb3V0cHV0LiBUaGUgbW9kZWwgaGFzICR7dGhpcy5vdXRwdXRzLmxlbmd0aH0gb3V0cHV0KHMpLCBgICtcbiAgICAgICAgICAgIGBidXQgeW91IHBhc3NlZCBsb3NzPSR7YXJncy5sb3NzfS5gKTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHRoZUxvc3NlcyA9IGFyZ3MubG9zcyBhcyBBcnJheTxzdHJpbmd8TG9zc09yTWV0cmljRm4+O1xuICAgICAgbG9zc0Z1bmN0aW9ucyA9IHRoZUxvc3Nlcy5tYXAobCA9PiBsb3NzZXMuZ2V0KGwpKTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgbG9zc0Z1bmN0aW9uID0gbG9zc2VzLmdldChhcmdzLmxvc3MpO1xuICAgICAgdGhpcy5vdXRwdXRzLmZvckVhY2goXyA9PiB7XG4gICAgICAgIGxvc3NGdW5jdGlvbnMucHVzaChsb3NzRnVuY3Rpb24pO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgdGhpcy5sb3NzRnVuY3Rpb25zID0gbG9zc0Z1bmN0aW9ucztcblxuICAgIHRoaXMuZmVlZE91dHB1dE5hbWVzID0gW107XG4gICAgdGhpcy5mZWVkT3V0cHV0U2hhcGVzID0gW107XG4gICAgdGhpcy5mZWVkTG9zc0ZucyA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5vdXRwdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBMb2dpYyBmb3Igc2tpcHBpbmcgdGFyZ2V0KHMpLlxuICAgICAgY29uc3Qgc2hhcGUgPSB0aGlzLmludGVybmFsT3V0cHV0U2hhcGVzW2ldO1xuICAgICAgY29uc3QgbmFtZSA9IHRoaXMub3V0cHV0TmFtZXNbaV07XG4gICAgICB0aGlzLmZlZWRPdXRwdXROYW1lcy5wdXNoKG5hbWUpO1xuICAgICAgdGhpcy5mZWVkT3V0cHV0U2hhcGVzLnB1c2goc2hhcGUpO1xuICAgICAgdGhpcy5mZWVkTG9zc0Zucy5wdXNoKHRoaXMubG9zc0Z1bmN0aW9uc1tpXSk7XG4gICAgfVxuXG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIGxvZ2ljIGZvciBvdXRwdXQgbWFza3MuXG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIGxvZ2ljIGZvciBzYW1wbGUgd2VpZ2h0cy5cbiAgICBjb25zdCBza2lwVGFyZ2V0SW5kaWNlczogbnVtYmVyW10gPSBbXTtcblxuICAgIC8vIFByZXBhcmUgbWV0cmljcy5cbiAgICB0aGlzLm1ldHJpY3MgPSBhcmdzLm1ldHJpY3M7XG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIHdlaWdodGVkTWV0cmljcy5cbiAgICB0aGlzLm1ldHJpY3NOYW1lcyA9IFsnbG9zcyddO1xuICAgIHRoaXMubWV0cmljc1RlbnNvcnMgPSBbXTtcblxuICAgIC8vIENvbXB1dGUgdG90YWwgbG9zcy5cbiAgICAvLyBQb3J0aW5nIE5vdGU6IEluIFB5S2VyYXMsIG1ldHJpY3NfdGVuc29ycyBhcmUgc3ltYm9saWMgdGVuc29yIG9iamVjdHMuXG4gICAgLy8gICBIZXJlLCBtZXRyaWNzVGVuc29ycyBhcmUgVHlwZVNjcmlwdCBmdW5jdGlvbnMuIFRoaXMgZGlmZmVyZW5jZSBpcyBkdWVcbiAgICAvLyAgIHRvIHRoZSBkaWZmZXJlbmNlIGluIHN5bWJvbGljL2ltcGVyYXRpdmUgcHJvcGVydHkgb2YgdGhlIGJhY2tlbmRzLlxuICAgIG5hbWVTY29wZSgnbG9zcycsICgpID0+IHtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5vdXRwdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIGlmIChza2lwVGFyZ2V0SW5kaWNlcy5pbmRleE9mKGkpICE9PSAtMSkge1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB3ZWlnaHRlZExvc3MsIHNhbXBsZVdlaWdodCBhbmQgbWFzay5cbiAgICAgICAgLy8gICBUaGUgZm9sbG93aW5nIGxpbmUgc2hvdWxkIGJlIHdlaWdodGVkTG9zc1xuICAgICAgICBjb25zdCB3ZWlnaHRlZExvc3MgPSB0aGlzLmxvc3NGdW5jdGlvbnNbaV07XG4gICAgICAgIGlmICh0aGlzLm91dHB1dHMubGVuZ3RoID4gMSkge1xuICAgICAgICAgIHRoaXMubWV0cmljc1RlbnNvcnMucHVzaChbd2VpZ2h0ZWRMb3NzLCBpXSk7XG4gICAgICAgICAgdGhpcy5tZXRyaWNzTmFtZXMucHVzaCh0aGlzLm91dHB1dE5hbWVzW2ldICsgJ19sb3NzJyk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgLy8gUG9ydGluZyBOb3RlOiBEdWUgdG8gdGhlIGltcGVyYXRpdmUgbmF0dXJlIG9mIHRoZSBiYWNrZW5kLCB3ZSBjYWxjdWxhdGVcbiAgICAgIC8vICAgdGhlIHJlZ3VsYXJpemVyIHBlbmFsdGllcyBpbiB0aGUgdG90YWxMb3NzRnVuY3Rpb24sIGluc3RlYWQgb2YgaGVyZS5cbiAgICB9KTtcblxuICAgIGNvbnN0IG5lc3RlZE1ldHJpY3MgPSBjb2xsZWN0TWV0cmljcyhhcmdzLm1ldHJpY3MsIHRoaXMub3V0cHV0TmFtZXMpO1xuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBuZXN0ZWRXZWlnaHRlZE1ldHJpY3MuXG5cbiAgICAvKipcbiAgICAgKiBIZWxwZXIgZnVuY3Rpb24gdXNlZCBpbiBsb29wIGJlbG93LlxuICAgICAqL1xuICAgIGNvbnN0IGFwcGVuZE1ldHJpYyA9XG4gICAgICAgIChvdXRwdXRJbmRleDogbnVtYmVyLCBtZXRyaWNOYW1lOiBzdHJpbmcsXG4gICAgICAgICBtZXRyaWNUZW5zb3I6IExvc3NPck1ldHJpY0ZuKSA9PiB7XG4gICAgICAgICAgaWYgKHRoaXMub3V0cHV0TmFtZXMubGVuZ3RoID4gMSkge1xuICAgICAgICAgICAgbWV0cmljTmFtZSA9IHRoaXMub3V0cHV0TmFtZXNbb3V0cHV0SW5kZXhdICsgJ18nICsgbWV0cmljTmFtZTtcbiAgICAgICAgICB9XG4gICAgICAgICAgdGhpcy5tZXRyaWNzTmFtZXMucHVzaChtZXRyaWNOYW1lKTtcbiAgICAgICAgICB0aGlzLm1ldHJpY3NUZW5zb3JzLnB1c2goW21ldHJpY1RlbnNvciwgb3V0cHV0SW5kZXhdKTtcbiAgICAgICAgfTtcblxuICAgIG5hbWVTY29wZSgnbWV0cmljJywgKCkgPT4ge1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLm91dHB1dHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgaWYgKHNraXBUYXJnZXRJbmRpY2VzLmluZGV4T2YoaSkgIT09IC0xKSB7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3Qgb3V0cHV0TWV0cmljcyA9IG5lc3RlZE1ldHJpY3NbaV07XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB3ZWlnaHRzIGFuZCBvdXRwdXRXZWlnaHRlZE1ldHJpY3MuXG5cbiAgICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIG9wdGlvbmFsIGFyZyBgd2VpZ2h0c2AgdG8gdGhlIGZvbGxvd2luZyBmdW5jdGlvbi5cbiAgICAgICAgY29uc3QgaGFuZGxlTWV0cmljcyA9IChtZXRyaWNzOiBBcnJheTxzdHJpbmd8TG9zc09yTWV0cmljRm4+KSA9PiB7XG4gICAgICAgICAgY29uc3QgbWV0cmljTmFtZVByZWZpeCA9ICcnO1xuICAgICAgICAgIGxldCBtZXRyaWNOYW1lOiBzdHJpbmc7XG4gICAgICAgICAgbGV0IGFjY0ZuOiBMb3NzT3JNZXRyaWNGbjtcbiAgICAgICAgICBsZXQgd2VpZ2h0ZWRNZXRyaWNGbjogTG9zc09yTWV0cmljRm47XG4gICAgICAgICAgLy8gIFRPRE8oY2Fpcyk6IFVzZSAnd2VpZ2h0c18nIGZvciB3ZWlnaHRlZCBtZXRyaWNzLlxuXG4gICAgICAgICAgZm9yIChjb25zdCBtZXRyaWMgb2YgbWV0cmljcykge1xuICAgICAgICAgICAgaWYgKHR5cGVvZiBtZXRyaWMgPT09ICdzdHJpbmcnICYmXG4gICAgICAgICAgICAgICAgWydhY2N1cmFjeScsICdhY2MnLCAnY3Jvc3NlbnRyb3B5JywgJ2NlJ10uaW5kZXhPZihtZXRyaWMpICE9PVxuICAgICAgICAgICAgICAgICAgICAtMSkge1xuICAgICAgICAgICAgICBjb25zdCBvdXRwdXRTaGFwZSA9IHRoaXMuaW50ZXJuYWxPdXRwdXRTaGFwZXNbaV07XG5cbiAgICAgICAgICAgICAgaWYgKG91dHB1dFNoYXBlW291dHB1dFNoYXBlLmxlbmd0aCAtIDFdID09PSAxIHx8XG4gICAgICAgICAgICAgICAgICB0aGlzLmxvc3NGdW5jdGlvbnNbaV0gPT09IGxvc3Nlcy5iaW5hcnlDcm9zc2VudHJvcHkpIHtcbiAgICAgICAgICAgICAgICAvLyBjYXNlOiBiaW5hcnkgYWNjdXJhY3kvY3Jvc3NlbnRyb3B5LlxuICAgICAgICAgICAgICAgIGlmIChbJ2FjY3VyYWN5JywgJ2FjYyddLmluZGV4T2YobWV0cmljKSAhPT0gLTEpIHtcbiAgICAgICAgICAgICAgICAgIGFjY0ZuID0gTWV0cmljcy5iaW5hcnlBY2N1cmFjeTtcbiAgICAgICAgICAgICAgICB9IGVsc2UgaWYgKFsnY3Jvc3NlbnRyb3B5JywgJ2NlJ10uaW5kZXhPZihtZXRyaWMpICE9PSAtMSkge1xuICAgICAgICAgICAgICAgICAgYWNjRm4gPSBNZXRyaWNzLmJpbmFyeUNyb3NzZW50cm9weTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH0gZWxzZSBpZiAoXG4gICAgICAgICAgICAgICAgICB0aGlzLmxvc3NGdW5jdGlvbnNbaV0gPT09XG4gICAgICAgICAgICAgICAgICBsb3NzZXMuc3BhcnNlQ2F0ZWdvcmljYWxDcm9zc2VudHJvcHkpIHtcbiAgICAgICAgICAgICAgICAvLyBjYXNlOiBjYXRlZ29yaWNhbCBhY2N1cmFjeSAvIGNyb3NzZW50cm9weSB3aXRoIHNwYXJzZVxuICAgICAgICAgICAgICAgIC8vIHRhcmdldHMuXG4gICAgICAgICAgICAgICAgaWYgKFsnYWNjdXJhY3knLCAnYWNjJ10uaW5kZXhPZihtZXRyaWMpICE9PSAtMSkge1xuICAgICAgICAgICAgICAgICAgYWNjRm4gPSBNZXRyaWNzLnNwYXJzZUNhdGVnb3JpY2FsQWNjdXJhY3k7XG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmIChbJ2Nyb3NzZW50cm9weScsICdjZSddLmluZGV4T2YobWV0cmljKSAhPT0gLTEpIHtcbiAgICAgICAgICAgICAgICAgIGFjY0ZuID0gTWV0cmljcy5zcGFyc2VDYXRlZ29yaWNhbENyb3NzZW50cm9weTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgLy8gY2FzZTogY2F0ZWdvcmljYWwgYWNjdXJhY3kgLyBjcm9zc2VudHJvcHkuXG4gICAgICAgICAgICAgICAgaWYgKFsnYWNjdXJhY3knLCAnYWNjJ10uaW5kZXhPZihtZXRyaWMpICE9PSAtMSkge1xuICAgICAgICAgICAgICAgICAgYWNjRm4gPSBNZXRyaWNzLmNhdGVnb3JpY2FsQWNjdXJhY3k7XG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmIChbJ2Nyb3NzZW50cm9weScsICdjZSddLmluZGV4T2YobWV0cmljKSAhPT0gLTEpIHtcbiAgICAgICAgICAgICAgICAgIGFjY0ZuID0gTWV0cmljcy5jYXRlZ29yaWNhbENyb3NzZW50cm9weTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgbGV0IHN1ZmZpeDogc3RyaW5nO1xuICAgICAgICAgICAgICBpZiAoWydhY2N1cmFjeScsICdhY2MnXS5pbmRleE9mKG1ldHJpYykgIT09IC0xKSB7XG4gICAgICAgICAgICAgICAgc3VmZml4ID0gJ2FjYyc7XG4gICAgICAgICAgICAgIH0gZWxzZSBpZiAoWydjcm9zc2VudHJvcHknLCAnY2UnXS5pbmRleE9mKG1ldHJpYykgIT09IC0xKSB7XG4gICAgICAgICAgICAgICAgc3VmZml4ID0gJ2NlJztcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgd2VpZ2h0aW5nIGFjdHVhbGx5LlxuICAgICAgICAgICAgICB3ZWlnaHRlZE1ldHJpY0ZuID0gYWNjRm47XG4gICAgICAgICAgICAgIG1ldHJpY05hbWUgPSBtZXRyaWNOYW1lUHJlZml4ICsgc3VmZml4O1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgY29uc3QgbWV0cmljRm4gPSBNZXRyaWNzLmdldChtZXRyaWMpO1xuICAgICAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgd2VpZ2h0aW5nIGFjdHVhbGx5LlxuICAgICAgICAgICAgICB3ZWlnaHRlZE1ldHJpY0ZuID0gbWV0cmljRm47XG4gICAgICAgICAgICAgIG1ldHJpY05hbWUgPVxuICAgICAgICAgICAgICAgICAgbWV0cmljTmFtZVByZWZpeCArIE1ldHJpY3MuZ2V0TG9zc09yTWV0cmljTmFtZShtZXRyaWMpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgd2VpZ2h0aW5nIGFuZCBtYXNraW5nIHRvIG1ldHJpY1Jlc3VsdC5cbiAgICAgICAgICAgIGxldCBtZXRyaWNSZXN1bHQ6IExvc3NPck1ldHJpY0ZuO1xuICAgICAgICAgICAgbmFtZVNjb3BlKG1ldHJpY05hbWUsICgpID0+IHtcbiAgICAgICAgICAgICAgbWV0cmljUmVzdWx0ID0gd2VpZ2h0ZWRNZXRyaWNGbjtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgYXBwZW5kTWV0cmljKGksIG1ldHJpY05hbWUsIG1ldHJpY1Jlc3VsdCk7XG4gICAgICAgICAgfVxuICAgICAgICB9O1xuXG4gICAgICAgIGhhbmRsZU1ldHJpY3Mob3V0cHV0TWV0cmljcyk7XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IENhbGwgaGFuZGxlTWV0cmljcyB3aXRoIHdlaWdodHMuXG4gICAgICB9XG4gICAgfSk7XG5cbiAgICAvLyBQb3J0aW5nIE5vdGVzOiBHaXZlbiB0aGUgaW1wZXJhdGl2ZSBiYWNrZW5kIG9mIHRmanMtY29yZSxcbiAgICAvLyAgIHRoZXJlIGlzIG5vIG5lZWQgZm9yIGNvbnN0cnVjdGluZyB0aGUgc3ltYm9saWMgZ3JhcGggYW5kIHBsYWNlaG9sZGVycy5cbiAgICB0aGlzLmNvbGxlY3RlZFRyYWluYWJsZVdlaWdodHMgPSB0aGlzLnRyYWluYWJsZVdlaWdodHM7XG4gIH1cblxuICAvKipcbiAgICogQ2hlY2sgdHJhaW5hYmxlIHdlaWdodHMgY291bnQgY29uc2lzdGVuY3kuXG4gICAqXG4gICAqIFRoaXMgd2lsbCByYWlzZSBhIHdhcm5pbmcgaWYgYHRoaXMudHJhaW5hYmxlV2VpZ2h0c2AgYW5kXG4gICAqIGB0aGlzLmNvbGxlY3RlZFRyYWluYWJsZVdlaWdodHNgIGFyZSBpbmNvbnNpc3RlbnQgKGkuZS4sIGhhdmUgZGlmZmVyZW50XG4gICAqIG51bWJlcnMgb2YgcGFyYW1ldGVycykuXG4gICAqIEluY29uc2lzdGVuY3kgd2lsbCB0eXBpY2FsbHkgYXJpc2Ugd2hlbiBvbmUgbW9kaWZpZXMgYG1vZGVsLnRyYWluYWJsZWBcbiAgICogd2l0aG91dCBjYWxsaW5nIGBtb2RlbC5jb21waWxlKClgIGFnYWluLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNoZWNrVHJhaW5hYmxlV2VpZ2h0c0NvbnNpc3RlbmN5KCk6IHZvaWQge1xuICAgIGlmICh0aGlzLmNvbGxlY3RlZFRyYWluYWJsZVdlaWdodHMgPT0gbnVsbCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAodGhpcy50cmFpbmFibGVXZWlnaHRzLmxlbmd0aCAhPT1cbiAgICAgICAgdGhpcy5jb2xsZWN0ZWRUcmFpbmFibGVXZWlnaHRzLmxlbmd0aCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdEaXNjcmVwYW5jeSBiZXR3ZWVuIHRyYWluYWJsZXdlaWdodHMgYW5kIGNvbGxlY3RlZCB0cmFpbmFibGUgJyArXG4gICAgICAgICAgJ3dlaWdodHMuIERpZCB5b3Ugc2V0IGBtb2RlbC50cmFpbmFibGVgIHdpdGhvdXQgY2FsbGluZyAnICtcbiAgICAgICAgICAnYG1vZGVsLmNvbXBpbGUoKWAgYWZ0ZXJ3YXJkcz8nKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyB0aGUgbG9zcyB2YWx1ZSAmIG1ldHJpY3MgdmFsdWVzIGZvciB0aGUgbW9kZWwgaW4gdGVzdCBtb2RlLlxuICAgKlxuICAgKiBMb3NzIGFuZCBtZXRyaWNzIGFyZSBzcGVjaWZpZWQgZHVyaW5nIGBjb21waWxlKClgLCB3aGljaCBuZWVkcyB0byBoYXBwZW5cbiAgICogYmVmb3JlIGNhbGxzIHRvIGBldmFsdWF0ZSgpYC5cbiAgICpcbiAgICogQ29tcHV0YXRpb24gaXMgZG9uZSBpbiBiYXRjaGVzLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoe1xuICAgKiAgIGxheWVyczogW3RmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMF19KV1cbiAgICogfSk7XG4gICAqIG1vZGVsLmNvbXBpbGUoe29wdGltaXplcjogJ3NnZCcsIGxvc3M6ICdtZWFuU3F1YXJlZEVycm9yJ30pO1xuICAgKiBjb25zdCByZXN1bHQgPSBtb2RlbC5ldmFsdWF0ZShcbiAgICogICAgIHRmLm9uZXMoWzgsIDEwXSksIHRmLm9uZXMoWzgsIDFdKSwge2JhdGNoU2l6ZTogNH0pO1xuICAgKiByZXN1bHQucHJpbnQoKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSB4IGB0Zi5UZW5zb3JgIG9mIHRlc3QgZGF0YSwgb3IgYW4gYEFycmF5YCBvZiBgdGYuVGVuc29yYHMgaWYgdGhlXG4gICAqIG1vZGVsIGhhcyBtdWx0aXBsZSBpbnB1dHMuXG4gICAqIEBwYXJhbSB5IGB0Zi5UZW5zb3JgIG9mIHRhcmdldCBkYXRhLCBvciBhbiBgQXJyYXlgIG9mIGB0Zi5UZW5zb3JgcyBpZiB0aGVcbiAgICogbW9kZWwgaGFzIG11bHRpcGxlIG91dHB1dHMuXG4gICAqIEBwYXJhbSBhcmdzIEEgYE1vZGVsRXZhbHVhdGVBcmdzYCwgY29udGFpbmluZyBvcHRpb25hbCBmaWVsZHMuXG4gICAqXG4gICAqIEByZXR1cm4gYFNjYWxhcmAgdGVzdCBsb3NzIChpZiB0aGUgbW9kZWwgaGFzIGEgc2luZ2xlIG91dHB1dCBhbmQgbm9cbiAgICogICBtZXRyaWNzKSBvciBgQXJyYXlgIG9mIGBTY2FsYXJgcyAoaWYgdGhlIG1vZGVsIGhhcyBtdWx0aXBsZSBvdXRwdXRzXG4gICAqICAgYW5kL29yIG1ldHJpY3MpLiBUaGUgYXR0cmlidXRlIGBtb2RlbC5tZXRyaWNzTmFtZXNgXG4gICAqICAgd2lsbCBnaXZlIHlvdSB0aGUgZGlzcGxheSBsYWJlbHMgZm9yIHRoZSBzY2FsYXIgb3V0cHV0cy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGV2YWx1YXRlKFxuICAgICAgeDogVGVuc29yfFRlbnNvcltdLCB5OiBUZW5zb3J8VGVuc29yW10sXG4gICAgICBhcmdzOiBNb2RlbEV2YWx1YXRlQXJncyA9IHt9KTogU2NhbGFyfFNjYWxhcltdIHtcbiAgICBjb25zdCBiYXRjaFNpemUgPSBhcmdzLmJhdGNoU2l6ZSA9PSBudWxsID8gMzIgOiBhcmdzLmJhdGNoU2l6ZTtcbiAgICBjaGVja0JhdGNoU2l6ZShiYXRjaFNpemUpO1xuXG4gICAgLy8gVE9ETyhjYWlzKTogU3RhbmRhcmRpemUgYGNvbmZpZy5zYW1wbGVXZWlnaHRzYCBhcyB3ZWxsLlxuICAgIC8vIFZhbGlkYXRlIHVzZXIgZGF0YS5cbiAgICBjb25zdCBjaGVja0JhdGNoQXhpcyA9IHRydWU7XG4gICAgY29uc3Qgc3RhbmRhcmRpemVkT3V0cyA9XG4gICAgICAgIHRoaXMuc3RhbmRhcmRpemVVc2VyRGF0YVhZKHgsIHksIGNoZWNrQmF0Y2hBeGlzLCBiYXRjaFNpemUpO1xuICAgIHRyeSB7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBJZiB1c2VzIGB1c2VMZWFybmluZ1BoYXNlYCwgc2V0IHRoZSBjb3JyZXNwb25kaW5nIGVsZW1lbnRcbiAgICAgIC8vIG9mIHRoZSBpbnB1dCB0byAwLlxuICAgICAgY29uc3QgaW5zID0gc3RhbmRhcmRpemVkT3V0c1swXS5jb25jYXQoc3RhbmRhcmRpemVkT3V0c1sxXSk7XG4gICAgICB0aGlzLm1ha2VUZXN0RnVuY3Rpb24oKTtcbiAgICAgIGNvbnN0IGYgPSB0aGlzLnRlc3RGdW5jdGlvbjtcbiAgICAgIGNvbnN0IHRlc3RPdXRzID1cbiAgICAgICAgICB0aGlzLnRlc3RMb29wKGYsIGlucywgYmF0Y2hTaXplLCBhcmdzLnZlcmJvc2UsIGFyZ3Muc3RlcHMpO1xuICAgICAgcmV0dXJuIHNpbmdsZXRvbk9yQXJyYXkodGVzdE91dHMpO1xuICAgIH0gZmluYWxseSB7XG4gICAgICBkaXNwb3NlTmV3VGVuc29ycyhzdGFuZGFyZGl6ZWRPdXRzWzBdLCB4KTtcbiAgICAgIGRpc3Bvc2VOZXdUZW5zb3JzKHN0YW5kYXJkaXplZE91dHNbMV0sIHkpO1xuICAgIH1cbiAgfVxuXG4gIC8vIFRPRE8oY2Fpcyk6IEFkZCBjb2RlIHNuaXBwZXQgYmVsb3cgb25jZSByZWFsIGRhdGFzZXQgb2JqZWN0cyBhcmVcbiAgLy8gICBhdmFpbGFibGUuXG4gIC8qKlxuICAgKiBFdmFsdWF0ZSBtb2RlbCB1c2luZyBhIGRhdGFzZXQgb2JqZWN0LlxuICAgKlxuICAgKiBOb3RlOiBVbmxpa2UgYGV2YWx1YXRlKClgLCB0aGlzIG1ldGhvZCBpcyBhc3luY2hyb25vdXMgKGBhc3luY2ApO1xuICAgKlxuICAgKiBAcGFyYW0gZGF0YXNldCBBIGRhdGFzZXQgb2JqZWN0LiBJdHMgYGl0ZXJhdG9yKClgIG1ldGhvZCBpcyBleHBlY3RlZFxuICAgKiAgIHRvIGdlbmVyYXRlIGEgZGF0YXNldCBpdGVyYXRvciBvYmplY3QsIHRoZSBgbmV4dCgpYCBtZXRob2Qgb2Ygd2hpY2hcbiAgICogICBpcyBleHBlY3RlZCB0byBwcm9kdWNlIGRhdGEgYmF0Y2hlcyBmb3IgZXZhbHVhdGlvbi4gVGhlIHJldHVybiB2YWx1ZVxuICAgKiAgIG9mIHRoZSBgbmV4dCgpYCBjYWxsIG91Z2h0IHRvIGNvbnRhaW4gYSBib29sZWFuIGBkb25lYCBmaWVsZCBhbmQgYVxuICAgKiAgIGB2YWx1ZWAgZmllbGQuIFRoZSBgdmFsdWVgIGZpZWxkIGlzIGV4cGVjdGVkIHRvIGJlIGFuIGFycmF5IG9mIHR3b1xuICAgKiAgIGB0Zi5UZW5zb3JgcyBvciBhbiBhcnJheSBvZiB0d28gbmVzdGVkIGB0Zi5UZW5zb3JgIHN0cnVjdHVyZXMuIFRoZSBmb3JtZXJcbiAgICogICBjYXNlIGlzIGZvciBtb2RlbHMgd2l0aCBleGFjdGx5IG9uZSBpbnB1dCBhbmQgb25lIG91dHB1dCAoZS5nLi5cbiAgICogICBhIHNlcXVlbnRpYWwgbW9kZWwpLiBUaGUgbGF0dGVyIGNhc2UgaXMgZm9yIG1vZGVscyB3aXRoIG11bHRpcGxlXG4gICAqICAgaW5wdXRzIGFuZC9vciBtdWx0aXBsZSBvdXRwdXRzLiBPZiB0aGUgdHdvIGl0ZW1zIGluIHRoZSBhcnJheSwgdGhlXG4gICAqICAgZmlyc3QgaXMgdGhlIGlucHV0IGZlYXR1cmUocykgYW5kIHRoZSBzZWNvbmQgaXMgdGhlIG91dHB1dCB0YXJnZXQocykuXG4gICAqIEBwYXJhbSBhcmdzIEEgY29uZmlndXJhdGlvbiBvYmplY3QgZm9yIHRoZSBkYXRhc2V0LWJhc2VkIGV2YWx1YXRpb24uXG4gICAqIEByZXR1cm5zIExvc3MgYW5kIG1ldHJpYyB2YWx1ZXMgYXMgYW4gQXJyYXkgb2YgYFNjYWxhcmAgb2JqZWN0cy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzeW5jIGV2YWx1YXRlRGF0YXNldChkYXRhc2V0OiBEYXRhc2V0PHt9PiwgYXJncz86IE1vZGVsRXZhbHVhdGVEYXRhc2V0QXJncyk6XG4gICAgICBQcm9taXNlPFNjYWxhcnxTY2FsYXJbXT4ge1xuICAgIHRoaXMubWFrZVRlc3RGdW5jdGlvbigpO1xuICAgIHJldHVybiBldmFsdWF0ZURhdGFzZXQodGhpcywgZGF0YXNldCwgYXJncyk7XG4gIH1cblxuICAvKipcbiAgICogR2V0IG51bWJlciBvZiBzYW1wbGVzIHByb3ZpZGVkIGZvciB0cmFpbmluZywgZXZhbHVhdGlvbiBvciBwcmVkaWN0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0gaW5zIElucHV0IGB0Zi5UZW5zb3JgLlxuICAgKiBAcGFyYW0gYmF0Y2hTaXplIEludGVnZXIgYmF0Y2ggc2l6ZSwgb3B0aW9uYWwuXG4gICAqIEBwYXJhbSBzdGVwcyBUb3RhbCBudW1iZXIgb2Ygc3RlcHMgKGJhdGNoZXMgb2Ygc2FtcGxlcykgYmVmb3JlXG4gICAqIGRlY2xhcmluZyBsb29wIGZpbmlzaGVkLiBPcHRpb25hbC5cbiAgICogQHBhcmFtIHN0ZXBzTmFtZSBUaGUgcHVibGljIEFQSSdzIHBhcmFtZXRlciBuYW1lIGZvciBgc3RlcHNgLlxuICAgKiBAcmV0dXJucyBOdW1iZXIgb2Ygc2FtcGxlcyBwcm92aWRlZC5cbiAgICovXG4gIHByaXZhdGUgY2hlY2tOdW1TYW1wbGVzKFxuICAgICAgaW5zOiBUZW5zb3J8VGVuc29yW10sIGJhdGNoU2l6ZT86IG51bWJlciwgc3RlcHM/OiBudW1iZXIsXG4gICAgICBzdGVwc05hbWUgPSAnc3RlcHMnKTogbnVtYmVyIHtcbiAgICBsZXQgbnVtU2FtcGxlczogbnVtYmVyO1xuICAgIGlmIChzdGVwcyAhPSBudWxsKSB7XG4gICAgICBudW1TYW1wbGVzID0gbnVsbDtcbiAgICAgIGlmIChiYXRjaFNpemUgIT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBJZiAke3N0ZXBzTmFtZX0gaXMgc2V0LCBiYXRjaFNpemUgbXVzdCBiZSBudWxsIG9yIHVuZGVmaW5lZC5gICtcbiAgICAgICAgICAgIGBHb3QgYmF0Y2hTaXplID0gJHtiYXRjaFNpemV9YCk7XG4gICAgICB9XG4gICAgfSBlbHNlIGlmIChpbnMgIT0gbnVsbCkge1xuICAgICAgaWYgKEFycmF5LmlzQXJyYXkoaW5zKSkge1xuICAgICAgICBudW1TYW1wbGVzID0gaW5zWzBdLnNoYXBlWzBdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgbnVtU2FtcGxlcyA9IGlucy5zaGFwZVswXTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYEVpdGhlciB0aGUgaW5wdXQgZGF0YSBzaG91bGQgaGF2ZSBhIGRlZmluZWQgc2hhcGUsIG9yIGAgK1xuICAgICAgICAgIGAke3N0ZXBzTmFtZX0gc2hvdWQgYmUgc3BlY2lmaWVkLmApO1xuICAgIH1cbiAgICByZXR1cm4gbnVtU2FtcGxlcztcbiAgfVxuXG4gIC8qKlxuICAgKiBFeGVjdXRlIGludGVybmFsIHRlbnNvcnMgb2YgdGhlIG1vZGVsIHdpdGggaW5wdXQgZGF0YSBmZWVkLlxuICAgKiBAcGFyYW0gaW5wdXRzIElucHV0IGRhdGEgZmVlZC4gTXVzdCBtYXRjaCB0aGUgaW5wdXRzIG9mIHRoZSBtb2RlbC5cbiAgICogQHBhcmFtIG91dHB1dHMgTmFtZXMgb2YgdGhlIG91dHB1dCB0ZW5zb3JzIHRvIGJlIGZldGNoZWQuIE11c3QgbWF0Y2hcbiAgICogICBuYW1lcyBvZiB0aGUgU3ltYm9saWNUZW5zb3JzIHRoYXQgYmVsb25nIHRvIHRoZSBncmFwaC5cbiAgICogQHJldHVybnMgRmV0Y2hlZCB2YWx1ZXMgZm9yIGBvdXRwdXRzYC5cbiAgICovXG4gIGV4ZWN1dGUoaW5wdXRzOiBUZW5zb3J8VGVuc29yW118TmFtZWRUZW5zb3JNYXAsIG91dHB1dHM6IHN0cmluZ3xzdHJpbmdbXSk6XG4gICAgICBUZW5zb3J8VGVuc29yW10ge1xuICAgIGlmIChBcnJheS5pc0FycmF5KG91dHB1dHMpICYmIG91dHB1dHMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnYG91dHB1dHNgIGlzIGFuIGVtcHR5IEFycmF5LCB3aGljaCBpcyBub3QgYWxsb3dlZC4nKTtcbiAgICB9XG5cbiAgICBjb25zdCBvdXRwdXRzSXNBcnJheSA9IEFycmF5LmlzQXJyYXkob3V0cHV0cyk7XG4gICAgY29uc3Qgb3V0cHV0TmFtZXMgPVxuICAgICAgICAob3V0cHV0c0lzQXJyYXkgPyBvdXRwdXRzIGFzIHN0cmluZ1tdIDogW291dHB1dHMgYXMgc3RyaW5nXSk7XG4gICAgY29uc3Qgb3V0cHV0U3ltYm9saWNUZW5zb3JzID0gdGhpcy5yZXRyaWV2ZVN5bWJvbGljVGVuc29ycyhvdXRwdXROYW1lcyk7XG5cbiAgICAvLyBGb3JtYXQgdGhlIGlucHV0IGludG8gYSBGZWVkRGljdC5cbiAgICBjb25zdCBmZWVkRGljdCA9IG5ldyBGZWVkRGljdCgpO1xuICAgIGlmIChpbnB1dHMgaW5zdGFuY2VvZiBUZW5zb3IpIHtcbiAgICAgIGlucHV0cyA9IFtpbnB1dHNdO1xuICAgIH1cbiAgICBpZiAoQXJyYXkuaXNBcnJheShpbnB1dHMpKSB7XG4gICAgICBpZiAoaW5wdXRzLmxlbmd0aCAhPT0gdGhpcy5pbnB1dHMubGVuZ3RoKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYFRoZSBudW1iZXIgb2YgaW5wdXRzIHByb3ZpZGVkICgke2lucHV0cy5sZW5ndGh9KSBgICtcbiAgICAgICAgICAgIGBkb2VzIG5vdCBtYXRjaCB0aGUgbnVtYmVyIG9mIGlucHV0cyBvZiB0aGlzIG1vZGVsIGAgK1xuICAgICAgICAgICAgYCgke3RoaXMuaW5wdXRzLmxlbmd0aH0pLmApO1xuICAgICAgfVxuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLmlucHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICBmZWVkRGljdC5hZGQodGhpcy5pbnB1dHNbaV0sIGlucHV0c1tpXSk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGZvciAoY29uc3QgaW5wdXQgb2YgdGhpcy5pbnB1dHMpIHtcbiAgICAgICAgY29uc3QgdGVuc29yVmFsdWUgPSBpbnB1dHNbaW5wdXQubmFtZV07XG4gICAgICAgIGlmICh0ZW5zb3JWYWx1ZSA9PSBudWxsKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBObyB2YWx1ZSBpcyBwcm92aWRlZCBmb3IgdGhlIG1vZGVsJ3MgaW5wdXQgJHtpbnB1dC5uYW1lfWApO1xuICAgICAgICB9XG4gICAgICAgIGZlZWREaWN0LmFkZChpbnB1dCwgdGVuc29yVmFsdWUpO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIFJ1biBleGVjdXRpb24uXG4gICAgY29uc3QgZXhlY3V0ZU91dHB1dHMgPSBleGVjdXRlKG91dHB1dFN5bWJvbGljVGVuc29ycywgZmVlZERpY3QpIGFzIFRlbnNvcltdO1xuICAgIHJldHVybiBvdXRwdXRzSXNBcnJheSA/IGV4ZWN1dGVPdXRwdXRzIDogZXhlY3V0ZU91dHB1dHNbMF07XG4gIH1cblxuICAvKipcbiAgICogUmV0cmlldmUgdGhlIG1vZGVsJ3MgaW50ZXJuYWwgc3ltYm9saWMgdGVuc29ycyBmcm9tIHN5bWJvbGljLXRlbnNvciBuYW1lcy5cbiAgICovXG4gIHByaXZhdGUgcmV0cmlldmVTeW1ib2xpY1RlbnNvcnMoc3ltYm9saWNUZW5zb3JOYW1lczogc3RyaW5nW10pOlxuICAgICAgU3ltYm9saWNUZW5zb3JbXSB7XG4gICAgY29uc3Qgb3V0cHV0U3ltYm9saWNUZW5zb3JzOiBTeW1ib2xpY1RlbnNvcltdID1cbiAgICAgICAgcHlMaXN0UmVwZWF0KG51bGwsIHN5bWJvbGljVGVuc29yTmFtZXMubGVuZ3RoKTtcbiAgICBsZXQgb3V0cHV0c1JlbWFpbmluZyA9IHN5bWJvbGljVGVuc29yTmFtZXMubGVuZ3RoO1xuICAgIGZvciAoY29uc3QgbGF5ZXIgb2YgdGhpcy5sYXllcnMpIHtcbiAgICAgIGNvbnN0IGxheWVyT3V0cHV0czogU3ltYm9saWNUZW5zb3JbXSA9XG4gICAgICAgICAgQXJyYXkuaXNBcnJheShsYXllci5vdXRwdXQpID8gbGF5ZXIub3V0cHV0IDogW2xheWVyLm91dHB1dF07XG4gICAgICBjb25zdCBsYXllck91dHB1dE5hbWVzID0gbGF5ZXJPdXRwdXRzLm1hcChvdXRwdXQgPT4gb3V0cHV0Lm5hbWUpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzeW1ib2xpY1RlbnNvck5hbWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIGNvbnN0IGluZGV4ID0gbGF5ZXJPdXRwdXROYW1lcy5pbmRleE9mKHN5bWJvbGljVGVuc29yTmFtZXNbaV0pO1xuICAgICAgICBpZiAoaW5kZXggIT09IC0xKSB7XG4gICAgICAgICAgb3V0cHV0U3ltYm9saWNUZW5zb3JzW2ldID0gbGF5ZXJPdXRwdXRzW2luZGV4XTtcbiAgICAgICAgICBvdXRwdXRzUmVtYWluaW5nLS07XG4gICAgICAgIH1cbiAgICAgICAgaWYgKG91dHB1dHNSZW1haW5pbmcgPT09IDApIHtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKG91dHB1dHNSZW1haW5pbmcgPT09IDApIHtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKG91dHB1dHNSZW1haW5pbmcgPiAwKSB7XG4gICAgICBjb25zdCByZW1haW5pbmdOYW1lczogc3RyaW5nW10gPSBbXTtcbiAgICAgIG91dHB1dFN5bWJvbGljVGVuc29ycy5mb3JFYWNoKCh0ZW5zb3IsIGkpID0+IHtcbiAgICAgICAgaWYgKHRlbnNvciA9PSBudWxsKSB7XG4gICAgICAgICAgcmVtYWluaW5nTmFtZXMucHVzaChzeW1ib2xpY1RlbnNvck5hbWVzW2ldKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgQ2Fubm90IGZpbmQgU3ltYm9saWNUZW5zb3JzIGZvciBvdXRwdXQgbmFtZShzKTogYCArXG4gICAgICAgICAgYCR7SlNPTi5zdHJpbmdpZnkocmVtYWluaW5nTmFtZXMpfWApO1xuICAgIH1cbiAgICByZXR1cm4gb3V0cHV0U3ltYm9saWNUZW5zb3JzO1xuICB9XG5cbiAgLyoqXG4gICAqIEhlbHBlciBtZXRob2QgdG8gbG9vcCBvdmVyIHNvbWUgZGF0YSBpbiBiYXRjaGVzLlxuICAgKlxuICAgKiBQb3J0aW5nIE5vdGU6IE5vdCB1c2luZyB0aGUgZnVuY3Rpb25hbCBhcHByb2FjaCBpbiB0aGUgUHl0aG9uIGVxdWl2YWxlbnRcbiAgICogICBkdWUgdG8gdGhlIGltcGVyYXRpdmUgYmFja2VuZC5cbiAgICogUG9ydGluZyBOb3RlOiBEb2VzIG5vdCBzdXBwb3J0IHN0ZXAgbW9kZSBjdXJyZW50bHkuXG4gICAqXG4gICAqIEBwYXJhbSBpbnM6IGlucHV0IGRhdGFcbiAgICogQHBhcmFtIGJhdGNoU2l6ZTogaW50ZWdlciBiYXRjaCBzaXplLlxuICAgKiBAcGFyYW0gdmVyYm9zZTogdmVyYm9zaXR5IG1vZGVsXG4gICAqIEByZXR1cm5zOiBQcmVkaWN0aW9ucyBhcyBgdGYuVGVuc29yYCAoaWYgYSBzaW5nbGUgb3V0cHV0KSBvciBhbiBgQXJyYXlgIG9mXG4gICAqICAgYHRmLlRlbnNvcmAgKGlmIG11bHRpcGUgb3V0cHV0cykuXG4gICAqL1xuICBwcml2YXRlIHByZWRpY3RMb29wKGluczogVGVuc29yfFRlbnNvcltdLCBiYXRjaFNpemUgPSAzMiwgdmVyYm9zZSA9IGZhbHNlKTpcbiAgICAgIFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IG51bVNhbXBsZXMgPSB0aGlzLmNoZWNrTnVtU2FtcGxlcyhpbnMpO1xuICAgICAgaWYgKHZlcmJvc2UpIHtcbiAgICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgICAnVmVyYm9zZSBwcmVkaWN0TG9vcCgpIGlzIG5vdCBpbXBsZW1lbnRlZCB5ZXQuJyk7XG4gICAgICB9XG5cbiAgICAgIC8vIFNhbXBsZS1iYXNlZCBwcmVkaWN0aW9ucy5cbiAgICAgIC8vIFBvcnRpbmcgTm90ZTogVGVuc29yIGN1cnJlbnRseSBkb2VzIG5vdCBzdXBwb3J0IHNsaWNlZCBhc3NpZ25tZW50cyBhc1xuICAgICAgLy8gICBpbiBudW1weSwgZS5nLiwgeFsxOjNdID0geS4gVGhlcmVmb3JlIHdlIHVzZSBjb25jYXRlbmF0aW9uIHdoaWxlXG4gICAgICAvLyAgIGl0ZXJhdGluZyBvdmVyIHRoZSBiYXRjaGVzLlxuXG4gICAgICBjb25zdCBiYXRjaGVzID0gbWFrZUJhdGNoZXMobnVtU2FtcGxlcywgYmF0Y2hTaXplKTtcbiAgICAgIGNvbnN0IG91dHNCYXRjaGVzOiBUZW5zb3JbXVtdID0gdGhpcy5vdXRwdXRzLm1hcChvdXRwdXQgPT4gW10pO1xuXG4gICAgICAvLyBUT0RPKGNhaXMpOiBDYW4gdGhlIHNjb3BlKCkgYmUgcHVzaGVkIGRvd24gaW5zaWRlIHRoZSBmb3IgbG9vcD9cbiAgICAgIGZvciAobGV0IGJhdGNoSW5kZXggPSAwOyBiYXRjaEluZGV4IDwgYmF0Y2hlcy5sZW5ndGg7ICsrYmF0Y2hJbmRleCkge1xuICAgICAgICBjb25zdCBiYXRjaE91dHMgPSB0ZmMudGlkeSgoKSA9PiB7XG4gICAgICAgICAgY29uc3QgYmF0Y2hTdGFydCA9IGJhdGNoZXNbYmF0Y2hJbmRleF1bMF07XG4gICAgICAgICAgY29uc3QgYmF0Y2hFbmQgPSBiYXRjaGVzW2JhdGNoSW5kZXhdWzFdO1xuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IFRha2UgY2FyZSBvZiB0aGUgY2FzZSBvZiB0aGUgbGFzdCBlbGVtZW50IGlzIGEgZmxhZyBmb3JcbiAgICAgICAgICAvLyAgIHRyYWluaW5nL3Rlc3QuXG4gICAgICAgICAgY29uc3QgaW5zQmF0Y2ggPSBzbGljZUFycmF5cyhpbnMsIGJhdGNoU3RhcnQsIGJhdGNoRW5kKTtcblxuICAgICAgICAgIC8vIENvbnN0cnVjdCB0aGUgZmVlZHMgZm9yIGV4ZWN1dGUoKTtcbiAgICAgICAgICBjb25zdCBmZWVkcyA9IFtdO1xuICAgICAgICAgIGlmIChBcnJheS5pc0FycmF5KGluc0JhdGNoKSkge1xuICAgICAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBpbnNCYXRjaC5sZW5ndGg7ICsraSkge1xuICAgICAgICAgICAgICBmZWVkcy5wdXNoKHtrZXk6IHRoaXMuaW5wdXRzW2ldLCB2YWx1ZTogaW5zQmF0Y2hbaV19KTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgZmVlZHMucHVzaCh7a2V5OiB0aGlzLmlucHV0c1swXSwgdmFsdWU6IGluc0JhdGNofSk7XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IGZlZWREaWN0ID0gbmV3IEZlZWREaWN0KGZlZWRzKTtcbiAgICAgICAgICByZXR1cm4gZXhlY3V0ZSh0aGlzLm91dHB1dHMsIGZlZWREaWN0KSBhcyBUZW5zb3JbXTtcbiAgICAgICAgfSk7XG4gICAgICAgIGJhdGNoT3V0cy5mb3JFYWNoKChiYXRjaE91dCwgaSkgPT4gb3V0c0JhdGNoZXNbaV0ucHVzaChiYXRjaE91dCkpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHNpbmdsZXRvbk9yQXJyYXkoXG4gICAgICAgICAgb3V0c0JhdGNoZXMubWFwKGJhdGNoZXMgPT4gdGZjLmNvbmNhdChiYXRjaGVzLCAwKSkpO1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIEdlbmVyYXRlcyBvdXRwdXQgcHJlZGljdGlvbnMgZm9yIHRoZSBpbnB1dCBzYW1wbGVzLlxuICAgKlxuICAgKiBDb21wdXRhdGlvbiBpcyBkb25lIGluIGJhdGNoZXMuXG4gICAqXG4gICAqIE5vdGU6IHRoZSBcInN0ZXBcIiBtb2RlIG9mIHByZWRpY3QoKSBpcyBjdXJyZW50bHkgbm90IHN1cHBvcnRlZC5cbiAgICogICBUaGlzIGlzIGJlY2F1c2UgdGhlIFRlbnNvckZsb3cuanMgY29yZSBiYWNrZW5kIGlzIGltcGVyYXRpdmUgb25seS5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKHtcbiAgICogICBsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbMTBdfSldXG4gICAqIH0pO1xuICAgKiBtb2RlbC5wcmVkaWN0KHRmLm9uZXMoWzgsIDEwXSksIHtiYXRjaFNpemU6IDR9KS5wcmludCgpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGRhdGEsIGFzIGEgVGVuc29yLCBvciBhbiBgQXJyYXlgIG9mIGB0Zi5UZW5zb3JgcyBpZlxuICAgKiAgIHRoZSBtb2RlbCBoYXMgbXVsdGlwbGUgaW5wdXRzLlxuICAgKiBAcGFyYW0gYXJncyBBIGBNb2RlbFByZWRpY3RBcmdzYCBvYmplY3QgY29udGFpbmluZyBvcHRpb25hbCBmaWVsZHMuXG4gICAqXG4gICAqIEByZXR1cm4gUHJlZGljdGlvbiByZXN1bHRzIGFzIGEgYHRmLlRlbnNvcmAocykuXG4gICAqXG4gICAqIEBleGNlcHRpb24gVmFsdWVFcnJvciBJbiBjYXNlIG9mIG1pc21hdGNoIGJldHdlZW4gdGhlIHByb3ZpZGVkIGlucHV0IGRhdGFcbiAgICogICBhbmQgdGhlIG1vZGVsJ3MgZXhwZWN0YXRpb25zLCBvciBpbiBjYXNlIGEgc3RhdGVmdWwgbW9kZWwgcmVjZWl2ZXMgYVxuICAgKiAgIG51bWJlciBvZiBzYW1wbGVzIHRoYXQgaXMgbm90IGEgbXVsdGlwbGUgb2YgdGhlIGJhdGNoIHNpemUuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBwcmVkaWN0KHg6IFRlbnNvcnxUZW5zb3JbXSwgYXJnczogTW9kZWxQcmVkaWN0QXJncyA9IHt9KTogVGVuc29yfFRlbnNvcltdIHtcbiAgICBjb25zdCB4c1JhbmsyT3JIaWdoZXIgPSBlbnN1cmVUZW5zb3JzUmFuazJPckhpZ2hlcih4KTtcbiAgICBjaGVja0lucHV0RGF0YShcbiAgICAgICAgeHNSYW5rMk9ySGlnaGVyLCB0aGlzLmlucHV0TmFtZXMsIHRoaXMuZmVlZElucHV0U2hhcGVzLCBmYWxzZSk7XG4gICAgdHJ5IHtcbiAgICAgIC8vIFRPRE8oY2Fpcyk6IFRha2UgY2FyZSBvZiBzdGF0ZWZ1bCBtb2RlbHMuXG4gICAgICAvLyAgIGlmICh0aGlzLnN0YXRlZnVsKSAuLi5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IFRha2UgY2FyZSBvZiB0aGUgbGVhcm5pbmdfcGhhc2UgYm9vbGVhbiBmbGFnLlxuICAgICAgLy8gICBpZiAodGhpcy51c2VMZWFybmluZ1BoYXNlKSAuLi5cbiAgICAgIGNvbnN0IGJhdGNoU2l6ZSA9IGFyZ3MuYmF0Y2hTaXplID09IG51bGwgPyAzMiA6IGFyZ3MuYmF0Y2hTaXplO1xuICAgICAgY2hlY2tCYXRjaFNpemUoYmF0Y2hTaXplKTtcbiAgICAgIHJldHVybiB0aGlzLnByZWRpY3RMb29wKHhzUmFuazJPckhpZ2hlciwgYmF0Y2hTaXplKTtcbiAgICB9IGZpbmFsbHkge1xuICAgICAgZGlzcG9zZU5ld1RlbnNvcnMoeHNSYW5rMk9ySGlnaGVyLCB4KTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBwcmVkaWN0aW9ucyBmb3IgYSBzaW5nbGUgYmF0Y2ggb2Ygc2FtcGxlcy5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKHtcbiAgICogICBsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbMTBdfSldXG4gICAqIH0pO1xuICAgKiBtb2RlbC5wcmVkaWN0T25CYXRjaCh0Zi5vbmVzKFs4LCAxMF0pKS5wcmludCgpO1xuICAgKiBgYGBcbiAgICogQHBhcmFtIHg6IElucHV0IHNhbXBsZXMsIGFzIGEgVGVuc29yIChmb3IgbW9kZWxzIHdpdGggZXhhY3RseSBvbmVcbiAgICogICBpbnB1dCkgb3IgYW4gYXJyYXkgb2YgVGVuc29ycyAoZm9yIG1vZGVscyB3aXRoIG1vcmUgdGhhbiBvbmUgaW5wdXQpLlxuICAgKiBAcmV0dXJuIFRlbnNvcihzKSBvZiBwcmVkaWN0aW9uc1xuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgcHJlZGljdE9uQmF0Y2goeDogVGVuc29yfFRlbnNvcltdKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICBjaGVja0lucHV0RGF0YSh4LCB0aGlzLmlucHV0TmFtZXMsIHRoaXMuZmVlZElucHV0U2hhcGVzLCB0cnVlKTtcbiAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgdGhlIGxlYXJuaW5nX3BoYXNlIGJvb2xlYW4gZmxhZy5cbiAgICAvLyAgIGlmICh0aGlzLnVzZUxlYXJuaW5nUGhhc2UpIC4uLlxuICAgIGNvbnN0IGJhdGNoU2l6ZSA9IChBcnJheS5pc0FycmF5KHgpID8geFswXSA6IHgpLnNoYXBlWzBdO1xuICAgIHJldHVybiB0aGlzLnByZWRpY3RMb29wKHgsIGJhdGNoU2l6ZSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3RhbmRhcmRpemVVc2VyRGF0YVhZKFxuICAgICAgeDogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgICAgeTogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LCBjaGVja0JhdGNoQXhpcyA9IHRydWUsXG4gICAgICBiYXRjaFNpemU/OiBudW1iZXIpOiBbVGVuc29yW10sIFRlbnNvcltdXSB7XG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIHNhbXBsZVdlaWdodCwgY2xhc3NXZWlnaHRcbiAgICBpZiAodGhpcy5vcHRpbWl6ZXJfID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBSdW50aW1lRXJyb3IoXG4gICAgICAgICAgJ1lvdSBtdXN0IGNvbXBpbGUgYSBtb2RlbCBiZWZvcmUgdHJhaW5pbmcvdGVzdGluZy4gVXNlICcgK1xuICAgICAgICAgICdMYXllcnNNb2RlbC5jb21waWxlKG1vZGVsQ29tcGlsZUFyZ3MpLicpO1xuICAgIH1cbiAgICBjb25zdCBvdXRwdXRTaGFwZXM6IFNoYXBlW10gPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuZmVlZE91dHB1dFNoYXBlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3Qgb3V0cHV0U2hhcGUgPSB0aGlzLmZlZWRPdXRwdXRTaGFwZXNbaV07XG4gICAgICBjb25zdCBsb3NzRm4gPSB0aGlzLmZlZWRMb3NzRm5zW2ldO1xuICAgICAgaWYgKGxvc3NGbiA9PT0gbG9zc2VzLnNwYXJzZUNhdGVnb3JpY2FsQ3Jvc3NlbnRyb3B5KSB7XG4gICAgICAgIG91dHB1dFNoYXBlcy5wdXNoKFxuICAgICAgICAgICAgb3V0cHV0U2hhcGUuc2xpY2UoMCwgb3V0cHV0U2hhcGUubGVuZ3RoIC0gMSkuY29uY2F0KFsxXSkpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgLy8gUG9ydGluZyBOb3RlOiBCZWNhdXNlIG9mIHN0cm9uZyB0eXBpbmcgYGxvc3NGbmAgbXVzdCBiZSBhIGZ1bmN0aW9uLlxuICAgICAgICBvdXRwdXRTaGFwZXMucHVzaChvdXRwdXRTaGFwZSk7XG4gICAgICB9XG4gICAgfVxuICAgIHggPSBzdGFuZGFyZGl6ZUlucHV0RGF0YShcbiAgICAgICAgeCwgdGhpcy5mZWVkSW5wdXROYW1lcywgdGhpcy5mZWVkSW5wdXRTaGFwZXMsIGZhbHNlLCAnaW5wdXQnKTtcbiAgICB5ID0gc3RhbmRhcmRpemVJbnB1dERhdGEoXG4gICAgICAgIHksIHRoaXMuZmVlZE91dHB1dE5hbWVzLCBvdXRwdXRTaGFwZXMsIGZhbHNlLCAndGFyZ2V0Jyk7XG4gICAgLy8gVE9ETyhjYWlzKTogU3RhbmRhcmRpemUgc2FtcGxlV2VpZ2h0cyAmIGNsYXNzV2VpZ2h0cy5cbiAgICBjaGVja0FycmF5TGVuZ3Rocyh4LCB5LCBudWxsKTtcbiAgICAvLyBUT0RPKGNhaXMpOiBDaGVjayBzYW1wbGVXZWlnaHRzIGFzIHdlbGwuXG4gICAgY2hlY2tMb3NzQW5kVGFyZ2V0Q29tcGF0aWJpbGl0eSh5LCB0aGlzLmZlZWRMb3NzRm5zLCB0aGlzLmZlZWRPdXRwdXRTaGFwZXMpO1xuICAgIGlmICh0aGlzLnN0YXRlZnVsICYmIGJhdGNoU2l6ZSAhPSBudWxsICYmIGJhdGNoU2l6ZSA+IDApIHtcbiAgICAgIGlmICh4WzBdLnNoYXBlWzBdICUgYmF0Y2hTaXplICE9PSAwKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYEluIGEgc3RhdGVmdWwgbmV0d29yaywgeW91IHNob3VsZCBvbmx5IHBhc3MgaW5wdXRzIHdpdGggYSBgICtcbiAgICAgICAgICAgIGBudW1iZXIgb2Ygc2FtcGxlcyB0aGF0IGlzIGRpdmlzaWJsZSBieSB0aGUgYmF0Y2ggc2l6ZSBgICtcbiAgICAgICAgICAgIGAke2JhdGNoU2l6ZX0uIEZvdW5kOiAke3hbMF0uc2hhcGVbMF19IHNhbXBsZShzKS5gKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFt4LCB5XTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhc3luYyBzdGFuZGFyZGl6ZVVzZXJEYXRhKFxuICAgICAgeDogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgICAgeTogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgICAgc2FtcGxlV2VpZ2h0PzogVGVuc29yfFRlbnNvcltdfHtbb3V0cHV0TmFtZTogc3RyaW5nXTogVGVuc29yfSxcbiAgICAgIGNsYXNzV2VpZ2h0PzogQ2xhc3NXZWlnaHR8Q2xhc3NXZWlnaHRbXXxDbGFzc1dlaWdodE1hcCxcbiAgICAgIGNoZWNrQmF0Y2hBeGlzID0gdHJ1ZSxcbiAgICAgIGJhdGNoU2l6ZT86IG51bWJlcik6IFByb21pc2U8W1RlbnNvcltdLCBUZW5zb3JbXSwgVGVuc29yW11dPiB7XG4gICAgY29uc3QgW3N0YW5kYXJkWHMsIHN0YW5kYXJkWXNdID1cbiAgICAgICAgdGhpcy5zdGFuZGFyZGl6ZVVzZXJEYXRhWFkoeCwgeSwgY2hlY2tCYXRjaEF4aXMsIGJhdGNoU2l6ZSk7XG4gICAgLy8gVE9ETyhjYWlzKTogSGFuZGxlIHNhbXBsZVdlaWdodHMuXG4gICAgaWYgKHNhbXBsZVdlaWdodCAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ3NhbXBsZSB3ZWlnaHQgaXMgbm90IHN1cHBvcnRlZCB5ZXQuJyk7XG4gICAgfVxuXG4gICAgbGV0IHN0YW5kYXJkU2FtcGxlV2VpZ2h0czogVGVuc29yW10gPSBudWxsO1xuICAgIGlmIChjbGFzc1dlaWdodCAhPSBudWxsKSB7XG4gICAgICBjb25zdCBjbGFzc1dlaWdodHMgPVxuICAgICAgICAgIHN0YW5kYXJkaXplQ2xhc3NXZWlnaHRzKGNsYXNzV2VpZ2h0LCB0aGlzLm91dHB1dE5hbWVzKTtcbiAgICAgIHN0YW5kYXJkU2FtcGxlV2VpZ2h0cyA9IFtdO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBjbGFzc1dlaWdodHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgc3RhbmRhcmRTYW1wbGVXZWlnaHRzLnB1c2goXG4gICAgICAgICAgICBhd2FpdCBzdGFuZGFyZGl6ZVdlaWdodHMoc3RhbmRhcmRZc1tpXSwgbnVsbCwgY2xhc3NXZWlnaHRzW2ldKSk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gVE9ETyhjYWlzKTogRGVhbCB3aXRoIHRoZSBjYXNlIG9mIG1vZGVsLnN0YXRlZnVsID09IHRydWUuXG4gICAgcmV0dXJuIFtzdGFuZGFyZFhzLCBzdGFuZGFyZFlzLCBzdGFuZGFyZFNhbXBsZVdlaWdodHNdO1xuICB9XG5cbiAgLyoqXG4gICAqIExvb3Agb3ZlciBzb21lIHRlc3QgZGF0YSBpbiBiYXRjaGVzLlxuICAgKiBAcGFyYW0gZiBBIEZ1bmN0aW9uIHJldHVybmluZyBhIGxpc3Qgb2YgdGVuc29ycy5cbiAgICogQHBhcmFtIGlucyBBcnJheSBvZiB0ZW5zb3JzIHRvIGJlIGZlZCB0byBgZmAuXG4gICAqIEBwYXJhbSBiYXRjaFNpemUgSW50ZWdlciBiYXRjaCBzaXplIG9yIGBudWxsYCAvIGB1bmRlZmluZWRgLlxuICAgKiBAcGFyYW0gdmVyYm9zZSB2ZXJib3NpdHkgbW9kZS5cbiAgICogQHBhcmFtIHN0ZXBzIFRvdGFsIG51bWJlciBvZiBzdGVwcyAoYmF0Y2hlcyBvZiBzYW1wbGVzKSBiZWZvcmVcbiAgICogZGVjbGFyaW5nIHRlc3QgZmluaXNoZWQuIElnbm9yZWQgd2l0aCB0aGUgZGVmYXVsdCB2YWx1ZSBvZiBgbnVsbGAgL1xuICAgKiBgdW5kZWZpbmVkYC5cbiAgICogQHJldHVybnMgQXJyYXkgb2YgU2NhbGFycy5cbiAgICovXG4gIHByaXZhdGUgdGVzdExvb3AoXG4gICAgICBmOiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdLCBpbnM6IFRlbnNvcltdLCBiYXRjaFNpemU/OiBudW1iZXIsXG4gICAgICB2ZXJib3NlID0gMCwgc3RlcHM/OiBudW1iZXIpOiBTY2FsYXJbXSB7XG4gICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IG51bVNhbXBsZXMgPSB0aGlzLmNoZWNrTnVtU2FtcGxlcyhpbnMsIGJhdGNoU2l6ZSwgc3RlcHMsICdzdGVwcycpO1xuICAgICAgY29uc3Qgb3V0czogU2NhbGFyW10gPSBbXTtcbiAgICAgIGlmICh2ZXJib3NlID4gMCkge1xuICAgICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcignVmVyYm9zZSBtb2RlIGlzIG5vdCBpbXBsZW1lbnRlZCB5ZXQuJyk7XG4gICAgICB9XG4gICAgICAvLyBUT0RPKGNhaXMpOiBVc2UgYGluZGljZXNGb3JDb252ZXJzaW9uVG9EZW5zZScgdG8gcHJldmVudCBzbG93IGRvd24uXG4gICAgICBpZiAoc3RlcHMgIT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAgICdzdGVwcyBtb2RlIGluIHRlc3RMb29wKCkgaXMgbm90IGltcGxlbWVudGVkIHlldCcpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgY29uc3QgYmF0Y2hlcyA9IG1ha2VCYXRjaGVzKG51bVNhbXBsZXMsIGJhdGNoU2l6ZSk7XG4gICAgICAgIGNvbnN0IGluZGV4QXJyYXkgPSB0ZW5zb3IxZChyYW5nZSgwLCBudW1TYW1wbGVzKSk7XG4gICAgICAgIGZvciAobGV0IGJhdGNoSW5kZXggPSAwOyBiYXRjaEluZGV4IDwgYmF0Y2hlcy5sZW5ndGg7ICsrYmF0Y2hJbmRleCkge1xuICAgICAgICAgIGNvbnN0IGJhdGNoU3RhcnQgPSBiYXRjaGVzW2JhdGNoSW5kZXhdWzBdO1xuICAgICAgICAgIGNvbnN0IGJhdGNoRW5kID0gYmF0Y2hlc1tiYXRjaEluZGV4XVsxXTtcbiAgICAgICAgICBjb25zdCBiYXRjaElkcyA9XG4gICAgICAgICAgICAgIEsuc2xpY2VBbG9uZ0ZpcnN0QXhpcyhcbiAgICAgICAgICAgICAgICAgIGluZGV4QXJyYXksIGJhdGNoU3RhcnQsIGJhdGNoRW5kIC0gYmF0Y2hTdGFydCkgYXMgVGVuc29yMUQ7XG4gICAgICAgICAgLy8gVE9ETyhjYWlzKTogSW4gaW5zLCB0cmFpbiBmbGFnIGNhbiBiZSBhIG51bWJlciwgaW5zdGVhZCBvZiBhblxuICAgICAgICAgIC8vICAgVGVuc29yPyBEbyB3ZSBuZWVkIHRvIGhhbmRsZSB0aGlzIGluIHRmanMtbGF5ZXJzP1xuICAgICAgICAgIGNvbnN0IGluc0JhdGNoID0gc2xpY2VBcnJheXNCeUluZGljZXMoaW5zLCBiYXRjaElkcykgYXMgU2NhbGFyW107XG4gICAgICAgICAgY29uc3QgYmF0Y2hPdXRzID0gZihpbnNCYXRjaCk7XG4gICAgICAgICAgaWYgKGJhdGNoSW5kZXggPT09IDApIHtcbiAgICAgICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmF0Y2hPdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgICAgIG91dHMucHVzaChzY2FsYXIoMCkpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IGJhdGNoT3V0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgICAgY29uc3QgYmF0Y2hPdXQgPSBiYXRjaE91dHNbaV07XG4gICAgICAgICAgICBvdXRzW2ldID1cbiAgICAgICAgICAgICAgICB0ZmMuYWRkKG91dHNbaV0sIHRmYy5tdWwoYmF0Y2hFbmQgLSBiYXRjaFN0YXJ0LCBiYXRjaE91dCkpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgICBvdXRzW2ldID0gdGZjLmRpdihvdXRzW2ldLCBudW1TYW1wbGVzKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgcmV0dXJuIG91dHM7XG4gICAgfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgZ2V0RGVkdXBlZE1ldHJpY3NOYW1lcygpOiBzdHJpbmdbXSB7XG4gICAgY29uc3Qgb3V0TGFiZWxzID0gdGhpcy5tZXRyaWNzTmFtZXM7XG4gICAgLy8gUmVuYW1lIGR1cGxpY2F0ZWQgbWV0cmljcyBuYW1lcyAoY2FuIGhhcHBlbiB3aXRoIGFuIG91dHB1dCBsYXllclxuICAgIC8vIHNoYXJlZCBhbW9uZyBtdWx0aXBsZSBkYXRhZmxvd3MpLlxuICAgIGNvbnN0IGRlZHVwZWRPdXRMYWJlbHMgPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dExhYmVscy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgbGFiZWwgPSBvdXRMYWJlbHNbaV07XG4gICAgICBsZXQgbmV3TGFiZWwgPSBsYWJlbDtcbiAgICAgIGlmIChjb3VudChvdXRMYWJlbHMsIGxhYmVsKSA+IDEpIHtcbiAgICAgICAgY29uc3QgZHVwSW5kZXggPSBjb3VudChvdXRMYWJlbHMuc2xpY2UoMCwgaSksIGxhYmVsKTtcbiAgICAgICAgbmV3TGFiZWwgKz0gYF8ke2R1cEluZGV4fWA7XG4gICAgICB9XG4gICAgICBkZWR1cGVkT3V0TGFiZWxzLnB1c2gobmV3TGFiZWwpO1xuICAgIH1cbiAgICByZXR1cm4gZGVkdXBlZE91dExhYmVscztcbiAgfVxuXG4gIC8qKlxuICAgKiBDcmVhdGVzIGEgZnVuY3Rpb24gdGhhdCBwZXJmb3JtcyB0aGUgZm9sbG93aW5nIGFjdGlvbnM6XG4gICAqXG4gICAqIDEuIGNvbXB1dGVzIHRoZSBsb3NzZXNcbiAgICogMi4gc3VtcyB0aGVtIHRvIGdldCB0aGUgdG90YWwgbG9zc1xuICAgKiAzLiBjYWxsIHRoZSBvcHRpbWl6ZXIgY29tcHV0ZXMgdGhlIGdyYWRpZW50cyBvZiB0aGUgTGF5ZXJzTW9kZWwnc1xuICAgKiAgICB0cmFpbmFibGUgd2VpZ2h0cyB3LnIudC4gdGhlIHRvdGFsIGxvc3MgYW5kIHVwZGF0ZSB0aGUgdmFyaWFibGVzXG4gICAqIDQuIGNhbGN1bGF0ZXMgdGhlIG1ldHJpY3NcbiAgICogNS4gcmV0dXJucyB0aGUgdmFsdWVzIG9mIHRoZSBsb3NzZXMgYW5kIG1ldHJpY3MuXG4gICAqL1xuICBwcm90ZWN0ZWQgbWFrZVRyYWluRnVuY3Rpb24oKTogKGRhdGE6IFRlbnNvcltdKSA9PiBTY2FsYXJbXSB7XG4gICAgcmV0dXJuIChkYXRhOiBUZW5zb3JbXSkgPT4ge1xuICAgICAgY29uc3QgbG9zc1ZhbHVlczogU2NhbGFyW10gPSBbXTtcblxuICAgICAgY29uc3QgaW5wdXRzID0gZGF0YS5zbGljZSgwLCB0aGlzLmlucHV0cy5sZW5ndGgpO1xuICAgICAgY29uc3QgdGFyZ2V0cyA9IGRhdGEuc2xpY2UoXG4gICAgICAgICAgdGhpcy5pbnB1dHMubGVuZ3RoLCB0aGlzLmlucHV0cy5sZW5ndGggKyB0aGlzLm91dHB1dHMubGVuZ3RoKTtcbiAgICAgIGNvbnN0IHNhbXBsZVdlaWdodHMgPSBkYXRhLnNsaWNlKFxuICAgICAgICAgIHRoaXMuaW5wdXRzLmxlbmd0aCArIHRoaXMub3V0cHV0cy5sZW5ndGgsXG4gICAgICAgICAgdGhpcy5pbnB1dHMubGVuZ3RoICsgdGhpcy5vdXRwdXRzLmxlbmd0aCAqIDIpO1xuXG4gICAgICBjb25zdCBtZXRyaWNzVmFsdWVzOiBTY2FsYXJbXSA9IFtdO1xuXG4gICAgICAvLyBDcmVhdGUgYSBmdW5jdGlvbiB0aGF0IGNvbXB1dGVzIHRoZSB0b3RhbCBsb3NzIGJhc2VkIG9uIHRoZVxuICAgICAgLy8gaW5wdXRzLiBUaGlzIGZ1bmN0aW9uIGlzIHVzZWQgZm9yIG9idGFpbmluZyBncmFkaWVudHMgdGhyb3VnaFxuICAgICAgLy8gYmFja3Byb3AuXG4gICAgICBjb25zdCB0b3RhbExvc3NGdW5jdGlvbiA9ICgpID0+IHtcbiAgICAgICAgY29uc3QgZmVlZHMgPSBbXTtcbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLmlucHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgIGZlZWRzLnB1c2goe2tleTogdGhpcy5pbnB1dHNbaV0sIHZhbHVlOiBpbnB1dHNbaV19KTtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBmZWVkRGljdCA9IG5ldyBGZWVkRGljdChmZWVkcyk7XG4gICAgICAgIGNvbnN0IG91dHB1dHMgPVxuICAgICAgICAgICAgZXhlY3V0ZSh0aGlzLm91dHB1dHMsIGZlZWREaWN0LCB7J3RyYWluaW5nJzogdHJ1ZX0pIGFzIFRlbnNvcltdO1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgdGhlIGNhc2Ugb2YgbXVsdGlwbGUgb3V0cHV0cyBmcm9tIGFcbiAgICAgICAgLy8gICBzaW5nbGUgbGF5ZXI/XG5cbiAgICAgICAgbGV0IHRvdGFsTG9zczogVGVuc29yO1xuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMubG9zc0Z1bmN0aW9ucy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgIGNvbnN0IGxvc3NGdW5jdGlvbiA9IHRoaXMubG9zc0Z1bmN0aW9uc1tpXTtcbiAgICAgICAgICBsZXQgbG9zcyA9IGxvc3NGdW5jdGlvbih0YXJnZXRzW2ldLCBvdXRwdXRzW2ldKTtcbiAgICAgICAgICBpZiAoc2FtcGxlV2VpZ2h0c1tpXSAhPSBudWxsKSB7XG4gICAgICAgICAgICBsb3NzID0gY29tcHV0ZVdlaWdodGVkTG9zcyhsb3NzLCBzYW1wbGVXZWlnaHRzW2ldKTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBwdXNoIFNjYWxhciBpbnN0ZWFkLlxuICAgICAgICAgIGNvbnN0IG1lYW5Mb3NzOiBTY2FsYXIgPSB0ZmMubWVhbihsb3NzKTtcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBVc2UgYSBzY29wZSgpIGluc3RlYWQsIHRvIGF2b2lkIG93bmVyc2hpcC5cbiAgICAgICAgICBsb3NzVmFsdWVzLnB1c2gobWVhbkxvc3MpO1xuICAgICAgICAgIGlmIChpID09PSAwKSB7XG4gICAgICAgICAgICB0b3RhbExvc3MgPSBsb3NzO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB0b3RhbExvc3MgPSB0ZmMuYWRkKHRvdGFsTG9zcywgbG9zcyk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG5cbiAgICAgICAgLy8gQ29tcHV0ZSB0aGUgbWV0cmljcy5cbiAgICAgICAgLy8gVE9ETyhjYWlzKTogVGhlc2Ugc2hvdWxkIHByb2JhYmx5IGJlIGNhbGN1bGF0ZWQgb3V0c2lkZVxuICAgICAgICAvLyAgIHRvdGFsTG9zc0Z1bmN0aW9uIHRvIGJlbmVmaXQgc3BlZWQ/XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5tZXRyaWNzVGVuc29ycy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgIGxldCB3ZWlnaHRlZE1ldHJpYzogU2NhbGFyO1xuXG4gICAgICAgICAgaWYgKHRoaXMub3V0cHV0cy5sZW5ndGggPiAxICYmIGkgPCB0aGlzLm91dHB1dHMubGVuZ3RoKSB7XG4gICAgICAgICAgICB3ZWlnaHRlZE1ldHJpYyA9IGxvc3NWYWx1ZXNbaV07XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGNvbnN0IG1ldHJpYyA9IHRoaXMubWV0cmljc1RlbnNvcnNbaV1bMF07XG4gICAgICAgICAgICBjb25zdCBvdXRwdXRJbmRleCA9IHRoaXMubWV0cmljc1RlbnNvcnNbaV1bMV07XG4gICAgICAgICAgICB3ZWlnaHRlZE1ldHJpYyA9XG4gICAgICAgICAgICAgICAgdGZjLm1lYW4obWV0cmljKHRhcmdldHNbb3V0cHV0SW5kZXhdLCBvdXRwdXRzW291dHB1dEluZGV4XSkpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIHRmYy5rZWVwKHdlaWdodGVkTWV0cmljKTtcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBVc2UgYSBzY29wZSgpIGluc3RlYWQsIHRvIGF2b2lkIG93bmVyc2hpcC5cbiAgICAgICAgICBtZXRyaWNzVmFsdWVzLnB1c2god2VpZ2h0ZWRNZXRyaWMpO1xuICAgICAgICB9XG5cbiAgICAgICAgdG90YWxMb3NzID0gdGZjLm1lYW4odG90YWxMb3NzKTtcblxuICAgICAgICAvLyBBZGQgcmVndWxhcml6ZXIgcGVuYWx0aWVzLlxuICAgICAgICB0aGlzLmNhbGN1bGF0ZUxvc3NlcygpLmZvckVhY2gocmVndWxhcml6ZXJMb3NzID0+IHtcbiAgICAgICAgICB0b3RhbExvc3MgPSB0ZmMuYWRkKHRvdGFsTG9zcywgcmVndWxhcml6ZXJMb3NzKTtcbiAgICAgICAgfSk7XG5cbiAgICAgICAgcmV0dXJuIHRvdGFsTG9zcyBhcyBTY2FsYXI7XG4gICAgICB9O1xuXG4gICAgICBjb25zdCB2YXJpYWJsZXMgPSB0aGlzLmNvbGxlY3RlZFRyYWluYWJsZVdlaWdodHMubWFwKFxuICAgICAgICAgIHBhcmFtID0+IHBhcmFtLnJlYWQoKSBhcyB0ZmMuVmFyaWFibGUpO1xuICAgICAgY29uc3QgcmV0dXJuQ29zdCA9IHRydWU7XG4gICAgICBjb25zdCB0b3RhbExvc3NWYWx1ZSA9XG4gICAgICAgICAgdGhpcy5vcHRpbWl6ZXJfLm1pbmltaXplKHRvdGFsTG9zc0Z1bmN0aW9uLCByZXR1cm5Db3N0LCB2YXJpYWJsZXMpO1xuXG4gICAgICByZXR1cm4gW3RvdGFsTG9zc1ZhbHVlXS5jb25jYXQobWV0cmljc1ZhbHVlcyk7XG4gICAgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDcmVhdGUgYSBmdW5jdGlvbiB3aGljaCwgd2hlbiBpbnZva2VkIHdpdGggYW4gYXJyYXkgb2YgYHRmLlRlbnNvcmBzIGFzIGFcbiAgICogYmF0Y2ggb2YgaW5wdXRzLCByZXR1cm5zIHRoZSBwcmVzcGVjaWZpZWQgbG9zcyBhbmQgbWV0cmljcyBvZiB0aGUgbW9kZWxcbiAgICogdW5kZXIgdGhlIGJhdGNoIG9mIGlucHV0IGRhdGEuXG4gICAqL1xuICBwcml2YXRlIG1ha2VUZXN0RnVuY3Rpb24oKSB7XG4gICAgdGhpcy50ZXN0RnVuY3Rpb24gPSAoZGF0YTogVGVuc29yW10pID0+IHtcbiAgICAgIHJldHVybiB0ZmMudGlkeSgoKSA9PiB7XG4gICAgICAgIGNvbnN0IHZhbE91dHB1dHM6IFNjYWxhcltdID0gW107XG4gICAgICAgIGxldCB0b3RhbExvc3M6IFNjYWxhcjtcbiAgICAgICAgY29uc3QgaW5wdXRzID0gZGF0YS5zbGljZSgwLCB0aGlzLmlucHV0cy5sZW5ndGgpO1xuICAgICAgICBjb25zdCB0YXJnZXRzID0gZGF0YS5zbGljZShcbiAgICAgICAgICAgIHRoaXMuaW5wdXRzLmxlbmd0aCwgdGhpcy5pbnB1dHMubGVuZ3RoICsgdGhpcy5vdXRwdXRzLmxlbmd0aCk7XG4gICAgICAgIGNvbnN0IGZlZWRzID0gW107XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5pbnB1dHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgICBmZWVkcy5wdXNoKHtrZXk6IHRoaXMuaW5wdXRzW2ldLCB2YWx1ZTogaW5wdXRzW2ldfSk7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgZmVlZERpY3QgPSBuZXcgRmVlZERpY3QoZmVlZHMpO1xuICAgICAgICBjb25zdCBvdXRwdXRzID0gZXhlY3V0ZSh0aGlzLm91dHB1dHMsIGZlZWREaWN0KSBhcyBUZW5zb3JbXTtcbiAgICAgICAgLy8gQ29tcHV0ZSB0b3RhbCBsb3NzLlxuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMubG9zc0Z1bmN0aW9ucy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgIGNvbnN0IGxvc3NGdW5jdGlvbiA9IHRoaXMubG9zc0Z1bmN0aW9uc1tpXTtcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgc2FtcGxlIHdlaWdodGluZyBhbmQgcmVwbGFjZSB0aGUgc2ltcGxlXG4gICAgICAgICAgLy8gYXZlcmFnaW5nLlxuICAgICAgICAgIGNvbnN0IGxvc3M6IFNjYWxhciA9IHRmYy5tZWFuKGxvc3NGdW5jdGlvbih0YXJnZXRzW2ldLCBvdXRwdXRzW2ldKSk7XG4gICAgICAgICAgaWYgKGkgPT09IDApIHtcbiAgICAgICAgICAgIHRvdGFsTG9zcyA9IGxvc3M7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHRvdGFsTG9zcyA9IHRmYy5hZGQodG90YWxMb3NzLCBsb3NzKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgdmFsT3V0cHV0cy5wdXNoKHRvdGFsTG9zcyk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gQ29tcHV0ZSB0aGUgbWV0cmljcy5cbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLm1ldHJpY3NUZW5zb3JzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgY29uc3QgbWV0cmljID0gdGhpcy5tZXRyaWNzVGVuc29yc1tpXVswXTtcbiAgICAgICAgICBjb25zdCBvdXRwdXRJbmRleCA9IHRoaXMubWV0cmljc1RlbnNvcnNbaV1bMV07XG4gICAgICAgICAgLy8gVE9ETyhjYWlzKTogUmVwbGFjZSBLLm1lYW4oKSB3aXRoIGEgcHJvcGVyIHdlaWdodGluZyBmdW5jdGlvbi5cbiAgICAgICAgICBjb25zdCBtZWFuTWV0cmljID1cbiAgICAgICAgICAgICAgdGZjLm1lYW4obWV0cmljKHRhcmdldHNbb3V0cHV0SW5kZXhdLCBvdXRwdXRzW291dHB1dEluZGV4XSkpO1xuICAgICAgICAgIHZhbE91dHB1dHMucHVzaChtZWFuTWV0cmljIGFzIFNjYWxhcik7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHZhbE91dHB1dHM7XG4gICAgICB9KTtcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIFRyYWlucyB0aGUgbW9kZWwgZm9yIGEgZml4ZWQgbnVtYmVyIG9mIGVwb2NocyAoaXRlcmF0aW9ucyBvbiBhXG4gICAqIGRhdGFzZXQpLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoe1xuICAgKiAgICAgbGF5ZXJzOiBbdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgaW5wdXRTaGFwZTogWzEwXX0pXVxuICAgKiB9KTtcbiAgICogbW9kZWwuY29tcGlsZSh7b3B0aW1pemVyOiAnc2dkJywgbG9zczogJ21lYW5TcXVhcmVkRXJyb3InfSk7XG4gICAqIGZvciAobGV0IGkgPSAxOyBpIDwgNSA7ICsraSkge1xuICAgKiAgIGNvbnN0IGggPSBhd2FpdCBtb2RlbC5maXQodGYub25lcyhbOCwgMTBdKSwgdGYub25lcyhbOCwgMV0pLCB7XG4gICAqICAgICAgIGJhdGNoU2l6ZTogNCxcbiAgICogICAgICAgZXBvY2hzOiAzXG4gICAqICAgfSk7XG4gICAqICAgY29uc29sZS5sb2coXCJMb3NzIGFmdGVyIEVwb2NoIFwiICsgaSArIFwiIDogXCIgKyBoLmhpc3RvcnkubG9zc1swXSk7XG4gICAqIH1cbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSB4IGB0Zi5UZW5zb3JgIG9mIHRyYWluaW5nIGRhdGEsIG9yIGFuIGFycmF5IG9mIGB0Zi5UZW5zb3JgcyBpZiB0aGVcbiAgICogbW9kZWwgaGFzIG11bHRpcGxlIGlucHV0cy4gSWYgYWxsIGlucHV0cyBpbiB0aGUgbW9kZWwgYXJlIG5hbWVkLCB5b3VcbiAgICogY2FuIGFsc28gcGFzcyBhIGRpY3Rpb25hcnkgbWFwcGluZyBpbnB1dCBuYW1lcyB0byBgdGYuVGVuc29yYHMuXG4gICAqIEBwYXJhbSB5IGB0Zi5UZW5zb3JgIG9mIHRhcmdldCAobGFiZWwpIGRhdGEsIG9yIGFuIGFycmF5IG9mIGB0Zi5UZW5zb3JgcyBpZlxuICAgKiB0aGUgbW9kZWwgaGFzIG11bHRpcGxlIG91dHB1dHMuIElmIGFsbCBvdXRwdXRzIGluIHRoZSBtb2RlbCBhcmUgbmFtZWQsXG4gICAqIHlvdSBjYW4gYWxzbyBwYXNzIGEgZGljdGlvbmFyeSBtYXBwaW5nIG91dHB1dCBuYW1lcyB0byBgdGYuVGVuc29yYHMuXG4gICAqIEBwYXJhbSBhcmdzIEEgYE1vZGVsRml0QXJnc2AsIGNvbnRhaW5pbmcgb3B0aW9uYWwgZmllbGRzLlxuICAgKlxuICAgKiBAcmV0dXJuIEEgYEhpc3RvcnlgIGluc3RhbmNlLiBJdHMgYGhpc3RvcnlgIGF0dHJpYnV0ZSBjb250YWlucyBhbGxcbiAgICogICBpbmZvcm1hdGlvbiBjb2xsZWN0ZWQgZHVyaW5nIHRyYWluaW5nLlxuICAgKlxuICAgKiBAZXhjZXB0aW9uIFZhbHVlRXJyb3IgSW4gY2FzZSBvZiBtaXNtYXRjaCBiZXR3ZWVuIHRoZSBwcm92aWRlZCBpbnB1dFxuICAgKiBkYXRhIGFuZCB3aGF0IHRoZSBtb2RlbCBleHBlY3RzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYXN5bmMgZml0KFxuICAgICAgeDogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgICAgeTogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgICAgYXJnczogTW9kZWxGaXRBcmdzID0ge30pOiBQcm9taXNlPEhpc3Rvcnk+IHtcbiAgICByZXR1cm4gZml0VGVuc29ycyh0aGlzLCB4LCB5LCBhcmdzKTtcbiAgfVxuXG4gIC8vIFRPRE8oY2Fpcyk6IEFkZCBjb2RlIHNuaXBwZXQgYmVsb3cgd2hlbiBpdCdzIHBvc3NpYmxlIHRvIGluc3RhbnRpYXRlXG4gIC8vICAgYWN0dWFsIGRhdGFzZXQgb2JqZWN0cy5cbiAgLyoqXG4gICAqIFRyYWlucyB0aGUgbW9kZWwgdXNpbmcgYSBkYXRhc2V0IG9iamVjdC5cbiAgICpcbiAgICogQHBhcmFtIGRhdGFzZXQgQSBkYXRhc2V0IG9iamVjdC4gSXRzIGBpdGVyYXRvcigpYCBtZXRob2QgaXMgZXhwZWN0ZWRcbiAgICogICB0byBnZW5lcmF0ZSBhIGRhdGFzZXQgaXRlcmF0b3Igb2JqZWN0LCB0aGUgYG5leHQoKWAgbWV0aG9kIG9mIHdoaWNoXG4gICAqICAgaXMgZXhwZWN0ZWQgdG8gcHJvZHVjZSBkYXRhIGJhdGNoZXMgZm9yIHRyYWluaW5nLiBUaGUgcmV0dXJuIHZhbHVlXG4gICAqICAgb2YgdGhlIGBuZXh0KClgIGNhbGwgb3VnaHQgdG8gY29udGFpbiBhIGJvb2xlYW4gYGRvbmVgIGZpZWxkIGFuZCBhXG4gICAqICAgYHZhbHVlYCBmaWVsZC4gVGhlIGB2YWx1ZWAgZmllbGQgaXMgZXhwZWN0ZWQgdG8gYmUgYW4gYXJyYXkgb2YgdHdvXG4gICAqICAgYHRmLlRlbnNvcmBzIG9yIGFuIGFycmF5IG9mIHR3byBuZXN0ZWQgYHRmLlRlbnNvcmAgc3RydWN0dXJlcy4gVGhlIGZvcm1lclxuICAgKiAgIGNhc2UgaXMgZm9yIG1vZGVscyB3aXRoIGV4YWN0bHkgb25lIGlucHV0IGFuZCBvbmUgb3V0cHV0IChlLmcuLlxuICAgKiAgIGEgc2VxdWVudGlhbCBtb2RlbCkuIFRoZSBsYXR0ZXIgY2FzZSBpcyBmb3IgbW9kZWxzIHdpdGggbXVsdGlwbGVcbiAgICogICBpbnB1dHMgYW5kL29yIG11bHRpcGxlIG91dHB1dHMuXG4gICAqICAgT2YgdGhlIHR3byBpdGVtcyBpbiB0aGUgYXJyYXksIHRoZSBmaXJzdCBpcyB0aGUgaW5wdXQgZmVhdHVyZShzKSBhbmRcbiAgICogICB0aGUgc2Vjb25kIGlzIHRoZSBvdXRwdXQgdGFyZ2V0KHMpLlxuICAgKiBAcGFyYW0gYXJncyBBIGBNb2RlbEZpdERhdGFzZXRBcmdzYCwgY29udGFpbmluZyBvcHRpb25hbCBmaWVsZHMuXG4gICAqXG4gICAqIEByZXR1cm4gQSBgSGlzdG9yeWAgaW5zdGFuY2UuIEl0cyBgaGlzdG9yeWAgYXR0cmlidXRlIGNvbnRhaW5zIGFsbFxuICAgKiAgIGluZm9ybWF0aW9uIGNvbGxlY3RlZCBkdXJpbmcgdHJhaW5pbmcuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhc3luYyBmaXREYXRhc2V0PFQ+KGRhdGFzZXQ6IERhdGFzZXQ8VD4sIGFyZ3M6IE1vZGVsRml0RGF0YXNldEFyZ3M8VD4pOlxuICAgICAgUHJvbWlzZTxIaXN0b3J5PiB7XG4gICAgcmV0dXJuIGZpdERhdGFzZXQodGhpcywgZGF0YXNldCwgYXJncyk7XG4gIH1cblxuICAvKipcbiAgICogUnVucyBhIHNpbmdsZSBncmFkaWVudCB1cGRhdGUgb24gYSBzaW5nbGUgYmF0Y2ggb2YgZGF0YS5cbiAgICpcbiAgICogVGhpcyBtZXRob2QgZGlmZmVycyBmcm9tIGBmaXQoKWAgYW5kIGBmaXREYXRhc2V0KClgIGluIHRoZSBmb2xsb3dpbmdcbiAgICogcmVnYXJkczpcbiAgICogICAtIEl0IG9wZXJhdGVzIG9uIGV4YWN0bHkgb25lIGJhdGNoIG9mIGRhdGEuXG4gICAqICAgLSBJdCByZXR1cm5zIG9ubHkgdGhlIGxvc3MgYW5kIG1hdHJpYyB2YWx1ZXMsIGluc3RlYWQgb2ZcbiAgICogICAgIHJldHVybmluZyB0aGUgYmF0Y2gtYnktYmF0Y2ggbG9zcyBhbmQgbWV0cmljIHZhbHVlcy5cbiAgICogICAtIEl0IGRvZXNuJ3Qgc3VwcG9ydCBmaW5lLWdyYWluZWQgb3B0aW9ucyBzdWNoIGFzIHZlcmJvc2l0eSBhbmRcbiAgICogICAgIGNhbGxiYWNrcy5cbiAgICpcbiAgICogQHBhcmFtIHggSW5wdXQgZGF0YS4gSXQgY291bGQgYmUgb25lIG9mIHRoZSBmb2xsb3dpbmc6XG4gICAqICAgLSBBIGB0Zi5UZW5zb3JgLCBvciBhbiBBcnJheSBvZiBgdGYuVGVuc29yYHMgKGluIGNhc2UgdGhlIG1vZGVsIGhhc1xuICAgKiAgICAgbXVsdGlwbGUgaW5wdXRzKS5cbiAgICogICAtIEFuIE9iamVjdCBtYXBwaW5nIGlucHV0IG5hbWVzIHRvIGNvcnJlc3BvbmRpbmcgYHRmLlRlbnNvcmAgKGlmIHRoZVxuICAgKiAgICAgbW9kZWwgaGFzIG5hbWVkIGlucHV0cykuXG4gICAqIEBwYXJhbSB5IFRhcmdldCBkYXJ0YS4gSXQgY291bGQgYmUgZWl0aGVyIGEgYHRmLlRlbnNvcmAgYSBtdWx0aXBsZVxuICAgKiAgIGB0Zi5UZW5zb3Jgcy4gSXQgc2hvdWxkIGJlIGNvbnNpc3RlbnQgd2l0aCBgeGAuXG4gICAqIEByZXR1cm5zIFRyYWluaW5nIGxvc3Mgb3IgbG9zc2VzIChpbiBjYXNlIHRoZSBtb2RlbCBoYXNcbiAgICogICBtdWx0aXBsZSBvdXRwdXRzKSwgYWxvbmcgd2l0aCBtZXRyaWNzIChpZiBhbnkpLCBhcyBudW1iZXJzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYXN5bmMgdHJhaW5PbkJhdGNoKFxuICAgICAgeDogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgICAgeTogVGVuc29yfFRlbnNvcltdfFxuICAgICAge1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0pOiBQcm9taXNlPG51bWJlcnxudW1iZXJbXT4ge1xuICAgIC8vIFRPRE8oY2Fpcyk6IFN1cHBvcnQgc2FtcGxlV2VpZ2h0IGFuZCBjbGFzc1dlaWdodC5cbiAgICAvLyBUT0RPKGNhaXMpOiBTdXBwb3J0IERhdGFzZXQgb2JqZWN0cy5cbiAgICBjb25zdCBzdGFuZGFyZGl6ZU91dCA9IGF3YWl0IHRoaXMuc3RhbmRhcmRpemVVc2VyRGF0YSh4LCB5KTtcbiAgICBjb25zdCBpbnB1dHMgPSBzdGFuZGFyZGl6ZU91dFswXTtcbiAgICBjb25zdCB0YXJnZXRzID0gc3RhbmRhcmRpemVPdXRbMV07XG4gICAgY29uc3QgdHJhaW5GdW5jdGlvbiA9IHRoaXMubWFrZVRyYWluRnVuY3Rpb24oKTtcbiAgICBjb25zdCBsb3NzZXMgPSB0cmFpbkZ1bmN0aW9uKGlucHV0cy5jb25jYXQodGFyZ2V0cykpO1xuICAgIGNvbnN0IGxvc3NWYWx1ZXM6IG51bWJlcltdID0gW107XG4gICAgZm9yIChjb25zdCBsb3NzIG9mIGxvc3Nlcykge1xuICAgICAgY29uc3QgdiA9IGF3YWl0IGxvc3MuZGF0YSgpO1xuICAgICAgbG9zc1ZhbHVlcy5wdXNoKHZbMF0pO1xuICAgIH1cbiAgICB0ZmMuZGlzcG9zZShsb3NzZXMpO1xuICAgIGRpc3Bvc2VOZXdUZW5zb3JzKHN0YW5kYXJkaXplT3V0WzBdLCB4KTtcbiAgICBkaXNwb3NlTmV3VGVuc29ycyhzdGFuZGFyZGl6ZU91dFsxXSwgeSk7XG4gICAgcmV0dXJuIHNpbmdsZXRvbk9yQXJyYXkobG9zc1ZhbHVlcyk7XG4gIH1cblxuICAvKipcbiAgICogRXh0cmFjdCB3ZWlnaHQgdmFsdWVzIG9mIHRoZSBtb2RlbC5cbiAgICpcbiAgICogQHBhcmFtIGNvbmZpZzogQW4gaW5zdGFuY2Ugb2YgYGlvLlNhdmVDb25maWdgLCB3aGljaCBzcGVjaWZpZXNcbiAgICogbW9kZWwtc2F2aW5nIG9wdGlvbnMgc3VjaCBhcyB3aGV0aGVyIG9ubHkgdHJhaW5hYmxlIHdlaWdodHMgYXJlIHRvIGJlXG4gICAqIHNhdmVkLlxuICAgKiBAcmV0dXJucyBBIGBOYW1lZFRlbnNvck1hcGAgbWFwcGluZyBvcmlnaW5hbCB3ZWlnaHQgbmFtZXMgKGkuZS4sXG4gICAqICAgbm9uLXVuaXF1ZWlmaWVkIHdlaWdodCBuYW1lcykgdG8gdGhlaXIgdmFsdWVzLlxuICAgKi9cbiAgcHJvdGVjdGVkIGdldE5hbWVkV2VpZ2h0cyhjb25maWc/OiBpby5TYXZlQ29uZmlnKTogTmFtZWRUZW5zb3JbXSB7XG4gICAgY29uc3QgbmFtZWRXZWlnaHRzOiBOYW1lZFRlbnNvcltdID0gW107XG5cbiAgICBjb25zdCB0cmFpbmFibGVPbmx5ID0gY29uZmlnICE9IG51bGwgJiYgY29uZmlnLnRyYWluYWJsZU9ubHk7XG4gICAgY29uc3Qgd2VpZ2h0cyA9IHRyYWluYWJsZU9ubHkgPyB0aGlzLnRyYWluYWJsZVdlaWdodHMgOiB0aGlzLndlaWdodHM7XG4gICAgY29uc3Qgd2VpZ2h0VmFsdWVzID0gdGhpcy5nZXRXZWlnaHRzKHRyYWluYWJsZU9ubHkpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgd2VpZ2h0cy5sZW5ndGg7ICsraSkge1xuICAgICAgaWYgKHRyYWluYWJsZU9ubHkgJiYgIXdlaWdodHNbaV0udHJhaW5hYmxlKSB7XG4gICAgICAgIC8vIE9wdGlvbmFsbHkgc2tpcCBub24tdHJhaW5hYmxlIHdlaWdodHMuXG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgbmFtZWRXZWlnaHRzLnB1c2goXG4gICAgICAgICAge25hbWU6IHdlaWdodHNbaV0ub3JpZ2luYWxOYW1lLCB0ZW5zb3I6IHdlaWdodFZhbHVlc1tpXX0pO1xuICAgIH1cbiAgICByZXR1cm4gbmFtZWRXZWlnaHRzO1xuICB9XG5cbiAgLyoqXG4gICAqIFNldHRlciB1c2VkIGZvciBmb3JjZSBzdG9wcGluZyBvZiBMYXllcnNNb2RlbC5maXQoKSAoaS5lLiwgdHJhaW5pbmcpLlxuICAgKlxuICAgKiBFeGFtcGxlOlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBpbnB1dCA9IHRmLmlucHV0KHtzaGFwZTogWzEwXX0pO1xuICAgKiBjb25zdCBvdXRwdXQgPSB0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxfSkuYXBwbHkoaW5wdXQpO1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLm1vZGVsKHtpbnB1dHM6IFtpbnB1dF0sIG91dHB1dHM6IFtvdXRwdXRdfSk7XG4gICAqIG1vZGVsLmNvbXBpbGUoe2xvc3M6ICdtZWFuU3F1YXJlZEVycm9yJywgb3B0aW1pemVyOiAnc2dkJ30pO1xuICAgKiBjb25zdCB4cyA9IHRmLm9uZXMoWzgsIDEwXSk7XG4gICAqIGNvbnN0IHlzID0gdGYuemVyb3MoWzgsIDFdKTtcbiAgICpcbiAgICogY29uc3QgaGlzdG9yeSA9IGF3YWl0IG1vZGVsLmZpdCh4cywgeXMsIHtcbiAgICogICBlcG9jaHM6IDEwLFxuICAgKiAgIGNhbGxiYWNrczoge1xuICAgKiAgICAgb25FcG9jaEVuZDogYXN5bmMgKGVwb2NoLCBsb2dzKSA9PiB7XG4gICAqICAgICAgIGlmIChlcG9jaCA9PT0gMikge1xuICAgKiAgICAgICAgIG1vZGVsLnN0b3BUcmFpbmluZyA9IHRydWU7XG4gICAqICAgICAgIH1cbiAgICogICAgIH1cbiAgICogICB9XG4gICAqIH0pO1xuICAgKlxuICAgKiAvLyBUaGVyZSBzaG91bGQgYmUgb25seSAzIHZhbHVlcyBpbiB0aGUgbG9zcyBhcnJheSwgaW5zdGVhZCBvZiAxMFxuICAgKiB2YWx1ZXMsXG4gICAqIC8vIGR1ZSB0byB0aGUgc3RvcHBpbmcgYWZ0ZXIgMyBlcG9jaHMuXG4gICAqIGNvbnNvbGUubG9nKGhpc3RvcnkuaGlzdG9yeS5sb3NzKTtcbiAgICogYGBgXG4gICAqL1xuICBzZXQgc3RvcFRyYWluaW5nKHN0b3A6IGJvb2xlYW4pIHtcbiAgICB0aGlzLnN0b3BUcmFpbmluZ18gPSBzdG9wO1xuICB9XG5cbiAgZ2V0IHN0b3BUcmFpbmluZygpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy5zdG9wVHJhaW5pbmdfO1xuICB9XG5cbiAgZ2V0IG9wdGltaXplcigpOiBPcHRpbWl6ZXIge1xuICAgIHJldHVybiB0aGlzLm9wdGltaXplcl87XG4gIH1cblxuICBzZXQgb3B0aW1pemVyKG9wdGltaXplcjogT3B0aW1pemVyKSB7XG4gICAgaWYgKHRoaXMub3B0aW1pemVyXyAhPT0gb3B0aW1pemVyKSB7XG4gICAgICB0aGlzLm9wdGltaXplcl8gPSBvcHRpbWl6ZXI7XG4gICAgICB0aGlzLmlzT3B0aW1pemVyT3duZWQgPSBmYWxzZTtcbiAgICB9XG4gIH1cblxuICBkaXNwb3NlKCk6IERpc3Bvc2VSZXN1bHQge1xuICAgIGNvbnN0IHJlc3VsdCA9IHN1cGVyLmRpc3Bvc2UoKTtcbiAgICBpZiAocmVzdWx0LnJlZkNvdW50QWZ0ZXJEaXNwb3NlID09PSAwICYmIHRoaXMub3B0aW1pemVyICE9IG51bGwgJiZcbiAgICAgICAgdGhpcy5pc09wdGltaXplck93bmVkKSB7XG4gICAgICBjb25zdCBudW1UZW5zb3JzQmVmb3JlT3B0bWl6ZXJEaXNwb3NhbCA9IHRmYy5tZW1vcnkoKS5udW1UZW5zb3JzO1xuICAgICAgdGhpcy5vcHRpbWl6ZXJfLmRpc3Bvc2UoKTtcbiAgICAgIHJlc3VsdC5udW1EaXNwb3NlZFZhcmlhYmxlcyArPVxuICAgICAgICAgIG51bVRlbnNvcnNCZWZvcmVPcHRtaXplckRpc3Bvc2FsIC0gdGZjLm1lbW9yeSgpLm51bVRlbnNvcnM7XG4gICAgfVxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcml2YXRlIGdldExvc3NJZGVudGlmaWVycygpOiBMb3NzSWRlbnRpZmllcnxMb3NzSWRlbnRpZmllcltdfFxuICAgICAge1tvdXRwdXROYW1lOiBzdHJpbmddOiBMb3NzSWRlbnRpZmllcn0ge1xuICAgIGxldCBsb3NzTmFtZXM6IExvc3NJZGVudGlmaWVyfExvc3NJZGVudGlmaWVyW118XG4gICAgICAgIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTG9zc0lkZW50aWZpZXJ9O1xuICAgIGlmICh0eXBlb2YgdGhpcy5sb3NzID09PSAnc3RyaW5nJykge1xuICAgICAgbG9zc05hbWVzID0gdG9TbmFrZUNhc2UodGhpcy5sb3NzKSBhcyBMb3NzSWRlbnRpZmllcjtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5sb3NzKSkge1xuICAgICAgZm9yIChjb25zdCBsb3NzIG9mIHRoaXMubG9zcykge1xuICAgICAgICBpZiAodHlwZW9mIGxvc3MgIT09ICdzdHJpbmcnKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdTZXJpYWxpemF0aW9uIG9mIG5vbi1zdHJpbmcgbG9zcyBpcyBub3Qgc3VwcG9ydGVkLicpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBsb3NzTmFtZXMgPSAodGhpcy5sb3NzIGFzIHN0cmluZ1tdKS5tYXAobmFtZSA9PiB0b1NuYWtlQ2FzZShuYW1lKSkgYXNcbiAgICAgICAgICBMb3NzSWRlbnRpZmllcltdO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBvdXRwdXROYW1lcyA9IE9iamVjdC5rZXlzKHRoaXMubG9zcyk7XG4gICAgICBsb3NzTmFtZXMgPSB7fSBhcyB7W291dHB1dE5hbWU6IHN0cmluZ106IExvc3NJZGVudGlmaWVyfTtcbiAgICAgIGNvbnN0IGxvc3NlcyA9XG4gICAgICAgICAgdGhpcy5sb3NzIGFzIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTG9zc09yTWV0cmljRm4gfCBzdHJpbmd9O1xuICAgICAgZm9yIChjb25zdCBvdXRwdXROYW1lIG9mIG91dHB1dE5hbWVzKSB7XG4gICAgICAgIGlmICh0eXBlb2YgbG9zc2VzW291dHB1dE5hbWVdID09PSAnc3RyaW5nJykge1xuICAgICAgICAgIGxvc3NOYW1lc1tvdXRwdXROYW1lXSA9XG4gICAgICAgICAgICAgIHRvU25ha2VDYXNlKGxvc3Nlc1tvdXRwdXROYW1lXSBhcyBzdHJpbmcpIGFzIExvc3NJZGVudGlmaWVyO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcignU2VyaWFsaXphdGlvbiBvZiBub24tc3RyaW5nIGxvc3MgaXMgbm90IHN1cHBvcnRlZC4nKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbG9zc05hbWVzO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXRNZXRyaWNJZGVudGlmaWVycygpOiBNZXRyaWNzSWRlbnRpZmllcltdfFxuICAgICAge1trZXk6IHN0cmluZ106IE1ldHJpY3NJZGVudGlmaWVyfSB7XG4gICAgaWYgKHR5cGVvZiB0aGlzLm1ldHJpY3MgPT09ICdzdHJpbmcnIHx8XG4gICAgICAgIHR5cGVvZiB0aGlzLm1ldHJpY3MgPT09ICdmdW5jdGlvbicpIHtcbiAgICAgIHJldHVybiBbdG9TbmFrZUNhc2UoTWV0cmljcy5nZXRMb3NzT3JNZXRyaWNOYW1lKHRoaXMubWV0cmljcykpXTtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5tZXRyaWNzKSkge1xuICAgICAgcmV0dXJuIHRoaXMubWV0cmljcy5tYXAoXG4gICAgICAgICAgbWV0cmljID0+IHRvU25ha2VDYXNlKE1ldHJpY3MuZ2V0TG9zc09yTWV0cmljTmFtZShtZXRyaWMpKSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IG1ldHJpY3NJZGVudGlmaWVyczoge1trZXk6IHN0cmluZ106IE1ldHJpY3NJZGVudGlmaWVyfSA9IHt9O1xuICAgICAgZm9yIChjb25zdCBrZXkgaW4gdGhpcy5tZXRyaWNzKSB7XG4gICAgICAgIG1ldHJpY3NJZGVudGlmaWVyc1trZXldID1cbiAgICAgICAgICAgIHRvU25ha2VDYXNlKE1ldHJpY3MuZ2V0TG9zc09yTWV0cmljTmFtZSh0aGlzLm1ldHJpY3Nba2V5XSkpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIG1ldHJpY3NJZGVudGlmaWVycztcbiAgICB9XG4gIH1cblxuICBwcm90ZWN0ZWQgZ2V0VHJhaW5pbmdDb25maWcoKTogVHJhaW5pbmdDb25maWcge1xuICAgIHJldHVybiB7XG4gICAgICBsb3NzOiB0aGlzLmdldExvc3NJZGVudGlmaWVycygpLFxuICAgICAgbWV0cmljczogdGhpcy5nZXRNZXRyaWNJZGVudGlmaWVycygpLFxuICAgICAgb3B0aW1pemVyX2NvbmZpZzoge1xuICAgICAgICBjbGFzc19uYW1lOiB0aGlzLm9wdGltaXplci5nZXRDbGFzc05hbWUoKSxcbiAgICAgICAgY29uZmlnOiB0aGlzLm9wdGltaXplci5nZXRDb25maWcoKVxuICAgICAgfSBhcyBPcHRpbWl6ZXJTZXJpYWxpemF0aW9uXG4gICAgfTtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgd2VpZ2h0X21ldHJpY3Mgd2hlbiB0aGV5IGFyZSBzdXBwb3J0ZWQuXG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIHNhbXBsZV93ZWlnaHRfbW9kZSB3aGVuIGl0J3Mgc3VwcG9ydGVkLlxuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBsb3NzX3dlaWdodHMgd2hlbiBpdCdzIHN1cHBvcnRlZC5cbiAgfVxuXG4gIGxvYWRUcmFpbmluZ0NvbmZpZyh0cmFpbmluZ0NvbmZpZzogVHJhaW5pbmdDb25maWcpIHtcbiAgICBpZiAodHJhaW5pbmdDb25maWcud2VpZ2h0ZWRfbWV0cmljcyAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0xvYWRpbmcgd2VpZ2h0X21ldHJpY3MgaXMgbm90IHN1cHBvcnRlZCB5ZXQuJyk7XG4gICAgfVxuICAgIGlmICh0cmFpbmluZ0NvbmZpZy5sb3NzX3dlaWdodHMgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdMb2FkaW5nIGxvc3Nfd2VpZ2h0cyBpcyBub3Qgc3VwcG9ydGVkIHlldC4nKTtcbiAgICB9XG4gICAgaWYgKHRyYWluaW5nQ29uZmlnLnNhbXBsZV93ZWlnaHRfbW9kZSAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0xvYWRpbmcgc2FtcGxlX3dlaWdodF9tb2RlIGlzIG5vdCBzdXBwb3J0ZWQgeWV0LicpO1xuICAgIH1cblxuICAgIGNvbnN0IHRzQ29uZmlnID0gY29udmVydFB5dGhvbmljVG9Ucyh0cmFpbmluZ0NvbmZpZy5vcHRpbWl6ZXJfY29uZmlnKSBhc1xuICAgICAgICBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Q7XG4gICAgY29uc3Qgb3B0aW1pemVyID0gZGVzZXJpYWxpemUodHNDb25maWcpIGFzIE9wdGltaXplcjtcblxuICAgIGxldCBsb3NzO1xuICAgIGlmICh0eXBlb2YgdHJhaW5pbmdDb25maWcubG9zcyA9PT0gJ3N0cmluZycpIHtcbiAgICAgIGxvc3MgPSB0b0NhbWVsQ2FzZSh0cmFpbmluZ0NvbmZpZy5sb3NzKTtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkodHJhaW5pbmdDb25maWcubG9zcykpIHtcbiAgICAgIGxvc3MgPSB0cmFpbmluZ0NvbmZpZy5sb3NzLm1hcChsb3NzRW50cnkgPT4gdG9DYW1lbENhc2UobG9zc0VudHJ5KSk7XG4gICAgfSBlbHNlIGlmICh0cmFpbmluZ0NvbmZpZy5sb3NzICE9IG51bGwpIHtcbiAgICAgIGxvc3MgPSB7fSBhcyB7W291dHB1dE5hbWU6IHN0cmluZ106IExvc3NJZGVudGlmaWVyfTtcbiAgICAgIGZvciAoY29uc3Qga2V5IGluIHRyYWluaW5nQ29uZmlnLmxvc3MpIHtcbiAgICAgICAgbG9zc1trZXldID0gdG9DYW1lbENhc2UodHJhaW5pbmdDb25maWcubG9zc1trZXldKSBhcyBMb3NzSWRlbnRpZmllcjtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBsZXQgbWV0cmljcztcbiAgICBpZiAoQXJyYXkuaXNBcnJheSh0cmFpbmluZ0NvbmZpZy5tZXRyaWNzKSkge1xuICAgICAgbWV0cmljcyA9IHRyYWluaW5nQ29uZmlnLm1ldHJpY3MubWFwKG1ldHJpYyA9PiB0b0NhbWVsQ2FzZShtZXRyaWMpKTtcbiAgICB9IGVsc2UgaWYgKHRyYWluaW5nQ29uZmlnLm1ldHJpY3MgIT0gbnVsbCkge1xuICAgICAgbWV0cmljcyA9IHt9IGFzIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTWV0cmljc0lkZW50aWZpZXJ9O1xuICAgICAgZm9yIChjb25zdCBrZXkgaW4gdHJhaW5pbmdDb25maWcubWV0cmljcykge1xuICAgICAgICBtZXRyaWNzW2tleV0gPSB0b0NhbWVsQ2FzZSh0cmFpbmluZ0NvbmZpZy5tZXRyaWNzW2tleV0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIHRoaXMuY29tcGlsZSh7bG9zcywgbWV0cmljcywgb3B0aW1pemVyfSk7XG4gIH1cblxuICAvKipcbiAgICogU2F2ZSB0aGUgY29uZmlndXJhdGlvbiBhbmQvb3Igd2VpZ2h0cyBvZiB0aGUgTGF5ZXJzTW9kZWwuXG4gICAqXG4gICAqIEFuIGBJT0hhbmRsZXJgIGlzIGFuIG9iamVjdCB0aGF0IGhhcyBhIGBzYXZlYCBtZXRob2Qgb2YgdGhlIHByb3BlclxuICAgKiBzaWduYXR1cmUgZGVmaW5lZC4gVGhlIGBzYXZlYCBtZXRob2QgbWFuYWdlcyB0aGUgc3RvcmluZyBvclxuICAgKiB0cmFuc21pc3Npb24gb2Ygc2VyaWFsaXplZCBkYXRhIChcImFydGlmYWN0c1wiKSB0aGF0IHJlcHJlc2VudCB0aGVcbiAgICogbW9kZWwncyB0b3BvbG9neSBhbmQgd2VpZ2h0cyBvbnRvIG9yIHZpYSBhIHNwZWNpZmljIG1lZGl1bSwgc3VjaCBhc1xuICAgKiBmaWxlIGRvd25sb2FkcywgbG9jYWwgc3RvcmFnZSwgSW5kZXhlZERCIGluIHRoZSB3ZWIgYnJvd3NlciBhbmQgSFRUUFxuICAgKiByZXF1ZXN0cyB0byBhIHNlcnZlci4gVGVuc29yRmxvdy5qcyBwcm92aWRlcyBgSU9IYW5kbGVyYFxuICAgKiBpbXBsZW1lbnRhdGlvbnMgZm9yIGEgbnVtYmVyIG9mIGZyZXF1ZW50bHkgdXNlZCBzYXZpbmcgbWVkaXVtcywgc3VjaCBhc1xuICAgKiBgdGYuaW8uYnJvd3NlckRvd25sb2Fkc2AgYW5kIGB0Zi5pby5icm93c2VyTG9jYWxTdG9yYWdlYC4gU2VlIGB0Zi5pb2BcbiAgICogZm9yIG1vcmUgZGV0YWlscy5cbiAgICpcbiAgICogVGhpcyBtZXRob2QgYWxzbyBhbGxvd3MgeW91IHRvIHJlZmVyIHRvIGNlcnRhaW4gdHlwZXMgb2YgYElPSGFuZGxlcmBzXG4gICAqIGFzIFVSTC1saWtlIHN0cmluZyBzaG9ydGN1dHMsIHN1Y2ggYXMgJ2xvY2Fsc3RvcmFnZTovLycgYW5kXG4gICAqICdpbmRleGVkZGI6Ly8nLlxuICAgKlxuICAgKiBFeGFtcGxlIDE6IFNhdmUgYG1vZGVsYCdzIHRvcG9sb2d5IGFuZCB3ZWlnaHRzIHRvIGJyb3dzZXIgW2xvY2FsXG4gICAqIHN0b3JhZ2VdKGh0dHBzOi8vZGV2ZWxvcGVyLm1vemlsbGEub3JnL2VuLVVTL2RvY3MvV2ViL0FQSS9XaW5kb3cvbG9jYWxTdG9yYWdlKTtcbiAgICogdGhlbiBsb2FkIGl0IGJhY2suXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbChcbiAgICogICAgIHtsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbM119KV19KTtcbiAgICogY29uc29sZS5sb2coJ1ByZWRpY3Rpb24gZnJvbSBvcmlnaW5hbCBtb2RlbDonKTtcbiAgICogbW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gICAqXG4gICAqIGNvbnN0IHNhdmVSZXN1bHRzID0gYXdhaXQgbW9kZWwuc2F2ZSgnbG9jYWxzdG9yYWdlOi8vbXktbW9kZWwtMScpO1xuICAgKlxuICAgKiBjb25zdCBsb2FkZWRNb2RlbCA9IGF3YWl0IHRmLmxvYWRMYXllcnNNb2RlbCgnbG9jYWxzdG9yYWdlOi8vbXktbW9kZWwtMScpO1xuICAgKiBjb25zb2xlLmxvZygnUHJlZGljdGlvbiBmcm9tIGxvYWRlZCBtb2RlbDonKTtcbiAgICogbG9hZGVkTW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBFeGFtcGxlIDIuIFNhdmluZyBgbW9kZWxgJ3MgdG9wb2xvZ3kgYW5kIHdlaWdodHMgdG8gYnJvd3NlclxuICAgKiBbSW5kZXhlZERCXShodHRwczovL2RldmVsb3Blci5tb3ppbGxhLm9yZy9lbi1VUy9kb2NzL1dlYi9BUEkvSW5kZXhlZERCX0FQSSk7XG4gICAqIHRoZW4gbG9hZCBpdCBiYWNrLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoXG4gICAqICAgICB7bGF5ZXJzOiBbdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgaW5wdXRTaGFwZTogWzNdfSldfSk7XG4gICAqIGNvbnNvbGUubG9nKCdQcmVkaWN0aW9uIGZyb20gb3JpZ2luYWwgbW9kZWw6Jyk7XG4gICAqIG1vZGVsLnByZWRpY3QodGYub25lcyhbMSwgM10pKS5wcmludCgpO1xuICAgKlxuICAgKiBjb25zdCBzYXZlUmVzdWx0cyA9IGF3YWl0IG1vZGVsLnNhdmUoJ2luZGV4ZWRkYjovL215LW1vZGVsLTEnKTtcbiAgICpcbiAgICogY29uc3QgbG9hZGVkTW9kZWwgPSBhd2FpdCB0Zi5sb2FkTGF5ZXJzTW9kZWwoJ2luZGV4ZWRkYjovL215LW1vZGVsLTEnKTtcbiAgICogY29uc29sZS5sb2coJ1ByZWRpY3Rpb24gZnJvbSBsb2FkZWQgbW9kZWw6Jyk7XG4gICAqIGxvYWRlZE1vZGVsLnByZWRpY3QodGYub25lcyhbMSwgM10pKS5wcmludCgpO1xuICAgKiBgYGBcbiAgICpcbiAgICogRXhhbXBsZSAzLiBTYXZpbmcgYG1vZGVsYCdzIHRvcG9sb2d5IGFuZCB3ZWlnaHRzIGFzIHR3byBmaWxlc1xuICAgKiAoYG15LW1vZGVsLTEuanNvbmAgYW5kIGBteS1tb2RlbC0xLndlaWdodHMuYmluYCkgZG93bmxvYWRlZCBmcm9tXG4gICAqIGJyb3dzZXIuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbChcbiAgICogICAgIHtsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbM119KV19KTtcbiAgICogY29uc3Qgc2F2ZVJlc3VsdHMgPSBhd2FpdCBtb2RlbC5zYXZlKCdkb3dubG9hZHM6Ly9teS1tb2RlbC0xJyk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBFeGFtcGxlIDQuIFNlbmQgIGBtb2RlbGAncyB0b3BvbG9neSBhbmQgd2VpZ2h0cyB0byBhbiBIVFRQIHNlcnZlci5cbiAgICogU2VlIHRoZSBkb2N1bWVudGF0aW9uIG9mIGB0Zi5pby5odHRwYCBmb3IgbW9yZSBkZXRhaWxzXG4gICAqIGluY2x1ZGluZyBzcGVjaWZ5aW5nIHJlcXVlc3QgcGFyYW1ldGVycyBhbmQgaW1wbGVtZW50YXRpb24gb2YgdGhlXG4gICAqIHNlcnZlci5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKFxuICAgKiAgICAge2xheWVyczogW3RmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFszXX0pXX0pO1xuICAgKiBjb25zdCBzYXZlUmVzdWx0cyA9IGF3YWl0IG1vZGVsLnNhdmUoJ2h0dHA6Ly9teS1zZXJ2ZXIvbW9kZWwvdXBsb2FkJyk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gaGFuZGxlck9yVVJMIEFuIGluc3RhbmNlIG9mIGBJT0hhbmRsZXJgIG9yIGEgVVJMLWxpa2UsXG4gICAqIHNjaGVtZS1iYXNlZCBzdHJpbmcgc2hvcnRjdXQgZm9yIGBJT0hhbmRsZXJgLlxuICAgKiBAcGFyYW0gY29uZmlnIE9wdGlvbnMgZm9yIHNhdmluZyB0aGUgbW9kZWwuXG4gICAqIEByZXR1cm5zIEEgYFByb21pc2VgIG9mIGBTYXZlUmVzdWx0YCwgd2hpY2ggc3VtbWFyaXplcyB0aGUgcmVzdWx0IG9mXG4gICAqIHRoZSBzYXZpbmcsIHN1Y2ggYXMgYnl0ZSBzaXplcyBvZiB0aGUgc2F2ZWQgYXJ0aWZhY3RzIGZvciB0aGUgbW9kZWwnc1xuICAgKiAgIHRvcG9sb2d5IGFuZCB3ZWlnaHQgdmFsdWVzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnLCBpZ25vcmVDSTogdHJ1ZX1cbiAgICovXG4gIGFzeW5jIHNhdmUoaGFuZGxlck9yVVJMOiBpby5JT0hhbmRsZXJ8c3RyaW5nLCBjb25maWc/OiBpby5TYXZlQ29uZmlnKTpcbiAgICAgIFByb21pc2U8aW8uU2F2ZVJlc3VsdD4ge1xuICAgIGlmICh0eXBlb2YgaGFuZGxlck9yVVJMID09PSAnc3RyaW5nJykge1xuICAgICAgY29uc3QgaGFuZGxlcnMgPSBpby5nZXRTYXZlSGFuZGxlcnMoaGFuZGxlck9yVVJMKTtcbiAgICAgIGlmIChoYW5kbGVycy5sZW5ndGggPT09IDApIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgQ2Fubm90IGZpbmQgYW55IHNhdmUgaGFuZGxlcnMgZm9yIFVSTCAnJHtoYW5kbGVyT3JVUkx9J2ApO1xuICAgICAgfSBlbHNlIGlmIChoYW5kbGVycy5sZW5ndGggPiAxKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYEZvdW5kIG1vcmUgdGhhbiBvbmUgKCR7aGFuZGxlcnMubGVuZ3RofSkgc2F2ZSBoYW5kbGVycyBmb3IgYCArXG4gICAgICAgICAgICBgVVJMICcke2hhbmRsZXJPclVSTH0nYCk7XG4gICAgICB9XG4gICAgICBoYW5kbGVyT3JVUkwgPSBoYW5kbGVyc1swXTtcbiAgICB9XG4gICAgaWYgKGhhbmRsZXJPclVSTC5zYXZlID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdMYXllcnNNb2RlbC5zYXZlKCkgY2Fubm90IHByb2NlZWQgYmVjYXVzZSB0aGUgSU9IYW5kbGVyICcgK1xuICAgICAgICAgICdwcm92aWRlZCBkb2VzIG5vdCBoYXZlIHRoZSBgc2F2ZWAgYXR0cmlidXRlIGRlZmluZWQuJyk7XG4gICAgfVxuXG4gICAgY29uc3Qgd2VpZ2h0RGF0YUFuZFNwZWNzID1cbiAgICAgICAgYXdhaXQgaW8uZW5jb2RlV2VpZ2h0cyh0aGlzLmdldE5hbWVkV2VpZ2h0cyhjb25maWcpKTtcblxuICAgIGNvbnN0IHJldHVyblN0cmluZyA9IGZhbHNlO1xuICAgIGNvbnN0IHVudXNlZEFyZzoge30gPSBudWxsO1xuICAgIGNvbnN0IG1vZGVsQ29uZmlnID0gdGhpcy50b0pTT04odW51c2VkQXJnLCByZXR1cm5TdHJpbmcpO1xuICAgIGNvbnN0IG1vZGVsQXJ0aWZhY3RzOiBpby5Nb2RlbEFydGlmYWN0cyA9IHtcbiAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsQ29uZmlnLFxuICAgICAgZm9ybWF0OiBMQVlFUlNfTU9ERUxfRk9STUFUX05BTUUsXG4gICAgICBnZW5lcmF0ZWRCeTogYFRlbnNvckZsb3cuanMgdGZqcy1sYXllcnMgdiR7dmVyc2lvbn1gLFxuICAgICAgY29udmVydGVkQnk6IG51bGwsXG4gICAgfTtcblxuICAgIGNvbnN0IGluY2x1ZGVPcHRpbWl6ZXIgPSBjb25maWcgPT0gbnVsbCA/IGZhbHNlIDogY29uZmlnLmluY2x1ZGVPcHRpbWl6ZXI7XG4gICAgaWYgKGluY2x1ZGVPcHRpbWl6ZXIgJiYgdGhpcy5vcHRpbWl6ZXIgIT0gbnVsbCkge1xuICAgICAgbW9kZWxBcnRpZmFjdHMudHJhaW5pbmdDb25maWcgPSB0aGlzLmdldFRyYWluaW5nQ29uZmlnKCk7XG4gICAgICBjb25zdCB3ZWlnaHRUeXBlID0gJ29wdGltaXplcic7XG4gICAgICBjb25zdCB7ZGF0YTogb3B0aW1pemVyV2VpZ2h0RGF0YSwgc3BlY3M6IG9wdGltaXplcldlaWdodFNwZWNzfSA9XG4gICAgICAgICAgYXdhaXQgaW8uZW5jb2RlV2VpZ2h0cyhhd2FpdCB0aGlzLm9wdGltaXplci5nZXRXZWlnaHRzKCksIHdlaWdodFR5cGUpO1xuICAgICAgd2VpZ2h0RGF0YUFuZFNwZWNzLnNwZWNzLnB1c2goLi4ub3B0aW1pemVyV2VpZ2h0U3BlY3MpO1xuICAgICAgd2VpZ2h0RGF0YUFuZFNwZWNzLmRhdGEgPSBpby5jb25jYXRlbmF0ZUFycmF5QnVmZmVycyhcbiAgICAgICAgICBbd2VpZ2h0RGF0YUFuZFNwZWNzLmRhdGEsIG9wdGltaXplcldlaWdodERhdGFdKTtcbiAgICB9XG5cbiAgICBpZiAodGhpcy51c2VyRGVmaW5lZE1ldGFkYXRhICE9IG51bGwpIHtcbiAgICAgIC8vIENoZWNrIHNlcmlhbGl6ZWQgc2l6ZSBvZiB1c2VyLWRlZmluZWQgbWV0YWRhdGEuXG4gICAgICBjb25zdCBjaGVja1NpemUgPSB0cnVlO1xuICAgICAgY2hlY2tVc2VyRGVmaW5lZE1ldGFkYXRhKHRoaXMudXNlckRlZmluZWRNZXRhZGF0YSwgdGhpcy5uYW1lLCBjaGVja1NpemUpO1xuICAgICAgbW9kZWxBcnRpZmFjdHMudXNlckRlZmluZWRNZXRhZGF0YSA9IHRoaXMudXNlckRlZmluZWRNZXRhZGF0YTtcbiAgICB9XG5cbiAgICBtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhID0gd2VpZ2h0RGF0YUFuZFNwZWNzLmRhdGE7XG4gICAgbW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MgPSB3ZWlnaHREYXRhQW5kU3BlY3Muc3BlY3M7XG4gICAgcmV0dXJuIGhhbmRsZXJPclVSTC5zYXZlKG1vZGVsQXJ0aWZhY3RzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXQgdXNlci1kZWZpbmVkIG1ldGFkYXRhLlxuICAgKlxuICAgKiBUaGUgc2V0IG1ldGFkYXRhIHdpbGwgYmUgc2VyaWFsaXplZCB0b2dldGhlciB3aXRoIHRoZSB0b3BvbG9neVxuICAgKiBhbmQgd2VpZ2h0cyBvZiB0aGUgbW9kZWwgZHVyaW5nIGBzYXZlKClgIGNhbGxzLlxuICAgKlxuICAgKiBAcGFyYW0gc2V0VXNlckRlZmluZWRNZXRhZGF0YVxuICAgKi9cbiAgc2V0VXNlckRlZmluZWRNZXRhZGF0YSh1c2VyRGVmaW5lZE1ldGFkYXRhOiB7fSk6IHZvaWQge1xuICAgIGNoZWNrVXNlckRlZmluZWRNZXRhZGF0YSh1c2VyRGVmaW5lZE1ldGFkYXRhLCB0aGlzLm5hbWUpO1xuICAgIHRoaXMudXNlckRlZmluZWRNZXRhZGF0YSA9IHVzZXJEZWZpbmVkTWV0YWRhdGE7XG4gIH1cblxuICAvKipcbiAgICogR2V0IHVzZXItZGVmaW5lZCBtZXRhZGF0YS5cbiAgICpcbiAgICogVGhlIG1ldGFkYXRhIGlzIHN1cHBsaWVkIHZpYSBvbmUgb2YgdGhlIHR3byByb3V0ZXM6XG4gICAqICAgMS4gQnkgY2FsbGluZyBgc2V0VXNlckRlZmluZWRNZXRhZGF0YSgpYC5cbiAgICogICAyLiBMb2FkZWQgZHVyaW5nIG1vZGVsIGxvYWRpbmcgKGlmIHRoZSBtb2RlbCBpcyBjb25zdHJ1Y3RlZFxuICAgKiAgICAgIHZpYSBgdGYubG9hZExheWVyc01vZGVsKClgLilcbiAgICpcbiAgICogSWYgbm8gdXNlci1kZWZpbmVkIG1ldGFkYXRhIGlzIGF2YWlsYWJsZSBmcm9tIGVpdGhlciBvZiB0aGVcbiAgICogdHdvIHJvdXRlcywgdGhpcyBmdW5jdGlvbiB3aWxsIHJldHVybiBgdW5kZWZpbmVkYC5cbiAgICovXG4gIGdldFVzZXJEZWZpbmVkTWV0YWRhdGEoKToge30ge1xuICAgIHJldHVybiB0aGlzLnVzZXJEZWZpbmVkTWV0YWRhdGE7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhMYXllcnNNb2RlbCk7XG5cbi8qKlxuICogQSBgdGYuRnVuY3Rpb25hbGAgaXMgYW4gYWxpYXMgdG8gYHRmLkxheWVyc01vZGVsYC5cbiAqXG4gKiBTZWUgYWxzbzpcbiAqICAgYHRmLkxheWVyc01vZGVsYCwgYHRmLlNlcXVlbnRpYWxgLCBgdGYubG9hZExheWVyc01vZGVsYC5cbiAqL1xuLyoqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9ICovXG5leHBvcnQgY2xhc3MgRnVuY3Rpb25hbCBleHRlbmRzIExheWVyc01vZGVsIHtcbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdGdW5jdGlvbmFsJztcbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhGdW5jdGlvbmFsKTtcbiJdfQ==
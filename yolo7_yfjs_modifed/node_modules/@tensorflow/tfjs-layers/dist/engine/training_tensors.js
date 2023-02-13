/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * Interfaces and methods for training models using tf.Tensor objects.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { Tensor, tensor1d, util } from '@tensorflow/tfjs-core';
import { expandDims, gather, sliceAlongFirstAxis } from '../backend/tfjs_backend';
import { configureCallbacks, standardizeCallbacks } from '../base_callbacks';
import { NotImplementedError, ValueError } from '../errors';
import { disposeTensorsInLogs } from '../logs';
import { range } from '../utils/math_utils';
export function checkBatchSize(batchSize) {
    tfc.util.assert(batchSize > 0 && Number.isInteger(batchSize), () => `batchSize is required to be a positive integer, but got ${batchSize}`);
}
/**
 * Slice a Tensor or an Array of Tensors, by start and stop indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArraysByIndices()` together.
 *
 * @param arrays: the input.
 * @param start: the starting index (inclusive).
 * @param stop: the stopping index (exclusive).
 * @returns The result of the slicing. If `arrays` is an `Array` of
 *   `tf.Tensor`s, the slicing will be applied to all elements of the `Array`
 *   in the same way.
 */
export function sliceArrays(arrays, start, stop) {
    if (arrays == null) {
        return [null];
    }
    else if (Array.isArray(arrays)) {
        return arrays.map(array => sliceAlongFirstAxis(array, start, stop - start));
    }
    else { // Tensor.
        return sliceAlongFirstAxis(arrays, start, stop - start);
    }
}
/**
 * Slice a Tensor or an Array of Tensors, by random-order indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArrays()` together.
 *
 * @param arrays The input `tf.Tensor` or `Array` of `tf.Tensor`s to slice.
 *   If an `Array` of `tf.Tensor`s, all `tf.Tensor`s will be sliced in the
 *   same fashion.
 * @param indices The indices to use for slicing along the first (batch)
 *   dimension.
 * @returns Result(s) of the slicing.
 */
export function sliceArraysByIndices(arrays, indices) {
    return tfc.tidy(() => {
        if (arrays == null) {
            return null;
        }
        else if (Array.isArray(arrays)) {
            return arrays.map(array => sliceArraysByIndices(array, indices));
        }
        else {
            // TODO(cais): indices should be a pre-constructed Tensor1D to avoid
            //   tensor1d() calls.
            return gather(arrays, indices.dtype === 'int32' ? indices : tfc.cast(indices, 'int32'));
        }
    });
}
/**
 * Returns a list of batch indices (tuples of indices).
 * @param size: Integer, total size of the data to slice into batches.
 * @param batchSize: Integer, batch size.
 * @returns An Array of [batchStart, batchEnd] tuples. batchStart is
 *   inclusive; batchEnd is exclusive. I.e., each batch consists of indices x
 *   that satisfy batchStart <= x < batchEnd.
 */
export function makeBatches(size, batchSize) {
    const output = [];
    let batchStart = 0;
    let batchEnd = null;
    while (batchStart < size) {
        batchEnd = batchStart + batchSize;
        if (batchEnd >= size) {
            batchEnd = size;
        }
        output.push([batchStart, batchEnd]);
        batchStart = batchEnd;
    }
    return output;
}
/**
 * Abstract fit function for `f(ins)`.
 * @param f A Function returning a list of tensors. For training, this
 *   function is expected to perform the updates to the variables.
 * @param ins List of tensors to be fed to `f`.
 * @param outLabels List of strings, display names of the outputs of `f`.
 * @param batchSize Integer batch size or `== null` if unknown. Default : 32.
 * @param epochs Number of times to iterate over the data. Default : 1.
 * @param verbose Verbosity mode: 0, 1, or 2. Default: 1.
 * @param callbacks List of callbacks to be called during training.
 * @param valF Function to call for validation.
 * @param valIns List of tensors to be fed to `valF`.
 * @param shuffle Whether to shuffle the data at the beginning of every
 * epoch. Default : true.
 * @param callbackMetrics List of strings, the display names of the metrics
 *   passed to the callbacks. They should be the concatenation of the
 *   display names of the outputs of `f` and the list of display names
 *   of the outputs of `valF`.
 * @param initialEpoch Epoch at which to start training (useful for
 *   resuming a previous training run). Default : 0.
 * @param stepsPerEpoch Total number of steps (batches on samples) before
 *   declaring one epoch finished and starting the next epoch. Ignored with
 *   the default value of `undefined` or `null`.
 * @param validationSteps Number of steps to run validation for (only if
 *   doing validation from data tensors). Not applicable for tfjs-layers.
 * @returns A `History` object.
 */
async function fitLoop(
// Type `model` as `any` here to avoid circular dependency w/ training.ts.
// tslint:disable-next-line:no-any
model, f, ins, outLabels, batchSize, epochs, verbose, callbacks, valF, valIns, shuffle, callbackMetrics, initialEpoch, stepsPerEpoch, validationSteps) {
    if (batchSize == null) {
        batchSize = 32;
    }
    if (epochs == null) {
        epochs = 1;
    }
    if (shuffle == null) {
        shuffle = true;
    }
    if (initialEpoch == null) {
        initialEpoch = 0;
    }
    // TODO(cais): Change const to let below when implementing validation.
    let doValidation = false;
    if (valF != null && valIns != null) {
        doValidation = true;
        // TODO(cais): verbose message.
    }
    if (validationSteps != null) {
        doValidation = true;
        if (stepsPerEpoch == null) {
            throw new ValueError('Can only use `validationSteps` when doing step-wise training, ' +
                'i.e., `stepsPerEpoch` must be set.');
        }
    }
    const numTrainSamples = model.checkNumSamples(ins, batchSize, stepsPerEpoch, 'steps_per_epoch');
    let indexArray;
    if (numTrainSamples != null) {
        indexArray = range(0, numTrainSamples);
    }
    if (verbose == null) {
        verbose = 1;
    }
    const { callbackList, history } = configureCallbacks(callbacks, verbose, epochs, initialEpoch, numTrainSamples, stepsPerEpoch, batchSize, doValidation, callbackMetrics);
    callbackList.setModel(model);
    model.history = history;
    await callbackList.onTrainBegin();
    model.stopTraining_ = false;
    // TODO(cais): Take care of callbacks.validation_data as in PyKeras.
    // TODO(cais): Pre-convert feeds for performance as in PyKeras.
    for (let epoch = initialEpoch; epoch < epochs; ++epoch) {
        await callbackList.onEpochBegin(epoch);
        const epochLogs = {};
        if (stepsPerEpoch != null) {
            throw new NotImplementedError('stepsPerEpoch mode is not implemented yet.');
        }
        else {
            if (shuffle === 'batch') {
                throw new NotImplementedError('batch shuffling is not implemneted yet');
            }
            else if (shuffle) {
                util.shuffle(indexArray);
            }
            // Convert the potentially shuffled indices to Tensor1D, to avoid the
            // cost of repeated creation of Array1Ds later on.
            const epochIndexArray1D = tensor1d(indexArray);
            const batches = makeBatches(numTrainSamples, batchSize);
            for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
                const batchLogs = {};
                await callbackList.onBatchBegin(batchIndex, batchLogs);
                tfc.tidy(() => {
                    const batchStart = batches[batchIndex][0];
                    const batchEnd = batches[batchIndex][1];
                    const batchIds = sliceAlongFirstAxis(epochIndexArray1D, batchStart, batchEnd - batchStart);
                    batchLogs['batch'] = batchIndex;
                    batchLogs['size'] = batchEnd - batchStart;
                    // TODO(cais): In ins, train flag can be a number, instead of an
                    //   Tensor? Do we need to handle this in tfjs-layers?
                    const insBatch = sliceArraysByIndices(ins, batchIds);
                    const outs = f(insBatch);
                    for (let i = 0; i < outLabels.length; ++i) {
                        const label = outLabels[i];
                        const out = outs[i];
                        batchLogs[label] = out;
                        tfc.keep(out);
                        // TODO(cais): Use scope() to avoid ownership.
                    }
                    if (batchIndex === batches.length - 1) { // Last batch.
                        if (doValidation) {
                            const valOuts = model.testLoop(valF, valIns, batchSize);
                            // Porting Notes: In tfjs-layers, valOuts is always an Array.
                            for (let i = 0; i < outLabels.length; ++i) {
                                const label = outLabels[i];
                                const out = valOuts[i];
                                tfc.keep(out);
                                // TODO(cais): Use scope() to avoid ownership.
                                epochLogs['val_' + label] = out;
                            }
                        }
                    }
                });
                await callbackList.onBatchEnd(batchIndex, batchLogs);
                disposeTensorsInLogs(batchLogs);
                if (model.stopTraining_) {
                    break;
                }
                // TODO(cais): return outs as list of Tensor.
            }
            epochIndexArray1D.dispose();
        }
        // TODO(cais): Run validation at the end of the epoch.
        await callbackList.onEpochEnd(epoch, epochLogs);
        if (model.stopTraining_) {
            break;
        }
    }
    await callbackList.onTrainEnd();
    await model.history.syncData();
    return model.history;
}
export async function fitTensors(
// Type `model` as `any` here to avoid circular dependency w/ training.ts.
// tslint:disable-next-line:no-any
model, x, y, args = {}) {
    if (model.isTraining) {
        throw new Error('Cannot start training because another fit() call is ongoing.');
    }
    model.isTraining = true;
    let inputs;
    let targets;
    let originalInputs;
    let originalTargets;
    let inputValX;
    let inputValY;
    let valX;
    let valY;
    let sampleWeights;
    try {
        const batchSize = args.batchSize == null ? 32 : args.batchSize;
        checkBatchSize(batchSize);
        // Validate user data.
        // TODO(cais): Support sampleWeight.
        const checkBatchAxis = false;
        const standardizedOuts = await model.standardizeUserData(x, y, args.sampleWeight, args.classWeight, checkBatchAxis, batchSize);
        inputs = standardizedOuts[0];
        targets = standardizedOuts[1];
        sampleWeights = standardizedOuts[2];
        // Prepare validation data.
        let doValidation = false;
        let valIns;
        if (args.validationData != null && args.validationData.length > 0) {
            doValidation = true;
            if (args.validationData.length === 2) {
                // config.validationData consists of valX and valY.
                inputValX = args.validationData[0];
                inputValY = args.validationData[1];
            }
            else if (args.validationData.length === 3) {
                throw new NotImplementedError('validationData including sample weights is not supported yet.');
            }
            else {
                throw new ValueError(`When passing validation data, it must contain 2 (valX, valY) ` +
                    `or 3 (valX, valY, valSampleWeight) items; ` +
                    `${args.validationData} is invalid.`);
            }
            const checkBatchAxis = true;
            const valStandardized = await model.standardizeUserData(inputValX, inputValY, null, /** Unused sample weights. */ null, /** Unused class weights. */ checkBatchAxis, batchSize);
            valX = valStandardized[0];
            valY = valStandardized[1];
            valIns = valX.concat(valY);
            // TODO(cais): Add useLearningPhase data properly.
        }
        else if (args.validationSplit != null && args.validationSplit > 0 &&
            args.validationSplit < 1) {
            doValidation = true;
            // Porting Note: In tfjs-layers, inputs[0] is always a Tensor.
            const splitAt = Math.floor(inputs[0].shape[0] * (1 - args.validationSplit));
            const originalBatchSize = inputs[0].shape[0];
            valX = sliceArrays(inputs, splitAt, originalBatchSize);
            originalInputs = inputs;
            inputs = sliceArrays(inputs, 0, splitAt);
            valY = sliceArrays(targets, splitAt, originalBatchSize);
            originalTargets = targets;
            targets = sliceArrays(targets, 0, splitAt);
            // TODO(cais): Once sampleWeights becomes available, slice it to get
            //   valSampleWeights.
            valIns = valX.concat(valY);
            // TODO(cais): Add useLearningPhase data properly.
        }
        else if (args.validationSteps != null) {
            doValidation = true;
            // TODO(cais): Add useLearningPhase.
        }
        const ins = inputs.concat(targets).concat(sampleWeights);
        model.checkTrainableWeightsConsistency();
        // TODO(cais): Handle use_learning_phase and learning_phase?
        // Porting Note: Here we see a key deviation of tfjs-layers from
        // Keras.
        //  Due to the imperative nature of tfjs-layers' backend (tfjs-core),
        //  we do not construct symbolic computation graphs to embody the
        //  training process. Instead, we define a function that performs the
        //  training action. In PyKeras, the data (inputs and targets) are fed
        //  through graph placeholders. In tfjs-layers, the data are fed as
        //  function arguments. Since the function are defined below in the
        //  scope, we don't have equivalents of PyKeras's
        //  `_make_train_funciton`.
        const trainFunction = model.makeTrainFunction();
        const outLabels = model.getDedupedMetricsNames();
        let valFunction;
        let callbackMetrics;
        if (doValidation) {
            model.makeTestFunction();
            valFunction = model.testFunction;
            callbackMetrics =
                outLabels.slice().concat(outLabels.map(n => 'val_' + n));
        }
        else {
            valFunction = null;
            valIns = [];
            callbackMetrics = outLabels.slice();
        }
        const callbacks = standardizeCallbacks(args.callbacks, args.yieldEvery);
        const out = await fitLoop(model, trainFunction, ins, outLabels, batchSize, args.epochs, args.verbose, callbacks, valFunction, valIns, args.shuffle, callbackMetrics, args.initialEpoch, null, null);
        return out;
    }
    finally {
        model.isTraining = false;
        // Memory clean up.
        disposeNewTensors(inputs, x);
        disposeNewTensors(targets, y);
        disposeNewTensors(originalInputs, x);
        disposeNewTensors(originalTargets, y);
        disposeNewTensors(valX, inputValX);
        disposeNewTensors(valY, inputValY);
        if (sampleWeights != null) {
            tfc.dispose(sampleWeights);
        }
    }
    // TODO(cais): Add value to outLabels.
}
/**
 * Ensure tensors all have a rank of at least 2.
 *
 * If a tensor has a rank of 1, it is dimension-expanded to rank 2.
 * If any tensor has a rank of 0 (i.e., is a scalar), an error will be thrown.
 */
export function ensureTensorsRank2OrHigher(tensors) {
    const outs = [];
    if (tensors instanceof Tensor) {
        tensors = [tensors];
    }
    // Make Tensors at least 2D.
    for (let i = 0; i < tensors.length; ++i) {
        const tensor = tensors[i];
        if (tensor.rank === 1) {
            outs.push(expandDims(tensor, 1));
        }
        else if (tensor.rank === 0) {
            throw new Error('Expected tensor to be at least 1D, but received a 0D tensor ' +
                '(scalar).');
        }
        else {
            outs.push(tensor);
        }
    }
    return outs;
}
/**
 * Compare a set of tensors with a reference (old) set, discard the ones
 * in the new set that are not present in the reference set.
 *
 * This method is used for memory clenaup during calls such as
 * LayersModel.fit().
 *
 * @param tensors New set which may contain Tensors not present in
 *   `refTensors`.
 * @param refTensors Reference Tensor set.
 */
// TODO(cais, kangyizhang): Deduplicate with tfjs-data.
export function disposeNewTensors(tensors, refTensors) {
    if (tensors == null) {
        return;
    }
    const oldTensorIds = [];
    if (refTensors instanceof Tensor) {
        oldTensorIds.push(refTensors.id);
    }
    else if (Array.isArray(refTensors)) {
        refTensors.forEach(t => oldTensorIds.push(t.id));
    }
    else if (refTensors != null) {
        // `oldTensors` is a map from string name to Tensor.
        for (const name in refTensors) {
            const oldTensor = refTensors[name];
            oldTensorIds.push(oldTensor.id);
        }
    }
    const tensorsToDispose = [];
    if (tensors instanceof Tensor) {
        if (oldTensorIds.indexOf(tensors.id) === -1) {
            tensorsToDispose.push(tensors);
        }
    }
    else if (Array.isArray(tensors)) {
        tensors.forEach(t => {
            if (oldTensorIds.indexOf(t.id) === -1) {
                tensorsToDispose.push(t);
            }
        });
    }
    else if (tensors != null) {
        // `oldTensors` is a map from string name to Tensor.
        for (const name in tensors) {
            const tensor = tensors[name];
            if (oldTensorIds.indexOf(tensor.id) === -1) {
                tensorsToDispose.push(tensor);
            }
        }
    }
    tensorsToDispose.forEach(t => {
        if (!t.isDisposed) {
            t.dispose();
        }
    });
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhaW5pbmdfdGVuc29ycy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9lbmdpbmUvdHJhaW5pbmdfdGVuc29ycy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVIOztHQUVHO0FBRUgsT0FBTyxLQUFLLEdBQUcsTUFBTSx1QkFBdUIsQ0FBQztBQUM3QyxPQUFPLEVBQVMsTUFBTSxFQUFZLFFBQVEsRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUUvRSxPQUFPLEVBQUMsVUFBVSxFQUFFLE1BQU0sRUFBRSxtQkFBbUIsRUFBQyxNQUFNLHlCQUF5QixDQUFDO0FBQ2hGLE9BQU8sRUFBZSxrQkFBa0IsRUFBc0Qsb0JBQW9CLEVBQW9CLE1BQU0sbUJBQW1CLENBQUM7QUFDaEssT0FBTyxFQUFDLG1CQUFtQixFQUFFLFVBQVUsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUMxRCxPQUFPLEVBQUMsb0JBQW9CLEVBQWlCLE1BQU0sU0FBUyxDQUFDO0FBQzdELE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQTRJMUMsTUFBTSxVQUFVLGNBQWMsQ0FBQyxTQUFpQjtJQUM5QyxHQUFHLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FDWCxTQUFTLEdBQUcsQ0FBQyxJQUFJLE1BQU0sQ0FBQyxTQUFTLENBQUMsU0FBUyxDQUFDLEVBQzVDLEdBQUcsRUFBRSxDQUFDLDJEQUNGLFNBQVMsRUFBRSxDQUFDLENBQUM7QUFDdkIsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7R0FZRztBQUNILE1BQU0sVUFBVSxXQUFXLENBQ3ZCLE1BQXVCLEVBQUUsS0FBYSxFQUFFLElBQVk7SUFDdEQsSUFBSSxNQUFNLElBQUksSUFBSSxFQUFFO1FBQ2xCLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztLQUNmO1NBQU0sSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLE1BQU0s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RjtLQUNGO0lBRUQsZ0JBQWdCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFO1FBQzNCLElBQUksQ0FBQyxDQUFDLENBQUMsVUFBVSxFQUFFO1lBQ2pCLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztTQUNiO0lBQ0gsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZVxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XG4gKiBodHRwczovL29wZW5zb3VyY2Uub3JnL2xpY2Vuc2VzL01JVC5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyoqXG4gKiBJbnRlcmZhY2VzIGFuZCBtZXRob2RzIGZvciB0cmFpbmluZyBtb2RlbHMgdXNpbmcgdGYuVGVuc29yIG9iamVjdHMuXG4gKi9cblxuaW1wb3J0ICogYXMgdGZjIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge1NjYWxhciwgVGVuc29yLCBUZW5zb3IxRCwgdGVuc29yMWQsIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7ZXhwYW5kRGltcywgZ2F0aGVyLCBzbGljZUFsb25nRmlyc3RBeGlzfSBmcm9tICcuLi9iYWNrZW5kL3RmanNfYmFja2VuZCc7XG5pbXBvcnQge0Jhc2VDYWxsYmFjaywgY29uZmlndXJlQ2FsbGJhY2tzLCBDdXN0b21DYWxsYmFja0FyZ3MsIEhpc3RvcnksIE1vZGVsTG9nZ2luZ1ZlcmJvc2l0eSwgc3RhbmRhcmRpemVDYWxsYmFja3MsIFlpZWxkRXZlcnlPcHRpb25zfSBmcm9tICcuLi9iYXNlX2NhbGxiYWNrcyc7XG5pbXBvcnQge05vdEltcGxlbWVudGVkRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge2Rpc3Bvc2VUZW5zb3JzSW5Mb2dzLCBVbnJlc29sdmVkTG9nc30gZnJvbSAnLi4vbG9ncyc7XG5pbXBvcnQge3JhbmdlfSBmcm9tICcuLi91dGlscy9tYXRoX3V0aWxzJztcbmltcG9ydCB7Q2xhc3NXZWlnaHQsIENsYXNzV2VpZ2h0TWFwfSBmcm9tICcuL3RyYWluaW5nX3V0aWxzJztcblxuLyoqXG4gKiBJbnRlcmZhY2UgY29uZmlndXJhdGlvbiBtb2RlbCB0cmFpbmluZyBiYXNlZCBvbiBkYXRhIGFzIGB0Zi5UZW5zb3Jgcy5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBNb2RlbEZpdEFyZ3Mge1xuICAvKipcbiAgICogTnVtYmVyIG9mIHNhbXBsZXMgcGVyIGdyYWRpZW50IHVwZGF0ZS4gSWYgdW5zcGVjaWZpZWQsIGl0XG4gICAqIHdpbGwgZGVmYXVsdCB0byAzMi5cbiAgICovXG4gIGJhdGNoU2l6ZT86IG51bWJlcjtcblxuICAvKipcbiAgICogSW50ZWdlciBudW1iZXIgb2YgdGltZXMgdG8gaXRlcmF0ZSBvdmVyIHRoZSB0cmFpbmluZyBkYXRhIGFycmF5cy5cbiAgICovXG4gIGVwb2Nocz86IG51bWJlcjtcblxuICAvKipcbiAgICogVmVyYm9zaXR5IGxldmVsLlxuICAgKlxuICAgKiBFeHBlY3RlZCB0byBiZSAwLCAxLCBvciAyLiBEZWZhdWx0OiAxLlxuICAgKlxuICAgKiAwIC0gTm8gcHJpbnRlZCBtZXNzYWdlIGR1cmluZyBmaXQoKSBjYWxsLlxuICAgKiAxIC0gSW4gTm9kZS5qcyAodGZqcy1ub2RlKSwgcHJpbnRzIHRoZSBwcm9ncmVzcyBiYXIsIHRvZ2V0aGVyIHdpdGhcbiAgICogICAgIHJlYWwtdGltZSB1cGRhdGVzIG9mIGxvc3MgYW5kIG1ldHJpYyB2YWx1ZXMgYW5kIHRyYWluaW5nIHNwZWVkLlxuICAgKiAgICAgSW4gdGhlIGJyb3dzZXI6IG5vIGFjdGlvbi4gVGhpcyBpcyB0aGUgZGVmYXVsdC5cbiAgICogMiAtIE5vdCBpbXBsZW1lbnRlZCB5ZXQuXG4gICAqL1xuICB2ZXJib3NlPzogTW9kZWxMb2dnaW5nVmVyYm9zaXR5O1xuXG4gIC8qKlxuICAgKiBMaXN0IG9mIGNhbGxiYWNrcyB0byBiZSBjYWxsZWQgZHVyaW5nIHRyYWluaW5nLlxuICAgKiBDYW4gaGF2ZSBvbmUgb3IgbW9yZSBvZiB0aGUgZm9sbG93aW5nIGNhbGxiYWNrczpcbiAgICogICAtIGBvblRyYWluQmVnaW4obG9ncylgOiBjYWxsZWQgd2hlbiB0cmFpbmluZyBzdGFydHMuXG4gICAqICAgLSBgb25UcmFpbkVuZChsb2dzKWA6IGNhbGxlZCB3aGVuIHRyYWluaW5nIGVuZHMuXG4gICAqICAgLSBgb25FcG9jaEJlZ2luKGVwb2NoLCBsb2dzKWA6IGNhbGxlZCBhdCB0aGUgc3RhcnQgb2YgZXZlcnkgZXBvY2guXG4gICAqICAgLSBgb25FcG9jaEVuZChlcG9jaCwgbG9ncylgOiBjYWxsZWQgYXQgdGhlIGVuZCBvZiBldmVyeSBlcG9jaC5cbiAgICogICAtIGBvbkJhdGNoQmVnaW4oYmF0Y2gsIGxvZ3MpYDogY2FsbGVkIGF0IHRoZSBzdGFydCBvZiBldmVyeSBiYXRjaC5cbiAgICogICAtIGBvbkJhdGNoRW5kKGJhdGNoLCBsb2dzKWA6IGNhbGxlZCBhdCB0aGUgZW5kIG9mIGV2ZXJ5IGJhdGNoLlxuICAgKiAgIC0gYG9uWWllbGQoZXBvY2gsIGJhdGNoLCBsb2dzKWA6IGNhbGxlZCBldmVyeSBgeWllbGRFdmVyeWAgbWlsbGlzZWNvbmRzXG4gICAqICAgICAgd2l0aCB0aGUgY3VycmVudCBlcG9jaCwgYmF0Y2ggYW5kIGxvZ3MuIFRoZSBsb2dzIGFyZSB0aGUgc2FtZVxuICAgKiAgICAgIGFzIGluIGBvbkJhdGNoRW5kKClgLiBOb3RlIHRoYXQgYG9uWWllbGRgIGNhbiBza2lwIGJhdGNoZXMgb3JcbiAgICogICAgICBlcG9jaHMuIFNlZSBhbHNvIGRvY3MgZm9yIGB5aWVsZEV2ZXJ5YCBiZWxvdy5cbiAgICovXG4gIGNhbGxiYWNrcz86IEJhc2VDYWxsYmFja1tdfEN1c3RvbUNhbGxiYWNrQXJnc3xDdXN0b21DYWxsYmFja0FyZ3NbXTtcblxuICAvKipcbiAgICogRmxvYXQgYmV0d2VlbiAwIGFuZCAxOiBmcmFjdGlvbiBvZiB0aGUgdHJhaW5pbmcgZGF0YVxuICAgKiB0byBiZSB1c2VkIGFzIHZhbGlkYXRpb24gZGF0YS4gVGhlIG1vZGVsIHdpbGwgc2V0IGFwYXJ0IHRoaXMgZnJhY3Rpb24gb2ZcbiAgICogdGhlIHRyYWluaW5nIGRhdGEsIHdpbGwgbm90IHRyYWluIG9uIGl0LCBhbmQgd2lsbCBldmFsdWF0ZSB0aGUgbG9zcyBhbmRcbiAgICogYW55IG1vZGVsIG1ldHJpY3Mgb24gdGhpcyBkYXRhIGF0IHRoZSBlbmQgb2YgZWFjaCBlcG9jaC5cbiAgICogVGhlIHZhbGlkYXRpb24gZGF0YSBpcyBzZWxlY3RlZCBmcm9tIHRoZSBsYXN0IHNhbXBsZXMgaW4gdGhlIGB4YCBhbmQgYHlgXG4gICAqIGRhdGEgcHJvdmlkZWQsIGJlZm9yZSBzaHVmZmxpbmcuXG4gICAqL1xuICB2YWxpZGF0aW9uU3BsaXQ/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIERhdGEgb24gd2hpY2ggdG8gZXZhbHVhdGUgdGhlIGxvc3MgYW5kIGFueSBtb2RlbFxuICAgKiBtZXRyaWNzIGF0IHRoZSBlbmQgb2YgZWFjaCBlcG9jaC4gVGhlIG1vZGVsIHdpbGwgbm90IGJlIHRyYWluZWQgb24gdGhpc1xuICAgKiBkYXRhLiBUaGlzIGNvdWxkIGJlIGEgdHVwbGUgW3hWYWwsIHlWYWxdIG9yIGEgdHVwbGUgW3hWYWwsIHlWYWwsXG4gICAqIHZhbFNhbXBsZVdlaWdodHNdLiBUaGUgbW9kZWwgd2lsbCBub3QgYmUgdHJhaW5lZCBvbiB0aGlzIGRhdGEuXG4gICAqIGB2YWxpZGF0aW9uRGF0YWAgd2lsbCBvdmVycmlkZSBgdmFsaWRhdGlvblNwbGl0YC5cbiAgICovXG4gIHZhbGlkYXRpb25EYXRhPzogW1xuICAgIFRlbnNvcnxUZW5zb3JbXSwgVGVuc29yfFRlbnNvcltdXG4gIF18W1RlbnNvciB8IFRlbnNvcltdLCBUZW5zb3J8VGVuc29yW10sIFRlbnNvcnxUZW5zb3JbXV07XG5cbiAgLyoqXG4gICAqIFdoZXRoZXIgdG8gc2h1ZmZsZSB0aGUgdHJhaW5pbmcgZGF0YSBiZWZvcmUgZWFjaCBlcG9jaC4gSGFzXG4gICAqIG5vIGVmZmVjdCB3aGVuIGBzdGVwc1BlckVwb2NoYCBpcyBub3QgYG51bGxgLlxuICAgKi9cbiAgc2h1ZmZsZT86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIE9wdGlvbmFsIG9iamVjdCBtYXBwaW5nIGNsYXNzIGluZGljZXMgKGludGVnZXJzKSB0b1xuICAgKiBhIHdlaWdodCAoZmxvYXQpIHRvIGFwcGx5IHRvIHRoZSBtb2RlbCdzIGxvc3MgZm9yIHRoZSBzYW1wbGVzIGZyb20gdGhpc1xuICAgKiBjbGFzcyBkdXJpbmcgdHJhaW5pbmcuIFRoaXMgY2FuIGJlIHVzZWZ1bCB0byB0ZWxsIHRoZSBtb2RlbCB0byBcInBheSBtb3JlXG4gICAqIGF0dGVudGlvblwiIHRvIHNhbXBsZXMgZnJvbSBhbiB1bmRlci1yZXByZXNlbnRlZCBjbGFzcy5cbiAgICpcbiAgICogSWYgdGhlIG1vZGVsIGhhcyBtdWx0aXBsZSBvdXRwdXRzLCBhIGNsYXNzIHdlaWdodCBjYW4gYmUgc3BlY2lmaWVkIGZvclxuICAgKiBlYWNoIG9mIHRoZSBvdXRwdXRzIGJ5IHNldHRpbmcgdGhpcyBmaWVsZCBhbiBhcnJheSBvZiB3ZWlnaHQgb2JqZWN0XG4gICAqIG9yIGEgb2JqZWN0IHRoYXQgbWFwcyBtb2RlbCBvdXRwdXQgbmFtZXMgKGUuZy4sIGBtb2RlbC5vdXRwdXROYW1lc1swXWApXG4gICAqIHRvIHdlaWdodCBvYmplY3RzLlxuICAgKi9cbiAgY2xhc3NXZWlnaHQ/OiBDbGFzc1dlaWdodHxDbGFzc1dlaWdodFtdfENsYXNzV2VpZ2h0TWFwO1xuXG4gIC8qKlxuICAgKiBPcHRpb25hbCBhcnJheSBvZiB0aGUgc2FtZSBsZW5ndGggYXMgeCwgY29udGFpbmluZ1xuICAgKiB3ZWlnaHRzIHRvIGFwcGx5IHRvIHRoZSBtb2RlbCdzIGxvc3MgZm9yIGVhY2ggc2FtcGxlLiBJbiB0aGUgY2FzZSBvZlxuICAgKiB0ZW1wb3JhbCBkYXRhLCB5b3UgY2FuIHBhc3MgYSAyRCBhcnJheSB3aXRoIHNoYXBlIChzYW1wbGVzLFxuICAgKiBzZXF1ZW5jZUxlbmd0aCksIHRvIGFwcGx5IGEgZGlmZmVyZW50IHdlaWdodCB0byBldmVyeSB0aW1lc3RlcCBvZiBldmVyeVxuICAgKiBzYW1wbGUuIEluIHRoaXMgY2FzZSB5b3Ugc2hvdWxkIG1ha2Ugc3VyZSB0byBzcGVjaWZ5XG4gICAqIHNhbXBsZVdlaWdodE1vZGU9XCJ0ZW1wb3JhbFwiIGluIGNvbXBpbGUoKS5cbiAgICovXG4gIHNhbXBsZVdlaWdodD86IFRlbnNvcjtcblxuICAvKipcbiAgICogRXBvY2ggYXQgd2hpY2ggdG8gc3RhcnQgdHJhaW5pbmcgKHVzZWZ1bCBmb3IgcmVzdW1pbmcgYSBwcmV2aW91cyB0cmFpbmluZ1xuICAgKiBydW4pLiBXaGVuIHRoaXMgaXMgdXNlZCwgYGVwb2Noc2AgaXMgdGhlIGluZGV4IG9mIHRoZSBcImZpbmFsIGVwb2NoXCIuXG4gICAqIFRoZSBtb2RlbCBpcyBub3QgdHJhaW5lZCBmb3IgYSBudW1iZXIgb2YgaXRlcmF0aW9ucyBnaXZlbiBieSBgZXBvY2hzYCxcbiAgICogYnV0IG1lcmVseSB1bnRpbCB0aGUgZXBvY2ggb2YgaW5kZXggYGVwb2Noc2AgaXMgcmVhY2hlZC5cbiAgICovXG4gIGluaXRpYWxFcG9jaD86IG51bWJlcjtcblxuICAvKipcbiAgICogVG90YWwgbnVtYmVyIG9mIHN0ZXBzIChiYXRjaGVzIG9mIHNhbXBsZXMpIGJlZm9yZVxuICAgKiBkZWNsYXJpbmcgb25lIGVwb2NoIGZpbmlzaGVkIGFuZCBzdGFydGluZyB0aGUgbmV4dCBlcG9jaC4gV2hlbiB0cmFpbmluZ1xuICAgKiB3aXRoIElucHV0IFRlbnNvcnMgc3VjaCBhcyBUZW5zb3JGbG93IGRhdGEgdGVuc29ycywgdGhlIGRlZmF1bHQgYG51bGxgIGlzXG4gICAqIGVxdWFsIHRvIHRoZSBudW1iZXIgb2YgdW5pcXVlIHNhbXBsZXMgaW4geW91ciBkYXRhc2V0IGRpdmlkZWQgYnkgdGhlXG4gICAqIGJhdGNoIHNpemUsIG9yIDEgaWYgdGhhdCBjYW5ub3QgYmUgZGV0ZXJtaW5lZC5cbiAgICovXG4gIHN0ZXBzUGVyRXBvY2g/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE9ubHkgcmVsZXZhbnQgaWYgYHN0ZXBzUGVyRXBvY2hgIGlzIHNwZWNpZmllZC4gVG90YWwgbnVtYmVyIG9mIHN0ZXBzXG4gICAqIChiYXRjaGVzIG9mIHNhbXBsZXMpIHRvIHZhbGlkYXRlIGJlZm9yZSBzdG9wcGluZy5cbiAgICovXG4gIHZhbGlkYXRpb25TdGVwcz86IG51bWJlcjtcblxuICAvKipcbiAgICogQ29uZmlndXJlcyB0aGUgZnJlcXVlbmN5IG9mIHlpZWxkaW5nIHRoZSBtYWluIHRocmVhZCB0byBvdGhlciB0YXNrcy5cbiAgICpcbiAgICogSW4gdGhlIGJyb3dzZXIgZW52aXJvbm1lbnQsIHlpZWxkaW5nIHRoZSBtYWluIHRocmVhZCBjYW4gaW1wcm92ZSB0aGVcbiAgICogcmVzcG9uc2l2ZW5lc3Mgb2YgdGhlIHBhZ2UgZHVyaW5nIHRyYWluaW5nLiBJbiB0aGUgTm9kZS5qcyBlbnZpcm9ubWVudCxcbiAgICogaXQgY2FuIGVuc3VyZSB0YXNrcyBxdWV1ZWQgaW4gdGhlIGV2ZW50IGxvb3AgY2FuIGJlIGhhbmRsZWQgaW4gYSB0aW1lbHlcbiAgICogbWFubmVyLlxuICAgKlxuICAgKiBUaGUgdmFsdWUgY2FuIGJlIG9uZSBvZiB0aGUgZm9sbG93aW5nOlxuICAgKiAgIC0gYCdhdXRvJ2A6IFRoZSB5aWVsZGluZyBoYXBwZW5zIGF0IGEgY2VydGFpbiBmcmFtZSByYXRlIChjdXJyZW50bHkgc2V0XG4gICAqICAgICAgICAgICAgICAgYXQgMTI1bXMpLiBUaGlzIGlzIHRoZSBkZWZhdWx0LlxuICAgKiAgIC0gYCdiYXRjaCdgOiB5aWVsZCBldmVyeSBiYXRjaC5cbiAgICogICAtIGAnZXBvY2gnYDogeWllbGQgZXZlcnkgZXBvY2guXG4gICAqICAgLSBhbnkgYG51bWJlcmA6IHlpZWxkIGV2ZXJ5IGBudW1iZXJgIG1pbGxpc2Vjb25kcy5cbiAgICogICAtIGAnbmV2ZXInYDogbmV2ZXIgeWllbGQuICh5aWVsZGluZyBjYW4gc3RpbGwgaGFwcGVuIHRocm91Z2ggYGF3YWl0XG4gICAqICAgICAgbmV4dEZyYW1lKClgIGNhbGxzIGluIGN1c3RvbSBjYWxsYmFja3MuKVxuICAgKi9cbiAgeWllbGRFdmVyeT86IFlpZWxkRXZlcnlPcHRpb25zO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tCYXRjaFNpemUoYmF0Y2hTaXplOiBudW1iZXIpIHtcbiAgdGZjLnV0aWwuYXNzZXJ0KFxuICAgICAgYmF0Y2hTaXplID4gMCAmJiBOdW1iZXIuaXNJbnRlZ2VyKGJhdGNoU2l6ZSksXG4gICAgICAoKSA9PiBgYmF0Y2hTaXplIGlzIHJlcXVpcmVkIHRvIGJlIGEgcG9zaXRpdmUgaW50ZWdlciwgYnV0IGdvdCAke1xuICAgICAgICAgIGJhdGNoU2l6ZX1gKTtcbn1cblxuLyoqXG4gKiBTbGljZSBhIFRlbnNvciBvciBhbiBBcnJheSBvZiBUZW5zb3JzLCBieSBzdGFydCBhbmQgc3RvcCBpbmRpY2VzLlxuICpcbiAqIFBvcnRpbmcgTm90ZTogVGhlIGBfc2xpY2VfYXJyYXlzYCBmdW5jdGlvbiBpbiBQeUtlcmFzIGlzIGNvdmVyZWQgYnkgdGhpc1xuICogICBmdW5jdGlvbiBhbmQgYHNsaWNlQXJyYXlzQnlJbmRpY2VzKClgIHRvZ2V0aGVyLlxuICpcbiAqIEBwYXJhbSBhcnJheXM6IHRoZSBpbnB1dC5cbiAqIEBwYXJhbSBzdGFydDogdGhlIHN0YXJ0aW5nIGluZGV4IChpbmNsdXNpdmUpLlxuICogQHBhcmFtIHN0b3A6IHRoZSBzdG9wcGluZyBpbmRleCAoZXhjbHVzaXZlKS5cbiAqIEByZXR1cm5zIFRoZSByZXN1bHQgb2YgdGhlIHNsaWNpbmcuIElmIGBhcnJheXNgIGlzIGFuIGBBcnJheWAgb2ZcbiAqICAgYHRmLlRlbnNvcmBzLCB0aGUgc2xpY2luZyB3aWxsIGJlIGFwcGxpZWQgdG8gYWxsIGVsZW1lbnRzIG9mIHRoZSBgQXJyYXlgXG4gKiAgIGluIHRoZSBzYW1lIHdheS5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNsaWNlQXJyYXlzKFxuICAgIGFycmF5czogVGVuc29yfFRlbnNvcltdLCBzdGFydDogbnVtYmVyLCBzdG9wOiBudW1iZXIpOiBUZW5zb3J8VGVuc29yW10ge1xuICBpZiAoYXJyYXlzID09IG51bGwpIHtcbiAgICByZXR1cm4gW251bGxdO1xuICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkoYXJyYXlzKSkge1xuICAgIHJldHVybiBhcnJheXMubWFwKGFycmF5ID0+IHNsaWNlQWxvbmdGaXJzdEF4aXMoYXJyYXksIHN0YXJ0LCBzdG9wIC0gc3RhcnQpKTtcbiAgfSBlbHNlIHsgIC8vIFRlbnNvci5cbiAgICByZXR1cm4gc2xpY2VBbG9uZ0ZpcnN0QXhpcyhhcnJheXMsIHN0YXJ0LCBzdG9wIC0gc3RhcnQpO1xuICB9XG59XG5cbi8qKlxuICogU2xpY2UgYSBUZW5zb3Igb3IgYW4gQXJyYXkgb2YgVGVuc29ycywgYnkgcmFuZG9tLW9yZGVyIGluZGljZXMuXG4gKlxuICogUG9ydGluZyBOb3RlOiBUaGUgYF9zbGljZV9hcnJheXNgIGZ1bmN0aW9uIGluIFB5S2VyYXMgaXMgY292ZXJlZCBieSB0aGlzXG4gKiAgIGZ1bmN0aW9uIGFuZCBgc2xpY2VBcnJheXMoKWAgdG9nZXRoZXIuXG4gKlxuICogQHBhcmFtIGFycmF5cyBUaGUgaW5wdXQgYHRmLlRlbnNvcmAgb3IgYEFycmF5YCBvZiBgdGYuVGVuc29yYHMgdG8gc2xpY2UuXG4gKiAgIElmIGFuIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzLCBhbGwgYHRmLlRlbnNvcmBzIHdpbGwgYmUgc2xpY2VkIGluIHRoZVxuICogICBzYW1lIGZhc2hpb24uXG4gKiBAcGFyYW0gaW5kaWNlcyBUaGUgaW5kaWNlcyB0byB1c2UgZm9yIHNsaWNpbmcgYWxvbmcgdGhlIGZpcnN0IChiYXRjaClcbiAqICAgZGltZW5zaW9uLlxuICogQHJldHVybnMgUmVzdWx0KHMpIG9mIHRoZSBzbGljaW5nLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc2xpY2VBcnJheXNCeUluZGljZXMoXG4gICAgYXJyYXlzOiBUZW5zb3J8VGVuc29yW10sIGluZGljZXM6IFRlbnNvcjFEKTogVGVuc29yfFRlbnNvcltdIHtcbiAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICBpZiAoYXJyYXlzID09IG51bGwpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheShhcnJheXMpKSB7XG4gICAgICByZXR1cm4gYXJyYXlzLm1hcChcbiAgICAgICAgICBhcnJheSA9PiAoc2xpY2VBcnJheXNCeUluZGljZXMoYXJyYXksIGluZGljZXMpIGFzIFRlbnNvcikpO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBpbmRpY2VzIHNob3VsZCBiZSBhIHByZS1jb25zdHJ1Y3RlZCBUZW5zb3IxRCB0byBhdm9pZFxuICAgICAgLy8gICB0ZW5zb3IxZCgpIGNhbGxzLlxuICAgICAgcmV0dXJuIGdhdGhlcihcbiAgICAgICAgICBhcnJheXMsXG4gICAgICAgICAgaW5kaWNlcy5kdHlwZSA9PT0gJ2ludDMyJyA/IGluZGljZXMgOiB0ZmMuY2FzdChpbmRpY2VzLCAnaW50MzInKSk7XG4gICAgfVxuICB9KTtcbn1cblxuLyoqXG4gKiBSZXR1cm5zIGEgbGlzdCBvZiBiYXRjaCBpbmRpY2VzICh0dXBsZXMgb2YgaW5kaWNlcykuXG4gKiBAcGFyYW0gc2l6ZTogSW50ZWdlciwgdG90YWwgc2l6ZSBvZiB0aGUgZGF0YSB0byBzbGljZSBpbnRvIGJhdGNoZXMuXG4gKiBAcGFyYW0gYmF0Y2hTaXplOiBJbnRlZ2VyLCBiYXRjaCBzaXplLlxuICogQHJldHVybnMgQW4gQXJyYXkgb2YgW2JhdGNoU3RhcnQsIGJhdGNoRW5kXSB0dXBsZXMuIGJhdGNoU3RhcnQgaXNcbiAqICAgaW5jbHVzaXZlOyBiYXRjaEVuZCBpcyBleGNsdXNpdmUuIEkuZS4sIGVhY2ggYmF0Y2ggY29uc2lzdHMgb2YgaW5kaWNlcyB4XG4gKiAgIHRoYXQgc2F0aXNmeSBiYXRjaFN0YXJ0IDw9IHggPCBiYXRjaEVuZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1ha2VCYXRjaGVzKFxuICAgIHNpemU6IG51bWJlciwgYmF0Y2hTaXplOiBudW1iZXIpOiBBcnJheTxbbnVtYmVyLCBudW1iZXJdPiB7XG4gIGNvbnN0IG91dHB1dDogQXJyYXk8W251bWJlciwgbnVtYmVyXT4gPSBbXTtcbiAgbGV0IGJhdGNoU3RhcnQgPSAwO1xuICBsZXQgYmF0Y2hFbmQ6IG51bWJlciA9IG51bGw7XG4gIHdoaWxlIChiYXRjaFN0YXJ0IDwgc2l6ZSkge1xuICAgIGJhdGNoRW5kID0gYmF0Y2hTdGFydCArIGJhdGNoU2l6ZTtcbiAgICBpZiAoYmF0Y2hFbmQgPj0gc2l6ZSkge1xuICAgICAgYmF0Y2hFbmQgPSBzaXplO1xuICAgIH1cbiAgICBvdXRwdXQucHVzaChbYmF0Y2hTdGFydCwgYmF0Y2hFbmRdKTtcbiAgICBiYXRjaFN0YXJ0ID0gYmF0Y2hFbmQ7XG4gIH1cbiAgcmV0dXJuIG91dHB1dDtcbn1cblxuLyoqXG4gKiBBYnN0cmFjdCBmaXQgZnVuY3Rpb24gZm9yIGBmKGlucylgLlxuICogQHBhcmFtIGYgQSBGdW5jdGlvbiByZXR1cm5pbmcgYSBsaXN0IG9mIHRlbnNvcnMuIEZvciB0cmFpbmluZywgdGhpc1xuICogICBmdW5jdGlvbiBpcyBleHBlY3RlZCB0byBwZXJmb3JtIHRoZSB1cGRhdGVzIHRvIHRoZSB2YXJpYWJsZXMuXG4gKiBAcGFyYW0gaW5zIExpc3Qgb2YgdGVuc29ycyB0byBiZSBmZWQgdG8gYGZgLlxuICogQHBhcmFtIG91dExhYmVscyBMaXN0IG9mIHN0cmluZ3MsIGRpc3BsYXkgbmFtZXMgb2YgdGhlIG91dHB1dHMgb2YgYGZgLlxuICogQHBhcmFtIGJhdGNoU2l6ZSBJbnRlZ2VyIGJhdGNoIHNpemUgb3IgYD09IG51bGxgIGlmIHVua25vd24uIERlZmF1bHQgOiAzMi5cbiAqIEBwYXJhbSBlcG9jaHMgTnVtYmVyIG9mIHRpbWVzIHRvIGl0ZXJhdGUgb3ZlciB0aGUgZGF0YS4gRGVmYXVsdCA6IDEuXG4gKiBAcGFyYW0gdmVyYm9zZSBWZXJib3NpdHkgbW9kZTogMCwgMSwgb3IgMi4gRGVmYXVsdDogMS5cbiAqIEBwYXJhbSBjYWxsYmFja3MgTGlzdCBvZiBjYWxsYmFja3MgdG8gYmUgY2FsbGVkIGR1cmluZyB0cmFpbmluZy5cbiAqIEBwYXJhbSB2YWxGIEZ1bmN0aW9uIHRvIGNhbGwgZm9yIHZhbGlkYXRpb24uXG4gKiBAcGFyYW0gdmFsSW5zIExpc3Qgb2YgdGVuc29ycyB0byBiZSBmZWQgdG8gYHZhbEZgLlxuICogQHBhcmFtIHNodWZmbGUgV2hldGhlciB0byBzaHVmZmxlIHRoZSBkYXRhIGF0IHRoZSBiZWdpbm5pbmcgb2YgZXZlcnlcbiAqIGVwb2NoLiBEZWZhdWx0IDogdHJ1ZS5cbiAqIEBwYXJhbSBjYWxsYmFja01ldHJpY3MgTGlzdCBvZiBzdHJpbmdzLCB0aGUgZGlzcGxheSBuYW1lcyBvZiB0aGUgbWV0cmljc1xuICogICBwYXNzZWQgdG8gdGhlIGNhbGxiYWNrcy4gVGhleSBzaG91bGQgYmUgdGhlIGNvbmNhdGVuYXRpb24gb2YgdGhlXG4gKiAgIGRpc3BsYXkgbmFtZXMgb2YgdGhlIG91dHB1dHMgb2YgYGZgIGFuZCB0aGUgbGlzdCBvZiBkaXNwbGF5IG5hbWVzXG4gKiAgIG9mIHRoZSBvdXRwdXRzIG9mIGB2YWxGYC5cbiAqIEBwYXJhbSBpbml0aWFsRXBvY2ggRXBvY2ggYXQgd2hpY2ggdG8gc3RhcnQgdHJhaW5pbmcgKHVzZWZ1bCBmb3JcbiAqICAgcmVzdW1pbmcgYSBwcmV2aW91cyB0cmFpbmluZyBydW4pLiBEZWZhdWx0IDogMC5cbiAqIEBwYXJhbSBzdGVwc1BlckVwb2NoIFRvdGFsIG51bWJlciBvZiBzdGVwcyAoYmF0Y2hlcyBvbiBzYW1wbGVzKSBiZWZvcmVcbiAqICAgZGVjbGFyaW5nIG9uZSBlcG9jaCBmaW5pc2hlZCBhbmQgc3RhcnRpbmcgdGhlIG5leHQgZXBvY2guIElnbm9yZWQgd2l0aFxuICogICB0aGUgZGVmYXVsdCB2YWx1ZSBvZiBgdW5kZWZpbmVkYCBvciBgbnVsbGAuXG4gKiBAcGFyYW0gdmFsaWRhdGlvblN0ZXBzIE51bWJlciBvZiBzdGVwcyB0byBydW4gdmFsaWRhdGlvbiBmb3IgKG9ubHkgaWZcbiAqICAgZG9pbmcgdmFsaWRhdGlvbiBmcm9tIGRhdGEgdGVuc29ycykuIE5vdCBhcHBsaWNhYmxlIGZvciB0ZmpzLWxheWVycy5cbiAqIEByZXR1cm5zIEEgYEhpc3RvcnlgIG9iamVjdC5cbiAqL1xuYXN5bmMgZnVuY3Rpb24gZml0TG9vcChcbiAgICAvLyBUeXBlIGBtb2RlbGAgYXMgYGFueWAgaGVyZSB0byBhdm9pZCBjaXJjdWxhciBkZXBlbmRlbmN5IHcvIHRyYWluaW5nLnRzLlxuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICBtb2RlbDogYW55LCBmOiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdLCBpbnM6IFRlbnNvcltdLFxuICAgIG91dExhYmVscz86IHN0cmluZ1tdLCBiYXRjaFNpemU/OiBudW1iZXIsIGVwb2Nocz86IG51bWJlciwgdmVyYm9zZT86IG51bWJlcixcbiAgICBjYWxsYmFja3M/OiBCYXNlQ2FsbGJhY2tbXSwgdmFsRj86IChkYXRhOiBUZW5zb3JbXSkgPT4gU2NhbGFyW10sXG4gICAgdmFsSW5zPzogVGVuc29yW10sIHNodWZmbGU/OiBib29sZWFufHN0cmluZywgY2FsbGJhY2tNZXRyaWNzPzogc3RyaW5nW10sXG4gICAgaW5pdGlhbEVwb2NoPzogbnVtYmVyLCBzdGVwc1BlckVwb2NoPzogbnVtYmVyLFxuICAgIHZhbGlkYXRpb25TdGVwcz86IG51bWJlcik6IFByb21pc2U8SGlzdG9yeT4ge1xuICBpZiAoYmF0Y2hTaXplID09IG51bGwpIHtcbiAgICBiYXRjaFNpemUgPSAzMjtcbiAgfVxuICBpZiAoZXBvY2hzID09IG51bGwpIHtcbiAgICBlcG9jaHMgPSAxO1xuICB9XG4gIGlmIChzaHVmZmxlID09IG51bGwpIHtcbiAgICBzaHVmZmxlID0gdHJ1ZTtcbiAgfVxuICBpZiAoaW5pdGlhbEVwb2NoID09IG51bGwpIHtcbiAgICBpbml0aWFsRXBvY2ggPSAwO1xuICB9XG5cbiAgLy8gVE9ETyhjYWlzKTogQ2hhbmdlIGNvbnN0IHRvIGxldCBiZWxvdyB3aGVuIGltcGxlbWVudGluZyB2YWxpZGF0aW9uLlxuICBsZXQgZG9WYWxpZGF0aW9uID0gZmFsc2U7XG4gIGlmICh2YWxGICE9IG51bGwgJiYgdmFsSW5zICE9IG51bGwpIHtcbiAgICBkb1ZhbGlkYXRpb24gPSB0cnVlO1xuICAgIC8vIFRPRE8oY2Fpcyk6IHZlcmJvc2UgbWVzc2FnZS5cbiAgfVxuICBpZiAodmFsaWRhdGlvblN0ZXBzICE9IG51bGwpIHtcbiAgICBkb1ZhbGlkYXRpb24gPSB0cnVlO1xuICAgIGlmIChzdGVwc1BlckVwb2NoID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdDYW4gb25seSB1c2UgYHZhbGlkYXRpb25TdGVwc2Agd2hlbiBkb2luZyBzdGVwLXdpc2UgdHJhaW5pbmcsICcgK1xuICAgICAgICAgICdpLmUuLCBgc3RlcHNQZXJFcG9jaGAgbXVzdCBiZSBzZXQuJyk7XG4gICAgfVxuICB9XG5cbiAgY29uc3QgbnVtVHJhaW5TYW1wbGVzID1cbiAgICAgIG1vZGVsLmNoZWNrTnVtU2FtcGxlcyhpbnMsIGJhdGNoU2l6ZSwgc3RlcHNQZXJFcG9jaCwgJ3N0ZXBzX3Blcl9lcG9jaCcpO1xuICBsZXQgaW5kZXhBcnJheTogbnVtYmVyW107XG4gIGlmIChudW1UcmFpblNhbXBsZXMgIT0gbnVsbCkge1xuICAgIGluZGV4QXJyYXkgPSByYW5nZSgwLCBudW1UcmFpblNhbXBsZXMpO1xuICB9XG5cbiAgaWYgKHZlcmJvc2UgPT0gbnVsbCkge1xuICAgIHZlcmJvc2UgPSAxO1xuICB9XG5cbiAgY29uc3Qge2NhbGxiYWNrTGlzdCwgaGlzdG9yeX0gPSBjb25maWd1cmVDYWxsYmFja3MoXG4gICAgICBjYWxsYmFja3MsIHZlcmJvc2UsIGVwb2NocywgaW5pdGlhbEVwb2NoLCBudW1UcmFpblNhbXBsZXMsIHN0ZXBzUGVyRXBvY2gsXG4gICAgICBiYXRjaFNpemUsIGRvVmFsaWRhdGlvbiwgY2FsbGJhY2tNZXRyaWNzKTtcbiAgY2FsbGJhY2tMaXN0LnNldE1vZGVsKG1vZGVsKTtcbiAgbW9kZWwuaGlzdG9yeSA9IGhpc3Rvcnk7XG4gIGF3YWl0IGNhbGxiYWNrTGlzdC5vblRyYWluQmVnaW4oKTtcbiAgbW9kZWwuc3RvcFRyYWluaW5nXyA9IGZhbHNlO1xuICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgY2FsbGJhY2tzLnZhbGlkYXRpb25fZGF0YSBhcyBpbiBQeUtlcmFzLlxuICAvLyBUT0RPKGNhaXMpOiBQcmUtY29udmVydCBmZWVkcyBmb3IgcGVyZm9ybWFuY2UgYXMgaW4gUHlLZXJhcy5cblxuICBmb3IgKGxldCBlcG9jaCA9IGluaXRpYWxFcG9jaDsgZXBvY2ggPCBlcG9jaHM7ICsrZXBvY2gpIHtcbiAgICBhd2FpdCBjYWxsYmFja0xpc3Qub25FcG9jaEJlZ2luKGVwb2NoKTtcbiAgICBjb25zdCBlcG9jaExvZ3M6IFVucmVzb2x2ZWRMb2dzID0ge307XG4gICAgaWYgKHN0ZXBzUGVyRXBvY2ggIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgJ3N0ZXBzUGVyRXBvY2ggbW9kZSBpcyBub3QgaW1wbGVtZW50ZWQgeWV0LicpO1xuICAgIH0gZWxzZSB7XG4gICAgICBpZiAoc2h1ZmZsZSA9PT0gJ2JhdGNoJykge1xuICAgICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcignYmF0Y2ggc2h1ZmZsaW5nIGlzIG5vdCBpbXBsZW1uZXRlZCB5ZXQnKTtcbiAgICAgIH0gZWxzZSBpZiAoc2h1ZmZsZSkge1xuICAgICAgICB1dGlsLnNodWZmbGUoaW5kZXhBcnJheSk7XG4gICAgICB9XG4gICAgICAvLyBDb252ZXJ0IHRoZSBwb3RlbnRpYWxseSBzaHVmZmxlZCBpbmRpY2VzIHRvIFRlbnNvcjFELCB0byBhdm9pZCB0aGVcbiAgICAgIC8vIGNvc3Qgb2YgcmVwZWF0ZWQgY3JlYXRpb24gb2YgQXJyYXkxRHMgbGF0ZXIgb24uXG4gICAgICBjb25zdCBlcG9jaEluZGV4QXJyYXkxRCA9IHRlbnNvcjFkKGluZGV4QXJyYXkpO1xuXG4gICAgICBjb25zdCBiYXRjaGVzID0gbWFrZUJhdGNoZXMobnVtVHJhaW5TYW1wbGVzLCBiYXRjaFNpemUpO1xuICAgICAgZm9yIChsZXQgYmF0Y2hJbmRleCA9IDA7IGJhdGNoSW5kZXggPCBiYXRjaGVzLmxlbmd0aDsgKytiYXRjaEluZGV4KSB7XG4gICAgICAgIGNvbnN0IGJhdGNoTG9nczogVW5yZXNvbHZlZExvZ3MgPSB7fTtcbiAgICAgICAgYXdhaXQgY2FsbGJhY2tMaXN0Lm9uQmF0Y2hCZWdpbihiYXRjaEluZGV4LCBiYXRjaExvZ3MpO1xuXG4gICAgICAgIHRmYy50aWR5KCgpID0+IHtcbiAgICAgICAgICBjb25zdCBiYXRjaFN0YXJ0ID0gYmF0Y2hlc1tiYXRjaEluZGV4XVswXTtcbiAgICAgICAgICBjb25zdCBiYXRjaEVuZCA9IGJhdGNoZXNbYmF0Y2hJbmRleF1bMV07XG4gICAgICAgICAgY29uc3QgYmF0Y2hJZHMgPSBzbGljZUFsb25nRmlyc3RBeGlzKFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGVwb2NoSW5kZXhBcnJheTFELCBiYXRjaFN0YXJ0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJhdGNoRW5kIC0gYmF0Y2hTdGFydCkgYXMgVGVuc29yMUQ7XG4gICAgICAgICAgYmF0Y2hMb2dzWydiYXRjaCddID0gYmF0Y2hJbmRleDtcbiAgICAgICAgICBiYXRjaExvZ3NbJ3NpemUnXSA9IGJhdGNoRW5kIC0gYmF0Y2hTdGFydDtcblxuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IEluIGlucywgdHJhaW4gZmxhZyBjYW4gYmUgYSBudW1iZXIsIGluc3RlYWQgb2YgYW5cbiAgICAgICAgICAvLyAgIFRlbnNvcj8gRG8gd2UgbmVlZCB0byBoYW5kbGUgdGhpcyBpbiB0ZmpzLWxheWVycz9cbiAgICAgICAgICBjb25zdCBpbnNCYXRjaCA9IHNsaWNlQXJyYXlzQnlJbmRpY2VzKGlucywgYmF0Y2hJZHMpIGFzIFRlbnNvcltdO1xuICAgICAgICAgIGNvbnN0IG91dHMgPSBmKGluc0JhdGNoKTtcbiAgICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dExhYmVscy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgICAgY29uc3QgbGFiZWwgPSBvdXRMYWJlbHNbaV07XG4gICAgICAgICAgICBjb25zdCBvdXQgPSBvdXRzW2ldO1xuICAgICAgICAgICAgYmF0Y2hMb2dzW2xhYmVsXSA9IG91dDtcbiAgICAgICAgICAgIHRmYy5rZWVwKG91dCk7XG4gICAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBVc2Ugc2NvcGUoKSB0byBhdm9pZCBvd25lcnNoaXAuXG4gICAgICAgICAgfVxuXG4gICAgICAgICAgaWYgKGJhdGNoSW5kZXggPT09IGJhdGNoZXMubGVuZ3RoIC0gMSkgeyAgLy8gTGFzdCBiYXRjaC5cbiAgICAgICAgICAgIGlmIChkb1ZhbGlkYXRpb24pIHtcbiAgICAgICAgICAgICAgY29uc3QgdmFsT3V0cyA9IG1vZGVsLnRlc3RMb29wKHZhbEYsIHZhbElucywgYmF0Y2hTaXplKTtcbiAgICAgICAgICAgICAgLy8gUG9ydGluZyBOb3RlczogSW4gdGZqcy1sYXllcnMsIHZhbE91dHMgaXMgYWx3YXlzIGFuIEFycmF5LlxuICAgICAgICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dExhYmVscy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgICAgICAgIGNvbnN0IGxhYmVsID0gb3V0TGFiZWxzW2ldO1xuICAgICAgICAgICAgICAgIGNvbnN0IG91dCA9IHZhbE91dHNbaV07XG4gICAgICAgICAgICAgICAgdGZjLmtlZXAob3V0KTtcbiAgICAgICAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBVc2Ugc2NvcGUoKSB0byBhdm9pZCBvd25lcnNoaXAuXG4gICAgICAgICAgICAgICAgZXBvY2hMb2dzWyd2YWxfJyArIGxhYmVsXSA9IG91dDtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfSk7XG5cbiAgICAgICAgYXdhaXQgY2FsbGJhY2tMaXN0Lm9uQmF0Y2hFbmQoYmF0Y2hJbmRleCwgYmF0Y2hMb2dzKTtcbiAgICAgICAgZGlzcG9zZVRlbnNvcnNJbkxvZ3MoYmF0Y2hMb2dzKTtcblxuICAgICAgICBpZiAobW9kZWwuc3RvcFRyYWluaW5nXykge1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IHJldHVybiBvdXRzIGFzIGxpc3Qgb2YgVGVuc29yLlxuICAgICAgfVxuXG4gICAgICBlcG9jaEluZGV4QXJyYXkxRC5kaXNwb3NlKCk7XG4gICAgfVxuICAgIC8vIFRPRE8oY2Fpcyk6IFJ1biB2YWxpZGF0aW9uIGF0IHRoZSBlbmQgb2YgdGhlIGVwb2NoLlxuICAgIGF3YWl0IGNhbGxiYWNrTGlzdC5vbkVwb2NoRW5kKGVwb2NoLCBlcG9jaExvZ3MpO1xuICAgIGlmIChtb2RlbC5zdG9wVHJhaW5pbmdfKSB7XG4gICAgICBicmVhaztcbiAgICB9XG4gIH1cbiAgYXdhaXQgY2FsbGJhY2tMaXN0Lm9uVHJhaW5FbmQoKTtcblxuICBhd2FpdCBtb2RlbC5oaXN0b3J5LnN5bmNEYXRhKCk7XG4gIHJldHVybiBtb2RlbC5oaXN0b3J5O1xufVxuXG5leHBvcnQgYXN5bmMgZnVuY3Rpb24gZml0VGVuc29ycyhcbiAgICAvLyBUeXBlIGBtb2RlbGAgYXMgYGFueWAgaGVyZSB0byBhdm9pZCBjaXJjdWxhciBkZXBlbmRlbmN5IHcvIHRyYWluaW5nLnRzLlxuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICBtb2RlbDogYW55LCB4OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgeTogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgIGFyZ3M6IE1vZGVsRml0QXJncyA9IHt9KTogUHJvbWlzZTxIaXN0b3J5PiB7XG4gIGlmIChtb2RlbC5pc1RyYWluaW5nKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnQ2Fubm90IHN0YXJ0IHRyYWluaW5nIGJlY2F1c2UgYW5vdGhlciBmaXQoKSBjYWxsIGlzIG9uZ29pbmcuJyk7XG4gIH1cbiAgbW9kZWwuaXNUcmFpbmluZyA9IHRydWU7XG4gIGxldCBpbnB1dHM6IFRlbnNvcltdO1xuICBsZXQgdGFyZ2V0czogVGVuc29yW107XG4gIGxldCBvcmlnaW5hbElucHV0czogVGVuc29yW107XG4gIGxldCBvcmlnaW5hbFRhcmdldHM6IFRlbnNvcltdO1xuICBsZXQgaW5wdXRWYWxYOiBUZW5zb3J8VGVuc29yW107XG4gIGxldCBpbnB1dFZhbFk6IFRlbnNvcnxUZW5zb3JbXTtcbiAgbGV0IHZhbFg6IFRlbnNvcnxUZW5zb3JbXTtcbiAgbGV0IHZhbFk6IFRlbnNvcnxUZW5zb3JbXTtcbiAgbGV0IHNhbXBsZVdlaWdodHM6IFRlbnNvcltdO1xuICB0cnkge1xuICAgIGNvbnN0IGJhdGNoU2l6ZSA9IGFyZ3MuYmF0Y2hTaXplID09IG51bGwgPyAzMiA6IGFyZ3MuYmF0Y2hTaXplO1xuICAgIGNoZWNrQmF0Y2hTaXplKGJhdGNoU2l6ZSk7XG5cbiAgICAvLyBWYWxpZGF0ZSB1c2VyIGRhdGEuXG4gICAgLy8gVE9ETyhjYWlzKTogU3VwcG9ydCBzYW1wbGVXZWlnaHQuXG4gICAgY29uc3QgY2hlY2tCYXRjaEF4aXMgPSBmYWxzZTtcbiAgICBjb25zdCBzdGFuZGFyZGl6ZWRPdXRzID1cbiAgICAgICAgYXdhaXQgbW9kZWwuc3RhbmRhcmRpemVVc2VyRGF0YShcbiAgICAgICAgICAgIHgsIHksIGFyZ3Muc2FtcGxlV2VpZ2h0LCBhcmdzLmNsYXNzV2VpZ2h0LCBjaGVja0JhdGNoQXhpcyxcbiAgICAgICAgICAgIGJhdGNoU2l6ZSkgYXMgW1RlbnNvcltdLCBUZW5zb3JbXSwgVGVuc29yW11dO1xuICAgIGlucHV0cyA9IHN0YW5kYXJkaXplZE91dHNbMF07XG4gICAgdGFyZ2V0cyA9IHN0YW5kYXJkaXplZE91dHNbMV07XG4gICAgc2FtcGxlV2VpZ2h0cyA9IHN0YW5kYXJkaXplZE91dHNbMl07XG5cbiAgICAvLyBQcmVwYXJlIHZhbGlkYXRpb24gZGF0YS5cbiAgICBsZXQgZG9WYWxpZGF0aW9uID0gZmFsc2U7XG4gICAgbGV0IHZhbEluczogVGVuc29yW107XG4gICAgaWYgKGFyZ3MudmFsaWRhdGlvbkRhdGEgIT0gbnVsbCAmJiBhcmdzLnZhbGlkYXRpb25EYXRhLmxlbmd0aCA+IDApIHtcbiAgICAgIGRvVmFsaWRhdGlvbiA9IHRydWU7XG4gICAgICBpZiAoYXJncy52YWxpZGF0aW9uRGF0YS5sZW5ndGggPT09IDIpIHtcbiAgICAgICAgLy8gY29uZmlnLnZhbGlkYXRpb25EYXRhIGNvbnNpc3RzIG9mIHZhbFggYW5kIHZhbFkuXG4gICAgICAgIGlucHV0VmFsWCA9IGFyZ3MudmFsaWRhdGlvbkRhdGFbMF07XG4gICAgICAgIGlucHV0VmFsWSA9IGFyZ3MudmFsaWRhdGlvbkRhdGFbMV07XG4gICAgICB9IGVsc2UgaWYgKGFyZ3MudmFsaWRhdGlvbkRhdGEubGVuZ3RoID09PSAzKSB7XG4gICAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICAgJ3ZhbGlkYXRpb25EYXRhIGluY2x1ZGluZyBzYW1wbGUgd2VpZ2h0cyBpcyBub3Qgc3VwcG9ydGVkIHlldC4nKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYFdoZW4gcGFzc2luZyB2YWxpZGF0aW9uIGRhdGEsIGl0IG11c3QgY29udGFpbiAyICh2YWxYLCB2YWxZKSBgICtcbiAgICAgICAgICAgIGBvciAzICh2YWxYLCB2YWxZLCB2YWxTYW1wbGVXZWlnaHQpIGl0ZW1zOyBgICtcbiAgICAgICAgICAgIGAke2FyZ3MudmFsaWRhdGlvbkRhdGF9IGlzIGludmFsaWQuYCk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IGNoZWNrQmF0Y2hBeGlzID0gdHJ1ZTtcbiAgICAgIGNvbnN0IHZhbFN0YW5kYXJkaXplZCA9XG4gICAgICAgICAgYXdhaXQgbW9kZWwuc3RhbmRhcmRpemVVc2VyRGF0YShcbiAgICAgICAgICAgICAgaW5wdXRWYWxYLCBpbnB1dFZhbFksIG51bGwsIC8qKiBVbnVzZWQgc2FtcGxlIHdlaWdodHMuICovXG4gICAgICAgICAgICAgIG51bGwsICAgICAgICAgICAgICAgICAgICAgICAvKiogVW51c2VkIGNsYXNzIHdlaWdodHMuICovXG4gICAgICAgICAgICAgIGNoZWNrQmF0Y2hBeGlzLCBiYXRjaFNpemUpIGFzIFtUZW5zb3JbXSwgVGVuc29yW10sIFRlbnNvcltdXTtcbiAgICAgIHZhbFggPSB2YWxTdGFuZGFyZGl6ZWRbMF07XG4gICAgICB2YWxZID0gdmFsU3RhbmRhcmRpemVkWzFdO1xuICAgICAgdmFsSW5zID0gdmFsWC5jb25jYXQodmFsWSk7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgdXNlTGVhcm5pbmdQaGFzZSBkYXRhIHByb3Blcmx5LlxuICAgIH0gZWxzZSBpZiAoXG4gICAgICAgIGFyZ3MudmFsaWRhdGlvblNwbGl0ICE9IG51bGwgJiYgYXJncy52YWxpZGF0aW9uU3BsaXQgPiAwICYmXG4gICAgICAgIGFyZ3MudmFsaWRhdGlvblNwbGl0IDwgMSkge1xuICAgICAgZG9WYWxpZGF0aW9uID0gdHJ1ZTtcbiAgICAgIC8vIFBvcnRpbmcgTm90ZTogSW4gdGZqcy1sYXllcnMsIGlucHV0c1swXSBpcyBhbHdheXMgYSBUZW5zb3IuXG4gICAgICBjb25zdCBzcGxpdEF0ID1cbiAgICAgICAgICBNYXRoLmZsb29yKGlucHV0c1swXS5zaGFwZVswXSAqICgxIC0gYXJncy52YWxpZGF0aW9uU3BsaXQpKTtcbiAgICAgIGNvbnN0IG9yaWdpbmFsQmF0Y2hTaXplID0gaW5wdXRzWzBdLnNoYXBlWzBdO1xuICAgICAgdmFsWCA9IHNsaWNlQXJyYXlzKGlucHV0cywgc3BsaXRBdCwgb3JpZ2luYWxCYXRjaFNpemUpIGFzIFRlbnNvcltdO1xuICAgICAgb3JpZ2luYWxJbnB1dHMgPSBpbnB1dHM7XG4gICAgICBpbnB1dHMgPSBzbGljZUFycmF5cyhpbnB1dHMsIDAsIHNwbGl0QXQpIGFzIFRlbnNvcltdO1xuICAgICAgdmFsWSA9IHNsaWNlQXJyYXlzKHRhcmdldHMsIHNwbGl0QXQsIG9yaWdpbmFsQmF0Y2hTaXplKSBhcyBUZW5zb3JbXTtcbiAgICAgIG9yaWdpbmFsVGFyZ2V0cyA9IHRhcmdldHM7XG4gICAgICB0YXJnZXRzID0gc2xpY2VBcnJheXModGFyZ2V0cywgMCwgc3BsaXRBdCkgYXMgVGVuc29yW107XG4gICAgICAvLyBUT0RPKGNhaXMpOiBPbmNlIHNhbXBsZVdlaWdodHMgYmVjb21lcyBhdmFpbGFibGUsIHNsaWNlIGl0IHRvIGdldFxuICAgICAgLy8gICB2YWxTYW1wbGVXZWlnaHRzLlxuICAgICAgdmFsSW5zID0gdmFsWC5jb25jYXQodmFsWSk7XG5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB1c2VMZWFybmluZ1BoYXNlIGRhdGEgcHJvcGVybHkuXG4gICAgfSBlbHNlIGlmIChhcmdzLnZhbGlkYXRpb25TdGVwcyAhPSBudWxsKSB7XG4gICAgICBkb1ZhbGlkYXRpb24gPSB0cnVlO1xuICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIHVzZUxlYXJuaW5nUGhhc2UuXG4gICAgfVxuXG4gICAgY29uc3QgaW5zID0gaW5wdXRzLmNvbmNhdCh0YXJnZXRzKS5jb25jYXQoc2FtcGxlV2VpZ2h0cyk7XG5cbiAgICBtb2RlbC5jaGVja1RyYWluYWJsZVdlaWdodHNDb25zaXN0ZW5jeSgpO1xuXG4gICAgLy8gVE9ETyhjYWlzKTogSGFuZGxlIHVzZV9sZWFybmluZ19waGFzZSBhbmQgbGVhcm5pbmdfcGhhc2U/XG5cbiAgICAvLyBQb3J0aW5nIE5vdGU6IEhlcmUgd2Ugc2VlIGEga2V5IGRldmlhdGlvbiBvZiB0ZmpzLWxheWVycyBmcm9tXG4gICAgLy8gS2VyYXMuXG4gICAgLy8gIER1ZSB0byB0aGUgaW1wZXJhdGl2ZSBuYXR1cmUgb2YgdGZqcy1sYXllcnMnIGJhY2tlbmQgKHRmanMtY29yZSksXG4gICAgLy8gIHdlIGRvIG5vdCBjb25zdHJ1Y3Qgc3ltYm9saWMgY29tcHV0YXRpb24gZ3JhcGhzIHRvIGVtYm9keSB0aGVcbiAgICAvLyAgdHJhaW5pbmcgcHJvY2Vzcy4gSW5zdGVhZCwgd2UgZGVmaW5lIGEgZnVuY3Rpb24gdGhhdCBwZXJmb3JtcyB0aGVcbiAgICAvLyAgdHJhaW5pbmcgYWN0aW9uLiBJbiBQeUtlcmFzLCB0aGUgZGF0YSAoaW5wdXRzIGFuZCB0YXJnZXRzKSBhcmUgZmVkXG4gICAgLy8gIHRocm91Z2ggZ3JhcGggcGxhY2Vob2xkZXJzLiBJbiB0ZmpzLWxheWVycywgdGhlIGRhdGEgYXJlIGZlZCBhc1xuICAgIC8vICBmdW5jdGlvbiBhcmd1bWVudHMuIFNpbmNlIHRoZSBmdW5jdGlvbiBhcmUgZGVmaW5lZCBiZWxvdyBpbiB0aGVcbiAgICAvLyAgc2NvcGUsIHdlIGRvbid0IGhhdmUgZXF1aXZhbGVudHMgb2YgUHlLZXJhcydzXG4gICAgLy8gIGBfbWFrZV90cmFpbl9mdW5jaXRvbmAuXG4gICAgY29uc3QgdHJhaW5GdW5jdGlvbiA9IG1vZGVsLm1ha2VUcmFpbkZ1bmN0aW9uKCk7XG4gICAgY29uc3Qgb3V0TGFiZWxzID0gbW9kZWwuZ2V0RGVkdXBlZE1ldHJpY3NOYW1lcygpIGFzIHN0cmluZ1tdO1xuXG4gICAgbGV0IHZhbEZ1bmN0aW9uOiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdO1xuICAgIGxldCBjYWxsYmFja01ldHJpY3M6IHN0cmluZ1tdO1xuICAgIGlmIChkb1ZhbGlkYXRpb24pIHtcbiAgICAgIG1vZGVsLm1ha2VUZXN0RnVuY3Rpb24oKTtcbiAgICAgIHZhbEZ1bmN0aW9uID0gbW9kZWwudGVzdEZ1bmN0aW9uO1xuICAgICAgY2FsbGJhY2tNZXRyaWNzID1cbiAgICAgICAgICBvdXRMYWJlbHMuc2xpY2UoKS5jb25jYXQob3V0TGFiZWxzLm1hcChuID0+ICd2YWxfJyArIG4pKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdmFsRnVuY3Rpb24gPSBudWxsO1xuICAgICAgdmFsSW5zID0gW107XG4gICAgICBjYWxsYmFja01ldHJpY3MgPSBvdXRMYWJlbHMuc2xpY2UoKTtcbiAgICB9XG5cbiAgICBjb25zdCBjYWxsYmFja3MgPSBzdGFuZGFyZGl6ZUNhbGxiYWNrcyhhcmdzLmNhbGxiYWNrcywgYXJncy55aWVsZEV2ZXJ5KTtcbiAgICBjb25zdCBvdXQgPSBhd2FpdCBmaXRMb29wKFxuICAgICAgICBtb2RlbCwgdHJhaW5GdW5jdGlvbiwgaW5zLCBvdXRMYWJlbHMsIGJhdGNoU2l6ZSwgYXJncy5lcG9jaHMsXG4gICAgICAgIGFyZ3MudmVyYm9zZSwgY2FsbGJhY2tzLCB2YWxGdW5jdGlvbiwgdmFsSW5zLCBhcmdzLnNodWZmbGUsXG4gICAgICAgIGNhbGxiYWNrTWV0cmljcywgYXJncy5pbml0aWFsRXBvY2gsIG51bGwsIG51bGwpO1xuICAgIHJldHVybiBvdXQ7XG4gIH0gZmluYWxseSB7XG4gICAgbW9kZWwuaXNUcmFpbmluZyA9IGZhbHNlO1xuICAgIC8vIE1lbW9yeSBjbGVhbiB1cC5cbiAgICBkaXNwb3NlTmV3VGVuc29ycyhpbnB1dHMsIHgpO1xuICAgIGRpc3Bvc2VOZXdUZW5zb3JzKHRhcmdldHMsIHkpO1xuICAgIGRpc3Bvc2VOZXdUZW5zb3JzKG9yaWdpbmFsSW5wdXRzLCB4KTtcbiAgICBkaXNwb3NlTmV3VGVuc29ycyhvcmlnaW5hbFRhcmdldHMsIHkpO1xuICAgIGRpc3Bvc2VOZXdUZW5zb3JzKHZhbFggYXMgVGVuc29yW10sIGlucHV0VmFsWCk7XG4gICAgZGlzcG9zZU5ld1RlbnNvcnModmFsWSBhcyBUZW5zb3JbXSwgaW5wdXRWYWxZKTtcbiAgICBpZiAoc2FtcGxlV2VpZ2h0cyAhPSBudWxsKSB7XG4gICAgICB0ZmMuZGlzcG9zZShzYW1wbGVXZWlnaHRzKTtcbiAgICB9XG4gIH1cbiAgLy8gVE9ETyhjYWlzKTogQWRkIHZhbHVlIHRvIG91dExhYmVscy5cbn1cblxuLyoqXG4gKiBFbnN1cmUgdGVuc29ycyBhbGwgaGF2ZSBhIHJhbmsgb2YgYXQgbGVhc3QgMi5cbiAqXG4gKiBJZiBhIHRlbnNvciBoYXMgYSByYW5rIG9mIDEsIGl0IGlzIGRpbWVuc2lvbi1leHBhbmRlZCB0byByYW5rIDIuXG4gKiBJZiBhbnkgdGVuc29yIGhhcyBhIHJhbmsgb2YgMCAoaS5lLiwgaXMgYSBzY2FsYXIpLCBhbiBlcnJvciB3aWxsIGJlIHRocm93bi5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGVuc3VyZVRlbnNvcnNSYW5rMk9ySGlnaGVyKHRlbnNvcnM6IFRlbnNvcnxUZW5zb3JbXSk6IFRlbnNvcltdIHtcbiAgY29uc3Qgb3V0czogVGVuc29yW10gPSBbXTtcbiAgaWYgKHRlbnNvcnMgaW5zdGFuY2VvZiBUZW5zb3IpIHtcbiAgICB0ZW5zb3JzID0gW3RlbnNvcnNdO1xuICB9XG5cbiAgLy8gTWFrZSBUZW5zb3JzIGF0IGxlYXN0IDJELlxuICBmb3IgKGxldCBpID0gMDsgaSA8IHRlbnNvcnMubGVuZ3RoOyArK2kpIHtcbiAgICBjb25zdCB0ZW5zb3IgPSB0ZW5zb3JzW2ldO1xuICAgIGlmICh0ZW5zb3IucmFuayA9PT0gMSkge1xuICAgICAgb3V0cy5wdXNoKGV4cGFuZERpbXModGVuc29yLCAxKSk7XG4gICAgfSBlbHNlIGlmICh0ZW5zb3IucmFuayA9PT0gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICdFeHBlY3RlZCB0ZW5zb3IgdG8gYmUgYXQgbGVhc3QgMUQsIGJ1dCByZWNlaXZlZCBhIDBEIHRlbnNvciAnICtcbiAgICAgICAgICAnKHNjYWxhcikuJyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIG91dHMucHVzaCh0ZW5zb3IpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gb3V0cztcbn1cblxuLyoqXG4gKiBDb21wYXJlIGEgc2V0IG9mIHRlbnNvcnMgd2l0aCBhIHJlZmVyZW5jZSAob2xkKSBzZXQsIGRpc2NhcmQgdGhlIG9uZXNcbiAqIGluIHRoZSBuZXcgc2V0IHRoYXQgYXJlIG5vdCBwcmVzZW50IGluIHRoZSByZWZlcmVuY2Ugc2V0LlxuICpcbiAqIFRoaXMgbWV0aG9kIGlzIHVzZWQgZm9yIG1lbW9yeSBjbGVuYXVwIGR1cmluZyBjYWxscyBzdWNoIGFzXG4gKiBMYXllcnNNb2RlbC5maXQoKS5cbiAqXG4gKiBAcGFyYW0gdGVuc29ycyBOZXcgc2V0IHdoaWNoIG1heSBjb250YWluIFRlbnNvcnMgbm90IHByZXNlbnQgaW5cbiAqICAgYHJlZlRlbnNvcnNgLlxuICogQHBhcmFtIHJlZlRlbnNvcnMgUmVmZXJlbmNlIFRlbnNvciBzZXQuXG4gKi9cbi8vIFRPRE8oY2Fpcywga2FuZ3lpemhhbmcpOiBEZWR1cGxpY2F0ZSB3aXRoIHRmanMtZGF0YS5cbmV4cG9ydCBmdW5jdGlvbiBkaXNwb3NlTmV3VGVuc29ycyhcbiAgICB0ZW5zb3JzOiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgcmVmVGVuc29yczogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9KTogdm9pZCB7XG4gIGlmICh0ZW5zb3JzID09IG51bGwpIHtcbiAgICByZXR1cm47XG4gIH1cbiAgY29uc3Qgb2xkVGVuc29ySWRzOiBudW1iZXJbXSA9IFtdO1xuICBpZiAocmVmVGVuc29ycyBpbnN0YW5jZW9mIFRlbnNvcikge1xuICAgIG9sZFRlbnNvcklkcy5wdXNoKHJlZlRlbnNvcnMuaWQpO1xuICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkocmVmVGVuc29ycykpIHtcbiAgICByZWZUZW5zb3JzLmZvckVhY2godCA9PiBvbGRUZW5zb3JJZHMucHVzaCh0LmlkKSk7XG4gIH0gZWxzZSBpZiAocmVmVGVuc29ycyAhPSBudWxsKSB7XG4gICAgLy8gYG9sZFRlbnNvcnNgIGlzIGEgbWFwIGZyb20gc3RyaW5nIG5hbWUgdG8gVGVuc29yLlxuICAgIGZvciAoY29uc3QgbmFtZSBpbiByZWZUZW5zb3JzKSB7XG4gICAgICBjb25zdCBvbGRUZW5zb3IgPSByZWZUZW5zb3JzW25hbWVdO1xuICAgICAgb2xkVGVuc29ySWRzLnB1c2gob2xkVGVuc29yLmlkKTtcbiAgICB9XG4gIH1cblxuICBjb25zdCB0ZW5zb3JzVG9EaXNwb3NlOiBUZW5zb3JbXSA9IFtdO1xuICBpZiAodGVuc29ycyBpbnN0YW5jZW9mIFRlbnNvcikge1xuICAgIGlmIChvbGRUZW5zb3JJZHMuaW5kZXhPZih0ZW5zb3JzLmlkKSA9PT0gLTEpIHtcbiAgICAgIHRlbnNvcnNUb0Rpc3Bvc2UucHVzaCh0ZW5zb3JzKTtcbiAgICB9XG4gIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheSh0ZW5zb3JzKSkge1xuICAgIHRlbnNvcnMuZm9yRWFjaCh0ID0+IHtcbiAgICAgIGlmIChvbGRUZW5zb3JJZHMuaW5kZXhPZih0LmlkKSA9PT0gLTEpIHtcbiAgICAgICAgdGVuc29yc1RvRGlzcG9zZS5wdXNoKHQpO1xuICAgICAgfVxuICAgIH0pO1xuICB9IGVsc2UgaWYgKHRlbnNvcnMgIT0gbnVsbCkge1xuICAgIC8vIGBvbGRUZW5zb3JzYCBpcyBhIG1hcCBmcm9tIHN0cmluZyBuYW1lIHRvIFRlbnNvci5cbiAgICBmb3IgKGNvbnN0IG5hbWUgaW4gdGVuc29ycykge1xuICAgICAgY29uc3QgdGVuc29yID0gdGVuc29yc1tuYW1lXTtcbiAgICAgIGlmIChvbGRUZW5zb3JJZHMuaW5kZXhPZih0ZW5zb3IuaWQpID09PSAtMSkge1xuICAgICAgICB0ZW5zb3JzVG9EaXNwb3NlLnB1c2godGVuc29yKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICB0ZW5zb3JzVG9EaXNwb3NlLmZvckVhY2godCA9PiB7XG4gICAgaWYgKCF0LmlzRGlzcG9zZWQpIHtcbiAgICAgIHQuZGlzcG9zZSgpO1xuICAgIH1cbiAgfSk7XG59XG4iXX0=
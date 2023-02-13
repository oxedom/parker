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
 * TensorFlow.js Layers: Recurrent Neural Network Layers.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy, util } from '@tensorflow/tfjs-core';
import { getActivation, serializeActivation } from '../activations';
import * as K from '../backend/tfjs_backend';
import { nameScope } from '../common';
import { getConstraint, serializeConstraint } from '../constraints';
import { InputSpec, SymbolicTensor } from '../engine/topology';
import { Layer } from '../engine/topology';
import { AttributeError, NotImplementedError, ValueError } from '../errors';
import { getInitializer, Initializer, Ones, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import { assertPositiveInteger } from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';
import { getExactlyOneShape, getExactlyOneTensor, isArrayOfShapes } from '../utils/types_utils';
import { batchGetValue, batchSetValue } from '../variables';
import { deserialize } from './serialization';
/**
 * Standardize `apply()` args to a single list of tensor inputs.
 *
 * When running a model loaded from file, the input tensors `initialState` and
 * `constants` are passed to `RNN.apply()` as part of `inputs` instead of the
 * dedicated kwargs fields. `inputs` consists of
 * `[inputs, initialState0, initialState1, ..., constant0, constant1]` in this
 * case.
 * This method makes sure that arguments are
 * separated and that `initialState` and `constants` are `Array`s of tensors
 * (or None).
 *
 * @param inputs Tensor or `Array` of  tensors.
 * @param initialState Tensor or `Array` of tensors or `null`/`undefined`.
 * @param constants Tensor or `Array` of tensors or `null`/`undefined`.
 * @returns An object consisting of
 *   inputs: A tensor.
 *   initialState: `Array` of tensors or `null`.
 *   constants: `Array` of tensors or `null`.
 * @throws ValueError, if `inputs` is an `Array` but either `initialState` or
 *   `constants` is provided.
 */
export function standardizeArgs(inputs, initialState, constants, numConstants) {
    if (Array.isArray(inputs)) {
        if (initialState != null || constants != null) {
            throw new ValueError('When inputs is an array, neither initialState or constants ' +
                'should be provided');
        }
        if (numConstants != null) {
            constants = inputs.slice(inputs.length - numConstants, inputs.length);
            inputs = inputs.slice(0, inputs.length - numConstants);
        }
        if (inputs.length > 1) {
            initialState = inputs.slice(1, inputs.length);
        }
        inputs = inputs[0];
    }
    function toListOrNull(x) {
        if (x == null || Array.isArray(x)) {
            return x;
        }
        else {
            return [x];
        }
    }
    initialState = toListOrNull(initialState);
    constants = toListOrNull(constants);
    return { inputs, initialState, constants };
}
/**
 * Iterates over the time dimension of a tensor.
 *
 * @param stepFunction RNN step function.
 *   Parameters:
 *     inputs: tensor with shape `[samples, ...]` (no time dimension),
 *       representing input for the batch of samples at a certain time step.
 *     states: an Array of tensors.
 *   Returns:
 *     outputs: tensor with shape `[samples, outputDim]` (no time dimension).
 *     newStates: list of tensors, same length and shapes as `states`. The first
 *       state in the list must be the output tensor at the previous timestep.
 * @param inputs Tensor of temporal data of shape `[samples, time, ...]` (at
 *   least 3D).
 * @param initialStates Tensor with shape `[samples, outputDim]` (no time
 *   dimension), containing the initial values of the states used in the step
 *   function.
 * @param goBackwards If `true`, do the iteration over the time dimension in
 *   reverse order and return the reversed sequence.
 * @param mask Binary tensor with shape `[sample, time, 1]`, with a zero for
 *   every element that is masked.
 * @param constants An Array of constant values passed at each step.
 * @param unroll Whether to unroll the RNN or to use a symbolic loop. *Not*
 *   applicable to this imperative deeplearn.js backend. Its value is ignored.
 * @param needPerStepOutputs Whether the per-step outputs are to be
 *   concatenated into a single tensor and returned (as the second return
 *   value). Default: `false`. This arg is included so that the relatively
 *   expensive concatenation of the stepwise outputs can be omitted unless
 *   the stepwise outputs need to be kept (e.g., for an LSTM layer of which
 *   `returnSequence` is `true`.)
 * @returns An Array: `[lastOutput, outputs, newStates]`.
 *   lastOutput: the lastest output of the RNN, of shape `[samples, ...]`.
 *   outputs: tensor with shape `[samples, time, ...]` where each entry
 *     `output[s, t]` is the output of the step function at time `t` for sample
 *     `s`. This return value is provided if and only if the
 *     `needPerStepOutputs` is set as `true`. If it is set as `false`, this
 *     return value will be `undefined`.
 *   newStates: Array of tensors, latest states returned by the step function,
 *      of shape `(samples, ...)`.
 * @throws ValueError If input dimension is less than 3.
 *
 * TODO(nielsene): This needs to be tidy-ed.
 */
export function rnn(stepFunction, inputs, initialStates, goBackwards = false, mask, constants, unroll = false, needPerStepOutputs = false) {
    return tfc.tidy(() => {
        const ndim = inputs.shape.length;
        if (ndim < 3) {
            throw new ValueError(`Input should be at least 3D, but is ${ndim}D.`);
        }
        // Transpose to time-major, i.e., from [batch, time, ...] to [time, batch,
        // ...].
        const axes = [1, 0].concat(math_utils.range(2, ndim));
        inputs = tfc.transpose(inputs, axes);
        if (constants != null) {
            throw new NotImplementedError('The rnn() functoin of the deeplearn.js backend does not support ' +
                'constants yet.');
        }
        // Porting Note: the unroll option is ignored by the imperative backend.
        if (unroll) {
            console.warn('Backend rnn(): the unroll = true option is not applicable to the ' +
                'imperative deeplearn.js backend.');
        }
        if (mask != null) {
            mask = tfc.cast(tfc.cast(mask, 'bool'), 'float32');
            if (mask.rank === ndim - 1) {
                mask = tfc.expandDims(mask, -1);
            }
            mask = tfc.transpose(mask, axes);
        }
        if (goBackwards) {
            inputs = tfc.reverse(inputs, 0);
            if (mask != null) {
                mask = tfc.reverse(mask, 0);
            }
        }
        // Porting Note: PyKeras with TensorFlow backend uses a symbolic loop
        //   (tf.while_loop). But for the imperative deeplearn.js backend, we just
        //   use the usual TypeScript control flow to iterate over the time steps in
        //   the inputs.
        // Porting Note: PyKeras patches a "_use_learning_phase" attribute to
        // outputs.
        //   This is not idiomatic in TypeScript. The info regarding whether we are
        //   in a learning (i.e., training) phase for RNN is passed in a different
        //   way.
        const perStepOutputs = [];
        let lastOutput;
        let states = initialStates;
        const timeSteps = inputs.shape[0];
        const perStepInputs = tfc.unstack(inputs);
        let perStepMasks;
        if (mask != null) {
            perStepMasks = tfc.unstack(mask);
        }
        for (let t = 0; t < timeSteps; ++t) {
            const currentInput = perStepInputs[t];
            const stepOutputs = tfc.tidy(() => stepFunction(currentInput, states));
            if (mask == null) {
                lastOutput = stepOutputs[0];
                states = stepOutputs[1];
            }
            else {
                const maskedOutputs = tfc.tidy(() => {
                    const stepMask = perStepMasks[t];
                    const negStepMask = tfc.sub(tfc.onesLike(stepMask), stepMask);
                    // TODO(cais): Would tfc.where() be better for performance?
                    const output = tfc.add(tfc.mul(stepOutputs[0], stepMask), tfc.mul(states[0], negStepMask));
                    const newStates = states.map((state, i) => {
                        return tfc.add(tfc.mul(stepOutputs[1][i], stepMask), tfc.mul(state, negStepMask));
                    });
                    return { output, newStates };
                });
                lastOutput = maskedOutputs.output;
                states = maskedOutputs.newStates;
            }
            if (needPerStepOutputs) {
                perStepOutputs.push(lastOutput);
            }
        }
        let outputs;
        if (needPerStepOutputs) {
            const axis = 1;
            outputs = tfc.stack(perStepOutputs, axis);
        }
        return [lastOutput, outputs, states];
    });
}
export class RNN extends Layer {
    constructor(args) {
        super(args);
        let cell;
        if (args.cell == null) {
            throw new ValueError('cell property is missing for the constructor of RNN.');
        }
        else if (Array.isArray(args.cell)) {
            cell = new StackedRNNCells({ cells: args.cell });
        }
        else {
            cell = args.cell;
        }
        if (cell.stateSize == null) {
            throw new ValueError('The RNN cell should have an attribute `stateSize` (tuple of ' +
                'integers, one integer per RNN state).');
        }
        this.cell = cell;
        this.returnSequences =
            args.returnSequences == null ? false : args.returnSequences;
        this.returnState = args.returnState == null ? false : args.returnState;
        this.goBackwards = args.goBackwards == null ? false : args.goBackwards;
        this._stateful = args.stateful == null ? false : args.stateful;
        this.unroll = args.unroll == null ? false : args.unroll;
        this.supportsMasking = true;
        this.inputSpec = [new InputSpec({ ndim: 3 })];
        this.stateSpec = null;
        this.states_ = null;
        // TODO(cais): Add constantsSpec and numConstants.
        this.numConstants = null;
        // TODO(cais): Look into the use of initial_state in the kwargs of the
        //   constructor.
        this.keptStates = [];
    }
    // Porting Note: This is the equivalent of `RNN.states` property getter in
    //   PyKeras.
    getStates() {
        if (this.states_ == null) {
            const numStates = Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
            return math_utils.range(0, numStates).map(x => null);
        }
        else {
            return this.states_;
        }
    }
    // Porting Note: This is the equivalent of the `RNN.states` property setter in
    //   PyKeras.
    setStates(states) {
        this.states_ = states;
    }
    computeOutputShape(inputShape) {
        if (isArrayOfShapes(inputShape)) {
            inputShape = inputShape[0];
        }
        inputShape = inputShape;
        // TODO(cais): Remove the casting once stacked RNN cells become supported.
        let stateSize = this.cell.stateSize;
        if (!Array.isArray(stateSize)) {
            stateSize = [stateSize];
        }
        const outputDim = stateSize[0];
        let outputShape;
        if (this.returnSequences) {
            outputShape = [inputShape[0], inputShape[1], outputDim];
        }
        else {
            outputShape = [inputShape[0], outputDim];
        }
        if (this.returnState) {
            const stateShape = [];
            for (const dim of stateSize) {
                stateShape.push([inputShape[0], dim]);
            }
            return [outputShape].concat(stateShape);
        }
        else {
            return outputShape;
        }
    }
    computeMask(inputs, mask) {
        return tfc.tidy(() => {
            if (Array.isArray(mask)) {
                mask = mask[0];
            }
            const outputMask = this.returnSequences ? mask : null;
            if (this.returnState) {
                const stateMask = this.states.map(s => null);
                return [outputMask].concat(stateMask);
            }
            else {
                return outputMask;
            }
        });
    }
    /**
     * Get the current state tensors of the RNN.
     *
     * If the state hasn't been set, return an array of `null`s of the correct
     * length.
     */
    get states() {
        if (this.states_ == null) {
            const numStates = Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
            const output = [];
            for (let i = 0; i < numStates; ++i) {
                output.push(null);
            }
            return output;
        }
        else {
            return this.states_;
        }
    }
    set states(s) {
        this.states_ = s;
    }
    build(inputShape) {
        // Note inputShape will be an Array of Shapes of initial states and
        // constants if these are passed in apply().
        const constantShape = null;
        if (this.numConstants != null) {
            throw new NotImplementedError('Constants support is not implemented in RNN yet.');
        }
        if (isArrayOfShapes(inputShape)) {
            inputShape = inputShape[0];
        }
        inputShape = inputShape;
        const batchSize = this.stateful ? inputShape[0] : null;
        const inputDim = inputShape.slice(2);
        this.inputSpec[0] = new InputSpec({ shape: [batchSize, null, ...inputDim] });
        // Allow cell (if RNNCell Layer) to build before we set or validate
        // stateSpec.
        const stepInputShape = [inputShape[0]].concat(inputShape.slice(2));
        if (constantShape != null) {
            throw new NotImplementedError('Constants support is not implemented in RNN yet.');
        }
        else {
            this.cell.build(stepInputShape);
        }
        // Set or validate stateSpec.
        let stateSize;
        if (Array.isArray(this.cell.stateSize)) {
            stateSize = this.cell.stateSize;
        }
        else {
            stateSize = [this.cell.stateSize];
        }
        if (this.stateSpec != null) {
            if (!util.arraysEqual(this.stateSpec.map(spec => spec.shape[spec.shape.length - 1]), stateSize)) {
                throw new ValueError(`An initialState was passed that is not compatible with ` +
                    `cell.stateSize. Received stateSpec=${this.stateSpec}; ` +
                    `However cell.stateSize is ${this.cell.stateSize}`);
            }
        }
        else {
            this.stateSpec =
                stateSize.map(dim => new InputSpec({ shape: [null, dim] }));
        }
        if (this.stateful) {
            this.resetStates();
        }
    }
    /**
     * Reset the state tensors of the RNN.
     *
     * If the `states` argument is `undefined` or `null`, will set the
     * state tensor(s) of the RNN to all-zero tensors of the appropriate
     * shape(s).
     *
     * If `states` is provided, will set the state tensors of the RNN to its
     * value.
     *
     * @param states Optional externally-provided initial states.
     * @param training Whether this call is done during training. For stateful
     *   RNNs, this affects whether the old states are kept or discarded. In
     *   particular, if `training` is `true`, the old states will be kept so
     *   that subsequent backpropgataion through time (BPTT) may work properly.
     *   Else, the old states will be discarded.
     */
    resetStates(states, training = false) {
        tidy(() => {
            if (!this.stateful) {
                throw new AttributeError('Cannot call resetStates() on an RNN Layer that is not stateful.');
            }
            const batchSize = this.inputSpec[0].shape[0];
            if (batchSize == null) {
                throw new ValueError('If an RNN is stateful, it needs to know its batch size. Specify ' +
                    'the batch size of your input tensors: \n' +
                    '- If using a Sequential model, specify the batch size by ' +
                    'passing a `batchInputShape` option to your first layer.\n' +
                    '- If using the functional API, specify the batch size by ' +
                    'passing a `batchShape` option to your Input layer.');
            }
            // Initialize state if null.
            if (this.states_ == null) {
                if (Array.isArray(this.cell.stateSize)) {
                    this.states_ =
                        this.cell.stateSize.map(dim => tfc.zeros([batchSize, dim]));
                }
                else {
                    this.states_ = [tfc.zeros([batchSize, this.cell.stateSize])];
                }
            }
            else if (states == null) {
                // Dispose old state tensors.
                tfc.dispose(this.states_);
                // For stateful RNNs, fully dispose kept old states.
                if (this.keptStates != null) {
                    tfc.dispose(this.keptStates);
                    this.keptStates = [];
                }
                if (Array.isArray(this.cell.stateSize)) {
                    this.states_ =
                        this.cell.stateSize.map(dim => tfc.zeros([batchSize, dim]));
                }
                else {
                    this.states_[0] = tfc.zeros([batchSize, this.cell.stateSize]);
                }
            }
            else {
                if (!Array.isArray(states)) {
                    states = [states];
                }
                if (states.length !== this.states_.length) {
                    throw new ValueError(`Layer ${this.name} expects ${this.states_.length} state(s), ` +
                        `but it received ${states.length} state value(s). Input ` +
                        `received: ${states}`);
                }
                if (training === true) {
                    // Store old state tensors for complete disposal later, i.e., during
                    // the next no-arg call to this method. We do not dispose the old
                    // states immediately because that BPTT (among other things) require
                    // them.
                    this.keptStates.push(this.states_.slice());
                }
                else {
                    tfc.dispose(this.states_);
                }
                for (let index = 0; index < this.states_.length; ++index) {
                    const value = states[index];
                    const dim = Array.isArray(this.cell.stateSize) ?
                        this.cell.stateSize[index] :
                        this.cell.stateSize;
                    const expectedShape = [batchSize, dim];
                    if (!util.arraysEqual(value.shape, expectedShape)) {
                        throw new ValueError(`State ${index} is incompatible with layer ${this.name}: ` +
                            `expected shape=${expectedShape}, received shape=${value.shape}`);
                    }
                    this.states_[index] = value;
                }
            }
            this.states_ = this.states_.map(state => tfc.keep(state.clone()));
        });
    }
    apply(inputs, kwargs) {
        // TODO(cais): Figure out whether initialState is in kwargs or inputs.
        let initialState = kwargs == null ? null : kwargs['initialState'];
        let constants = kwargs == null ? null : kwargs['constants'];
        if (kwargs == null) {
            kwargs = {};
        }
        const standardized = standardizeArgs(inputs, initialState, constants, this.numConstants);
        inputs = standardized.inputs;
        initialState = standardized.initialState;
        constants = standardized.constants;
        // If any of `initial_state` or `constants` are specified and are
        // `tf.SymbolicTensor`s, then add them to the inputs and temporarily modify
        // the input_spec to include them.
        let additionalInputs = [];
        let additionalSpecs = [];
        if (initialState != null) {
            kwargs['initialState'] = initialState;
            additionalInputs = additionalInputs.concat(initialState);
            this.stateSpec = [];
            for (const state of initialState) {
                this.stateSpec.push(new InputSpec({ shape: state.shape }));
            }
            // TODO(cais): Use the following instead.
            // this.stateSpec = initialState.map(state => new InputSpec({shape:
            // state.shape}));
            additionalSpecs = additionalSpecs.concat(this.stateSpec);
        }
        if (constants != null) {
            kwargs['constants'] = constants;
            additionalInputs = additionalInputs.concat(constants);
            // TODO(cais): Add this.constantsSpec.
            this.numConstants = constants.length;
        }
        const isTensor = additionalInputs[0] instanceof SymbolicTensor;
        if (isTensor) {
            // Compute full input spec, including state and constants.
            const fullInput = [inputs].concat(additionalInputs);
            const fullInputSpec = this.inputSpec.concat(additionalSpecs);
            // Perform the call with temporarily replaced inputSpec.
            const originalInputSpec = this.inputSpec;
            this.inputSpec = fullInputSpec;
            const output = super.apply(fullInput, kwargs);
            this.inputSpec = originalInputSpec;
            return output;
        }
        else {
            return super.apply(inputs, kwargs);
        }
    }
    // tslint:disable-next-line:no-any
    call(inputs, kwargs) {
        // Input shape: `[samples, time (padded with zeros), input_dim]`.
        // Note that the .build() method of subclasses **must** define
        // this.inputSpec and this.stateSpec owith complete input shapes.
        return tidy(() => {
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            let initialState = kwargs == null ? null : kwargs['initialState'];
            inputs = getExactlyOneTensor(inputs);
            if (initialState == null) {
                if (this.stateful) {
                    initialState = this.states_;
                }
                else {
                    initialState = this.getInitialState(inputs);
                }
            }
            const numStates = Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
            if (initialState.length !== numStates) {
                throw new ValueError(`RNN Layer has ${numStates} state(s) but was passed ` +
                    `${initialState.length} initial state(s).`);
            }
            if (this.unroll) {
                console.warn('Ignoring unroll = true for RNN layer, due to imperative backend.');
            }
            const cellCallKwargs = { training };
            // TODO(cais): Add support for constants.
            const step = (inputs, states) => {
                // `inputs` and `states` are concatenated to form a single `Array` of
                // `tf.Tensor`s as the input to `cell.call()`.
                const outputs = this.cell.call([inputs].concat(states), cellCallKwargs);
                // Marshall the return value into output and new states.
                return [outputs[0], outputs.slice(1)];
            };
            // TODO(cais): Add support for constants.
            const rnnOutputs = rnn(step, inputs, initialState, this.goBackwards, mask, null, this.unroll, this.returnSequences);
            const lastOutput = rnnOutputs[0];
            const outputs = rnnOutputs[1];
            const states = rnnOutputs[2];
            if (this.stateful) {
                this.resetStates(states, training);
            }
            const output = this.returnSequences ? outputs : lastOutput;
            // TODO(cais): Porperty set learning phase flag.
            if (this.returnState) {
                return [output].concat(states);
            }
            else {
                return output;
            }
        });
    }
    getInitialState(inputs) {
        return tidy(() => {
            // Build an all-zero tensor of shape [samples, outputDim].
            // [Samples, timeSteps, inputDim].
            let initialState = tfc.zeros(inputs.shape);
            // [Samples].
            initialState = tfc.sum(initialState, [1, 2]);
            initialState = K.expandDims(initialState); // [Samples, 1].
            if (Array.isArray(this.cell.stateSize)) {
                return this.cell.stateSize.map(dim => dim > 1 ? K.tile(initialState, [1, dim]) : initialState);
            }
            else {
                return this.cell.stateSize > 1 ?
                    [K.tile(initialState, [1, this.cell.stateSize])] :
                    [initialState];
            }
        });
    }
    get trainableWeights() {
        if (!this.trainable) {
            return [];
        }
        // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
        return this.cell.trainableWeights;
    }
    get nonTrainableWeights() {
        // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
        if (!this.trainable) {
            return this.cell.weights;
        }
        return this.cell.nonTrainableWeights;
    }
    setFastWeightInitDuringBuild(value) {
        super.setFastWeightInitDuringBuild(value);
        if (this.cell != null) {
            this.cell.setFastWeightInitDuringBuild(value);
        }
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            returnSequences: this.returnSequences,
            returnState: this.returnState,
            goBackwards: this.goBackwards,
            stateful: this.stateful,
            unroll: this.unroll,
        };
        if (this.numConstants != null) {
            config['numConstants'] = this.numConstants;
        }
        const cellConfig = this.cell.getConfig();
        if (this.getClassName() === RNN.className) {
            config['cell'] = {
                'className': this.cell.getClassName(),
                'config': cellConfig,
            };
        }
        // this order is necessary, to prevent cell name from replacing layer name
        return Object.assign({}, cellConfig, baseConfig, config);
    }
    /** @nocollapse */
    static fromConfig(cls, config, customObjects = {}) {
        const cellConfig = config['cell'];
        const cell = deserialize(cellConfig, customObjects);
        return new cls(Object.assign(config, { cell }));
    }
}
/** @nocollapse */
RNN.className = 'RNN';
serialization.registerClass(RNN);
// Porting Note: This is a common parent class for RNN cells. There is no
// equivalent of this in PyKeras. Having a common parent class forgoes the
//  need for `has_attr(cell, ...)` checks or its TypeScript equivalent.
/**
 * An RNNCell layer.
 *
 * @doc {heading: 'Layers', subheading: 'Classes'}
 */
export class RNNCell extends Layer {
}
export class SimpleRNNCell extends RNNCell {
    constructor(args) {
        super(args);
        this.DEFAULT_ACTIVATION = 'tanh';
        this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        this.DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
        this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        this.units = args.units;
        assertPositiveInteger(this.units, `units`);
        this.activation = getActivation(args.activation == null ? this.DEFAULT_ACTIVATION : args.activation);
        this.useBias = args.useBias == null ? true : args.useBias;
        this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
        this.recurrentInitializer = getInitializer(args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);
        this.biasInitializer =
            getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
        this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
        this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
        this.biasRegularizer = getRegularizer(args.biasRegularizer);
        this.kernelConstraint = getConstraint(args.kernelConstraint);
        this.recurrentConstraint = getConstraint(args.recurrentConstraint);
        this.biasConstraint = getConstraint(args.biasConstraint);
        this.dropout = math_utils.min([1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
        this.recurrentDropout = math_utils.min([
            1,
            math_utils.max([0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
        ]);
        this.dropoutFunc = args.dropoutFunc;
        this.stateSize = this.units;
        this.dropoutMask = null;
        this.recurrentDropoutMask = null;
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        // TODO(cais): Use regularizer.
        this.kernel = this.addWeight('kernel', [inputShape[inputShape.length - 1], this.units], null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        this.recurrentKernel = this.addWeight('recurrent_kernel', [this.units, this.units], null, this.recurrentInitializer, this.recurrentRegularizer, true, this.recurrentConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.units], null, this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        this.built = true;
    }
    // Porting Note: PyKeras' equivalent of this method takes two tensor inputs:
    //   `inputs` and `states`. Here, the two tensors are combined into an
    //   `Tensor[]` Array as the first input argument.
    //   Similarly, PyKeras' equivalent of this method returns two values:
    //    `output` and `[output]`. Here the two are combined into one length-2
    //    `Tensor[]`, consisting of `output` repeated.
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = inputs;
            if (inputs.length !== 2) {
                throw new ValueError(`SimpleRNNCell expects 2 input Tensors, got ${inputs.length}.`);
            }
            let prevOutput = inputs[1];
            inputs = inputs[0];
            const training = kwargs['training'] == null ? false : kwargs['training'];
            if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
                this.dropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(inputs),
                    rate: this.dropout,
                    training,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
                this.recurrentDropoutMask == null) {
                this.recurrentDropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(prevOutput),
                    rate: this.recurrentDropout,
                    training,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            let h;
            const dpMask = this.dropoutMask;
            const recDpMask = this.recurrentDropoutMask;
            if (dpMask != null) {
                h = K.dot(tfc.mul(inputs, dpMask), this.kernel.read());
            }
            else {
                h = K.dot(inputs, this.kernel.read());
            }
            if (this.bias != null) {
                h = K.biasAdd(h, this.bias.read());
            }
            if (recDpMask != null) {
                prevOutput = tfc.mul(prevOutput, recDpMask);
            }
            let output = tfc.add(h, K.dot(prevOutput, this.recurrentKernel.read()));
            if (this.activation != null) {
                output = this.activation.apply(output);
            }
            // TODO(cais): Properly set learning phase on output tensor?
            return [output, output];
        });
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            units: this.units,
            activation: serializeActivation(this.activation),
            useBias: this.useBias,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            recurrentInitializer: serializeInitializer(this.recurrentInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint),
            recurrentConstraint: serializeConstraint(this.recurrentConstraint),
            biasConstraint: serializeConstraint(this.biasConstraint),
            dropout: this.dropout,
            recurrentDropout: this.recurrentDropout,
        };
        return Object.assign({}, baseConfig, config);
    }
}
/** @nocollapse */
SimpleRNNCell.className = 'SimpleRNNCell';
serialization.registerClass(SimpleRNNCell);
export class SimpleRNN extends RNN {
    constructor(args) {
        args.cell = new SimpleRNNCell(args);
        super(args);
        // TODO(cais): Add activityRegularizer.
    }
    call(inputs, kwargs) {
        return tidy(() => {
            if (this.cell.dropoutMask != null) {
                tfc.dispose(this.cell.dropoutMask);
                this.cell.dropoutMask = null;
            }
            if (this.cell.recurrentDropoutMask != null) {
                tfc.dispose(this.cell.recurrentDropoutMask);
                this.cell.recurrentDropoutMask = null;
            }
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            const initialState = kwargs == null ? null : kwargs['initialState'];
            return super.call(inputs, { mask, training, initialState });
        });
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config);
    }
}
/** @nocollapse */
SimpleRNN.className = 'SimpleRNN';
serialization.registerClass(SimpleRNN);
export class GRUCell extends RNNCell {
    constructor(args) {
        super(args);
        this.DEFAULT_ACTIVATION = 'tanh';
        this.DEFAULT_RECURRENT_ACTIVATION = 'hardSigmoid';
        this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        this.DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
        this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        if (args.resetAfter) {
            throw new ValueError(`GRUCell does not support reset_after parameter set to true.`);
        }
        this.units = args.units;
        assertPositiveInteger(this.units, 'units');
        this.activation = getActivation(args.activation === undefined ? this.DEFAULT_ACTIVATION :
            args.activation);
        this.recurrentActivation = getActivation(args.recurrentActivation === undefined ?
            this.DEFAULT_RECURRENT_ACTIVATION :
            args.recurrentActivation);
        this.useBias = args.useBias == null ? true : args.useBias;
        this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
        this.recurrentInitializer = getInitializer(args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);
        this.biasInitializer =
            getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
        this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
        this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
        this.biasRegularizer = getRegularizer(args.biasRegularizer);
        this.kernelConstraint = getConstraint(args.kernelConstraint);
        this.recurrentConstraint = getConstraint(args.recurrentConstraint);
        this.biasConstraint = getConstraint(args.biasConstraint);
        this.dropout = math_utils.min([1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
        this.recurrentDropout = math_utils.min([
            1,
            math_utils.max([0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
        ]);
        this.dropoutFunc = args.dropoutFunc;
        this.implementation = args.implementation;
        this.stateSize = this.units;
        this.dropoutMask = null;
        this.recurrentDropoutMask = null;
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const inputDim = inputShape[inputShape.length - 1];
        this.kernel = this.addWeight('kernel', [inputDim, this.units * 3], null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        this.recurrentKernel = this.addWeight('recurrent_kernel', [this.units, this.units * 3], null, this.recurrentInitializer, this.recurrentRegularizer, true, this.recurrentConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.units * 3], null, this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        // Porting Notes: Unlike the PyKeras implementation, we perform slicing
        //   of the weights and bias in the call() method, at execution time.
        this.built = true;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = inputs;
            if (inputs.length !== 2) {
                throw new ValueError(`GRUCell expects 2 input Tensors (inputs, h, c), got ` +
                    `${inputs.length}.`);
            }
            const training = kwargs['training'] == null ? false : kwargs['training'];
            let hTMinus1 = inputs[1]; // Previous memory state.
            inputs = inputs[0];
            // Note: For superior performance, TensorFlow.js always uses
            // implementation 2, regardless of the actual value of
            // config.implementation.
            if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
                this.dropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(inputs),
                    rate: this.dropout,
                    training,
                    count: 3,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
                this.recurrentDropoutMask == null) {
                this.recurrentDropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(hTMinus1),
                    rate: this.recurrentDropout,
                    training,
                    count: 3,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            const dpMask = this.dropoutMask;
            const recDpMask = this.recurrentDropoutMask;
            let z;
            let r;
            let hh;
            if (0 < this.dropout && this.dropout < 1) {
                inputs = tfc.mul(inputs, dpMask[0]);
            }
            let matrixX = K.dot(inputs, this.kernel.read());
            if (this.useBias) {
                matrixX = K.biasAdd(matrixX, this.bias.read());
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1) {
                hTMinus1 = tfc.mul(hTMinus1, recDpMask[0]);
            }
            const recurrentKernelValue = this.recurrentKernel.read();
            const [rk1, rk2] = tfc.split(recurrentKernelValue, [2 * this.units, this.units], recurrentKernelValue.rank - 1);
            const matrixInner = K.dot(hTMinus1, rk1);
            const [xZ, xR, xH] = tfc.split(matrixX, 3, matrixX.rank - 1);
            const [recurrentZ, recurrentR] = tfc.split(matrixInner, 2, matrixInner.rank - 1);
            z = this.recurrentActivation.apply(tfc.add(xZ, recurrentZ));
            r = this.recurrentActivation.apply(tfc.add(xR, recurrentR));
            const recurrentH = K.dot(tfc.mul(r, hTMinus1), rk2);
            hh = this.activation.apply(tfc.add(xH, recurrentH));
            const h = tfc.add(tfc.mul(z, hTMinus1), tfc.mul(tfc.add(1, tfc.neg(z)), hh));
            // TODO(cais): Add use_learning_phase flag properly.
            return [h, h];
        });
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            units: this.units,
            activation: serializeActivation(this.activation),
            recurrentActivation: serializeActivation(this.recurrentActivation),
            useBias: this.useBias,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            recurrentInitializer: serializeInitializer(this.recurrentInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint),
            recurrentConstraint: serializeConstraint(this.recurrentConstraint),
            biasConstraint: serializeConstraint(this.biasConstraint),
            dropout: this.dropout,
            recurrentDropout: this.recurrentDropout,
            implementation: this.implementation,
            resetAfter: false
        };
        return Object.assign({}, baseConfig, config);
    }
}
/** @nocollapse */
GRUCell.className = 'GRUCell';
serialization.registerClass(GRUCell);
export class GRU extends RNN {
    constructor(args) {
        if (args.implementation === 0) {
            console.warn('`implementation=0` has been deprecated, and now defaults to ' +
                '`implementation=1`. Please update your layer call.');
        }
        args.cell = new GRUCell(args);
        super(args);
        // TODO(cais): Add activityRegularizer.
    }
    call(inputs, kwargs) {
        return tidy(() => {
            if (this.cell.dropoutMask != null) {
                tfc.dispose(this.cell.dropoutMask);
                this.cell.dropoutMask = null;
            }
            if (this.cell.recurrentDropoutMask != null) {
                tfc.dispose(this.cell.recurrentDropoutMask);
                this.cell.recurrentDropoutMask = null;
            }
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            const initialState = kwargs == null ? null : kwargs['initialState'];
            return super.call(inputs, { mask, training, initialState });
        });
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        if (config['implmentation'] === 0) {
            config['implementation'] = 1;
        }
        return new cls(config);
    }
}
/** @nocollapse */
GRU.className = 'GRU';
serialization.registerClass(GRU);
export class LSTMCell extends RNNCell {
    constructor(args) {
        super(args);
        this.DEFAULT_ACTIVATION = 'tanh';
        this.DEFAULT_RECURRENT_ACTIVATION = 'hardSigmoid';
        this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        this.DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
        this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        this.units = args.units;
        assertPositiveInteger(this.units, 'units');
        this.activation = getActivation(args.activation === undefined ? this.DEFAULT_ACTIVATION :
            args.activation);
        this.recurrentActivation = getActivation(args.recurrentActivation === undefined ?
            this.DEFAULT_RECURRENT_ACTIVATION :
            args.recurrentActivation);
        this.useBias = args.useBias == null ? true : args.useBias;
        this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
        this.recurrentInitializer = getInitializer(args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);
        this.biasInitializer =
            getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
        this.unitForgetBias = args.unitForgetBias;
        this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
        this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
        this.biasRegularizer = getRegularizer(args.biasRegularizer);
        this.kernelConstraint = getConstraint(args.kernelConstraint);
        this.recurrentConstraint = getConstraint(args.recurrentConstraint);
        this.biasConstraint = getConstraint(args.biasConstraint);
        this.dropout = math_utils.min([1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
        this.recurrentDropout = math_utils.min([
            1,
            math_utils.max([0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
        ]);
        this.dropoutFunc = args.dropoutFunc;
        this.implementation = args.implementation;
        this.stateSize = [this.units, this.units];
        this.dropoutMask = null;
        this.recurrentDropoutMask = null;
    }
    build(inputShape) {
        var _a;
        inputShape = getExactlyOneShape(inputShape);
        const inputDim = inputShape[inputShape.length - 1];
        this.kernel = this.addWeight('kernel', [inputDim, this.units * 4], null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        this.recurrentKernel = this.addWeight('recurrent_kernel', [this.units, this.units * 4], null, this.recurrentInitializer, this.recurrentRegularizer, true, this.recurrentConstraint);
        let biasInitializer;
        if (this.useBias) {
            if (this.unitForgetBias) {
                const capturedBiasInit = this.biasInitializer;
                const capturedUnits = this.units;
                biasInitializer = new (_a = class CustomInit extends Initializer {
                        apply(shape, dtype) {
                            // TODO(cais): More informative variable names?
                            const bI = capturedBiasInit.apply([capturedUnits]);
                            const bF = (new Ones()).apply([capturedUnits]);
                            const bCAndH = capturedBiasInit.apply([capturedUnits * 2]);
                            return K.concatAlongFirstAxis(K.concatAlongFirstAxis(bI, bF), bCAndH);
                        }
                    },
                    /** @nocollapse */
                    _a.className = 'CustomInit',
                    _a)();
            }
            else {
                biasInitializer = this.biasInitializer;
            }
            this.bias = this.addWeight('bias', [this.units * 4], null, biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        // Porting Notes: Unlike the PyKeras implementation, we perform slicing
        //   of the weights and bias in the call() method, at execution time.
        this.built = true;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const training = kwargs['training'] == null ? false : kwargs['training'];
            inputs = inputs;
            if (inputs.length !== 3) {
                throw new ValueError(`LSTMCell expects 3 input Tensors (inputs, h, c), got ` +
                    `${inputs.length}.`);
            }
            let hTMinus1 = inputs[1]; // Previous memory state.
            const cTMinus1 = inputs[2]; // Previous carry state.
            inputs = inputs[0];
            if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
                this.dropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(inputs),
                    rate: this.dropout,
                    training,
                    count: 4,
                    dropoutFunc: this.dropoutFunc
                });
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
                this.recurrentDropoutMask == null) {
                this.recurrentDropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(hTMinus1),
                    rate: this.recurrentDropout,
                    training,
                    count: 4,
                    dropoutFunc: this.dropoutFunc
                });
            }
            const dpMask = this.dropoutMask;
            const recDpMask = this.recurrentDropoutMask;
            // Note: For superior performance, TensorFlow.js always uses
            // implementation 2 regardless of the actual value of
            // config.implementation.
            let i;
            let f;
            let c;
            let o;
            if (0 < this.dropout && this.dropout < 1) {
                inputs = tfc.mul(inputs, dpMask[0]);
            }
            let z = K.dot(inputs, this.kernel.read());
            if (0 < this.recurrentDropout && this.recurrentDropout < 1) {
                hTMinus1 = tfc.mul(hTMinus1, recDpMask[0]);
            }
            z = tfc.add(z, K.dot(hTMinus1, this.recurrentKernel.read()));
            if (this.useBias) {
                z = K.biasAdd(z, this.bias.read());
            }
            const [z0, z1, z2, z3] = tfc.split(z, 4, z.rank - 1);
            i = this.recurrentActivation.apply(z0);
            f = this.recurrentActivation.apply(z1);
            c = tfc.add(tfc.mul(f, cTMinus1), tfc.mul(i, this.activation.apply(z2)));
            o = this.recurrentActivation.apply(z3);
            const h = tfc.mul(o, this.activation.apply(c));
            // TODO(cais): Add use_learning_phase flag properly.
            return [h, h, c];
        });
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            units: this.units,
            activation: serializeActivation(this.activation),
            recurrentActivation: serializeActivation(this.recurrentActivation),
            useBias: this.useBias,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            recurrentInitializer: serializeInitializer(this.recurrentInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
            unitForgetBias: this.unitForgetBias,
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint),
            recurrentConstraint: serializeConstraint(this.recurrentConstraint),
            biasConstraint: serializeConstraint(this.biasConstraint),
            dropout: this.dropout,
            recurrentDropout: this.recurrentDropout,
            implementation: this.implementation,
        };
        return Object.assign({}, baseConfig, config);
    }
}
/** @nocollapse */
LSTMCell.className = 'LSTMCell';
serialization.registerClass(LSTMCell);
export class LSTM extends RNN {
    constructor(args) {
        if (args.implementation === 0) {
            console.warn('`implementation=0` has been deprecated, and now defaults to ' +
                '`implementation=1`. Please update your layer call.');
        }
        args.cell = new LSTMCell(args);
        super(args);
        // TODO(cais): Add activityRegularizer.
    }
    call(inputs, kwargs) {
        return tidy(() => {
            if (this.cell.dropoutMask != null) {
                tfc.dispose(this.cell.dropoutMask);
                this.cell.dropoutMask = null;
            }
            if (this.cell.recurrentDropoutMask != null) {
                tfc.dispose(this.cell.recurrentDropoutMask);
                this.cell.recurrentDropoutMask = null;
            }
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            const initialState = kwargs == null ? null : kwargs['initialState'];
            return super.call(inputs, { mask, training, initialState });
        });
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        if (config['implmentation'] === 0) {
            config['implementation'] = 1;
        }
        return new cls(config);
    }
}
/** @nocollapse */
LSTM.className = 'LSTM';
serialization.registerClass(LSTM);
export class StackedRNNCells extends RNNCell {
    constructor(args) {
        super(args);
        this.cells = args.cells;
    }
    get stateSize() {
        // States are a flat list in reverse order of the cell stack.
        // This allows perserving the requirement `stack.statesize[0] ===
        // outputDim`. E.g., states of a 2-layer LSTM would be `[h2, c2, h1, c1]`,
        // assuming one LSTM has states `[h, c]`.
        const stateSize = [];
        for (const cell of this.cells.slice().reverse()) {
            if (Array.isArray(cell.stateSize)) {
                stateSize.push(...cell.stateSize);
            }
            else {
                stateSize.push(cell.stateSize);
            }
        }
        return stateSize;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = inputs;
            let states = inputs.slice(1);
            // Recover per-cell states.
            const nestedStates = [];
            for (const cell of this.cells.slice().reverse()) {
                if (Array.isArray(cell.stateSize)) {
                    nestedStates.push(states.splice(0, cell.stateSize.length));
                }
                else {
                    nestedStates.push(states.splice(0, 1));
                }
            }
            nestedStates.reverse();
            // Call the cells in order and store the returned states.
            const newNestedStates = [];
            let callInputs;
            for (let i = 0; i < this.cells.length; ++i) {
                const cell = this.cells[i];
                states = nestedStates[i];
                // TODO(cais): Take care of constants.
                if (i === 0) {
                    callInputs = [inputs[0]].concat(states);
                }
                else {
                    callInputs = [callInputs[0]].concat(states);
                }
                callInputs = cell.call(callInputs, kwargs);
                newNestedStates.push(callInputs.slice(1));
            }
            // Format the new states as a flat list in reverse cell order.
            states = [];
            for (const cellStates of newNestedStates.slice().reverse()) {
                states.push(...cellStates);
            }
            return [callInputs[0]].concat(states);
        });
    }
    build(inputShape) {
        if (isArrayOfShapes(inputShape)) {
            // TODO(cais): Take care of input constants.
            // const constantShape = inputShape.slice(1);
            inputShape = inputShape[0];
        }
        inputShape = inputShape;
        let outputDim;
        this.cells.forEach((cell, i) => {
            nameScope(`RNNCell_${i}`, () => {
                // TODO(cais): Take care of input constants.
                cell.build(inputShape);
                if (Array.isArray(cell.stateSize)) {
                    outputDim = cell.stateSize[0];
                }
                else {
                    outputDim = cell.stateSize;
                }
                inputShape = [inputShape[0], outputDim];
            });
        });
        this.built = true;
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const getCellConfig = (cell) => {
            return {
                'className': cell.getClassName(),
                'config': cell.getConfig(),
            };
        };
        const cellConfigs = this.cells.map(getCellConfig);
        const config = { 'cells': cellConfigs };
        return Object.assign({}, baseConfig, config);
    }
    /** @nocollapse */
    static fromConfig(cls, config, customObjects = {}) {
        const cells = [];
        for (const cellConfig of config['cells']) {
            cells.push(deserialize(cellConfig, customObjects));
        }
        return new cls({ cells });
    }
    get trainableWeights() {
        if (!this.trainable) {
            return [];
        }
        const weights = [];
        for (const cell of this.cells) {
            weights.push(...cell.trainableWeights);
        }
        return weights;
    }
    get nonTrainableWeights() {
        const weights = [];
        for (const cell of this.cells) {
            weights.push(...cell.nonTrainableWeights);
        }
        if (!this.trainable) {
            const trainableWeights = [];
            for (const cell of this.cells) {
                trainableWeights.push(...cell.trainableWeights);
            }
            return trainableWeights.concat(weights);
        }
        return weights;
    }
    /**
     * Retrieve the weights of a the model.
     *
     * @returns A flat `Array` of `tf.Tensor`s.
     */
    getWeights() {
        const weights = [];
        for (const cell of this.cells) {
            weights.push(...cell.weights);
        }
        return batchGetValue(weights);
    }
    /**
     * Set the weights of the model.
     *
     * @param weights An `Array` of `tf.Tensor`s with shapes and types matching
     *     the output of `getWeights()`.
     */
    setWeights(weights) {
        const tuples = [];
        for (const cell of this.cells) {
            const numParams = cell.weights.length;
            const inputWeights = weights.splice(numParams);
            for (let i = 0; i < cell.weights.length; ++i) {
                tuples.push([cell.weights[i], inputWeights[i]]);
            }
        }
        batchSetValue(tuples);
    }
}
/** @nocollapse */
StackedRNNCells.className = 'StackedRNNCells';
serialization.registerClass(StackedRNNCells);
export function generateDropoutMask(args) {
    const { ones, rate, training = false, count = 1, dropoutFunc } = args;
    const droppedInputs = () => dropoutFunc != null ? dropoutFunc(ones(), rate) : K.dropout(ones(), rate);
    const createMask = () => K.inTrainPhase(droppedInputs, ones, training);
    // just in case count is provided with null or undefined
    if (!count || count <= 1) {
        return tfc.keep(createMask().clone());
    }
    const masks = Array(count).fill(undefined).map(createMask);
    return masks.map(m => tfc.keep(m.clone()));
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVjdXJyZW50LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL2xheWVycy9yZWN1cnJlbnQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSDs7R0FFRztBQUVILE9BQU8sS0FBSyxHQUFHLE1BQU0sdUJBQXVCLENBQUM7QUFDN0MsT0FBTyxFQUFXLGFBQWEsRUFBVSxJQUFJLEVBQUUsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFbEYsT0FBTyxFQUFhLGFBQWEsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQzlFLE9BQU8sS0FBSyxDQUFDLE1BQU0seUJBQXlCLENBQUM7QUFDN0MsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNwQyxPQUFPLEVBQW1DLGFBQWEsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQ3BHLE9BQU8sRUFBQyxTQUFTLEVBQUUsY0FBYyxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDN0QsT0FBTyxFQUFDLEtBQUssRUFBWSxNQUFNLG9CQUFvQixDQUFDO0FBQ3BELE9BQU8sRUFBQyxjQUFjLEVBQUUsbUJBQW1CLEVBQUUsVUFBVSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQzFFLE9BQU8sRUFBQyxjQUFjLEVBQUUsV0FBVyxFQUF5QixJQUFJLEVBQUUsb0JBQW9CLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUcvRyxPQUFPLEVBQUMsY0FBYyxFQUFzQyxvQkFBb0IsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBRXpHLE9BQU8sRUFBQyxxQkFBcUIsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQzdELE9BQU8sS0FBSyxVQUFVLE1BQU0scUJBQXFCLENBQUM7QUFDbEQsT0FBTyxFQUFDLGtCQUFrQixFQUFFLG1CQUFtQixFQUFFLGVBQWUsRUFBQyxNQUFNLHNCQUFzQixDQUFD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QUFDO0lBRXZFLHdEQUF3RDtJQUN4RCxJQUFJLENBQUMsS0FBSyxJQUFJLEtBQUssSUFBSSxDQUFDLEVBQUU7UUFDeEIsT0FBTyxHQUFHLENBQUMsSUFBSSxDQUFDLFVBQVUsRUFBRSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDdkM7SUFFRCxNQUFNLEtBQUssR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxVQUFVLENBQUMsQ0FBQztJQUUzRCxPQUFPLEtBQUssQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLENBQUM7QUFDN0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogVGVuc29yRmxvdy5qcyBMYXllcnM6IFJlY3VycmVudCBOZXVyYWwgTmV0d29yayBMYXllcnMuXG4gKi9cblxuaW1wb3J0ICogYXMgdGZjIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge0RhdGFUeXBlLCBzZXJpYWxpemF0aW9uLCBUZW5zb3IsIHRpZHksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QWN0aXZhdGlvbiwgZ2V0QWN0aXZhdGlvbiwgc2VyaWFsaXplQWN0aXZhdGlvbn0gZnJvbSAnLi4vYWN0aXZhdGlvbnMnO1xuaW1wb3J0ICogYXMgSyBmcm9tICcuLi9iYWNrZW5kL3RmanNfYmFja2VuZCc7XG5pbXBvcnQge25hbWVTY29wZX0gZnJvbSAnLi4vY29tbW9uJztcbmltcG9ydCB7Q29uc3RyYWludCwgQ29uc3RyYWludElkZW50aWZpZXIsIGdldENvbnN0cmFpbnQsIHNlcmlhbGl6ZUNvbnN0cmFpbnR9IGZyb20gJy4uL2NvbnN0cmFpbnRzJztcbmltcG9ydCB7SW5wdXRTcGVjLCBTeW1ib2xpY1RlbnNvcn0gZnJvbSAnLi4vZW5naW5lL3RvcG9sb2d5JztcbmltcG9ydCB7TGF5ZXIsIExheWVyQXJnc30gZnJvbSAnLi4vZW5naW5lL3RvcG9sb2d5JztcbmltcG9ydCB7QXR0cmlidXRlRXJyb3IsIE5vdEltcGxlbWVudGVkRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge2dldEluaXRpYWxpemVyLCBJbml0aWFsaXplciwgSW5pdGlhbGl6ZXJJZGVudGlmaWVyLCBPbmVzLCBzZXJpYWxpemVJbml0aWFsaXplcn0gZnJvbSAnLi4vaW5pdGlhbGl6ZXJzJztcbmltcG9ydCB7QWN0aXZhdGlvbklkZW50aWZpZXJ9IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9hY3RpdmF0aW9uX2NvbmZpZyc7XG5pbXBvcnQge1NoYXBlfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvY29tbW9uJztcbmltcG9ydCB7Z2V0UmVndWxhcml6ZXIsIFJlZ3VsYXJpemVyLCBSZWd1bGFyaXplcklkZW50aWZpZXIsIHNlcmlhbGl6ZVJlZ3VsYXJpemVyfSBmcm9tICcuLi9yZWd1bGFyaXplcnMnO1xuaW1wb3J0IHtLd2FyZ3MsIFJublN0ZXBGdW5jdGlvbn0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHthc3NlcnRQb3NpdGl2ZUludGVnZXJ9IGZyb20gJy4uL3V0aWxzL2dlbmVyaWNfdXRpbHMnO1xuaW1wb3J0ICogYXMgbWF0aF91dGlscyBmcm9tICcuLi91dGlscy9tYXRoX3V0aWxzJztcbmltcG9ydCB7Z2V0RXhhY3RseU9uZVNoYXBlLCBnZXRFeGFjdGx5T25lVGVuc29yLCBpc0FycmF5T2ZTaGFwZXN9IGZyb20gJy4uL3V0aWxzL3R5cGVzX3V0aWxzJztcbmltcG9ydCB7YmF0Y2hHZXRWYWx1ZSwgYmF0Y2hTZXRWYWx1ZSwgTGF5ZXJWYXJpYWJsZX0gZnJvbSAnLi4vdmFyaWFibGVzJztcblxuaW1wb3J0IHtkZXNlcmlhbGl6ZX0gZnJvbSAnLi9zZXJpYWxpemF0aW9uJztcblxuLyoqXG4gKiBTdGFuZGFyZGl6ZSBgYXBwbHkoKWAgYXJncyB0byBhIHNpbmdsZSBsaXN0IG9mIHRlbnNvciBpbnB1dHMuXG4gKlxuICogV2hlbiBydW5uaW5nIGEgbW9kZWwgbG9hZGVkIGZyb20gZmlsZSwgdGhlIGlucHV0IHRlbnNvcnMgYGluaXRpYWxTdGF0ZWAgYW5kXG4gKiBgY29uc3RhbnRzYCBhcmUgcGFzc2VkIHRvIGBSTk4uYXBwbHkoKWAgYXMgcGFydCBvZiBgaW5wdXRzYCBpbnN0ZWFkIG9mIHRoZVxuICogZGVkaWNhdGVkIGt3YXJncyBmaWVsZHMuIGBpbnB1dHNgIGNvbnNpc3RzIG9mXG4gKiBgW2lucHV0cywgaW5pdGlhbFN0YXRlMCwgaW5pdGlhbFN0YXRlMSwgLi4uLCBjb25zdGFudDAsIGNvbnN0YW50MV1gIGluIHRoaXNcbiAqIGNhc2UuXG4gKiBUaGlzIG1ldGhvZCBtYWtlcyBzdXJlIHRoYXQgYXJndW1lbnRzIGFyZVxuICogc2VwYXJhdGVkIGFuZCB0aGF0IGBpbml0aWFsU3RhdGVgIGFuZCBgY29uc3RhbnRzYCBhcmUgYEFycmF5YHMgb2YgdGVuc29yc1xuICogKG9yIE5vbmUpLlxuICpcbiAqIEBwYXJhbSBpbnB1dHMgVGVuc29yIG9yIGBBcnJheWAgb2YgIHRlbnNvcnMuXG4gKiBAcGFyYW0gaW5pdGlhbFN0YXRlIFRlbnNvciBvciBgQXJyYXlgIG9mIHRlbnNvcnMgb3IgYG51bGxgL2B1bmRlZmluZWRgLlxuICogQHBhcmFtIGNvbnN0YW50cyBUZW5zb3Igb3IgYEFycmF5YCBvZiB0ZW5zb3JzIG9yIGBudWxsYC9gdW5kZWZpbmVkYC5cbiAqIEByZXR1cm5zIEFuIG9iamVjdCBjb25zaXN0aW5nIG9mXG4gKiAgIGlucHV0czogQSB0ZW5zb3IuXG4gKiAgIGluaXRpYWxTdGF0ZTogYEFycmF5YCBvZiB0ZW5zb3JzIG9yIGBudWxsYC5cbiAqICAgY29uc3RhbnRzOiBgQXJyYXlgIG9mIHRlbnNvcnMgb3IgYG51bGxgLlxuICogQHRocm93cyBWYWx1ZUVycm9yLCBpZiBgaW5wdXRzYCBpcyBhbiBgQXJyYXlgIGJ1dCBlaXRoZXIgYGluaXRpYWxTdGF0ZWAgb3JcbiAqICAgYGNvbnN0YW50c2AgaXMgcHJvdmlkZWQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzdGFuZGFyZGl6ZUFyZ3MoXG4gICAgaW5wdXRzOiBUZW5zb3J8VGVuc29yW118U3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSxcbiAgICBpbml0aWFsU3RhdGU6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdLFxuICAgIGNvbnN0YW50czogVGVuc29yfFRlbnNvcltdfFN5bWJvbGljVGVuc29yfFN5bWJvbGljVGVuc29yW10sXG4gICAgbnVtQ29uc3RhbnRzPzogbnVtYmVyKToge1xuICBpbnB1dHM6IFRlbnNvcnxTeW1ib2xpY1RlbnNvcixcbiAgaW5pdGlhbFN0YXRlOiBUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcltdLFxuICBjb25zdGFudHM6IFRlbnNvcltdfFN5bWJvbGljVGVuc29yW11cbn0ge1xuICBpZiAoQXJyYXkuaXNBcnJheShpbnB1dHMpKSB7XG4gICAgaWYgKGluaXRpYWxTdGF0ZSAhPSBudWxsIHx8IGNvbnN0YW50cyAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnV2hlbiBpbnB1dHMgaXMgYW4gYXJyYXksIG5laXRoZXIgaW5pdGlhbFN0YXRlIG9yIGNvbnN0YW50cyAnICtcbiAgICAgICAgICAnc2hvdWxkIGJlIHByb3ZpZGVkJyk7XG4gICAgfVxuICAgIGlmIChudW1Db25zdGFudHMgIT0gbnVsbCkge1xuICAgICAgY29uc3RhbnRzID0gaW5wdXRzLnNsaWNlKGlucHV0cy5sZW5ndGggLSBudW1Db25zdGFudHMsIGlucHV0cy5sZW5ndGgpO1xuICAgICAgaW5wdXRzID0gaW5wdXRzLnNsaWNlKDAsIGlucHV0cy5sZW5ndGggLSBudW1Db25zdGFudHMpO1xuICAgIH1cbiAgICBpZiAoaW5wdXRzLmxlbmd0aCA+IDEpIHtcbiAgICAgIGluaXRpYWxTdGF0ZSA9IGlucHV0cy5zbGljZSgxLCBpbnB1dHMubGVuZ3RoKTtcbiAgICB9XG4gICAgaW5wdXRzID0gaW5wdXRzWzBdO1xuICB9XG5cbiAgZnVuY3Rpb24gdG9MaXN0T3JOdWxsKHg6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxcbiAgICAgICAgICAgICAgICAgICAgICAgIFN5bWJvbGljVGVuc29yW10pOiBUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcltdIHtcbiAgICBpZiAoeCA9PSBudWxsIHx8IEFycmF5LmlzQXJyYXkoeCkpIHtcbiAgICAgIHJldHVybiB4IGFzIFRlbnNvcltdIHwgU3ltYm9saWNUZW5zb3JbXTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIFt4XSBhcyBUZW5zb3JbXSB8IFN5bWJvbGljVGVuc29yW107XG4gICAgfVxuICB9XG5cbiAgaW5pdGlhbFN0YXRlID0gdG9MaXN0T3JOdWxsKGluaXRpYWxTdGF0ZSk7XG4gIGNvbnN0YW50cyA9IHRvTGlzdE9yTnVsbChjb25zdGFudHMpO1xuXG4gIHJldHVybiB7aW5wdXRzLCBpbml0aWFsU3RhdGUsIGNvbnN0YW50c307XG59XG5cbi8qKlxuICogSXRlcmF0ZXMgb3ZlciB0aGUgdGltZSBkaW1lbnNpb24gb2YgYSB0ZW5zb3IuXG4gKlxuICogQHBhcmFtIHN0ZXBGdW5jdGlvbiBSTk4gc3RlcCBmdW5jdGlvbi5cbiAqICAgUGFyYW1ldGVyczpcbiAqICAgICBpbnB1dHM6IHRlbnNvciB3aXRoIHNoYXBlIGBbc2FtcGxlcywgLi4uXWAgKG5vIHRpbWUgZGltZW5zaW9uKSxcbiAqICAgICAgIHJlcHJlc2VudGluZyBpbnB1dCBmb3IgdGhlIGJhdGNoIG9mIHNhbXBsZXMgYXQgYSBjZXJ0YWluIHRpbWUgc3RlcC5cbiAqICAgICBzdGF0ZXM6IGFuIEFycmF5IG9mIHRlbnNvcnMuXG4gKiAgIFJldHVybnM6XG4gKiAgICAgb3V0cHV0czogdGVuc29yIHdpdGggc2hhcGUgYFtzYW1wbGVzLCBvdXRwdXREaW1dYCAobm8gdGltZSBkaW1lbnNpb24pLlxuICogICAgIG5ld1N0YXRlczogbGlzdCBvZiB0ZW5zb3JzLCBzYW1lIGxlbmd0aCBhbmQgc2hhcGVzIGFzIGBzdGF0ZXNgLiBUaGUgZmlyc3RcbiAqICAgICAgIHN0YXRlIGluIHRoZSBsaXN0IG11c3QgYmUgdGhlIG91dHB1dCB0ZW5zb3IgYXQgdGhlIHByZXZpb3VzIHRpbWVzdGVwLlxuICogQHBhcmFtIGlucHV0cyBUZW5zb3Igb2YgdGVtcG9yYWwgZGF0YSBvZiBzaGFwZSBgW3NhbXBsZXMsIHRpbWUsIC4uLl1gIChhdFxuICogICBsZWFzdCAzRCkuXG4gKiBAcGFyYW0gaW5pdGlhbFN0YXRlcyBUZW5zb3Igd2l0aCBzaGFwZSBgW3NhbXBsZXMsIG91dHB1dERpbV1gIChubyB0aW1lXG4gKiAgIGRpbWVuc2lvbiksIGNvbnRhaW5pbmcgdGhlIGluaXRpYWwgdmFsdWVzIG9mIHRoZSBzdGF0ZXMgdXNlZCBpbiB0aGUgc3RlcFxuICogICBmdW5jdGlvbi5cbiAqIEBwYXJhbSBnb0JhY2t3YXJkcyBJZiBgdHJ1ZWAsIGRvIHRoZSBpdGVyYXRpb24gb3ZlciB0aGUgdGltZSBkaW1lbnNpb24gaW5cbiAqICAgcmV2ZXJzZSBvcmRlciBhbmQgcmV0dXJuIHRoZSByZXZlcnNlZCBzZXF1ZW5jZS5cbiAqIEBwYXJhbSBtYXNrIEJpbmFyeSB0ZW5zb3Igd2l0aCBzaGFwZSBgW3NhbXBsZSwgdGltZSwgMV1gLCB3aXRoIGEgemVybyBmb3JcbiAqICAgZXZlcnkgZWxlbWVudCB0aGF0IGlzIG1hc2tlZC5cbiAqIEBwYXJhbSBjb25zdGFudHMgQW4gQXJyYXkgb2YgY29uc3RhbnQgdmFsdWVzIHBhc3NlZCBhdCBlYWNoIHN0ZXAuXG4gKiBAcGFyYW0gdW5yb2xsIFdoZXRoZXIgdG8gdW5yb2xsIHRoZSBSTk4gb3IgdG8gdXNlIGEgc3ltYm9saWMgbG9vcC4gKk5vdCpcbiAqICAgYXBwbGljYWJsZSB0byB0aGlzIGltcGVyYXRpdmUgZGVlcGxlYXJuLmpzIGJhY2tlbmQuIEl0cyB2YWx1ZSBpcyBpZ25vcmVkLlxuICogQHBhcmFtIG5lZWRQZXJTdGVwT3V0cHV0cyBXaGV0aGVyIHRoZSBwZXItc3RlcCBvdXRwdXRzIGFyZSB0byBiZVxuICogICBjb25jYXRlbmF0ZWQgaW50byBhIHNpbmdsZSB0ZW5zb3IgYW5kIHJldHVybmVkIChhcyB0aGUgc2Vjb25kIHJldHVyblxuICogICB2YWx1ZSkuIERlZmF1bHQ6IGBmYWxzZWAuIFRoaXMgYXJnIGlzIGluY2x1ZGVkIHNvIHRoYXQgdGhlIHJlbGF0aXZlbHlcbiAqICAgZXhwZW5zaXZlIGNvbmNhdGVuYXRpb24gb2YgdGhlIHN0ZXB3aXNlIG91dHB1dHMgY2FuIGJlIG9taXR0ZWQgdW5sZXNzXG4gKiAgIHRoZSBzdGVwd2lzZSBvdXRwdXRzIG5lZWQgdG8gYmUga2VwdCAoZS5nLiwgZm9yIGFuIExTVE0gbGF5ZXIgb2Ygd2hpY2hcbiAqICAgYHJldHVyblNlcXVlbmNlYCBpcyBgdHJ1ZWAuKVxuICogQHJldHVybnMgQW4gQXJyYXk6IGBbbGFzdE91dHB1dCwgb3V0cHV0cywgbmV3U3RhdGVzXWAuXG4gKiAgIGxhc3RPdXRwdXQ6IHRoZSBsYXN0ZXN0IG91dHB1dCBvZiB0aGUgUk5OLCBvZiBzaGFwZSBgW3NhbXBsZXMsIC4uLl1gLlxuICogICBvdXRwdXRzOiB0ZW5zb3Igd2l0aCBzaGFwZSBgW3NhbXBsZXMsIHRpbWUsIC4uLl1gIHdoZXJlIGVhY2ggZW50cnlcbiAqICAgICBgb3V0cHV0W3MsIHRdYCBpcyB0aGUgb3V0cHV0IG9mIHRoZSBzdGVwIGZ1bmN0aW9uIGF0IHRpbWUgYHRgIGZvciBzYW1wbGVcbiAqICAgICBgc2AuIFRoaXMgcmV0dXJuIHZhbHVlIGlzIHByb3ZpZGVkIGlmIGFuZCBvbmx5IGlmIHRoZVxuICogICAgIGBuZWVkUGVyU3RlcE91dHB1dHNgIGlzIHNldCBhcyBgdHJ1ZWAuIElmIGl0IGlzIHNldCBhcyBgZmFsc2VgLCB0aGlzXG4gKiAgICAgcmV0dXJuIHZhbHVlIHdpbGwgYmUgYHVuZGVmaW5lZGAuXG4gKiAgIG5ld1N0YXRlczogQXJyYXkgb2YgdGVuc29ycywgbGF0ZXN0IHN0YXRlcyByZXR1cm5lZCBieSB0aGUgc3RlcCBmdW5jdGlvbixcbiAqICAgICAgb2Ygc2hhcGUgYChzYW1wbGVzLCAuLi4pYC5cbiAqIEB0aHJvd3MgVmFsdWVFcnJvciBJZiBpbnB1dCBkaW1lbnNpb24gaXMgbGVzcyB0aGFuIDMuXG4gKlxuICogVE9ETyhuaWVsc2VuZSk6IFRoaXMgbmVlZHMgdG8gYmUgdGlkeS1lZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHJubihcbiAgICBzdGVwRnVuY3Rpb246IFJublN0ZXBGdW5jdGlvbiwgaW5wdXRzOiBUZW5zb3IsIGluaXRpYWxTdGF0ZXM6IFRlbnNvcltdLFxuICAgIGdvQmFja3dhcmRzID0gZmFsc2UsIG1hc2s/OiBUZW5zb3IsIGNvbnN0YW50cz86IFRlbnNvcltdLCB1bnJvbGwgPSBmYWxzZSxcbiAgICBuZWVkUGVyU3RlcE91dHB1dHMgPSBmYWxzZSk6IFtUZW5zb3IsIFRlbnNvciwgVGVuc29yW11dIHtcbiAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICBjb25zdCBuZGltID0gaW5wdXRzLnNoYXBlLmxlbmd0aDtcbiAgICBpZiAobmRpbSA8IDMpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKGBJbnB1dCBzaG91bGQgYmUgYXQgbGVhc3QgM0QsIGJ1dCBpcyAke25kaW19RC5gKTtcbiAgICB9XG5cbiAgICAvLyBUcmFuc3Bvc2UgdG8gdGltZS1tYWpvciwgaS5lLiwgZnJvbSBbYmF0Y2gsIHRpbWUsIC4uLl0gdG8gW3RpbWUsIGJhdGNoLFxuICAgIC8vIC4uLl0uXG4gICAgY29uc3QgYXhlcyA9IFsxLCAwXS5jb25jYXQobWF0aF91dGlscy5yYW5nZSgyLCBuZGltKSk7XG4gICAgaW5wdXRzID0gdGZjLnRyYW5zcG9zZShpbnB1dHMsIGF4ZXMpO1xuXG4gICAgaWYgKGNvbnN0YW50cyAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAnVGhlIHJubigpIGZ1bmN0b2luIG9mIHRoZSBkZWVwbGVhcm4uanMgYmFja2VuZCBkb2VzIG5vdCBzdXBwb3J0ICcgK1xuICAgICAgICAgICdjb25zdGFudHMgeWV0LicpO1xuICAgIH1cblxuICAgIC8vIFBvcnRpbmcgTm90ZTogdGhlIHVucm9sbCBvcHRpb24gaXMgaWdub3JlZCBieSB0aGUgaW1wZXJhdGl2ZSBiYWNrZW5kLlxuICAgIGlmICh1bnJvbGwpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnQmFja2VuZCBybm4oKTogdGhlIHVucm9sbCA9IHRydWUgb3B0aW9uIGlzIG5vdCBhcHBsaWNhYmxlIHRvIHRoZSAnICtcbiAgICAgICAgICAnaW1wZXJhdGl2ZSBkZWVwbGVhcm4uanMgYmFja2VuZC4nKTtcbiAgICB9XG5cbiAgICBpZiAobWFzayAhPSBudWxsKSB7XG4gICAgICBtYXNrID0gdGZjLmNhc3QodGZjLmNhc3QobWFzaywgJ2Jvb2wnKSwgJ2Zsb2F0MzInKTtcbiAgICAgIGlmIChtYXNrLnJhbmsgPT09IG5kaW0gLSAxKSB7XG4gICAgICAgIG1hc2sgPSB0ZmMuZXhwYW5kRGltcyhtYXNrLCAtMSk7XG4gICAgICB9XG4gICAgICBtYXNrID0gdGZjLnRyYW5zcG9zZShtYXNrLCBheGVzKTtcbiAgICB9XG5cbiAgICBpZiAoZ29CYWNrd2FyZHMpIHtcbiAgICAgIGlucHV0cyA9IHRmYy5yZXZlcnNlKGlucHV0cywgMCk7XG4gICAgICBpZiAobWFzayAhPSBudWxsKSB7XG4gICAgICAgIG1hc2sgPSB0ZmMucmV2ZXJzZShtYXNrLCAwKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBQb3J0aW5nIE5vdGU6IFB5S2VyYXMgd2l0aCBUZW5zb3JGbG93IGJhY2tlbmQgdXNlcyBhIHN5bWJvbGljIGxvb3BcbiAgICAvLyAgICh0Zi53aGlsZV9sb29wKS4gQnV0IGZvciB0aGUgaW1wZXJhdGl2ZSBkZWVwbGVhcm4uanMgYmFja2VuZCwgd2UganVzdFxuICAgIC8vICAgdXNlIHRoZSB1c3VhbCBUeXBlU2NyaXB0IGNvbnRyb2wgZmxvdyB0byBpdGVyYXRlIG92ZXIgdGhlIHRpbWUgc3RlcHMgaW5cbiAgICAvLyAgIHRoZSBpbnB1dHMuXG4gICAgLy8gUG9ydGluZyBOb3RlOiBQeUtlcmFzIHBhdGNoZXMgYSBcIl91c2VfbGVhcm5pbmdfcGhhc2VcIiBhdHRyaWJ1dGUgdG9cbiAgICAvLyBvdXRwdXRzLlxuICAgIC8vICAgVGhpcyBpcyBub3QgaWRpb21hdGljIGluIFR5cGVTY3JpcHQuIFRoZSBpbmZvIHJlZ2FyZGluZyB3aGV0aGVyIHdlIGFyZVxuICAgIC8vICAgaW4gYSBsZWFybmluZyAoaS5lLiwgdHJhaW5pbmcpIHBoYXNlIGZvciBSTk4gaXMgcGFzc2VkIGluIGEgZGlmZmVyZW50XG4gICAgLy8gICB3YXkuXG5cbiAgICBjb25zdCBwZXJTdGVwT3V0cHV0czogVGVuc29yW10gPSBbXTtcbiAgICBsZXQgbGFzdE91dHB1dDogVGVuc29yO1xuICAgIGxldCBzdGF0ZXMgPSBpbml0aWFsU3RhdGVzO1xuICAgIGNvbnN0IHRpbWVTdGVwcyA9IGlucHV0cy5zaGFwZVswXTtcbiAgICBjb25zdCBwZXJTdGVwSW5wdXRzID0gdGZjLnVuc3RhY2soaW5wdXRzKTtcbiAgICBsZXQgcGVyU3RlcE1hc2tzOiBUZW5zb3JbXTtcbiAgICBpZiAobWFzayAhPSBudWxsKSB7XG4gICAgICBwZXJTdGVwTWFza3MgPSB0ZmMudW5zdGFjayhtYXNrKTtcbiAgICB9XG5cbiAgICBmb3IgKGxldCB0ID0gMDsgdCA8IHRpbWVTdGVwczsgKyt0KSB7XG4gICAgICBjb25zdCBjdXJyZW50SW5wdXQgPSBwZXJTdGVwSW5wdXRzW3RdO1xuICAgICAgY29uc3Qgc3RlcE91dHB1dHMgPSB0ZmMudGlkeSgoKSA9PiBzdGVwRnVuY3Rpb24oY3VycmVudElucHV0LCBzdGF0ZXMpKTtcblxuICAgICAgaWYgKG1hc2sgPT0gbnVsbCkge1xuICAgICAgICBsYXN0T3V0cHV0ID0gc3RlcE91dHB1dHNbMF07XG4gICAgICAgIHN0YXRlcyA9IHN0ZXBPdXRwdXRzWzFdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgY29uc3QgbWFza2VkT3V0cHV0cyA9IHRmYy50aWR5KCgpID0+IHtcbiAgICAgICAgICBjb25zdCBzdGVwTWFzayA9IHBlclN0ZXBNYXNrc1t0XTtcbiAgICAgICAgICBjb25zdCBuZWdTdGVwTWFzayA9IHRmYy5zdWIodGZjLm9uZXNMaWtlKHN0ZXBNYXNrKSwgc3RlcE1hc2spO1xuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IFdvdWxkIHRmYy53aGVyZSgpIGJlIGJldHRlciBmb3IgcGVyZm9ybWFuY2U/XG4gICAgICAgICAgY29uc3Qgb3V0cHV0ID0gdGZjLmFkZChcbiAgICAgICAgICAgICAgdGZjLm11bChzdGVwT3V0cHV0c1swXSwgc3RlcE1hc2spLFxuICAgICAgICAgICAgICB0ZmMubXVsKHN0YXRlc1swXSwgbmVnU3RlcE1hc2spKTtcbiAgICAgICAgICBjb25zdCBuZXdTdGF0ZXMgPSBzdGF0ZXMubWFwKChzdGF0ZSwgaSkgPT4ge1xuICAgICAgICAgICAgcmV0dXJuIHRmYy5hZGQoXG4gICAgICAgICAgICAgICAgdGZjLm11bChzdGVwT3V0cHV0c1sxXVtpXSwgc3RlcE1hc2spLFxuICAgICAgICAgICAgICAgIHRmYy5tdWwoc3RhdGUsIG5lZ1N0ZXBNYXNrKSk7XG4gICAgICAgICAgfSk7XG4gICAgICAgICAgcmV0dXJuIHtvdXRwdXQsIG5ld1N0YXRlc307XG4gICAgICAgIH0pO1xuICAgICAgICBsYXN0T3V0cHV0ID0gbWFza2VkT3V0cHV0cy5vdXRwdXQ7XG4gICAgICAgIHN0YXRlcyA9IG1hc2tlZE91dHB1dHMubmV3U3RhdGVzO1xuICAgICAgfVxuXG4gICAgICBpZiAobmVlZFBlclN0ZXBPdXRwdXRzKSB7XG4gICAgICAgIHBlclN0ZXBPdXRwdXRzLnB1c2gobGFzdE91dHB1dCk7XG4gICAgICB9XG4gICAgfVxuICAgIGxldCBvdXRwdXRzOiBUZW5zb3I7XG4gICAgaWYgKG5lZWRQZXJTdGVwT3V0cHV0cykge1xuICAgICAgY29uc3QgYXhpcyA9IDE7XG4gICAgICBvdXRwdXRzID0gdGZjLnN0YWNrKHBlclN0ZXBPdXRwdXRzLCBheGlzKTtcbiAgICB9XG4gICAgcmV0dXJuIFtsYXN0T3V0cHV0LCBvdXRwdXRzLCBzdGF0ZXNdIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yW11dO1xuICB9KTtcbn1cblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIEJhc2VSTk5MYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogQSBSTk4gY2VsbCBpbnN0YW5jZS4gQSBSTk4gY2VsbCBpcyBhIGNsYXNzIHRoYXQgaGFzOlxuICAgKiAgIC0gYSBgY2FsbCgpYCBtZXRob2QsIHdoaWNoIHRha2VzIGBbVGVuc29yLCBUZW5zb3JdYCBhcyB0aGVcbiAgICogICAgIGZpcnN0IGlucHV0IGFyZ3VtZW50LiBUaGUgZmlyc3QgaXRlbSBpcyB0aGUgaW5wdXQgYXQgdGltZSB0LCBhbmRcbiAgICogICAgIHNlY29uZCBpdGVtIGlzIHRoZSBjZWxsIHN0YXRlIGF0IHRpbWUgdC5cbiAgICogICAgIFRoZSBgY2FsbCgpYCBtZXRob2QgcmV0dXJucyBgW291dHB1dEF0VCwgc3RhdGVzQXRUUGx1czFdYC5cbiAgICogICAgIFRoZSBgY2FsbCgpYCBtZXRob2Qgb2YgdGhlIGNlbGwgY2FuIGFsc28gdGFrZSB0aGUgYXJndW1lbnQgYGNvbnN0YW50c2AsXG4gICAqICAgICBzZWUgc2VjdGlvbiBcIk5vdGUgb24gcGFzc2luZyBleHRlcm5hbCBjb25zdGFudHNcIiBiZWxvdy5cbiAgICogICAgIFBvcnRpbmcgTm9kZTogUHlLZXJhcyBvdmVycmlkZXMgdGhlIGBjYWxsKClgIHNpZ25hdHVyZSBvZiBSTk4gY2VsbHMsXG4gICAqICAgICAgIHdoaWNoIGFyZSBMYXllciBzdWJ0eXBlcywgdG8gYWNjZXB0IHR3byBhcmd1bWVudHMuIHRmanMtbGF5ZXJzIGRvZXNcbiAgICogICAgICAgbm90IGRvIHN1Y2ggb3ZlcnJpZGluZy4gSW5zdGVhZCB3ZSBwcmVzZXZlIHRoZSBgY2FsbCgpYCBzaWduYXR1cmUsXG4gICAqICAgICAgIHdoaWNoIGR1ZSB0byBpdHMgYFRlbnNvcnxUZW5zb3JbXWAgYXJndW1lbnQgYW5kIHJldHVybiB2YWx1ZSwgaXNcbiAgICogICAgICAgZmxleGlibGUgZW5vdWdoIHRvIGhhbmRsZSB0aGUgaW5wdXRzIGFuZCBzdGF0ZXMuXG4gICAqICAgLSBhIGBzdGF0ZVNpemVgIGF0dHJpYnV0ZS4gVGhpcyBjYW4gYmUgYSBzaW5nbGUgaW50ZWdlciAoc2luZ2xlIHN0YXRlKVxuICAgKiAgICAgaW4gd2hpY2ggY2FzZSBpdCBpcyB0aGUgc2l6ZSBvZiB0aGUgcmVjdXJyZW50IHN0YXRlICh3aGljaCBzaG91bGQgYmVcbiAgICogICAgIHRoZSBzYW1lIGFzIHRoZSBzaXplIG9mIHRoZSBjZWxsIG91dHB1dCkuIFRoaXMgY2FuIGFsc28gYmUgYW4gQXJyYXkgb2ZcbiAgICogICAgIGludGVnZXJzIChvbmUgc2l6ZSBwZXIgc3RhdGUpLiBJbiB0aGlzIGNhc2UsIHRoZSBmaXJzdCBlbnRyeVxuICAgKiAgICAgKGBzdGF0ZVNpemVbMF1gKSBzaG91bGQgYmUgdGhlIHNhbWUgYXMgdGhlIHNpemUgb2YgdGhlIGNlbGwgb3V0cHV0LlxuICAgKiBJdCBpcyBhbHNvIHBvc3NpYmxlIGZvciBgY2VsbGAgdG8gYmUgYSBsaXN0IG9mIFJOTiBjZWxsIGluc3RhbmNlcywgaW4gd2hpY2hcbiAgICogY2FzZSB0aGUgY2VsbHMgZ2V0IHN0YWNrZWQgb24gYWZ0ZXIgdGhlIG90aGVyIGluIHRoZSBSTk4sIGltcGxlbWVudGluZyBhblxuICAgKiBlZmZpY2llbnQgc3RhY2tlZCBSTk4uXG4gICAqL1xuICBjZWxsPzogUk5OQ2VsbHxSTk5DZWxsW107XG5cbiAgLyoqXG4gICAqIFdoZXRoZXIgdG8gcmV0dXJuIHRoZSBsYXN0IG91dHB1dCBpbiB0aGUgb3V0cHV0IHNlcXVlbmNlLCBvciB0aGUgZnVsbFxuICAgKiBzZXF1ZW5jZS5cbiAgICovXG4gIHJldHVyblNlcXVlbmNlcz86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIFdoZXRoZXIgdG8gcmV0dXJuIHRoZSBsYXN0IHN0YXRlIGluIGFkZGl0aW9uIHRvIHRoZSBvdXRwdXQuXG4gICAqL1xuICByZXR1cm5TdGF0ZT86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIElmIGB0cnVlYCwgcHJvY2VzcyB0aGUgaW5wdXQgc2VxdWVuY2UgYmFja3dhcmRzIGFuZCByZXR1cm4gdGhlIHJldmVyc2VkXG4gICAqIHNlcXVlbmNlIChkZWZhdWx0OiBgZmFsc2VgKS5cbiAgICovXG4gIGdvQmFja3dhcmRzPzogYm9vbGVhbjtcblxuICAvKipcbiAgICogSWYgYHRydWVgLCB0aGUgbGFzdCBzdGF0ZSBmb3IgZWFjaCBzYW1wbGUgYXQgaW5kZXggaSBpbiBhIGJhdGNoIHdpbGwgYmVcbiAgICogdXNlZCBhcyBpbml0aWFsIHN0YXRlIG9mIHRoZSBzYW1wbGUgb2YgaW5kZXggaSBpbiB0aGUgZm9sbG93aW5nIGJhdGNoXG4gICAqIChkZWZhdWx0OiBgZmFsc2VgKS5cbiAgICpcbiAgICogWW91IGNhbiBzZXQgUk5OIGxheWVycyB0byBiZSBcInN0YXRlZnVsXCIsIHdoaWNoIG1lYW5zIHRoYXQgdGhlIHN0YXRlc1xuICAgKiBjb21wdXRlZCBmb3IgdGhlIHNhbXBsZXMgaW4gb25lIGJhdGNoIHdpbGwgYmUgcmV1c2VkIGFzIGluaXRpYWwgc3RhdGVzXG4gICAqIGZvciB0aGUgc2FtcGxlcyBpbiB0aGUgbmV4dCBiYXRjaC4gVGhpcyBhc3N1bWVzIGEgb25lLXRvLW9uZSBtYXBwaW5nXG4gICAqIGJldHdlZW4gc2FtcGxlcyBpbiBkaWZmZXJlbnQgc3VjY2Vzc2l2ZSBiYXRjaGVzLlxuICAgKlxuICAgKiBUbyBlbmFibGUgXCJzdGF0ZWZ1bG5lc3NcIjpcbiAgICogICAtIHNwZWNpZnkgYHN0YXRlZnVsOiB0cnVlYCBpbiB0aGUgbGF5ZXIgY29uc3RydWN0b3IuXG4gICAqICAgLSBzcGVjaWZ5IGEgZml4ZWQgYmF0Y2ggc2l6ZSBmb3IgeW91ciBtb2RlbCwgYnkgcGFzc2luZ1xuICAgKiAgICAgLSBpZiBzZXF1ZW50aWFsIG1vZGVsOlxuICAgKiAgICAgICBgYmF0Y2hJbnB1dFNoYXBlOiBbLi4uXWAgdG8gdGhlIGZpcnN0IGxheWVyIGluIHlvdXIgbW9kZWwuXG4gICAqICAgICAtIGVsc2UgZm9yIGZ1bmN0aW9uYWwgbW9kZWwgd2l0aCAxIG9yIG1vcmUgSW5wdXQgbGF5ZXJzOlxuICAgKiAgICAgICBgYmF0Y2hTaGFwZTogWy4uLl1gIHRvIGFsbCB0aGUgZmlyc3QgbGF5ZXJzIGluIHlvdXIgbW9kZWwuXG4gICAqICAgICBUaGlzIGlzIHRoZSBleHBlY3RlZCBzaGFwZSBvZiB5b3VyIGlucHV0c1xuICAgKiAgICAgKmluY2x1ZGluZyB0aGUgYmF0Y2ggc2l6ZSouXG4gICAqICAgICBJdCBzaG91bGQgYmUgYSB0dXBsZSBvZiBpbnRlZ2VycywgZS5nLiwgYFszMiwgMTAsIDEwMF1gLlxuICAgKiAgIC0gc3BlY2lmeSBgc2h1ZmZsZTogZmFsc2VgIHdoZW4gY2FsbGluZyBgTGF5ZXJzTW9kZWwuZml0KClgLlxuICAgKlxuICAgKiBUbyByZXNldCB0aGUgc3RhdGUgb2YgeW91ciBtb2RlbCwgY2FsbCBgcmVzZXRTdGF0ZXMoKWAgb24gZWl0aGVyIHRoZVxuICAgKiBzcGVjaWZpYyBsYXllciBvciBvbiB0aGUgZW50aXJlIG1vZGVsLlxuICAgKi9cbiAgc3RhdGVmdWw/OiBib29sZWFuO1xuICAvLyBUT0RPKGNhaXMpOiBFeHBsb3JlIHdoZXRoZXIgd2UgY2FuIHdhcm4gdXNlcnMgd2hlbiB0aGV5IGZhaWwgdG8gc2V0XG4gIC8vICAgYHNodWZmbGU6IGZhbHNlYCB3aGVuIHRyYWluaW5nIGEgbW9kZWwgY29uc2lzdGluZyBvZiBzdGF0ZWZ1bCBSTk5zXG4gIC8vICAgYW5kIGFueSBzdGF0ZWZ1bCBMYXllcnMgaW4gZ2VuZXJhbC5cblxuICAvKipcbiAgICogSWYgYHRydWVgLCB0aGUgbmV0d29yayB3aWxsIGJlIHVucm9sbGVkLCBlbHNlIGEgc3ltYm9saWMgbG9vcCB3aWxsIGJlXG4gICAqIHVzZWQuIFVucm9sbGluZyBjYW4gc3BlZWQtdXAgYSBSTk4sIGFsdGhvdWdoIGl0IHRlbmRzIHRvIGJlIG1vcmUgbWVtb3J5LVxuICAgKiBpbnRlbnNpdmUuIFVucm9sbGluZyBpcyBvbmx5IHN1aXRhYmxlIGZvciBzaG9ydCBzZXF1ZW5jZXMgKGRlZmF1bHQ6XG4gICAqIGBmYWxzZWApLlxuICAgKiBQb3J0aW5nIE5vdGU6IHRmanMtbGF5ZXJzIGhhcyBhbiBpbXBlcmF0aXZlIGJhY2tlbmQuIFJOTnMgYXJlIGV4ZWN1dGVkIHdpdGhcbiAgICogICBub3JtYWwgVHlwZVNjcmlwdCBjb250cm9sIGZsb3cuIEhlbmNlIHRoaXMgcHJvcGVydHkgaXMgaW5hcHBsaWNhYmxlIGFuZFxuICAgKiAgIGlnbm9yZWQgaW4gdGZqcy1sYXllcnMuXG4gICAqL1xuICB1bnJvbGw/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBEaW1lbnNpb25hbGl0eSBvZiB0aGUgaW5wdXQgKGludGVnZXIpLlxuICAgKiAgIFRoaXMgb3B0aW9uIChvciBhbHRlcm5hdGl2ZWx5LCB0aGUgb3B0aW9uIGBpbnB1dFNoYXBlYCkgaXMgcmVxdWlyZWQgd2hlblxuICAgKiAgIHRoaXMgbGF5ZXIgaXMgdXNlZCBhcyB0aGUgZmlyc3QgbGF5ZXIgaW4gYSBtb2RlbC5cbiAgICovXG4gIGlucHV0RGltPzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBMZW5ndGggb2YgdGhlIGlucHV0IHNlcXVlbmNlcywgdG8gYmUgc3BlY2lmaWVkIHdoZW4gaXQgaXMgY29uc3RhbnQuXG4gICAqIFRoaXMgYXJndW1lbnQgaXMgcmVxdWlyZWQgaWYgeW91IGFyZSBnb2luZyB0byBjb25uZWN0IGBGbGF0dGVuYCB0aGVuXG4gICAqIGBEZW5zZWAgbGF5ZXJzIHVwc3RyZWFtICh3aXRob3V0IGl0LCB0aGUgc2hhcGUgb2YgdGhlIGRlbnNlIG91dHB1dHMgY2Fubm90XG4gICAqIGJlIGNvbXB1dGVkKS4gTm90ZSB0aGF0IGlmIHRoZSByZWN1cnJlbnQgbGF5ZXIgaXMgbm90IHRoZSBmaXJzdCBsYXllciBpblxuICAgKiB5b3VyIG1vZGVsLCB5b3Ugd291bGQgbmVlZCB0byBzcGVjaWZ5IHRoZSBpbnB1dCBsZW5ndGggYXQgdGhlIGxldmVsIG9mIHRoZVxuICAgKiBmaXJzdCBsYXllciAoZS5nLiwgdmlhIHRoZSBgaW5wdXRTaGFwZWAgb3B0aW9uKS5cbiAgICovXG4gIGlucHV0TGVuZ3RoPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgUk5OIGV4dGVuZHMgTGF5ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdSTk4nO1xuICBwdWJsaWMgcmVhZG9ubHkgY2VsbDogUk5OQ2VsbDtcbiAgcHVibGljIHJlYWRvbmx5IHJldHVyblNlcXVlbmNlczogYm9vbGVhbjtcbiAgcHVibGljIHJlYWRvbmx5IHJldHVyblN0YXRlOiBib29sZWFuO1xuICBwdWJsaWMgcmVhZG9ubHkgZ29CYWNrd2FyZHM6IGJvb2xlYW47XG4gIHB1YmxpYyByZWFkb25seSB1bnJvbGw6IGJvb2xlYW47XG5cbiAgcHVibGljIHN0YXRlU3BlYzogSW5wdXRTcGVjW107XG4gIHByb3RlY3RlZCBzdGF0ZXNfOiBUZW5zb3JbXTtcblxuICAvLyBOT1RFKGNhaXMpOiBGb3Igc3RhdGVmdWwgUk5OcywgdGhlIG9sZCBzdGF0ZXMgY2Fubm90IGJlIGRpc3Bvc2VkIHJpZ2h0XG4gIC8vIGF3YXkgd2hlbiBuZXcgc3RhdGVzIGFyZSBzZXQsIGJlY2F1c2UgdGhlIG9sZCBzdGF0ZXMgbWF5IG5lZWQgdG8gYmUgdXNlZFxuICAvLyBsYXRlciBmb3IgYmFja3Byb3BhZ2F0aW9uIHRocm91Z2ggdGltZSAoQlBUVCkgYW5kIG90aGVyIHB1cnBvc2VzLiBTbyB3ZVxuICAvLyBrZWVwIHRoZW0gaGVyZSBmb3IgZmluYWwgZGlzcG9zYWwgd2hlbiB0aGUgc3RhdGUgaXMgcmVzZXQgY29tcGxldGVseVxuICAvLyAoaS5lLiwgdGhyb3VnaCBuby1hcmcgY2FsbCB0byBgcmVzZXRTdGF0ZXMoKWApLlxuICBwcm90ZWN0ZWQga2VwdFN0YXRlczogVGVuc29yW11bXTtcblxuICBwcml2YXRlIG51bUNvbnN0YW50czogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFJOTkxheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIGxldCBjZWxsOiBSTk5DZWxsO1xuICAgIGlmIChhcmdzLmNlbGwgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ2NlbGwgcHJvcGVydHkgaXMgbWlzc2luZyBmb3IgdGhlIGNvbnN0cnVjdG9yIG9mIFJOTi4nKTtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkoYXJncy5jZWxsKSkge1xuICAgICAgY2VsbCA9IG5ldyBTdGFja2VkUk5OQ2VsbHMoe2NlbGxzOiBhcmdzLmNlbGx9KTtcbiAgICB9IGVsc2Uge1xuICAgICAgY2VsbCA9IGFyZ3MuY2VsbDtcbiAgICB9XG4gICAgaWYgKGNlbGwuc3RhdGVTaXplID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdUaGUgUk5OIGNlbGwgc2hvdWxkIGhhdmUgYW4gYXR0cmlidXRlIGBzdGF0ZVNpemVgICh0dXBsZSBvZiAnICtcbiAgICAgICAgICAnaW50ZWdlcnMsIG9uZSBpbnRlZ2VyIHBlciBSTk4gc3RhdGUpLicpO1xuICAgIH1cbiAgICB0aGlzLmNlbGwgPSBjZWxsO1xuICAgIHRoaXMucmV0dXJuU2VxdWVuY2VzID1cbiAgICAgICAgYXJncy5yZXR1cm5TZXF1ZW5jZXMgPT0gbnVsbCA/IGZhbHNlIDogYXJncy5yZXR1cm5TZXF1ZW5jZXM7XG4gICAgdGhpcy5yZXR1cm5TdGF0ZSA9IGFyZ3MucmV0dXJuU3RhdGUgPT0gbnVsbCA/IGZhbHNlIDogYXJncy5yZXR1cm5TdGF0ZTtcbiAgICB0aGlzLmdvQmFja3dhcmRzID0gYXJncy5nb0JhY2t3YXJkcyA9PSBudWxsID8gZmFsc2UgOiBhcmdzLmdvQmFja3dhcmRzO1xuICAgIHRoaXMuX3N0YXRlZnVsID0gYXJncy5zdGF0ZWZ1bCA9PSBudWxsID8gZmFsc2UgOiBhcmdzLnN0YXRlZnVsO1xuICAgIHRoaXMudW5yb2xsID0gYXJncy51bnJvbGwgPT0gbnVsbCA/IGZhbHNlIDogYXJncy51bnJvbGw7XG5cbiAgICB0aGlzLnN1cHBvcnRzTWFza2luZyA9IHRydWU7XG4gICAgdGhpcy5pbnB1dFNwZWMgPSBbbmV3IElucHV0U3BlYyh7bmRpbTogM30pXTtcbiAgICB0aGlzLnN0YXRlU3BlYyA9IG51bGw7XG4gICAgdGhpcy5zdGF0ZXNfID0gbnVsbDtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgY29uc3RhbnRzU3BlYyBhbmQgbnVtQ29uc3RhbnRzLlxuICAgIHRoaXMubnVtQ29uc3RhbnRzID0gbnVsbDtcbiAgICAvLyBUT0RPKGNhaXMpOiBMb29rIGludG8gdGhlIHVzZSBvZiBpbml0aWFsX3N0YXRlIGluIHRoZSBrd2FyZ3Mgb2YgdGhlXG4gICAgLy8gICBjb25zdHJ1Y3Rvci5cblxuICAgIHRoaXMua2VwdFN0YXRlcyA9IFtdO1xuICB9XG5cbiAgLy8gUG9ydGluZyBOb3RlOiBUaGlzIGlzIHRoZSBlcXVpdmFsZW50IG9mIGBSTk4uc3RhdGVzYCBwcm9wZXJ0eSBnZXR0ZXIgaW5cbiAgLy8gICBQeUtlcmFzLlxuICBnZXRTdGF0ZXMoKTogVGVuc29yW10ge1xuICAgIGlmICh0aGlzLnN0YXRlc18gPT0gbnVsbCkge1xuICAgICAgY29uc3QgbnVtU3RhdGVzID1cbiAgICAgICAgICBBcnJheS5pc0FycmF5KHRoaXMuY2VsbC5zdGF0ZVNpemUpID8gdGhpcy5jZWxsLnN0YXRlU2l6ZS5sZW5ndGggOiAxO1xuICAgICAgcmV0dXJuIG1hdGhfdXRpbHMucmFuZ2UoMCwgbnVtU3RhdGVzKS5tYXAoeCA9PiBudWxsKTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHRoaXMuc3RhdGVzXztcbiAgICB9XG4gIH1cblxuICAvLyBQb3J0aW5nIE5vdGU6IFRoaXMgaXMgdGhlIGVxdWl2YWxlbnQgb2YgdGhlIGBSTk4uc3RhdGVzYCBwcm9wZXJ0eSBzZXR0ZXIgaW5cbiAgLy8gICBQeUtlcmFzLlxuICBzZXRTdGF0ZXMoc3RhdGVzOiBUZW5zb3JbXSk6IHZvaWQge1xuICAgIHRoaXMuc3RhdGVzXyA9IHN0YXRlcztcbiAgfVxuXG4gIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgaWYgKGlzQXJyYXlPZlNoYXBlcyhpbnB1dFNoYXBlKSkge1xuICAgICAgaW5wdXRTaGFwZSA9IChpbnB1dFNoYXBlIGFzIFNoYXBlW10pWzBdO1xuICAgIH1cbiAgICBpbnB1dFNoYXBlID0gaW5wdXRTaGFwZSBhcyBTaGFwZTtcblxuICAgIC8vIFRPRE8oY2Fpcyk6IFJlbW92ZSB0aGUgY2FzdGluZyBvbmNlIHN0YWNrZWQgUk5OIGNlbGxzIGJlY29tZSBzdXBwb3J0ZWQuXG4gICAgbGV0IHN0YXRlU2l6ZSA9IHRoaXMuY2VsbC5zdGF0ZVNpemU7XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KHN0YXRlU2l6ZSkpIHtcbiAgICAgIHN0YXRlU2l6ZSA9IFtzdGF0ZVNpemVdO1xuICAgIH1cbiAgICBjb25zdCBvdXRwdXREaW0gPSBzdGF0ZVNpemVbMF07XG4gICAgbGV0IG91dHB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdO1xuICAgIGlmICh0aGlzLnJldHVyblNlcXVlbmNlcykge1xuICAgICAgb3V0cHV0U2hhcGUgPSBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVsxXSwgb3V0cHV0RGltXTtcbiAgICB9IGVsc2Uge1xuICAgICAgb3V0cHV0U2hhcGUgPSBbaW5wdXRTaGFwZVswXSwgb3V0cHV0RGltXTtcbiAgICB9XG5cbiAgICBpZiAodGhpcy5yZXR1cm5TdGF0ZSkge1xuICAgICAgY29uc3Qgc3RhdGVTaGFwZTogU2hhcGVbXSA9IFtdO1xuICAgICAgZm9yIChjb25zdCBkaW0gb2Ygc3RhdGVTaXplKSB7XG4gICAgICAgIHN0YXRlU2hhcGUucHVzaChbaW5wdXRTaGFwZVswXSwgZGltXSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gW291dHB1dFNoYXBlXS5jb25jYXQoc3RhdGVTaGFwZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiBvdXRwdXRTaGFwZTtcbiAgICB9XG4gIH1cblxuICBjb21wdXRlTWFzayhpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwgbWFzaz86IFRlbnNvcnxUZW5zb3JbXSk6IFRlbnNvclxuICAgICAgfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGZjLnRpZHkoKCkgPT4ge1xuICAgICAgaWYgKEFycmF5LmlzQXJyYXkobWFzaykpIHtcbiAgICAgICAgbWFzayA9IG1hc2tbMF07XG4gICAgICB9XG4gICAgICBjb25zdCBvdXRwdXRNYXNrID0gdGhpcy5yZXR1cm5TZXF1ZW5jZXMgPyBtYXNrIDogbnVsbDtcblxuICAgICAgaWYgKHRoaXMucmV0dXJuU3RhdGUpIHtcbiAgICAgICAgY29uc3Qgc3RhdGVNYXNrID0gdGhpcy5zdGF0ZXMubWFwKHMgPT4gbnVsbCk7XG4gICAgICAgIHJldHVybiBbb3V0cHV0TWFza10uY29uY2F0KHN0YXRlTWFzayk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gb3V0cHV0TWFzaztcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBHZXQgdGhlIGN1cnJlbnQgc3RhdGUgdGVuc29ycyBvZiB0aGUgUk5OLlxuICAgKlxuICAgKiBJZiB0aGUgc3RhdGUgaGFzbid0IGJlZW4gc2V0LCByZXR1cm4gYW4gYXJyYXkgb2YgYG51bGxgcyBvZiB0aGUgY29ycmVjdFxuICAgKiBsZW5ndGguXG4gICAqL1xuICBnZXQgc3RhdGVzKCk6IFRlbnNvcltdIHtcbiAgICBpZiAodGhpcy5zdGF0ZXNfID09IG51bGwpIHtcbiAgICAgIGNvbnN0IG51bVN0YXRlcyA9XG4gICAgICAgICAgQXJyYXkuaXNBcnJheSh0aGlzLmNlbGwuc3RhdGVTaXplKSA/IHRoaXMuY2VsbC5zdGF0ZVNpemUubGVuZ3RoIDogMTtcbiAgICAgIGNvbnN0IG91dHB1dDogVGVuc29yW10gPSBbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbnVtU3RhdGVzOyArK2kpIHtcbiAgICAgICAgb3V0cHV0LnB1c2gobnVsbCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gb3V0cHV0O1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gdGhpcy5zdGF0ZXNfO1xuICAgIH1cbiAgfVxuXG4gIHNldCBzdGF0ZXMoczogVGVuc29yW10pIHtcbiAgICB0aGlzLnN0YXRlc18gPSBzO1xuICB9XG5cbiAgcHVibGljIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICAvLyBOb3RlIGlucHV0U2hhcGUgd2lsbCBiZSBhbiBBcnJheSBvZiBTaGFwZXMgb2YgaW5pdGlhbCBzdGF0ZXMgYW5kXG4gICAgLy8gY29uc3RhbnRzIGlmIHRoZXNlIGFyZSBwYXNzZWQgaW4gYXBwbHkoKS5cbiAgICBjb25zdCBjb25zdGFudFNoYXBlOiBTaGFwZVtdID0gbnVsbDtcbiAgICBpZiAodGhpcy5udW1Db25zdGFudHMgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgJ0NvbnN0YW50cyBzdXBwb3J0IGlzIG5vdCBpbXBsZW1lbnRlZCBpbiBSTk4geWV0LicpO1xuICAgIH1cblxuICAgIGlmIChpc0FycmF5T2ZTaGFwZXMoaW5wdXRTaGFwZSkpIHtcbiAgICAgIGlucHV0U2hhcGUgPSAoaW5wdXRTaGFwZSBhcyBTaGFwZVtdKVswXTtcbiAgICB9XG4gICAgaW5wdXRTaGFwZSA9IGlucHV0U2hhcGUgYXMgU2hhcGU7XG5cbiAgICBjb25zdCBiYXRjaFNpemU6IG51bWJlciA9IHRoaXMuc3RhdGVmdWwgPyBpbnB1dFNoYXBlWzBdIDogbnVsbDtcbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGUuc2xpY2UoMik7XG4gICAgdGhpcy5pbnB1dFNwZWNbMF0gPSBuZXcgSW5wdXRTcGVjKHtzaGFwZTogW2JhdGNoU2l6ZSwgbnVsbCwgLi4uaW5wdXREaW1dfSk7XG5cbiAgICAvLyBBbGxvdyBjZWxsIChpZiBSTk5DZWxsIExheWVyKSB0byBidWlsZCBiZWZvcmUgd2Ugc2V0IG9yIHZhbGlkYXRlXG4gICAgLy8gc3RhdGVTcGVjLlxuICAgIGNvbnN0IHN0ZXBJbnB1dFNoYXBlID0gW2lucHV0U2hhcGVbMF1dLmNvbmNhdChpbnB1dFNoYXBlLnNsaWNlKDIpKTtcbiAgICBpZiAoY29uc3RhbnRTaGFwZSAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAnQ29uc3RhbnRzIHN1cHBvcnQgaXMgbm90IGltcGxlbWVudGVkIGluIFJOTiB5ZXQuJyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuY2VsbC5idWlsZChzdGVwSW5wdXRTaGFwZSk7XG4gICAgfVxuXG4gICAgLy8gU2V0IG9yIHZhbGlkYXRlIHN0YXRlU3BlYy5cbiAgICBsZXQgc3RhdGVTaXplOiBudW1iZXJbXTtcbiAgICBpZiAoQXJyYXkuaXNBcnJheSh0aGlzLmNlbGwuc3RhdGVTaXplKSkge1xuICAgICAgc3RhdGVTaXplID0gdGhpcy5jZWxsLnN0YXRlU2l6ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgc3RhdGVTaXplID0gW3RoaXMuY2VsbC5zdGF0ZVNpemVdO1xuICAgIH1cblxuICAgIGlmICh0aGlzLnN0YXRlU3BlYyAhPSBudWxsKSB7XG4gICAgICBpZiAoIXV0aWwuYXJyYXlzRXF1YWwoXG4gICAgICAgICAgICAgIHRoaXMuc3RhdGVTcGVjLm1hcChzcGVjID0+IHNwZWMuc2hhcGVbc3BlYy5zaGFwZS5sZW5ndGggLSAxXSksXG4gICAgICAgICAgICAgIHN0YXRlU2l6ZSkpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgQW4gaW5pdGlhbFN0YXRlIHdhcyBwYXNzZWQgdGhhdCBpcyBub3QgY29tcGF0aWJsZSB3aXRoIGAgK1xuICAgICAgICAgICAgYGNlbGwuc3RhdGVTaXplLiBSZWNlaXZlZCBzdGF0ZVNwZWM9JHt0aGlzLnN0YXRlU3BlY307IGAgK1xuICAgICAgICAgICAgYEhvd2V2ZXIgY2VsbC5zdGF0ZVNpemUgaXMgJHt0aGlzLmNlbGwuc3RhdGVTaXplfWApO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnN0YXRlU3BlYyA9XG4gICAgICAgICAgc3RhdGVTaXplLm1hcChkaW0gPT4gbmV3IElucHV0U3BlYyh7c2hhcGU6IFtudWxsLCBkaW1dfSkpO1xuICAgIH1cbiAgICBpZiAodGhpcy5zdGF0ZWZ1bCkge1xuICAgICAgdGhpcy5yZXNldFN0YXRlcygpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBSZXNldCB0aGUgc3RhdGUgdGVuc29ycyBvZiB0aGUgUk5OLlxuICAgKlxuICAgKiBJZiB0aGUgYHN0YXRlc2AgYXJndW1lbnQgaXMgYHVuZGVmaW5lZGAgb3IgYG51bGxgLCB3aWxsIHNldCB0aGVcbiAgICogc3RhdGUgdGVuc29yKHMpIG9mIHRoZSBSTk4gdG8gYWxsLXplcm8gdGVuc29ycyBvZiB0aGUgYXBwcm9wcmlhdGVcbiAgICogc2hhcGUocykuXG4gICAqXG4gICAqIElmIGBzdGF0ZXNgIGlzIHByb3ZpZGVkLCB3aWxsIHNldCB0aGUgc3RhdGUgdGVuc29ycyBvZiB0aGUgUk5OIHRvIGl0c1xuICAgKiB2YWx1ZS5cbiAgICpcbiAgICogQHBhcmFtIHN0YXRlcyBPcHRpb25hbCBleHRlcm5hbGx5LXByb3ZpZGVkIGluaXRpYWwgc3RhdGVzLlxuICAgKiBAcGFyYW0gdHJhaW5pbmcgV2hldGhlciB0aGlzIGNhbGwgaXMgZG9uZSBkdXJpbmcgdHJhaW5pbmcuIEZvciBzdGF0ZWZ1bFxuICAgKiAgIFJOTnMsIHRoaXMgYWZmZWN0cyB3aGV0aGVyIHRoZSBvbGQgc3RhdGVzIGFyZSBrZXB0IG9yIGRpc2NhcmRlZC4gSW5cbiAgICogICBwYXJ0aWN1bGFyLCBpZiBgdHJhaW5pbmdgIGlzIGB0cnVlYCwgdGhlIG9sZCBzdGF0ZXMgd2lsbCBiZSBrZXB0IHNvXG4gICAqICAgdGhhdCBzdWJzZXF1ZW50IGJhY2twcm9wZ2F0YWlvbiB0aHJvdWdoIHRpbWUgKEJQVFQpIG1heSB3b3JrIHByb3Blcmx5LlxuICAgKiAgIEVsc2UsIHRoZSBvbGQgc3RhdGVzIHdpbGwgYmUgZGlzY2FyZGVkLlxuICAgKi9cbiAgcmVzZXRTdGF0ZXMoc3RhdGVzPzogVGVuc29yfFRlbnNvcltdLCB0cmFpbmluZyA9IGZhbHNlKTogdm9pZCB7XG4gICAgdGlkeSgoKSA9PiB7XG4gICAgICBpZiAoIXRoaXMuc3RhdGVmdWwpIHtcbiAgICAgICAgdGhyb3cgbmV3IEF0dHJpYnV0ZUVycm9yKFxuICAgICAgICAgICAgJ0Nhbm5vdCBjYWxsIHJlc2V0U3RhdGVzKCkgb24gYW4gUk5OIExheWVyIHRoYXQgaXMgbm90IHN0YXRlZnVsLicpO1xuICAgICAgfVxuICAgICAgY29uc3QgYmF0Y2hTaXplID0gdGhpcy5pbnB1dFNwZWNbMF0uc2hhcGVbMF07XG4gICAgICBpZiAoYmF0Y2hTaXplID09IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAnSWYgYW4gUk5OIGlzIHN0YXRlZnVsLCBpdCBuZWVkcyB0byBrbm93IGl0cyBiYXRjaCBzaXplLiBTcGVjaWZ5ICcgK1xuICAgICAgICAgICAgJ3RoZSBiYXRjaCBzaXplIG9mIHlvdXIgaW5wdXQgdGVuc29yczogXFxuJyArXG4gICAgICAgICAgICAnLSBJZiB1c2luZyBhIFNlcXVlbnRpYWwgbW9kZWwsIHNwZWNpZnkgdGhlIGJhdGNoIHNpemUgYnkgJyArXG4gICAgICAgICAgICAncGFzc2luZyBhIGBiYXRjaElucHV0U2hhcGVgIG9wdGlvbiB0byB5b3VyIGZpcnN0IGxheWVyLlxcbicgK1xuICAgICAgICAgICAgJy0gSWYgdXNpbmcgdGhlIGZ1bmN0aW9uYWwgQVBJLCBzcGVjaWZ5IHRoZSBiYXRjaCBzaXplIGJ5ICcgK1xuICAgICAgICAgICAgJ3Bhc3NpbmcgYSBgYmF0Y2hTaGFwZWAgb3B0aW9uIHRvIHlvdXIgSW5wdXQgbGF5ZXIuJyk7XG4gICAgICB9XG4gICAgICAvLyBJbml0aWFsaXplIHN0YXRlIGlmIG51bGwuXG4gICAgICBpZiAodGhpcy5zdGF0ZXNfID09IG51bGwpIHtcbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5jZWxsLnN0YXRlU2l6ZSkpIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc18gPVxuICAgICAgICAgICAgICB0aGlzLmNlbGwuc3RhdGVTaXplLm1hcChkaW0gPT4gdGZjLnplcm9zKFtiYXRjaFNpemUsIGRpbV0pKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc18gPSBbdGZjLnplcm9zKFtiYXRjaFNpemUsIHRoaXMuY2VsbC5zdGF0ZVNpemVdKV07XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAoc3RhdGVzID09IG51bGwpIHtcbiAgICAgICAgLy8gRGlzcG9zZSBvbGQgc3RhdGUgdGVuc29ycy5cbiAgICAgICAgdGZjLmRpc3Bvc2UodGhpcy5zdGF0ZXNfKTtcbiAgICAgICAgLy8gRm9yIHN0YXRlZnVsIFJOTnMsIGZ1bGx5IGRpc3Bvc2Uga2VwdCBvbGQgc3RhdGVzLlxuICAgICAgICBpZiAodGhpcy5rZXB0U3RhdGVzICE9IG51bGwpIHtcbiAgICAgICAgICB0ZmMuZGlzcG9zZSh0aGlzLmtlcHRTdGF0ZXMpO1xuICAgICAgICAgIHRoaXMua2VwdFN0YXRlcyA9IFtdO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5jZWxsLnN0YXRlU2l6ZSkpIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc18gPVxuICAgICAgICAgICAgICB0aGlzLmNlbGwuc3RhdGVTaXplLm1hcChkaW0gPT4gdGZjLnplcm9zKFtiYXRjaFNpemUsIGRpbV0pKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc19bMF0gPSB0ZmMuemVyb3MoW2JhdGNoU2l6ZSwgdGhpcy5jZWxsLnN0YXRlU2l6ZV0pO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBpZiAoIUFycmF5LmlzQXJyYXkoc3RhdGVzKSkge1xuICAgICAgICAgIHN0YXRlcyA9IFtzdGF0ZXNdO1xuICAgICAgICB9XG4gICAgICAgIGlmIChzdGF0ZXMubGVuZ3RoICE9PSB0aGlzLnN0YXRlc18ubGVuZ3RoKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBMYXllciAke3RoaXMubmFtZX0gZXhwZWN0cyAke3RoaXMuc3RhdGVzXy5sZW5ndGh9IHN0YXRlKHMpLCBgICtcbiAgICAgICAgICAgICAgYGJ1dCBpdCByZWNlaXZlZCAke3N0YXRlcy5sZW5ndGh9IHN0YXRlIHZhbHVlKHMpLiBJbnB1dCBgICtcbiAgICAgICAgICAgICAgYHJlY2VpdmVkOiAke3N0YXRlc31gKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmICh0cmFpbmluZyA9PT0gdHJ1ZSkge1xuICAgICAgICAgIC8vIFN0b3JlIG9sZCBzdGF0ZSB0ZW5zb3JzIGZvciBjb21wbGV0ZSBkaXNwb3NhbCBsYXRlciwgaS5lLiwgZHVyaW5nXG4gICAgICAgICAgLy8gdGhlIG5leHQgbm8tYXJnIGNhbGwgdG8gdGhpcyBtZXRob2QuIFdlIGRvIG5vdCBkaXNwb3NlIHRoZSBvbGRcbiAgICAgICAgICAvLyBzdGF0ZXMgaW1tZWRpYXRlbHkgYmVjYXVzZSB0aGF0IEJQVFQgKGFtb25nIG90aGVyIHRoaW5ncykgcmVxdWlyZVxuICAgICAgICAgIC8vIHRoZW0uXG4gICAgICAgICAgdGhpcy5rZXB0U3RhdGVzLnB1c2godGhpcy5zdGF0ZXNfLnNsaWNlKCkpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRmYy5kaXNwb3NlKHRoaXMuc3RhdGVzXyk7XG4gICAgICAgIH1cblxuICAgICAgICBmb3IgKGxldCBpbmRleCA9IDA7IGluZGV4IDwgdGhpcy5zdGF0ZXNfLmxlbmd0aDsgKytpbmRleCkge1xuICAgICAgICAgIGNvbnN0IHZhbHVlID0gc3RhdGVzW2luZGV4XTtcbiAgICAgICAgICBjb25zdCBkaW0gPSBBcnJheS5pc0FycmF5KHRoaXMuY2VsbC5zdGF0ZVNpemUpID9cbiAgICAgICAgICAgICAgdGhpcy5jZWxsLnN0YXRlU2l6ZVtpbmRleF0gOlxuICAgICAgICAgICAgICB0aGlzLmNlbGwuc3RhdGVTaXplO1xuICAgICAgICAgIGNvbnN0IGV4cGVjdGVkU2hhcGUgPSBbYmF0Y2hTaXplLCBkaW1dO1xuICAgICAgICAgIGlmICghdXRpbC5hcnJheXNFcXVhbCh2YWx1ZS5zaGFwZSwgZXhwZWN0ZWRTaGFwZSkpIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICAgIGBTdGF0ZSAke2luZGV4fSBpcyBpbmNvbXBhdGlibGUgd2l0aCBsYXllciAke3RoaXMubmFtZX06IGAgK1xuICAgICAgICAgICAgICAgIGBleHBlY3RlZCBzaGFwZT0ke2V4cGVjdGVkU2hhcGV9LCByZWNlaXZlZCBzaGFwZT0ke1xuICAgICAgICAgICAgICAgICAgICB2YWx1ZS5zaGFwZX1gKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgdGhpcy5zdGF0ZXNfW2luZGV4XSA9IHZhbHVlO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICB0aGlzLnN0YXRlc18gPSB0aGlzLnN0YXRlc18ubWFwKHN0YXRlID0+IHRmYy5rZWVwKHN0YXRlLmNsb25lKCkpKTtcbiAgICB9KTtcbiAgfVxuXG4gIGFwcGx5KFxuICAgICAgaW5wdXRzOiBUZW5zb3J8VGVuc29yW118U3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSxcbiAgICAgIGt3YXJncz86IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdIHtcbiAgICAvLyBUT0RPKGNhaXMpOiBGaWd1cmUgb3V0IHdoZXRoZXIgaW5pdGlhbFN0YXRlIGlzIGluIGt3YXJncyBvciBpbnB1dHMuXG4gICAgbGV0IGluaXRpYWxTdGF0ZTogVGVuc29yW118U3ltYm9saWNUZW5zb3JbXSA9XG4gICAgICAgIGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snaW5pdGlhbFN0YXRlJ107XG4gICAgbGV0IGNvbnN0YW50czogVGVuc29yW118U3ltYm9saWNUZW5zb3JbXSA9XG4gICAgICAgIGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snY29uc3RhbnRzJ107XG4gICAgaWYgKGt3YXJncyA9PSBudWxsKSB7XG4gICAgICBrd2FyZ3MgPSB7fTtcbiAgICB9XG5cbiAgICBjb25zdCBzdGFuZGFyZGl6ZWQgPVxuICAgICAgICBzdGFuZGFyZGl6ZUFyZ3MoaW5wdXRzLCBpbml0aWFsU3RhdGUsIGNvbnN0YW50cywgdGhpcy5udW1Db25zdGFudHMpO1xuICAgIGlucHV0cyA9IHN0YW5kYXJkaXplZC5pbnB1dHM7XG4gICAgaW5pdGlhbFN0YXRlID0gc3RhbmRhcmRpemVkLmluaXRpYWxTdGF0ZTtcbiAgICBjb25zdGFudHMgPSBzdGFuZGFyZGl6ZWQuY29uc3RhbnRzO1xuXG4gICAgLy8gSWYgYW55IG9mIGBpbml0aWFsX3N0YXRlYCBvciBgY29uc3RhbnRzYCBhcmUgc3BlY2lmaWVkIGFuZCBhcmVcbiAgICAvLyBgdGYuU3ltYm9saWNUZW5zb3JgcywgdGhlbiBhZGQgdGhlbSB0byB0aGUgaW5wdXRzIGFuZCB0ZW1wb3JhcmlseSBtb2RpZnlcbiAgICAvLyB0aGUgaW5wdXRfc3BlYyB0byBpbmNsdWRlIHRoZW0uXG5cbiAgICBsZXQgYWRkaXRpb25hbElucHV0czogQXJyYXk8VGVuc29yfFN5bWJvbGljVGVuc29yPiA9IFtdO1xuICAgIGxldCBhZGRpdGlvbmFsU3BlY3M6IElucHV0U3BlY1tdID0gW107XG4gICAgaWYgKGluaXRpYWxTdGF0ZSAhPSBudWxsKSB7XG4gICAgICBrd2FyZ3NbJ2luaXRpYWxTdGF0ZSddID0gaW5pdGlhbFN0YXRlO1xuICAgICAgYWRkaXRpb25hbElucHV0cyA9IGFkZGl0aW9uYWxJbnB1dHMuY29uY2F0KGluaXRpYWxTdGF0ZSk7XG4gICAgICB0aGlzLnN0YXRlU3BlYyA9IFtdO1xuICAgICAgZm9yIChjb25zdCBzdGF0ZSBvZiBpbml0aWFsU3RhdGUpIHtcbiAgICAgICAgdGhpcy5zdGF0ZVNwZWMucHVzaChuZXcgSW5wdXRTcGVjKHtzaGFwZTogc3RhdGUuc2hhcGV9KSk7XG4gICAgICB9XG4gICAgICAvLyBUT0RPKGNhaXMpOiBVc2UgdGhlIGZvbGxvd2luZyBpbnN0ZWFkLlxuICAgICAgLy8gdGhpcy5zdGF0ZVNwZWMgPSBpbml0aWFsU3RhdGUubWFwKHN0YXRlID0+IG5ldyBJbnB1dFNwZWMoe3NoYXBlOlxuICAgICAgLy8gc3RhdGUuc2hhcGV9KSk7XG4gICAgICBhZGRpdGlvbmFsU3BlY3MgPSBhZGRpdGlvbmFsU3BlY3MuY29uY2F0KHRoaXMuc3RhdGVTcGVjKTtcbiAgICB9XG4gICAgaWYgKGNvbnN0YW50cyAhPSBudWxsKSB7XG4gICAgICBrd2FyZ3NbJ2NvbnN0YW50cyddID0gY29uc3RhbnRzO1xuICAgICAgYWRkaXRpb25hbElucHV0cyA9IGFkZGl0aW9uYWxJbnB1dHMuY29uY2F0KGNvbnN0YW50cyk7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgdGhpcy5jb25zdGFudHNTcGVjLlxuICAgICAgdGhpcy5udW1Db25zdGFudHMgPSBjb25zdGFudHMubGVuZ3RoO1xuICAgIH1cblxuICAgIGNvbnN0IGlzVGVuc29yID0gYWRkaXRpb25hbElucHV0c1swXSBpbnN0YW5jZW9mIFN5bWJvbGljVGVuc29yO1xuICAgIGlmIChpc1RlbnNvcikge1xuICAgICAgLy8gQ29tcHV0ZSBmdWxsIGlucHV0IHNwZWMsIGluY2x1ZGluZyBzdGF0ZSBhbmQgY29uc3RhbnRzLlxuICAgICAgY29uc3QgZnVsbElucHV0ID1cbiAgICAgICAgICBbaW5wdXRzXS5jb25jYXQoYWRkaXRpb25hbElucHV0cykgYXMgVGVuc29yW10gfCBTeW1ib2xpY1RlbnNvcltdO1xuICAgICAgY29uc3QgZnVsbElucHV0U3BlYyA9IHRoaXMuaW5wdXRTcGVjLmNvbmNhdChhZGRpdGlvbmFsU3BlY3MpO1xuICAgICAgLy8gUGVyZm9ybSB0aGUgY2FsbCB3aXRoIHRlbXBvcmFyaWx5IHJlcGxhY2VkIGlucHV0U3BlYy5cbiAgICAgIGNvbnN0IG9yaWdpbmFsSW5wdXRTcGVjID0gdGhpcy5pbnB1dFNwZWM7XG4gICAgICB0aGlzLmlucHV0U3BlYyA9IGZ1bGxJbnB1dFNwZWM7XG4gICAgICBjb25zdCBvdXRwdXQgPSBzdXBlci5hcHBseShmdWxsSW5wdXQsIGt3YXJncyk7XG4gICAgICB0aGlzLmlucHV0U3BlYyA9IG9yaWdpbmFsSW5wdXRTcGVjO1xuICAgICAgcmV0dXJuIG91dHB1dDtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHN1cGVyLmFwcGx5KGlucHV0cywga3dhcmdzKTtcbiAgICB9XG4gIH1cblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICAvLyBJbnB1dCBzaGFwZTogYFtzYW1wbGVzLCB0aW1lIChwYWRkZWQgd2l0aCB6ZXJvcyksIGlucHV0X2RpbV1gLlxuICAgIC8vIE5vdGUgdGhhdCB0aGUgLmJ1aWxkKCkgbWV0aG9kIG9mIHN1YmNsYXNzZXMgKiptdXN0KiogZGVmaW5lXG4gICAgLy8gdGhpcy5pbnB1dFNwZWMgYW5kIHRoaXMuc3RhdGVTcGVjIG93aXRoIGNvbXBsZXRlIGlucHV0IHNoYXBlcy5cbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBtYXNrID0ga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydtYXNrJ10gYXMgVGVuc29yO1xuICAgICAgY29uc3QgdHJhaW5pbmcgPSBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ3RyYWluaW5nJ107XG4gICAgICBsZXQgaW5pdGlhbFN0YXRlOiBUZW5zb3JbXSA9XG4gICAgICAgICAga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydpbml0aWFsU3RhdGUnXTtcblxuICAgICAgaW5wdXRzID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgICAgaWYgKGluaXRpYWxTdGF0ZSA9PSBudWxsKSB7XG4gICAgICAgIGlmICh0aGlzLnN0YXRlZnVsKSB7XG4gICAgICAgICAgaW5pdGlhbFN0YXRlID0gdGhpcy5zdGF0ZXNfO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGluaXRpYWxTdGF0ZSA9IHRoaXMuZ2V0SW5pdGlhbFN0YXRlKGlucHV0cyk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgY29uc3QgbnVtU3RhdGVzID1cbiAgICAgICAgICBBcnJheS5pc0FycmF5KHRoaXMuY2VsbC5zdGF0ZVNpemUpID8gdGhpcy5jZWxsLnN0YXRlU2l6ZS5sZW5ndGggOiAxO1xuICAgICAgaWYgKGluaXRpYWxTdGF0ZS5sZW5ndGggIT09IG51bVN0YXRlcykge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBSTk4gTGF5ZXIgaGFzICR7bnVtU3RhdGVzfSBzdGF0ZShzKSBidXQgd2FzIHBhc3NlZCBgICtcbiAgICAgICAgICAgIGAke2luaXRpYWxTdGF0ZS5sZW5ndGh9IGluaXRpYWwgc3RhdGUocykuYCk7XG4gICAgICB9XG4gICAgICBpZiAodGhpcy51bnJvbGwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgJ0lnbm9yaW5nIHVucm9sbCA9IHRydWUgZm9yIFJOTiBsYXllciwgZHVlIHRvIGltcGVyYXRpdmUgYmFja2VuZC4nKTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgY2VsbENhbGxLd2FyZ3M6IEt3YXJncyA9IHt0cmFpbmluZ307XG5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBzdXBwb3J0IGZvciBjb25zdGFudHMuXG4gICAgICBjb25zdCBzdGVwID0gKGlucHV0czogVGVuc29yLCBzdGF0ZXM6IFRlbnNvcltdKSA9PiB7XG4gICAgICAgIC8vIGBpbnB1dHNgIGFuZCBgc3RhdGVzYCBhcmUgY29uY2F0ZW5hdGVkIHRvIGZvcm0gYSBzaW5nbGUgYEFycmF5YCBvZlxuICAgICAgICAvLyBgdGYuVGVuc29yYHMgYXMgdGhlIGlucHV0IHRvIGBjZWxsLmNhbGwoKWAuXG4gICAgICAgIGNvbnN0IG91dHB1dHMgPVxuICAgICAgICAgICAgdGhpcy5jZWxsLmNhbGwoW2lucHV0c10uY29uY2F0KHN0YXRlcyksIGNlbGxDYWxsS3dhcmdzKSBhcyBUZW5zb3JbXTtcbiAgICAgICAgLy8gTWFyc2hhbGwgdGhlIHJldHVybiB2YWx1ZSBpbnRvIG91dHB1dCBhbmQgbmV3IHN0YXRlcy5cbiAgICAgICAgcmV0dXJuIFtvdXRwdXRzWzBdLCBvdXRwdXRzLnNsaWNlKDEpXSBhcyBbVGVuc29yLCBUZW5zb3JbXV07XG4gICAgICB9O1xuXG4gICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgc3VwcG9ydCBmb3IgY29uc3RhbnRzLlxuXG4gICAgICBjb25zdCBybm5PdXRwdXRzID1cbiAgICAgICAgICBybm4oc3RlcCwgaW5wdXRzLCBpbml0aWFsU3RhdGUsIHRoaXMuZ29CYWNrd2FyZHMsIG1hc2ssIG51bGwsXG4gICAgICAgICAgICAgIHRoaXMudW5yb2xsLCB0aGlzLnJldHVyblNlcXVlbmNlcyk7XG4gICAgICBjb25zdCBsYXN0T3V0cHV0ID0gcm5uT3V0cHV0c1swXTtcbiAgICAgIGNvbnN0IG91dHB1dHMgPSBybm5PdXRwdXRzWzFdO1xuICAgICAgY29uc3Qgc3RhdGVzID0gcm5uT3V0cHV0c1syXTtcblxuICAgICAgaWYgKHRoaXMuc3RhdGVmdWwpIHtcbiAgICAgICAgdGhpcy5yZXNldFN0YXRlcyhzdGF0ZXMsIHRyYWluaW5nKTtcbiAgICAgIH1cblxuICAgICAgY29uc3Qgb3V0cHV0ID0gdGhpcy5yZXR1cm5TZXF1ZW5jZXMgPyBvdXRwdXRzIDogbGFzdE91dHB1dDtcblxuICAgICAgLy8gVE9ETyhjYWlzKTogUG9ycGVydHkgc2V0IGxlYXJuaW5nIHBoYXNlIGZsYWcuXG5cbiAgICAgIGlmICh0aGlzLnJldHVyblN0YXRlKSB7XG4gICAgICAgIHJldHVybiBbb3V0cHV0XS5jb25jYXQoc3RhdGVzKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBvdXRwdXQ7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICBnZXRJbml0aWFsU3RhdGUoaW5wdXRzOiBUZW5zb3IpOiBUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgLy8gQnVpbGQgYW4gYWxsLXplcm8gdGVuc29yIG9mIHNoYXBlIFtzYW1wbGVzLCBvdXRwdXREaW1dLlxuICAgICAgLy8gW1NhbXBsZXMsIHRpbWVTdGVwcywgaW5wdXREaW1dLlxuICAgICAgbGV0IGluaXRpYWxTdGF0ZSA9IHRmYy56ZXJvcyhpbnB1dHMuc2hhcGUpO1xuICAgICAgLy8gW1NhbXBsZXNdLlxuICAgICAgaW5pdGlhbFN0YXRlID0gdGZjLnN1bShpbml0aWFsU3RhdGUsIFsxLCAyXSk7XG4gICAgICBpbml0aWFsU3RhdGUgPSBLLmV4cGFuZERpbXMoaW5pdGlhbFN0YXRlKTsgIC8vIFtTYW1wbGVzLCAxXS5cblxuICAgICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5jZWxsLnN0YXRlU2l6ZSkpIHtcbiAgICAgICAgcmV0dXJuIHRoaXMuY2VsbC5zdGF0ZVNpemUubWFwKFxuICAgICAgICAgICAgZGltID0+IGRpbSA+IDEgPyBLLnRpbGUoaW5pdGlhbFN0YXRlLCBbMSwgZGltXSkgOiBpbml0aWFsU3RhdGUpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIHRoaXMuY2VsbC5zdGF0ZVNpemUgPiAxID9cbiAgICAgICAgICAgIFtLLnRpbGUoaW5pdGlhbFN0YXRlLCBbMSwgdGhpcy5jZWxsLnN0YXRlU2l6ZV0pXSA6XG4gICAgICAgICAgICBbaW5pdGlhbFN0YXRlXTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIGdldCB0cmFpbmFibGVXZWlnaHRzKCk6IExheWVyVmFyaWFibGVbXSB7XG4gICAgaWYgKCF0aGlzLnRyYWluYWJsZSkge1xuICAgICAgcmV0dXJuIFtdO1xuICAgIH1cbiAgICAvLyBQb3J0aW5nIE5vdGU6IEluIFR5cGVTY3JpcHQsIGB0aGlzYCBpcyBhbHdheXMgYW4gaW5zdGFuY2Ugb2YgYExheWVyYC5cbiAgICByZXR1cm4gdGhpcy5jZWxsLnRyYWluYWJsZVdlaWdodHM7XG4gIH1cblxuICBnZXQgbm9uVHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIC8vIFBvcnRpbmcgTm90ZTogSW4gVHlwZVNjcmlwdCwgYHRoaXNgIGlzIGFsd2F5cyBhbiBpbnN0YW5jZSBvZiBgTGF5ZXJgLlxuICAgIGlmICghdGhpcy50cmFpbmFibGUpIHtcbiAgICAgIHJldHVybiB0aGlzLmNlbGwud2VpZ2h0cztcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuY2VsbC5ub25UcmFpbmFibGVXZWlnaHRzO1xuICB9XG5cbiAgc2V0RmFzdFdlaWdodEluaXREdXJpbmdCdWlsZCh2YWx1ZTogYm9vbGVhbikge1xuICAgIHN1cGVyLnNldEZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQodmFsdWUpO1xuICAgIGlmICh0aGlzLmNlbGwgIT0gbnVsbCkge1xuICAgICAgdGhpcy5jZWxsLnNldEZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQodmFsdWUpO1xuICAgIH1cbiAgfVxuXG4gIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcblxuICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge1xuICAgICAgcmV0dXJuU2VxdWVuY2VzOiB0aGlzLnJldHVyblNlcXVlbmNlcyxcbiAgICAgIHJldHVyblN0YXRlOiB0aGlzLnJldHVyblN0YXRlLFxuICAgICAgZ29CYWNrd2FyZHM6IHRoaXMuZ29CYWNrd2FyZHMsXG4gICAgICBzdGF0ZWZ1bDogdGhpcy5zdGF0ZWZ1bCxcbiAgICAgIHVucm9sbDogdGhpcy51bnJvbGwsXG4gICAgfTtcblxuICAgIGlmICh0aGlzLm51bUNvbnN0YW50cyAhPSBudWxsKSB7XG4gICAgICBjb25maWdbJ251bUNvbnN0YW50cyddID0gdGhpcy5udW1Db25zdGFudHM7XG4gICAgfVxuXG4gICAgY29uc3QgY2VsbENvbmZpZyA9IHRoaXMuY2VsbC5nZXRDb25maWcoKTtcblxuICAgIGlmICh0aGlzLmdldENsYXNzTmFtZSgpID09PSBSTk4uY2xhc3NOYW1lKSB7XG4gICAgICBjb25maWdbJ2NlbGwnXSA9IHtcbiAgICAgICAgJ2NsYXNzTmFtZSc6IHRoaXMuY2VsbC5nZXRDbGFzc05hbWUoKSxcbiAgICAgICAgJ2NvbmZpZyc6IGNlbGxDb25maWcsXG4gICAgICB9IGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdFZhbHVlO1xuICAgIH1cblxuICAgIC8vIHRoaXMgb3JkZXIgaXMgbmVjZXNzYXJ5LCB0byBwcmV2ZW50IGNlbGwgbmFtZSBmcm9tIHJlcGxhY2luZyBsYXllciBuYW1lXG4gICAgcmV0dXJuIHsuLi5jZWxsQ29uZmlnLCAuLi5iYXNlQ29uZmlnLCAuLi5jb25maWd9O1xuICB9XG5cbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBmcm9tQ29uZmlnPFQgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sXG4gICAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICAgIGN1c3RvbU9iamVjdHMgPSB7fSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QpOiBUIHtcbiAgICBjb25zdCBjZWxsQ29uZmlnID0gY29uZmlnWydjZWxsJ10gYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0O1xuICAgIGNvbnN0IGNlbGwgPSBkZXNlcmlhbGl6ZShjZWxsQ29uZmlnLCBjdXN0b21PYmplY3RzKSBhcyBSTk5DZWxsO1xuICAgIHJldHVybiBuZXcgY2xzKE9iamVjdC5hc3NpZ24oY29uZmlnLCB7Y2VsbH0pKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFJOTik7XG5cbi8vIFBvcnRpbmcgTm90ZTogVGhpcyBpcyBhIGNvbW1vbiBwYXJlbnQgY2xhc3MgZm9yIFJOTiBjZWxscy4gVGhlcmUgaXMgbm9cbi8vIGVxdWl2YWxlbnQgb2YgdGhpcyBpbiBQeUtlcmFzLiBIYXZpbmcgYSBjb21tb24gcGFyZW50IGNsYXNzIGZvcmdvZXMgdGhlXG4vLyAgbmVlZCBmb3IgYGhhc19hdHRyKGNlbGwsIC4uLilgIGNoZWNrcyBvciBpdHMgVHlwZVNjcmlwdCBlcXVpdmFsZW50LlxuLyoqXG4gKiBBbiBSTk5DZWxsIGxheWVyLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBSTk5DZWxsIGV4dGVuZHMgTGF5ZXIge1xuICAvKipcbiAgICogU2l6ZShzKSBvZiB0aGUgc3RhdGVzLlxuICAgKiBGb3IgUk5OIGNlbGxzIHdpdGggb25seSBhIHNpbmdsZSBzdGF0ZSwgdGhpcyBpcyBhIHNpbmdsZSBpbnRlZ2VyLlxuICAgKi9cbiAgLy8gU2VlXG4gIC8vIGh0dHBzOi8vd3d3LnR5cGVzY3JpcHRsYW5nLm9yZy9kb2NzL2hhbmRib29rL3JlbGVhc2Utbm90ZXMvdHlwZXNjcmlwdC00LTAuaHRtbCNwcm9wZXJ0aWVzLW92ZXJyaWRpbmctYWNjZXNzb3JzLWFuZC12aWNlLXZlcnNhLWlzLWFuLWVycm9yXG4gIHB1YmxpYyBhYnN0cmFjdCBzdGF0ZVNpemU6IG51bWJlcnxudW1iZXJbXTtcbiAgcHVibGljIGRyb3BvdXRNYXNrOiBUZW5zb3J8VGVuc29yW107XG4gIHB1YmxpYyByZWN1cnJlbnREcm9wb3V0TWFzazogVGVuc29yfFRlbnNvcltdO1xufVxuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgU2ltcGxlUk5OQ2VsbExheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiB1bml0czogUG9zaXRpdmUgaW50ZWdlciwgZGltZW5zaW9uYWxpdHkgb2YgdGhlIG91dHB1dCBzcGFjZS5cbiAgICovXG4gIHVuaXRzOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIEFjdGl2YXRpb24gZnVuY3Rpb24gdG8gdXNlLlxuICAgKiBEZWZhdWx0OiBoeXBlcmJvbGljIHRhbmdlbnQgKCd0YW5oJykuXG4gICAqIElmIHlvdSBwYXNzIGBudWxsYCwgICdsaW5lYXInIGFjdGl2YXRpb24gd2lsbCBiZSBhcHBsaWVkLlxuICAgKi9cbiAgYWN0aXZhdGlvbj86IEFjdGl2YXRpb25JZGVudGlmaWVyO1xuXG4gIC8qKlxuICAgKiBXaGV0aGVyIHRoZSBsYXllciB1c2VzIGEgYmlhcyB2ZWN0b3IuXG4gICAqL1xuICB1c2VCaWFzPzogYm9vbGVhbjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBga2VybmVsYCB3ZWlnaHRzIG1hdHJpeCwgdXNlZCBmb3IgdGhlIGxpbmVhclxuICAgKiB0cmFuc2Zvcm1hdGlvbiBvZiB0aGUgaW5wdXRzLlxuICAgKi9cbiAga2VybmVsSW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgYHJlY3VycmVudEtlcm5lbGAgd2VpZ2h0cyBtYXRyaXgsIHVzZWQgZm9yXG4gICAqIGxpbmVhciB0cmFuc2Zvcm1hdGlvbiBvZiB0aGUgcmVjdXJyZW50IHN0YXRlLlxuICAgKi9cbiAgcmVjdXJyZW50SW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgYmlhcyB2ZWN0b3IuXG4gICAqL1xuICBiaWFzSW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGBrZXJuZWxgIHdlaWdodHMgbWF0cml4LlxuICAgKi9cbiAga2VybmVsUmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGByZWN1cnJlbnRfa2VybmVsYCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIHJlY3VycmVudFJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJJZGVudGlmaWVyfFJlZ3VsYXJpemVyO1xuXG4gIC8qKlxuICAgKiBSZWd1bGFyaXplciBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIGJpYXNSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcblxuICAvKipcbiAgICogQ29uc3RyYWludCBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBga2VybmVsYCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIGtlcm5lbENvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGByZWN1cnJlbnRLZXJuZWxgIHdlaWdodHMgbWF0cml4LlxuICAgKi9cbiAgcmVjdXJyZW50Q29uc3RyYWludD86IENvbnN0cmFpbnRJZGVudGlmaWVyfENvbnN0cmFpbnQ7XG5cbiAgLyoqXG4gICAqIENvbnN0cmFpbnRmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIGJpYXNDb25zdHJhaW50PzogQ29uc3RyYWludElkZW50aWZpZXJ8Q29uc3RyYWludDtcblxuICAvKipcbiAgICogRmxvYXQgbnVtYmVyIGJldHdlZW4gMCBhbmQgMS4gRnJhY3Rpb24gb2YgdGhlIHVuaXRzIHRvIGRyb3AgZm9yIHRoZSBsaW5lYXJcbiAgICogdHJhbnNmb3JtYXRpb24gb2YgdGhlIGlucHV0cy5cbiAgICovXG4gIGRyb3BvdXQ/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIEZsb2F0IG51bWJlciBiZXR3ZWVuIDAgYW5kIDEuIEZyYWN0aW9uIG9mIHRoZSB1bml0cyB0byBkcm9wIGZvciB0aGUgbGluZWFyXG4gICAqIHRyYW5zZm9ybWF0aW9uIG9mIHRoZSByZWN1cnJlbnQgc3RhdGUuXG4gICAqL1xuICByZWN1cnJlbnREcm9wb3V0PzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBUaGlzIGlzIGFkZGVkIGZvciB0ZXN0IERJIHB1cnBvc2UuXG4gICAqL1xuICBkcm9wb3V0RnVuYz86IEZ1bmN0aW9uO1xufVxuXG5leHBvcnQgY2xhc3MgU2ltcGxlUk5OQ2VsbCBleHRlbmRzIFJOTkNlbGwge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdTaW1wbGVSTk5DZWxsJztcbiAgcmVhZG9ubHkgdW5pdHM6IG51bWJlcjtcbiAgcmVhZG9ubHkgYWN0aXZhdGlvbjogQWN0aXZhdGlvbjtcbiAgcmVhZG9ubHkgdXNlQmlhczogYm9vbGVhbjtcblxuICByZWFkb25seSBrZXJuZWxJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudEluaXRpYWxpemVyOiBJbml0aWFsaXplcjtcbiAgcmVhZG9ubHkgYmlhc0luaXRpYWxpemVyOiBJbml0aWFsaXplcjtcblxuICByZWFkb25seSBrZXJuZWxDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSByZWN1cnJlbnRDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSBiaWFzQ29uc3RyYWludDogQ29uc3RyYWludDtcblxuICByZWFkb25seSBrZXJuZWxSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudFJlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcbiAgcmVhZG9ubHkgYmlhc1JlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcblxuICByZWFkb25seSBkcm9wb3V0OiBudW1iZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudERyb3BvdXQ6IG51bWJlcjtcbiAgcmVhZG9ubHkgZHJvcG91dEZ1bmM6IEZ1bmN0aW9uO1xuXG4gIHJlYWRvbmx5IHN0YXRlU2l6ZTogbnVtYmVyO1xuXG4gIGtlcm5lbDogTGF5ZXJWYXJpYWJsZTtcbiAgcmVjdXJyZW50S2VybmVsOiBMYXllclZhcmlhYmxlO1xuICBiaWFzOiBMYXllclZhcmlhYmxlO1xuXG4gIHJlYWRvbmx5IERFRkFVTFRfQUNUSVZBVElPTiA9ICd0YW5oJztcbiAgcmVhZG9ubHkgREVGQVVMVF9LRVJORUxfSU5JVElBTElaRVIgPSAnZ2xvcm90Tm9ybWFsJztcbiAgcmVhZG9ubHkgREVGQVVMVF9SRUNVUlJFTlRfSU5JVElBTElaRVIgPSAnb3J0aG9nb25hbCc7XG4gIHJlYWRvbmx5IERFRkFVTFRfQklBU19JTklUSUFMSVpFUjogSW5pdGlhbGl6ZXJJZGVudGlmaWVyID0gJ3plcm9zJztcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBTaW1wbGVSTk5DZWxsTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gICAgdGhpcy51bml0cyA9IGFyZ3MudW5pdHM7XG4gICAgYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMudW5pdHMsIGB1bml0c2ApO1xuICAgIHRoaXMuYWN0aXZhdGlvbiA9IGdldEFjdGl2YXRpb24oXG4gICAgICAgIGFyZ3MuYWN0aXZhdGlvbiA9PSBudWxsID8gdGhpcy5ERUZBVUxUX0FDVElWQVRJT04gOiBhcmdzLmFjdGl2YXRpb24pO1xuICAgIHRoaXMudXNlQmlhcyA9IGFyZ3MudXNlQmlhcyA9PSBudWxsID8gdHJ1ZSA6IGFyZ3MudXNlQmlhcztcblxuICAgIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIgPSBnZXRJbml0aWFsaXplcihcbiAgICAgICAgYXJncy5rZXJuZWxJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfS0VSTkVMX0lOSVRJQUxJWkVSKTtcbiAgICB0aGlzLnJlY3VycmVudEluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXIoXG4gICAgICAgIGFyZ3MucmVjdXJyZW50SW5pdGlhbGl6ZXIgfHwgdGhpcy5ERUZBVUxUX1JFQ1VSUkVOVF9JTklUSUFMSVpFUik7XG5cbiAgICB0aGlzLmJpYXNJbml0aWFsaXplciA9XG4gICAgICAgIGdldEluaXRpYWxpemVyKGFyZ3MuYmlhc0luaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9CSUFTX0lOSVRJQUxJWkVSKTtcblxuICAgIHRoaXMua2VybmVsUmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLmtlcm5lbFJlZ3VsYXJpemVyKTtcbiAgICB0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5yZWN1cnJlbnRSZWd1bGFyaXplcik7XG4gICAgdGhpcy5iaWFzUmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLmJpYXNSZWd1bGFyaXplcik7XG5cbiAgICB0aGlzLmtlcm5lbENvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGFyZ3Mua2VybmVsQ29uc3RyYWludCk7XG4gICAgdGhpcy5yZWN1cnJlbnRDb25zdHJhaW50ID0gZ2V0Q29uc3RyYWludChhcmdzLnJlY3VycmVudENvbnN0cmFpbnQpO1xuICAgIHRoaXMuYmlhc0NvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGFyZ3MuYmlhc0NvbnN0cmFpbnQpO1xuXG4gICAgdGhpcy5kcm9wb3V0ID0gbWF0aF91dGlscy5taW4oXG4gICAgICAgIFsxLCBtYXRoX3V0aWxzLm1heChbMCwgYXJncy5kcm9wb3V0ID09IG51bGwgPyAwIDogYXJncy5kcm9wb3V0XSldKTtcbiAgICB0aGlzLnJlY3VycmVudERyb3BvdXQgPSBtYXRoX3V0aWxzLm1pbihbXG4gICAgICAxLFxuICAgICAgbWF0aF91dGlscy5tYXgoXG4gICAgICAgICAgWzAsIGFyZ3MucmVjdXJyZW50RHJvcG91dCA9PSBudWxsID8gMCA6IGFyZ3MucmVjdXJyZW50RHJvcG91dF0pXG4gICAgXSk7XG4gICAgdGhpcy5kcm9wb3V0RnVuYyA9IGFyZ3MuZHJvcG91dEZ1bmM7XG4gICAgdGhpcy5zdGF0ZVNpemUgPSB0aGlzLnVuaXRzO1xuICAgIHRoaXMuZHJvcG91dE1hc2sgPSBudWxsO1xuICAgIHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgPSBudWxsO1xuICB9XG5cbiAgYnVpbGQoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IHZvaWQge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgLy8gVE9ETyhjYWlzKTogVXNlIHJlZ3VsYXJpemVyLlxuICAgIHRoaXMua2VybmVsID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICdrZXJuZWwnLCBbaW5wdXRTaGFwZVtpbnB1dFNoYXBlLmxlbmd0aCAtIDFdLCB0aGlzLnVuaXRzXSwgbnVsbCxcbiAgICAgICAgdGhpcy5rZXJuZWxJbml0aWFsaXplciwgdGhpcy5rZXJuZWxSZWd1bGFyaXplciwgdHJ1ZSxcbiAgICAgICAgdGhpcy5rZXJuZWxDb25zdHJhaW50KTtcbiAgICB0aGlzLnJlY3VycmVudEtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAncmVjdXJyZW50X2tlcm5lbCcsIFt0aGlzLnVuaXRzLCB0aGlzLnVuaXRzXSwgbnVsbCxcbiAgICAgICAgdGhpcy5yZWN1cnJlbnRJbml0aWFsaXplciwgdGhpcy5yZWN1cnJlbnRSZWd1bGFyaXplciwgdHJ1ZSxcbiAgICAgICAgdGhpcy5yZWN1cnJlbnRDb25zdHJhaW50KTtcbiAgICBpZiAodGhpcy51c2VCaWFzKSB7XG4gICAgICB0aGlzLmJpYXMgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgICAnYmlhcycsIFt0aGlzLnVuaXRzXSwgbnVsbCwgdGhpcy5iaWFzSW5pdGlhbGl6ZXIsXG4gICAgICAgICAgdGhpcy5iaWFzUmVndWxhcml6ZXIsIHRydWUsIHRoaXMuYmlhc0NvbnN0cmFpbnQpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmJpYXMgPSBudWxsO1xuICAgIH1cbiAgICB0aGlzLmJ1aWx0ID0gdHJ1ZTtcbiAgfVxuXG4gIC8vIFBvcnRpbmcgTm90ZTogUHlLZXJhcycgZXF1aXZhbGVudCBvZiB0aGlzIG1ldGhvZCB0YWtlcyB0d28gdGVuc29yIGlucHV0czpcbiAgLy8gICBgaW5wdXRzYCBhbmQgYHN0YXRlc2AuIEhlcmUsIHRoZSB0d28gdGVuc29ycyBhcmUgY29tYmluZWQgaW50byBhblxuICAvLyAgIGBUZW5zb3JbXWAgQXJyYXkgYXMgdGhlIGZpcnN0IGlucHV0IGFyZ3VtZW50LlxuICAvLyAgIFNpbWlsYXJseSwgUHlLZXJhcycgZXF1aXZhbGVudCBvZiB0aGlzIG1ldGhvZCByZXR1cm5zIHR3byB2YWx1ZXM6XG4gIC8vICAgIGBvdXRwdXRgIGFuZCBgW291dHB1dF1gLiBIZXJlIHRoZSB0d28gYXJlIGNvbWJpbmVkIGludG8gb25lIGxlbmd0aC0yXG4gIC8vICAgIGBUZW5zb3JbXWAsIGNvbnNpc3Rpbmcgb2YgYG91dHB1dGAgcmVwZWF0ZWQuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpbnB1dHMgPSBpbnB1dHMgYXMgVGVuc29yW107XG4gICAgICBpZiAoaW5wdXRzLmxlbmd0aCAhPT0gMikge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBTaW1wbGVSTk5DZWxsIGV4cGVjdHMgMiBpbnB1dCBUZW5zb3JzLCBnb3QgJHtpbnB1dHMubGVuZ3RofS5gKTtcbiAgICAgIH1cbiAgICAgIGxldCBwcmV2T3V0cHV0ID0gaW5wdXRzWzFdO1xuICAgICAgaW5wdXRzID0gaW5wdXRzWzBdO1xuICAgICAgY29uc3QgdHJhaW5pbmcgPSBrd2FyZ3NbJ3RyYWluaW5nJ10gPT0gbnVsbCA/IGZhbHNlIDoga3dhcmdzWyd0cmFpbmluZyddO1xuXG4gICAgICBpZiAoMCA8IHRoaXMuZHJvcG91dCAmJiB0aGlzLmRyb3BvdXQgPCAxICYmIHRoaXMuZHJvcG91dE1hc2sgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLmRyb3BvdXRNYXNrID0gZ2VuZXJhdGVEcm9wb3V0TWFzayh7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9uZXM6ICgpID0+IHRmYy5vbmVzTGlrZShpbnB1dHMgYXMgVGVuc29yKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmF0ZTogdGhpcy5kcm9wb3V0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0cmFpbmluZyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZHJvcG91dEZ1bmM6IHRoaXMuZHJvcG91dEZ1bmMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICB9KSBhcyBUZW5zb3I7XG4gICAgICB9XG4gICAgICBpZiAoMCA8IHRoaXMucmVjdXJyZW50RHJvcG91dCAmJiB0aGlzLnJlY3VycmVudERyb3BvdXQgPCAxICYmXG4gICAgICAgICAgdGhpcy5yZWN1cnJlbnREcm9wb3V0TWFzayA9PSBudWxsKSB7XG4gICAgICAgIHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgPSBnZW5lcmF0ZURyb3BvdXRNYXNrKHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb25lczogKCkgPT4gdGZjLm9uZXNMaWtlKHByZXZPdXRwdXQpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICByYXRlOiB0aGlzLnJlY3VycmVudERyb3BvdXQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRyYWluaW5nLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkcm9wb3V0RnVuYzogdGhpcy5kcm9wb3V0RnVuYyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pIGFzIFRlbnNvcjtcbiAgICAgIH1cbiAgICAgIGxldCBoOiBUZW5zb3I7XG4gICAgICBjb25zdCBkcE1hc2s6IFRlbnNvciA9IHRoaXMuZHJvcG91dE1hc2sgYXMgVGVuc29yO1xuICAgICAgY29uc3QgcmVjRHBNYXNrOiBUZW5zb3IgPSB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrIGFzIFRlbnNvcjtcbiAgICAgIGlmIChkcE1hc2sgIT0gbnVsbCkge1xuICAgICAgICBoID0gSy5kb3QodGZjLm11bChpbnB1dHMsIGRwTWFzayksIHRoaXMua2VybmVsLnJlYWQoKSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBoID0gSy5kb3QoaW5wdXRzLCB0aGlzLmtlcm5lbC5yZWFkKCkpO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMuYmlhcyAhPSBudWxsKSB7XG4gICAgICAgIGggPSBLLmJpYXNBZGQoaCwgdGhpcy5iaWFzLnJlYWQoKSk7XG4gICAgICB9XG4gICAgICBpZiAocmVjRHBNYXNrICE9IG51bGwpIHtcbiAgICAgICAgcHJldk91dHB1dCA9IHRmYy5tdWwocHJldk91dHB1dCwgcmVjRHBNYXNrKTtcbiAgICAgIH1cbiAgICAgIGxldCBvdXRwdXQgPSB0ZmMuYWRkKGgsIEsuZG90KHByZXZPdXRwdXQsIHRoaXMucmVjdXJyZW50S2VybmVsLnJlYWQoKSkpO1xuICAgICAgaWYgKHRoaXMuYWN0aXZhdGlvbiAhPSBudWxsKSB7XG4gICAgICAgIG91dHB1dCA9IHRoaXMuYWN0aXZhdGlvbi5hcHBseShvdXRwdXQpO1xuICAgICAgfVxuXG4gICAgICAvLyBUT0RPKGNhaXMpOiBQcm9wZXJseSBzZXQgbGVhcm5pbmcgcGhhc2Ugb24gb3V0cHV0IHRlbnNvcj9cbiAgICAgIHJldHVybiBbb3V0cHV0LCBvdXRwdXRdO1xuICAgIH0pO1xuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuXG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7XG4gICAgICB1bml0czogdGhpcy51bml0cyxcbiAgICAgIGFjdGl2YXRpb246IHNlcmlhbGl6ZUFjdGl2YXRpb24odGhpcy5hY3RpdmF0aW9uKSxcbiAgICAgIHVzZUJpYXM6IHRoaXMudXNlQmlhcyxcbiAgICAgIGtlcm5lbEluaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmtlcm5lbEluaXRpYWxpemVyKSxcbiAgICAgIHJlY3VycmVudEluaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLnJlY3VycmVudEluaXRpYWxpemVyKSxcbiAgICAgIGJpYXNJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5iaWFzSW5pdGlhbGl6ZXIpLFxuICAgICAga2VybmVsUmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMua2VybmVsUmVndWxhcml6ZXIpLFxuICAgICAgcmVjdXJyZW50UmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMucmVjdXJyZW50UmVndWxhcml6ZXIpLFxuICAgICAgYmlhc1JlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmJpYXNSZWd1bGFyaXplciksXG4gICAgICBhY3Rpdml0eVJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmFjdGl2aXR5UmVndWxhcml6ZXIpLFxuICAgICAga2VybmVsQ29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLmtlcm5lbENvbnN0cmFpbnQpLFxuICAgICAgcmVjdXJyZW50Q29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLnJlY3VycmVudENvbnN0cmFpbnQpLFxuICAgICAgYmlhc0NvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5iaWFzQ29uc3RyYWludCksXG4gICAgICBkcm9wb3V0OiB0aGlzLmRyb3BvdXQsXG4gICAgICByZWN1cnJlbnREcm9wb3V0OiB0aGlzLnJlY3VycmVudERyb3BvdXQsXG4gICAgfTtcblxuICAgIHJldHVybiB7Li4uYmFzZUNvbmZpZywgLi4uY29uZmlnfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFNpbXBsZVJOTkNlbGwpO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgU2ltcGxlUk5OTGF5ZXJBcmdzIGV4dGVuZHMgQmFzZVJOTkxheWVyQXJncyB7XG4gIC8qKlxuICAgKiBQb3NpdGl2ZSBpbnRlZ2VyLCBkaW1lbnNpb25hbGl0eSBvZiB0aGUgb3V0cHV0IHNwYWNlLlxuICAgKi9cbiAgdW5pdHM6IG51bWJlcjtcblxuICAvKipcbiAgICogQWN0aXZhdGlvbiBmdW5jdGlvbiB0byB1c2UuXG4gICAqXG4gICAqIERlZmF1bHRzIHRvICBoeXBlcmJvbGljIHRhbmdlbnQgKGB0YW5oYClcbiAgICpcbiAgICogSWYgeW91IHBhc3MgYG51bGxgLCBubyBhY3RpdmF0aW9uIHdpbGwgYmUgYXBwbGllZC5cbiAgICovXG4gIGFjdGl2YXRpb24/OiBBY3RpdmF0aW9uSWRlbnRpZmllcjtcblxuICAvKipcbiAgICogV2hldGhlciB0aGUgbGF5ZXIgdXNlcyBhIGJpYXMgdmVjdG9yLlxuICAgKi9cbiAgdXNlQmlhcz86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgYGtlcm5lbGAgd2VpZ2h0cyBtYXRyaXgsIHVzZWQgZm9yIHRoZSBsaW5lYXJcbiAgICogdHJhbnNmb3JtYXRpb24gb2YgdGhlIGlucHV0cy5cbiAgICovXG4gIGtlcm5lbEluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBJbml0aWFsaXplciBmb3IgdGhlIGByZWN1cnJlbnRLZXJuZWxgIHdlaWdodHMgbWF0cml4LCB1c2VkIGZvclxuICAgKiBsaW5lYXIgdHJhbnNmb3JtYXRpb24gb2YgdGhlIHJlY3VycmVudCBzdGF0ZS5cbiAgICovXG4gIHJlY3VycmVudEluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBJbml0aWFsaXplciBmb3IgdGhlIGJpYXMgdmVjdG9yLlxuICAgKi9cbiAgYmlhc0luaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBSZWd1bGFyaXplciBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBrZXJuZWwgd2VpZ2h0cyBtYXRyaXguXG4gICAqL1xuICBrZXJuZWxSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcblxuICAvKipcbiAgICogUmVndWxhcml6ZXIgZnVuY3Rpb24gYXBwbGllZCB0byB0aGUgcmVjdXJyZW50S2VybmVsIHdlaWdodHMgbWF0cml4LlxuICAgKi9cbiAgcmVjdXJyZW50UmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGJpYXMgdmVjdG9yLlxuICAgKi9cbiAgYmlhc1JlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJJZGVudGlmaWVyfFJlZ3VsYXJpemVyO1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGtlcm5lbCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIGtlcm5lbENvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIHJlY3VycmVudEtlcm5lbCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIHJlY3VycmVudENvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGJpYXMgdmVjdG9yLlxuICAgKi9cbiAgYmlhc0NvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xuXG4gIC8qKlxuICAgKiBOdW1iZXIgYmV0d2VlbiAwIGFuZCAxLiBGcmFjdGlvbiBvZiB0aGUgdW5pdHMgdG8gZHJvcCBmb3IgdGhlIGxpbmVhclxuICAgKiB0cmFuc2Zvcm1hdGlvbiBvZiB0aGUgaW5wdXRzLlxuICAgKi9cbiAgZHJvcG91dD86IG51bWJlcjtcblxuICAvKipcbiAgICogTnVtYmVyIGJldHdlZW4gMCBhbmQgMS4gRnJhY3Rpb24gb2YgdGhlIHVuaXRzIHRvIGRyb3AgZm9yIHRoZSBsaW5lYXJcbiAgICogdHJhbnNmb3JtYXRpb24gb2YgdGhlIHJlY3VycmVudCBzdGF0ZS5cbiAgICovXG4gIHJlY3VycmVudERyb3BvdXQ/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIFRoaXMgaXMgYWRkZWQgZm9yIHRlc3QgREkgcHVycG9zZS5cbiAgICovXG4gIGRyb3BvdXRGdW5jPzogRnVuY3Rpb247XG59XG5cbi8qKlxuICogUk5OTGF5ZXJDb25maWcgaXMgaWRlbnRpY2FsIHRvIEJhc2VSTk5MYXllckNvbmZpZywgZXhjZXB0IGl0IG1ha2VzIHRoZVxuICogYGNlbGxgIHByb3BlcnR5IHJlcXVpcmVkLiBUaGlzIGludGVyZmFjZSBpcyB0byBiZSB1c2VkIHdpdGggY29uc3RydWN0b3JzXG4gKiBvZiBjb25jcmV0ZSBSTk4gbGF5ZXIgc3VidHlwZXMuXG4gKi9cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBSTk5MYXllckFyZ3MgZXh0ZW5kcyBCYXNlUk5OTGF5ZXJBcmdzIHtcbiAgY2VsbDogUk5OQ2VsbHxSTk5DZWxsW107XG59XG5cbmV4cG9ydCBjbGFzcyBTaW1wbGVSTk4gZXh0ZW5kcyBSTk4ge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdTaW1wbGVSTk4nO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBTaW1wbGVSTk5MYXllckFyZ3MpIHtcbiAgICBhcmdzLmNlbGwgPSBuZXcgU2ltcGxlUk5OQ2VsbChhcmdzKTtcbiAgICBzdXBlcihhcmdzIGFzIFJOTkxheWVyQXJncyk7XG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIGFjdGl2aXR5UmVndWxhcml6ZXIuXG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaWYgKHRoaXMuY2VsbC5kcm9wb3V0TWFzayAhPSBudWxsKSB7XG4gICAgICAgIHRmYy5kaXNwb3NlKHRoaXMuY2VsbC5kcm9wb3V0TWFzayk7XG4gICAgICAgIHRoaXMuY2VsbC5kcm9wb3V0TWFzayA9IG51bGw7XG4gICAgICB9XG4gICAgICBpZiAodGhpcy5jZWxsLnJlY3VycmVudERyb3BvdXRNYXNrICE9IG51bGwpIHtcbiAgICAgICAgdGZjLmRpc3Bvc2UodGhpcy5jZWxsLnJlY3VycmVudERyb3BvdXRNYXNrKTtcbiAgICAgICAgdGhpcy5jZWxsLnJlY3VycmVudERyb3BvdXRNYXNrID0gbnVsbDtcbiAgICAgIH1cbiAgICAgIGNvbnN0IG1hc2sgPSBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ21hc2snXTtcbiAgICAgIGNvbnN0IHRyYWluaW5nID0ga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWyd0cmFpbmluZyddO1xuICAgICAgY29uc3QgaW5pdGlhbFN0YXRlOiBUZW5zb3JbXSA9XG4gICAgICAgICAga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydpbml0aWFsU3RhdGUnXTtcbiAgICAgIHJldHVybiBzdXBlci5jYWxsKGlucHV0cywge21hc2ssIHRyYWluaW5nLCBpbml0aWFsU3RhdGV9KTtcbiAgICB9KTtcbiAgfVxuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgZnJvbUNvbmZpZzxUIGV4dGVuZHMgc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGU+KFxuICAgICAgY2xzOiBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yPFQ+LFxuICAgICAgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QpOiBUIHtcbiAgICByZXR1cm4gbmV3IGNscyhjb25maWcpO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoU2ltcGxlUk5OKTtcblxuLy8gUG9ydGluZyBOb3RlOiBTaW5jZSB0aGlzIGlzIGEgc3VwZXJzZXQgb2YgU2ltcGxlUk5OTGF5ZXJDb25maWcsIHdlIGV4dGVuZFxuLy8gICB0aGF0IGludGVyZmFjZSBpbnN0ZWFkIG9mIHJlcGVhdGluZyB0aGUgZmllbGRzLlxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIEdSVUNlbGxMYXllckFyZ3MgZXh0ZW5kcyBTaW1wbGVSTk5DZWxsTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIEFjdGl2YXRpb24gZnVuY3Rpb24gdG8gdXNlIGZvciB0aGUgcmVjdXJyZW50IHN0ZXAuXG4gICAqXG4gICAqIERlZmF1bHRzIHRvIGhhcmQgc2lnbW9pZCAoYGhhcmRTaWdtb2lkYCkuXG4gICAqXG4gICAqIElmIGBudWxsYCwgbm8gYWN0aXZhdGlvbiBpcyBhcHBsaWVkLlxuICAgKi9cbiAgcmVjdXJyZW50QWN0aXZhdGlvbj86IEFjdGl2YXRpb25JZGVudGlmaWVyO1xuXG4gIC8qKlxuICAgKiBJbXBsZW1lbnRhdGlvbiBtb2RlLCBlaXRoZXIgMSBvciAyLlxuICAgKlxuICAgKiBNb2RlIDEgd2lsbCBzdHJ1Y3R1cmUgaXRzIG9wZXJhdGlvbnMgYXMgYSBsYXJnZXIgbnVtYmVyIG9mXG4gICAqICAgc21hbGxlciBkb3QgcHJvZHVjdHMgYW5kIGFkZGl0aW9ucy5cbiAgICpcbiAgICogTW9kZSAyIHdpbGwgYmF0Y2ggdGhlbSBpbnRvIGZld2VyLCBsYXJnZXIgb3BlcmF0aW9ucy4gVGhlc2UgbW9kZXMgd2lsbFxuICAgKiBoYXZlIGRpZmZlcmVudCBwZXJmb3JtYW5jZSBwcm9maWxlcyBvbiBkaWZmZXJlbnQgaGFyZHdhcmUgYW5kXG4gICAqIGZvciBkaWZmZXJlbnQgYXBwbGljYXRpb25zLlxuICAgKlxuICAgKiBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXMgaW1wbGVtZW50YXRpb25cbiAgICogMiwgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mIHRoaXMgY29uZmlndXJhdGlvbiBmaWVsZC5cbiAgICovXG4gIGltcGxlbWVudGF0aW9uPzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBHUlUgY29udmVudGlvbiAod2hldGhlciB0byBhcHBseSByZXNldCBnYXRlIGFmdGVyIG9yIGJlZm9yZSBtYXRyaXhcbiAgICogbXVsdGlwbGljYXRpb24pLiBmYWxzZSA9IFwiYmVmb3JlXCIsIHRydWUgPSBcImFmdGVyXCIgKG9ubHkgZmFsc2UgaXNcbiAgICogc3VwcG9ydGVkKS5cbiAgICovXG4gIHJlc2V0QWZ0ZXI/OiBib29sZWFuO1xufVxuXG5leHBvcnQgY2xhc3MgR1JVQ2VsbCBleHRlbmRzIFJOTkNlbGwge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdHUlVDZWxsJztcbiAgcmVhZG9ubHkgdW5pdHM6IG51bWJlcjtcbiAgcmVhZG9ubHkgYWN0aXZhdGlvbjogQWN0aXZhdGlvbjtcbiAgcmVhZG9ubHkgcmVjdXJyZW50QWN0aXZhdGlvbjogQWN0aXZhdGlvbjtcbiAgcmVhZG9ubHkgdXNlQmlhczogYm9vbGVhbjtcblxuICByZWFkb25seSBrZXJuZWxJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudEluaXRpYWxpemVyOiBJbml0aWFsaXplcjtcbiAgcmVhZG9ubHkgYmlhc0luaXRpYWxpemVyOiBJbml0aWFsaXplcjtcblxuICByZWFkb25seSBrZXJuZWxSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudFJlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcbiAgcmVhZG9ubHkgYmlhc1JlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcblxuICByZWFkb25seSBrZXJuZWxDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSByZWN1cnJlbnRDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSBiaWFzQ29uc3RyYWludDogQ29uc3RyYWludDtcblxuICByZWFkb25seSBkcm9wb3V0OiBudW1iZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudERyb3BvdXQ6IG51bWJlcjtcbiAgcmVhZG9ubHkgZHJvcG91dEZ1bmM6IEZ1bmN0aW9uO1xuXG4gIHJlYWRvbmx5IHN0YXRlU2l6ZTogbnVtYmVyO1xuICByZWFkb25seSBpbXBsZW1lbnRhdGlvbjogbnVtYmVyO1xuXG4gIHJlYWRvbmx5IERFRkFVTFRfQUNUSVZBVElPTiA9ICd0YW5oJztcbiAgcmVhZG9ubHkgREVGQVVMVF9SRUNVUlJFTlRfQUNUSVZBVElPTjogQWN0aXZhdGlvbklkZW50aWZpZXIgPSAnaGFyZFNpZ21vaWQnO1xuXG4gIHJlYWRvbmx5IERFRkFVTFRfS0VSTkVMX0lOSVRJQUxJWkVSID0gJ2dsb3JvdE5vcm1hbCc7XG4gIHJlYWRvbmx5IERFRkFVTFRfUkVDVVJSRU5UX0lOSVRJQUxJWkVSID0gJ29ydGhvZ29uYWwnO1xuICByZWFkb25seSBERUZBVUxUX0JJQVNfSU5JVElBTElaRVI6IEluaXRpYWxpemVySWRlbnRpZmllciA9ICd6ZXJvcyc7XG5cbiAga2VybmVsOiBMYXllclZhcmlhYmxlO1xuICByZWN1cnJlbnRLZXJuZWw6IExheWVyVmFyaWFibGU7XG4gIGJpYXM6IExheWVyVmFyaWFibGU7XG5cbiAgY29uc3RydWN0b3IoYXJnczogR1JVQ2VsbExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIGlmIChhcmdzLnJlc2V0QWZ0ZXIpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBHUlVDZWxsIGRvZXMgbm90IHN1cHBvcnQgcmVzZXRfYWZ0ZXIgcGFyYW1ldGVyIHNldCB0byB0cnVlLmApO1xuICAgIH1cbiAgICB0aGlzLnVuaXRzID0gYXJncy51bml0cztcbiAgICBhc3NlcnRQb3NpdGl2ZUludGVnZXIodGhpcy51bml0cywgJ3VuaXRzJyk7XG4gICAgdGhpcy5hY3RpdmF0aW9uID0gZ2V0QWN0aXZhdGlvbihcbiAgICAgICAgYXJncy5hY3RpdmF0aW9uID09PSB1bmRlZmluZWQgPyB0aGlzLkRFRkFVTFRfQUNUSVZBVElPTiA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYXJncy5hY3RpdmF0aW9uKTtcbiAgICB0aGlzLnJlY3VycmVudEFjdGl2YXRpb24gPSBnZXRBY3RpdmF0aW9uKFxuICAgICAgICBhcmdzLnJlY3VycmVudEFjdGl2YXRpb24gPT09IHVuZGVmaW5lZCA/XG4gICAgICAgICAgICB0aGlzLkRFRkFVTFRfUkVDVVJSRU5UX0FDVElWQVRJT04gOlxuICAgICAgICAgICAgYXJncy5yZWN1cnJlbnRBY3RpdmF0aW9uKTtcbiAgICB0aGlzLnVzZUJpYXMgPSBhcmdzLnVzZUJpYXMgPT0gbnVsbCA/IHRydWUgOiBhcmdzLnVzZUJpYXM7XG5cbiAgICB0aGlzLmtlcm5lbEluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXIoXG4gICAgICAgIGFyZ3Mua2VybmVsSW5pdGlhbGl6ZXIgfHwgdGhpcy5ERUZBVUxUX0tFUk5FTF9JTklUSUFMSVpFUik7XG4gICAgdGhpcy5yZWN1cnJlbnRJbml0aWFsaXplciA9IGdldEluaXRpYWxpemVyKFxuICAgICAgICBhcmdzLnJlY3VycmVudEluaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9SRUNVUlJFTlRfSU5JVElBTElaRVIpO1xuXG4gICAgdGhpcy5iaWFzSW5pdGlhbGl6ZXIgPVxuICAgICAgICBnZXRJbml0aWFsaXplcihhcmdzLmJpYXNJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfQklBU19JTklUSUFMSVpFUik7XG5cbiAgICB0aGlzLmtlcm5lbFJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5rZXJuZWxSZWd1bGFyaXplcik7XG4gICAgdGhpcy5yZWN1cnJlbnRSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MucmVjdXJyZW50UmVndWxhcml6ZXIpO1xuICAgIHRoaXMuYmlhc1JlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5iaWFzUmVndWxhcml6ZXIpO1xuXG4gICAgdGhpcy5rZXJuZWxDb25zdHJhaW50ID0gZ2V0Q29uc3RyYWludChhcmdzLmtlcm5lbENvbnN0cmFpbnQpO1xuICAgIHRoaXMucmVjdXJyZW50Q29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5yZWN1cnJlbnRDb25zdHJhaW50KTtcbiAgICB0aGlzLmJpYXNDb25zdHJhaW50ID0gZ2V0Q29uc3RyYWludChhcmdzLmJpYXNDb25zdHJhaW50KTtcblxuICAgIHRoaXMuZHJvcG91dCA9IG1hdGhfdXRpbHMubWluKFxuICAgICAgICBbMSwgbWF0aF91dGlscy5tYXgoWzAsIGFyZ3MuZHJvcG91dCA9PSBudWxsID8gMCA6IGFyZ3MuZHJvcG91dF0pXSk7XG4gICAgdGhpcy5yZWN1cnJlbnREcm9wb3V0ID0gbWF0aF91dGlscy5taW4oW1xuICAgICAgMSxcbiAgICAgIG1hdGhfdXRpbHMubWF4KFxuICAgICAgICAgIFswLCBhcmdzLnJlY3VycmVudERyb3BvdXQgPT0gbnVsbCA/IDAgOiBhcmdzLnJlY3VycmVudERyb3BvdXRdKVxuICAgIF0pO1xuICAgIHRoaXMuZHJvcG91dEZ1bmMgPSBhcmdzLmRyb3BvdXRGdW5jO1xuICAgIHRoaXMuaW1wbGVtZW50YXRpb24gPSBhcmdzLmltcGxlbWVudGF0aW9uO1xuICAgIHRoaXMuc3RhdGVTaXplID0gdGhpcy51bml0cztcbiAgICB0aGlzLmRyb3BvdXRNYXNrID0gbnVsbDtcbiAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gbnVsbDtcbiAgfVxuXG4gIHB1YmxpYyBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGVbaW5wdXRTaGFwZS5sZW5ndGggLSAxXTtcbiAgICB0aGlzLmtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAna2VybmVsJywgW2lucHV0RGltLCB0aGlzLnVuaXRzICogM10sIG51bGwsIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIsXG4gICAgICAgIHRoaXMua2VybmVsUmVndWxhcml6ZXIsIHRydWUsIHRoaXMua2VybmVsQ29uc3RyYWludCk7XG4gICAgdGhpcy5yZWN1cnJlbnRLZXJuZWwgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ3JlY3VycmVudF9rZXJuZWwnLCBbdGhpcy51bml0cywgdGhpcy51bml0cyAqIDNdLCBudWxsLFxuICAgICAgICB0aGlzLnJlY3VycmVudEluaXRpYWxpemVyLCB0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyLCB0cnVlLFxuICAgICAgICB0aGlzLnJlY3VycmVudENvbnN0cmFpbnQpO1xuICAgIGlmICh0aGlzLnVzZUJpYXMpIHtcbiAgICAgIHRoaXMuYmlhcyA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAgICdiaWFzJywgW3RoaXMudW5pdHMgKiAzXSwgbnVsbCwgdGhpcy5iaWFzSW5pdGlhbGl6ZXIsXG4gICAgICAgICAgdGhpcy5iaWFzUmVndWxhcml6ZXIsIHRydWUsIHRoaXMuYmlhc0NvbnN0cmFpbnQpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmJpYXMgPSBudWxsO1xuICAgIH1cbiAgICAvLyBQb3J0aW5nIE5vdGVzOiBVbmxpa2UgdGhlIFB5S2VyYXMgaW1wbGVtZW50YXRpb24sIHdlIHBlcmZvcm0gc2xpY2luZ1xuICAgIC8vICAgb2YgdGhlIHdlaWdodHMgYW5kIGJpYXMgaW4gdGhlIGNhbGwoKSBtZXRob2QsIGF0IGV4ZWN1dGlvbiB0aW1lLlxuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICB9XG5cbiAgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGlucHV0cyA9IGlucHV0cyBhcyBUZW5zb3JbXTtcbiAgICAgIGlmIChpbnB1dHMubGVuZ3RoICE9PSAyKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYEdSVUNlbGwgZXhwZWN0cyAyIGlucHV0IFRlbnNvcnMgKGlucHV0cywgaCwgYyksIGdvdCBgICtcbiAgICAgICAgICAgIGAke2lucHV0cy5sZW5ndGh9LmApO1xuICAgICAgfVxuXG4gICAgICBjb25zdCB0cmFpbmluZyA9IGt3YXJnc1sndHJhaW5pbmcnXSA9PSBudWxsID8gZmFsc2UgOiBrd2FyZ3NbJ3RyYWluaW5nJ107XG4gICAgICBsZXQgaFRNaW51czEgPSBpbnB1dHNbMV07ICAvLyBQcmV2aW91cyBtZW1vcnkgc3RhdGUuXG4gICAgICBpbnB1dHMgPSBpbnB1dHNbMF07XG5cbiAgICAgIC8vIE5vdGU6IEZvciBzdXBlcmlvciBwZXJmb3JtYW5jZSwgVGVuc29yRmxvdy5qcyBhbHdheXMgdXNlc1xuICAgICAgLy8gaW1wbGVtZW50YXRpb24gMiwgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mXG4gICAgICAvLyBjb25maWcuaW1wbGVtZW50YXRpb24uXG4gICAgICBpZiAoMCA8IHRoaXMuZHJvcG91dCAmJiB0aGlzLmRyb3BvdXQgPCAxICYmIHRoaXMuZHJvcG91dE1hc2sgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLmRyb3BvdXRNYXNrID0gZ2VuZXJhdGVEcm9wb3V0TWFzayh7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9uZXM6ICgpID0+IHRmYy5vbmVzTGlrZShpbnB1dHMgYXMgVGVuc29yKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmF0ZTogdGhpcy5kcm9wb3V0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0cmFpbmluZyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY291bnQ6IDMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRyb3BvdXRGdW5jOiB0aGlzLmRyb3BvdXRGdW5jLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgfSkgYXMgVGVuc29yW107XG4gICAgICB9XG4gICAgICBpZiAoMCA8IHRoaXMucmVjdXJyZW50RHJvcG91dCAmJiB0aGlzLnJlY3VycmVudERyb3BvdXQgPCAxICYmXG4gICAgICAgICAgdGhpcy5yZWN1cnJlbnREcm9wb3V0TWFzayA9PSBudWxsKSB7XG4gICAgICAgIHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgPSBnZW5lcmF0ZURyb3BvdXRNYXNrKHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb25lczogKCkgPT4gdGZjLm9uZXNMaWtlKGhUTWludXMxKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmF0ZTogdGhpcy5yZWN1cnJlbnREcm9wb3V0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0cmFpbmluZyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY291bnQ6IDMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRyb3BvdXRGdW5jOiB0aGlzLmRyb3BvdXRGdW5jLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSkgYXMgVGVuc29yW107XG4gICAgICB9XG4gICAgICBjb25zdCBkcE1hc2sgPSB0aGlzLmRyb3BvdXRNYXNrIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yXTtcbiAgICAgIGNvbnN0IHJlY0RwTWFzayA9IHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgYXMgW1RlbnNvciwgVGVuc29yLCBUZW5zb3JdO1xuICAgICAgbGV0IHo6IFRlbnNvcjtcbiAgICAgIGxldCByOiBUZW5zb3I7XG4gICAgICBsZXQgaGg6IFRlbnNvcjtcblxuICAgICAgaWYgKDAgPCB0aGlzLmRyb3BvdXQgJiYgdGhpcy5kcm9wb3V0IDwgMSkge1xuICAgICAgICBpbnB1dHMgPSB0ZmMubXVsKGlucHV0cywgZHBNYXNrWzBdKTtcbiAgICAgIH1cbiAgICAgIGxldCBtYXRyaXhYID0gSy5kb3QoaW5wdXRzLCB0aGlzLmtlcm5lbC5yZWFkKCkpO1xuICAgICAgaWYgKHRoaXMudXNlQmlhcykge1xuICAgICAgICBtYXRyaXhYID0gSy5iaWFzQWRkKG1hdHJpeFgsIHRoaXMuYmlhcy5yZWFkKCkpO1xuICAgICAgfVxuICAgICAgaWYgKDAgPCB0aGlzLnJlY3VycmVudERyb3BvdXQgJiYgdGhpcy5yZWN1cnJlbnREcm9wb3V0IDwgMSkge1xuICAgICAgICBoVE1pbnVzMSA9IHRmYy5tdWwoaFRNaW51czEsIHJlY0RwTWFza1swXSk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IHJlY3VycmVudEtlcm5lbFZhbHVlID0gdGhpcy5yZWN1cnJlbnRLZXJuZWwucmVhZCgpO1xuICAgICAgY29uc3QgW3JrMSwgcmsyXSA9IHRmYy5zcGxpdChcbiAgICAgICAgICByZWN1cnJlbnRLZXJuZWxWYWx1ZSwgWzIgKiB0aGlzLnVuaXRzLCB0aGlzLnVuaXRzXSxcbiAgICAgICAgICByZWN1cnJlbnRLZXJuZWxWYWx1ZS5yYW5rIC0gMSk7XG4gICAgICBjb25zdCBtYXRyaXhJbm5lciA9IEsuZG90KGhUTWludXMxLCByazEpO1xuXG4gICAgICBjb25zdCBbeFosIHhSLCB4SF0gPSB0ZmMuc3BsaXQobWF0cml4WCwgMywgbWF0cml4WC5yYW5rIC0gMSk7XG4gICAgICBjb25zdCBbcmVjdXJyZW50WiwgcmVjdXJyZW50Ul0gPVxuICAgICAgICAgIHRmYy5zcGxpdChtYXRyaXhJbm5lciwgMiwgbWF0cml4SW5uZXIucmFuayAtIDEpO1xuICAgICAgeiA9IHRoaXMucmVjdXJyZW50QWN0aXZhdGlvbi5hcHBseSh0ZmMuYWRkKHhaLCByZWN1cnJlbnRaKSk7XG4gICAgICByID0gdGhpcy5yZWN1cnJlbnRBY3RpdmF0aW9uLmFwcGx5KHRmYy5hZGQoeFIsIHJlY3VycmVudFIpKTtcblxuICAgICAgY29uc3QgcmVjdXJyZW50SCA9IEsuZG90KHRmYy5tdWwociwgaFRNaW51czEpLCByazIpO1xuICAgICAgaGggPSB0aGlzLmFjdGl2YXRpb24uYXBwbHkodGZjLmFkZCh4SCwgcmVjdXJyZW50SCkpO1xuXG4gICAgICBjb25zdCBoID1cbiAgICAgICAgICB0ZmMuYWRkKHRmYy5tdWwoeiwgaFRNaW51czEpLCB0ZmMubXVsKHRmYy5hZGQoMSwgdGZjLm5lZyh6KSksIGhoKSk7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgdXNlX2xlYXJuaW5nX3BoYXNlIGZsYWcgcHJvcGVybHkuXG4gICAgICByZXR1cm4gW2gsIGhdO1xuICAgIH0pO1xuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuXG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7XG4gICAgICB1bml0czogdGhpcy51bml0cyxcbiAgICAgIGFjdGl2YXRpb246IHNlcmlhbGl6ZUFjdGl2YXRpb24odGhpcy5hY3RpdmF0aW9uKSxcbiAgICAgIHJlY3VycmVudEFjdGl2YXRpb246IHNlcmlhbGl6ZUFjdGl2YXRpb24odGhpcy5yZWN1cnJlbnRBY3RpdmF0aW9uKSxcbiAgICAgIHVzZUJpYXM6IHRoaXMudXNlQmlhcyxcbiAgICAgIGtlcm5lbEluaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmtlcm5lbEluaXRpYWxpemVyKSxcbiAgICAgIHJlY3VycmVudEluaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLnJlY3VycmVudEluaXRpYWxpemVyKSxcbiAgICAgIGJpYXNJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5iaWFzSW5pdGlhbGl6ZXIpLFxuICAgICAga2VybmVsUmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMua2VybmVsUmVndWxhcml6ZXIpLFxuICAgICAgcmVjdXJyZW50UmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMucmVjdXJyZW50UmVndWxhcml6ZXIpLFxuICAgICAgYmlhc1JlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmJpYXNSZWd1bGFyaXplciksXG4gICAgICBhY3Rpdml0eVJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmFjdGl2aXR5UmVndWxhcml6ZXIpLFxuICAgICAga2VybmVsQ29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLmtlcm5lbENvbnN0cmFpbnQpLFxuICAgICAgcmVjdXJyZW50Q29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLnJlY3VycmVudENvbnN0cmFpbnQpLFxuICAgICAgYmlhc0NvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5iaWFzQ29uc3RyYWludCksXG4gICAgICBkcm9wb3V0OiB0aGlzLmRyb3BvdXQsXG4gICAgICByZWN1cnJlbnREcm9wb3V0OiB0aGlzLnJlY3VycmVudERyb3BvdXQsXG4gICAgICBpbXBsZW1lbnRhdGlvbjogdGhpcy5pbXBsZW1lbnRhdGlvbixcbiAgICAgIHJlc2V0QWZ0ZXI6IGZhbHNlXG4gICAgfTtcblxuICAgIHJldHVybiB7Li4uYmFzZUNvbmZpZywgLi4uY29uZmlnfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEdSVUNlbGwpO1xuXG4vLyBQb3J0aW5nIE5vdGU6IFNpbmNlIHRoaXMgaXMgYSBzdXBlcnNldCBvZiBTaW1wbGVSTk5MYXllckNvbmZpZywgd2UgaW5oZXJpdFxuLy8gICBmcm9tIHRoYXQgaW50ZXJmYWNlIGluc3RlYWQgb2YgcmVwZWF0aW5nIHRoZSBmaWVsZHMgaGVyZS5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBHUlVMYXllckFyZ3MgZXh0ZW5kcyBTaW1wbGVSTk5MYXllckFyZ3Mge1xuICAvKipcbiAgICogQWN0aXZhdGlvbiBmdW5jdGlvbiB0byB1c2UgZm9yIHRoZSByZWN1cnJlbnQgc3RlcC5cbiAgICpcbiAgICogRGVmYXVsdHMgdG8gaGFyZCBzaWdtb2lkIChgaGFyZFNpZ21vaWRgKS5cbiAgICpcbiAgICogSWYgYG51bGxgLCBubyBhY3RpdmF0aW9uIGlzIGFwcGxpZWQuXG4gICAqL1xuICByZWN1cnJlbnRBY3RpdmF0aW9uPzogQWN0aXZhdGlvbklkZW50aWZpZXI7XG5cbiAgLyoqXG4gICAqIEltcGxlbWVudGF0aW9uIG1vZGUsIGVpdGhlciAxIG9yIDIuXG4gICAqXG4gICAqIE1vZGUgMSB3aWxsIHN0cnVjdHVyZSBpdHMgb3BlcmF0aW9ucyBhcyBhIGxhcmdlciBudW1iZXIgb2ZcbiAgICogc21hbGxlciBkb3QgcHJvZHVjdHMgYW5kIGFkZGl0aW9ucy5cbiAgICpcbiAgICogTW9kZSAyIHdpbGwgYmF0Y2ggdGhlbSBpbnRvIGZld2VyLCBsYXJnZXIgb3BlcmF0aW9ucy4gVGhlc2UgbW9kZXMgd2lsbFxuICAgKiBoYXZlIGRpZmZlcmVudCBwZXJmb3JtYW5jZSBwcm9maWxlcyBvbiBkaWZmZXJlbnQgaGFyZHdhcmUgYW5kXG4gICAqIGZvciBkaWZmZXJlbnQgYXBwbGljYXRpb25zLlxuICAgKlxuICAgKiBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXMgaW1wbGVtZW50YXRpb25cbiAgICogMiwgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mIHRoaXMgY29uZmlndXJhdGlvbiBmaWVsZC5cbiAgICovXG4gIGltcGxlbWVudGF0aW9uPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgR1JVIGV4dGVuZHMgUk5OIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnR1JVJztcbiAgY29uc3RydWN0b3IoYXJnczogR1JVTGF5ZXJBcmdzKSB7XG4gICAgaWYgKGFyZ3MuaW1wbGVtZW50YXRpb24gPT09IDApIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnYGltcGxlbWVudGF0aW9uPTBgIGhhcyBiZWVuIGRlcHJlY2F0ZWQsIGFuZCBub3cgZGVmYXVsdHMgdG8gJyArXG4gICAgICAgICAgJ2BpbXBsZW1lbnRhdGlvbj0xYC4gUGxlYXNlIHVwZGF0ZSB5b3VyIGxheWVyIGNhbGwuJyk7XG4gICAgfVxuICAgIGFyZ3MuY2VsbCA9IG5ldyBHUlVDZWxsKGFyZ3MpO1xuICAgIHN1cGVyKGFyZ3MgYXMgUk5OTGF5ZXJBcmdzKTtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgYWN0aXZpdHlSZWd1bGFyaXplci5cbiAgfVxuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpZiAodGhpcy5jZWxsLmRyb3BvdXRNYXNrICE9IG51bGwpIHtcbiAgICAgICAgdGZjLmRpc3Bvc2UodGhpcy5jZWxsLmRyb3BvdXRNYXNrKTtcbiAgICAgICAgdGhpcy5jZWxsLmRyb3BvdXRNYXNrID0gbnVsbDtcbiAgICAgIH1cbiAgICAgIGlmICh0aGlzLmNlbGwucmVjdXJyZW50RHJvcG91dE1hc2sgIT0gbnVsbCkge1xuICAgICAgICB0ZmMuZGlzcG9zZSh0aGlzLmNlbGwucmVjdXJyZW50RHJvcG91dE1hc2spO1xuICAgICAgICB0aGlzLmNlbGwucmVjdXJyZW50RHJvcG91dE1hc2sgPSBudWxsO1xuICAgICAgfVxuICAgICAgY29uc3QgbWFzayA9IGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snbWFzayddO1xuICAgICAgY29uc3QgdHJhaW5pbmcgPSBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ3RyYWluaW5nJ107XG4gICAgICBjb25zdCBpbml0aWFsU3RhdGU6IFRlbnNvcltdID1cbiAgICAgICAgICBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ2luaXRpYWxTdGF0ZSddO1xuICAgICAgcmV0dXJuIHN1cGVyLmNhbGwoaW5wdXRzLCB7bWFzaywgdHJhaW5pbmcsIGluaXRpYWxTdGF0ZX0pO1xuICAgIH0pO1xuICB9XG5cbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBmcm9tQ29uZmlnPFQgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sXG4gICAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCk6IFQge1xuICAgIGlmIChjb25maWdbJ2ltcGxtZW50YXRpb24nXSA9PT0gMCkge1xuICAgICAgY29uZmlnWydpbXBsZW1lbnRhdGlvbiddID0gMTtcbiAgICB9XG4gICAgcmV0dXJuIG5ldyBjbHMoY29uZmlnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEdSVSk7XG5cbi8vIFBvcnRpbmcgTm90ZTogU2luY2UgdGhpcyBpcyBhIHN1cGVyc2V0IG9mIFNpbXBsZVJOTkxheWVyQ29uZmlnLCB3ZSBleHRlbmRcbi8vICAgdGhhdCBpbnRlcmZhY2UgaW5zdGVhZCBvZiByZXBlYXRpbmcgdGhlIGZpZWxkcy5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBMU1RNQ2VsbExheWVyQXJncyBleHRlbmRzIFNpbXBsZVJOTkNlbGxMYXllckFyZ3Mge1xuICAvKipcbiAgICogQWN0aXZhdGlvbiBmdW5jdGlvbiB0byB1c2UgZm9yIHRoZSByZWN1cnJlbnQgc3RlcC5cbiAgICpcbiAgICogRGVmYXVsdHMgdG8gaGFyZCBzaWdtb2lkIChgaGFyZFNpZ21vaWRgKS5cbiAgICpcbiAgICogSWYgYG51bGxgLCBubyBhY3RpdmF0aW9uIGlzIGFwcGxpZWQuXG4gICAqL1xuICByZWN1cnJlbnRBY3RpdmF0aW9uPzogQWN0aXZhdGlvbklkZW50aWZpZXI7XG5cbiAgLyoqXG4gICAqIElmIGB0cnVlYCwgYWRkIDEgdG8gdGhlIGJpYXMgb2YgdGhlIGZvcmdldCBnYXRlIGF0IGluaXRpYWxpemF0aW9uLlxuICAgKiBTZXR0aW5nIGl0IHRvIGB0cnVlYCB3aWxsIGFsc28gZm9yY2UgYGJpYXNJbml0aWFsaXplciA9ICd6ZXJvcydgLlxuICAgKiBUaGlzIGlzIHJlY29tbWVuZGVkIGluXG4gICAqIFtKb3plZm93aWN6IGV0XG4gICAqIGFsLl0oaHR0cDovL3d3dy5qbWxyLm9yZy9wcm9jZWVkaW5ncy9wYXBlcnMvdjM3L2pvemVmb3dpY3oxNS5wZGYpLlxuICAgKi9cbiAgdW5pdEZvcmdldEJpYXM/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBJbXBsZW1lbnRhdGlvbiBtb2RlLCBlaXRoZXIgMSBvciAyLlxuICAgKlxuICAgKiBNb2RlIDEgd2lsbCBzdHJ1Y3R1cmUgaXRzIG9wZXJhdGlvbnMgYXMgYSBsYXJnZXIgbnVtYmVyIG9mXG4gICAqICAgc21hbGxlciBkb3QgcHJvZHVjdHMgYW5kIGFkZGl0aW9ucy5cbiAgICpcbiAgICogTW9kZSAyIHdpbGwgYmF0Y2ggdGhlbSBpbnRvIGZld2VyLCBsYXJnZXIgb3BlcmF0aW9ucy4gVGhlc2UgbW9kZXMgd2lsbFxuICAgKiBoYXZlIGRpZmZlcmVudCBwZXJmb3JtYW5jZSBwcm9maWxlcyBvbiBkaWZmZXJlbnQgaGFyZHdhcmUgYW5kXG4gICAqIGZvciBkaWZmZXJlbnQgYXBwbGljYXRpb25zLlxuICAgKlxuICAgKiBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXMgaW1wbGVtZW50YXRpb25cbiAgICogMiwgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mIHRoaXMgY29uZmlndXJhdGlvbiBmaWVsZC5cbiAgICovXG4gIGltcGxlbWVudGF0aW9uPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgTFNUTUNlbGwgZXh0ZW5kcyBSTk5DZWxsIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTFNUTUNlbGwnO1xuICByZWFkb25seSB1bml0czogbnVtYmVyO1xuICByZWFkb25seSBhY3RpdmF0aW9uOiBBY3RpdmF0aW9uO1xuICByZWFkb25seSByZWN1cnJlbnRBY3RpdmF0aW9uOiBBY3RpdmF0aW9uO1xuICByZWFkb25seSB1c2VCaWFzOiBib29sZWFuO1xuXG4gIHJlYWRvbmx5IGtlcm5lbEluaXRpYWxpemVyOiBJbml0aWFsaXplcjtcbiAgcmVhZG9ubHkgcmVjdXJyZW50SW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICByZWFkb25seSBiaWFzSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICByZWFkb25seSB1bml0Rm9yZ2V0QmlhczogYm9vbGVhbjtcblxuICByZWFkb25seSBrZXJuZWxDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSByZWN1cnJlbnRDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSBiaWFzQ29uc3RyYWludDogQ29uc3RyYWludDtcblxuICByZWFkb25seSBrZXJuZWxSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudFJlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcbiAgcmVhZG9ubHkgYmlhc1JlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcblxuICByZWFkb25seSBkcm9wb3V0OiBudW1iZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudERyb3BvdXQ6IG51bWJlcjtcbiAgcmVhZG9ubHkgZHJvcG91dEZ1bmM6IEZ1bmN0aW9uO1xuXG4gIHJlYWRvbmx5IHN0YXRlU2l6ZTogbnVtYmVyW107XG4gIHJlYWRvbmx5IGltcGxlbWVudGF0aW9uOiBudW1iZXI7XG5cbiAgcmVhZG9ubHkgREVGQVVMVF9BQ1RJVkFUSU9OID0gJ3RhbmgnO1xuICByZWFkb25seSBERUZBVUxUX1JFQ1VSUkVOVF9BQ1RJVkFUSU9OID0gJ2hhcmRTaWdtb2lkJztcbiAgcmVhZG9ubHkgREVGQVVMVF9LRVJORUxfSU5JVElBTElaRVIgPSAnZ2xvcm90Tm9ybWFsJztcbiAgcmVhZG9ubHkgREVGQVVMVF9SRUNVUlJFTlRfSU5JVElBTElaRVIgPSAnb3J0aG9nb25hbCc7XG5cbiAgcmVhZG9ubHkgREVGQVVMVF9CSUFTX0lOSVRJQUxJWkVSID0gJ3plcm9zJztcblxuICBrZXJuZWw6IExheWVyVmFyaWFibGU7XG4gIHJlY3VycmVudEtlcm5lbDogTGF5ZXJWYXJpYWJsZTtcbiAgYmlhczogTGF5ZXJWYXJpYWJsZTtcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBMU1RNQ2VsbExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuXG4gICAgdGhpcy51bml0cyA9IGFyZ3MudW5pdHM7XG4gICAgYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMudW5pdHMsICd1bml0cycpO1xuICAgIHRoaXMuYWN0aXZhdGlvbiA9IGdldEFjdGl2YXRpb24oXG4gICAgICAgIGFyZ3MuYWN0aXZhdGlvbiA9PT0gdW5kZWZpbmVkID8gdGhpcy5ERUZBVUxUX0FDVElWQVRJT04gOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFyZ3MuYWN0aXZhdGlvbik7XG4gICAgdGhpcy5yZWN1cnJlbnRBY3RpdmF0aW9uID0gZ2V0QWN0aXZhdGlvbihcbiAgICAgICAgYXJncy5yZWN1cnJlbnRBY3RpdmF0aW9uID09PSB1bmRlZmluZWQgP1xuICAgICAgICAgICAgdGhpcy5ERUZBVUxUX1JFQ1VSUkVOVF9BQ1RJVkFUSU9OIDpcbiAgICAgICAgICAgIGFyZ3MucmVjdXJyZW50QWN0aXZhdGlvbik7XG4gICAgdGhpcy51c2VCaWFzID0gYXJncy51c2VCaWFzID09IG51bGwgPyB0cnVlIDogYXJncy51c2VCaWFzO1xuXG4gICAgdGhpcy5rZXJuZWxJbml0aWFsaXplciA9IGdldEluaXRpYWxpemVyKFxuICAgICAgICBhcmdzLmtlcm5lbEluaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9LRVJORUxfSU5JVElBTElaRVIpO1xuICAgIHRoaXMucmVjdXJyZW50SW5pdGlhbGl6ZXIgPSBnZXRJbml0aWFsaXplcihcbiAgICAgICAgYXJncy5yZWN1cnJlbnRJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfUkVDVVJSRU5UX0lOSVRJQUxJWkVSKTtcblxuICAgIHRoaXMuYmlhc0luaXRpYWxpemVyID1cbiAgICAgICAgZ2V0SW5pdGlhbGl6ZXIoYXJncy5iaWFzSW5pdGlhbGl6ZXIgfHwgdGhpcy5ERUZBVUxUX0JJQVNfSU5JVElBTElaRVIpO1xuICAgIHRoaXMudW5pdEZvcmdldEJpYXMgPSBhcmdzLnVuaXRGb3JnZXRCaWFzO1xuXG4gICAgdGhpcy5rZXJuZWxSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3Mua2VybmVsUmVndWxhcml6ZXIpO1xuICAgIHRoaXMucmVjdXJyZW50UmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLnJlY3VycmVudFJlZ3VsYXJpemVyKTtcbiAgICB0aGlzLmJpYXNSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MuYmlhc1JlZ3VsYXJpemVyKTtcblxuICAgIHRoaXMua2VybmVsQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5rZXJuZWxDb25zdHJhaW50KTtcbiAgICB0aGlzLnJlY3VycmVudENvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGFyZ3MucmVjdXJyZW50Q29uc3RyYWludCk7XG4gICAgdGhpcy5iaWFzQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5iaWFzQ29uc3RyYWludCk7XG5cbiAgICB0aGlzLmRyb3BvdXQgPSBtYXRoX3V0aWxzLm1pbihcbiAgICAgICAgWzEsIG1hdGhfdXRpbHMubWF4KFswLCBhcmdzLmRyb3BvdXQgPT0gbnVsbCA/IDAgOiBhcmdzLmRyb3BvdXRdKV0pO1xuICAgIHRoaXMucmVjdXJyZW50RHJvcG91dCA9IG1hdGhfdXRpbHMubWluKFtcbiAgICAgIDEsXG4gICAgICBtYXRoX3V0aWxzLm1heChcbiAgICAgICAgICBbMCwgYXJncy5yZWN1cnJlbnREcm9wb3V0ID09IG51bGwgPyAwIDogYXJncy5yZWN1cnJlbnREcm9wb3V0XSlcbiAgICBdKTtcbiAgICB0aGlzLmRyb3BvdXRGdW5jID0gYXJncy5kcm9wb3V0RnVuYztcbiAgICB0aGlzLmltcGxlbWVudGF0aW9uID0gYXJncy5pbXBsZW1lbnRhdGlvbjtcbiAgICB0aGlzLnN0YXRlU2l6ZSA9IFt0aGlzLnVuaXRzLCB0aGlzLnVuaXRzXTtcbiAgICB0aGlzLmRyb3BvdXRNYXNrID0gbnVsbDtcbiAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gbnVsbDtcbiAgfVxuXG4gIHB1YmxpYyBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGVbaW5wdXRTaGFwZS5sZW5ndGggLSAxXTtcbiAgICB0aGlzLmtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAna2VybmVsJywgW2lucHV0RGltLCB0aGlzLnVuaXRzICogNF0sIG51bGwsIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIsXG4gICAgICAgIHRoaXMua2VybmVsUmVndWxhcml6ZXIsIHRydWUsIHRoaXMua2VybmVsQ29uc3RyYWludCk7XG4gICAgdGhpcy5yZWN1cnJlbnRLZXJuZWwgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ3JlY3VycmVudF9rZXJuZWwnLCBbdGhpcy51bml0cywgdGhpcy51bml0cyAqIDRdLCBudWxsLFxuICAgICAgICB0aGlzLnJlY3VycmVudEluaXRpYWxpemVyLCB0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyLCB0cnVlLFxuICAgICAgICB0aGlzLnJlY3VycmVudENvbnN0cmFpbnQpO1xuICAgIGxldCBiaWFzSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICAgIGlmICh0aGlzLnVzZUJpYXMpIHtcbiAgICAgIGlmICh0aGlzLnVuaXRGb3JnZXRCaWFzKSB7XG4gICAgICAgIGNvbnN0IGNhcHR1cmVkQmlhc0luaXQgPSB0aGlzLmJpYXNJbml0aWFsaXplcjtcbiAgICAgICAgY29uc3QgY2FwdHVyZWRVbml0cyA9IHRoaXMudW5pdHM7XG4gICAgICAgIGJpYXNJbml0aWFsaXplciA9IG5ldyAoY2xhc3MgQ3VzdG9tSW5pdCBleHRlbmRzIEluaXRpYWxpemVyIHtcbiAgICAgICAgICAvKiogQG5vY29sbGFwc2UgKi9cbiAgICAgICAgICBzdGF0aWMgY2xhc3NOYW1lID0gJ0N1c3RvbUluaXQnO1xuXG4gICAgICAgICAgYXBwbHkoc2hhcGU6IFNoYXBlLCBkdHlwZT86IERhdGFUeXBlKTogVGVuc29yIHtcbiAgICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IE1vcmUgaW5mb3JtYXRpdmUgdmFyaWFibGUgbmFtZXM/XG4gICAgICAgICAgICBjb25zdCBiSSA9IGNhcHR1cmVkQmlhc0luaXQuYXBwbHkoW2NhcHR1cmVkVW5pdHNdKTtcbiAgICAgICAgICAgIGNvbnN0IGJGID0gKG5ldyBPbmVzKCkpLmFwcGx5KFtjYXB0dXJlZFVuaXRzXSk7XG4gICAgICAgICAgICBjb25zdCBiQ0FuZEggPSBjYXB0dXJlZEJpYXNJbml0LmFwcGx5KFtjYXB0dXJlZFVuaXRzICogMl0pO1xuICAgICAgICAgICAgcmV0dXJuIEsuY29uY2F0QWxvbmdGaXJzdEF4aXMoXG4gICAgICAgICAgICAgICAgSy5jb25jYXRBbG9uZ0ZpcnN0QXhpcyhiSSwgYkYpLCBiQ0FuZEgpO1xuICAgICAgICAgIH1cbiAgICAgICAgfSkoKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGJpYXNJbml0aWFsaXplciA9IHRoaXMuYmlhc0luaXRpYWxpemVyO1xuICAgICAgfVxuICAgICAgdGhpcy5iaWFzID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2JpYXMnLCBbdGhpcy51bml0cyAqIDRdLCBudWxsLCBiaWFzSW5pdGlhbGl6ZXIsIHRoaXMuYmlhc1JlZ3VsYXJpemVyLFxuICAgICAgICAgIHRydWUsIHRoaXMuYmlhc0NvbnN0cmFpbnQpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmJpYXMgPSBudWxsO1xuICAgIH1cbiAgICAvLyBQb3J0aW5nIE5vdGVzOiBVbmxpa2UgdGhlIFB5S2VyYXMgaW1wbGVtZW50YXRpb24sIHdlIHBlcmZvcm0gc2xpY2luZ1xuICAgIC8vICAgb2YgdGhlIHdlaWdodHMgYW5kIGJpYXMgaW4gdGhlIGNhbGwoKSBtZXRob2QsIGF0IGV4ZWN1dGlvbiB0aW1lLlxuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICB9XG5cbiAgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IHRyYWluaW5nID0ga3dhcmdzWyd0cmFpbmluZyddID09IG51bGwgPyBmYWxzZSA6IGt3YXJnc1sndHJhaW5pbmcnXTtcbiAgICAgIGlucHV0cyA9IGlucHV0cyBhcyBUZW5zb3JbXTtcbiAgICAgIGlmIChpbnB1dHMubGVuZ3RoICE9PSAzKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYExTVE1DZWxsIGV4cGVjdHMgMyBpbnB1dCBUZW5zb3JzIChpbnB1dHMsIGgsIGMpLCBnb3QgYCArXG4gICAgICAgICAgICBgJHtpbnB1dHMubGVuZ3RofS5gKTtcbiAgICAgIH1cbiAgICAgIGxldCBoVE1pbnVzMSA9IGlucHV0c1sxXTsgICAgLy8gUHJldmlvdXMgbWVtb3J5IHN0YXRlLlxuICAgICAgY29uc3QgY1RNaW51czEgPSBpbnB1dHNbMl07ICAvLyBQcmV2aW91cyBjYXJyeSBzdGF0ZS5cbiAgICAgIGlucHV0cyA9IGlucHV0c1swXTtcbiAgICAgIGlmICgwIDwgdGhpcy5kcm9wb3V0ICYmIHRoaXMuZHJvcG91dCA8IDEgJiYgdGhpcy5kcm9wb3V0TWFzayA9PSBudWxsKSB7XG4gICAgICAgIHRoaXMuZHJvcG91dE1hc2sgPSBnZW5lcmF0ZURyb3BvdXRNYXNrKHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb25lczogKCkgPT4gdGZjLm9uZXNMaWtlKGlucHV0cyBhcyBUZW5zb3IpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICByYXRlOiB0aGlzLmRyb3BvdXQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRyYWluaW5nLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb3VudDogNCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZHJvcG91dEZ1bmM6IHRoaXMuZHJvcG91dEZ1bmNcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pIGFzIFRlbnNvcltdO1xuICAgICAgfVxuICAgICAgaWYgKDAgPCB0aGlzLnJlY3VycmVudERyb3BvdXQgJiYgdGhpcy5yZWN1cnJlbnREcm9wb3V0IDwgMSAmJlxuICAgICAgICAgIHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gZ2VuZXJhdGVEcm9wb3V0TWFzayh7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9uZXM6ICgpID0+IHRmYy5vbmVzTGlrZShoVE1pbnVzMSksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJhdGU6IHRoaXMucmVjdXJyZW50RHJvcG91dCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdHJhaW5pbmcsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvdW50OiA0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkcm9wb3V0RnVuYzogdGhpcy5kcm9wb3V0RnVuY1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSkgYXMgVGVuc29yW107XG4gICAgICB9XG4gICAgICBjb25zdCBkcE1hc2sgPSB0aGlzLmRyb3BvdXRNYXNrIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yLCBUZW5zb3JdO1xuICAgICAgY29uc3QgcmVjRHBNYXNrID1cbiAgICAgICAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yLCBUZW5zb3JdO1xuXG4gICAgICAvLyBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXNcbiAgICAgIC8vIGltcGxlbWVudGF0aW9uIDIgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mXG4gICAgICAvLyBjb25maWcuaW1wbGVtZW50YXRpb24uXG4gICAgICBsZXQgaTogVGVuc29yO1xuICAgICAgbGV0IGY6IFRlbnNvcjtcbiAgICAgIGxldCBjOiBUZW5zb3I7XG4gICAgICBsZXQgbzogVGVuc29yO1xuICAgICAgaWYgKDAgPCB0aGlzLmRyb3BvdXQgJiYgdGhpcy5kcm9wb3V0IDwgMSkge1xuICAgICAgICBpbnB1dHMgPSB0ZmMubXVsKGlucHV0cywgZHBNYXNrWzBdKTtcbiAgICAgIH1cbiAgICAgIGxldCB6ID0gSy5kb3QoaW5wdXRzLCB0aGlzLmtlcm5lbC5yZWFkKCkpO1xuICAgICAgaWYgKDAgPCB0aGlzLnJlY3VycmVudERyb3BvdXQgJiYgdGhpcy5yZWN1cnJlbnREcm9wb3V0IDwgMSkge1xuICAgICAgICBoVE1pbnVzMSA9IHRmYy5tdWwoaFRNaW51czEsIHJlY0RwTWFza1swXSk7XG4gICAgICB9XG4gICAgICB6ID0gdGZjLmFkZCh6LCBLLmRvdChoVE1pbnVzMSwgdGhpcy5yZWN1cnJlbnRLZXJuZWwucmVhZCgpKSk7XG4gICAgICBpZiAodGhpcy51c2VCaWFzKSB7XG4gICAgICAgIHogPSBLLmJpYXNBZGQoeiwgdGhpcy5iaWFzLnJlYWQoKSk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IFt6MCwgejEsIHoyLCB6M10gPSB0ZmMuc3BsaXQoeiwgNCwgei5yYW5rIC0gMSk7XG5cbiAgICAgIGkgPSB0aGlzLnJlY3VycmVudEFjdGl2YXRpb24uYXBwbHkoejApO1xuICAgICAgZiA9IHRoaXMucmVjdXJyZW50QWN0aXZhdGlvbi5hcHBseSh6MSk7XG4gICAgICBjID0gdGZjLmFkZCh0ZmMubXVsKGYsIGNUTWludXMxKSwgdGZjLm11bChpLCB0aGlzLmFjdGl2YXRpb24uYXBwbHkoejIpKSk7XG4gICAgICBvID0gdGhpcy5yZWN1cnJlbnRBY3RpdmF0aW9uLmFwcGx5KHozKTtcblxuICAgICAgY29uc3QgaCA9IHRmYy5tdWwobywgdGhpcy5hY3RpdmF0aW9uLmFwcGx5KGMpKTtcbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB1c2VfbGVhcm5pbmdfcGhhc2UgZmxhZyBwcm9wZXJseS5cbiAgICAgIHJldHVybiBbaCwgaCwgY107XG4gICAgfSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG5cbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgIHVuaXRzOiB0aGlzLnVuaXRzLFxuICAgICAgYWN0aXZhdGlvbjogc2VyaWFsaXplQWN0aXZhdGlvbih0aGlzLmFjdGl2YXRpb24pLFxuICAgICAgcmVjdXJyZW50QWN0aXZhdGlvbjogc2VyaWFsaXplQWN0aXZhdGlvbih0aGlzLnJlY3VycmVudEFjdGl2YXRpb24pLFxuICAgICAgdXNlQmlhczogdGhpcy51c2VCaWFzLFxuICAgICAga2VybmVsSW5pdGlhbGl6ZXI6IHNlcmlhbGl6ZUluaXRpYWxpemVyKHRoaXMua2VybmVsSW5pdGlhbGl6ZXIpLFxuICAgICAgcmVjdXJyZW50SW5pdGlhbGl6ZXI6IHNlcmlhbGl6ZUluaXRpYWxpemVyKHRoaXMucmVjdXJyZW50SW5pdGlhbGl6ZXIpLFxuICAgICAgYmlhc0luaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmJpYXNJbml0aWFsaXplciksXG4gICAgICB1bml0Rm9yZ2V0QmlhczogdGhpcy51bml0Rm9yZ2V0QmlhcyxcbiAgICAgIGtlcm5lbFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmtlcm5lbFJlZ3VsYXJpemVyKSxcbiAgICAgIHJlY3VycmVudFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyKSxcbiAgICAgIGJpYXNSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5iaWFzUmVndWxhcml6ZXIpLFxuICAgICAgYWN0aXZpdHlSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5hY3Rpdml0eVJlZ3VsYXJpemVyKSxcbiAgICAgIGtlcm5lbENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5rZXJuZWxDb25zdHJhaW50KSxcbiAgICAgIHJlY3VycmVudENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5yZWN1cnJlbnRDb25zdHJhaW50KSxcbiAgICAgIGJpYXNDb25zdHJhaW50OiBzZXJpYWxpemVDb25zdHJhaW50KHRoaXMuYmlhc0NvbnN0cmFpbnQpLFxuICAgICAgZHJvcG91dDogdGhpcy5kcm9wb3V0LFxuICAgICAgcmVjdXJyZW50RHJvcG91dDogdGhpcy5yZWN1cnJlbnREcm9wb3V0LFxuICAgICAgaW1wbGVtZW50YXRpb246IHRoaXMuaW1wbGVtZW50YXRpb24sXG4gICAgfTtcblxuICAgIHJldHVybiB7Li4uYmFzZUNvbmZpZywgLi4uY29uZmlnfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKExTVE1DZWxsKTtcblxuLy8gUG9ydGluZyBOb3RlOiBTaW5jZSB0aGlzIGlzIGEgc3VwZXJzZXQgb2YgU2ltcGxlUk5OTGF5ZXJDb25maWcsIHdlIGluaGVyaXRcbi8vICAgZnJvbSB0aGF0IGludGVyZmFjZSBpbnN0ZWFkIG9mIHJlcGVhdGluZyB0aGUgZmllbGRzIGhlcmUuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgTFNUTUxheWVyQXJncyBleHRlbmRzIFNpbXBsZVJOTkxheWVyQXJncyB7XG4gIC8qKlxuICAgKiBBY3RpdmF0aW9uIGZ1bmN0aW9uIHRvIHVzZSBmb3IgdGhlIHJlY3VycmVudCBzdGVwLlxuICAgKlxuICAgKiBEZWZhdWx0cyB0byBoYXJkIHNpZ21vaWQgKGBoYXJkU2lnbW9pZGApLlxuICAgKlxuICAgKiBJZiBgbnVsbGAsIG5vIGFjdGl2YXRpb24gaXMgYXBwbGllZC5cbiAgICovXG4gIHJlY3VycmVudEFjdGl2YXRpb24/OiBBY3RpdmF0aW9uSWRlbnRpZmllcjtcblxuICAvKipcbiAgICogSWYgYHRydWVgLCBhZGQgMSB0byB0aGUgYmlhcyBvZiB0aGUgZm9yZ2V0IGdhdGUgYXQgaW5pdGlhbGl6YXRpb24uXG4gICAqIFNldHRpbmcgaXQgdG8gYHRydWVgIHdpbGwgYWxzbyBmb3JjZSBgYmlhc0luaXRpYWxpemVyID0gJ3plcm9zJ2AuXG4gICAqIFRoaXMgaXMgcmVjb21tZW5kZWQgaW5cbiAgICogW0pvemVmb3dpY3ogZXRcbiAgICogYWwuXShodHRwOi8vd3d3LmptbHIub3JnL3Byb2NlZWRpbmdzL3BhcGVycy92Mzcvam96ZWZvd2ljejE1LnBkZikuXG4gICAqL1xuICB1bml0Rm9yZ2V0Qmlhcz86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIEltcGxlbWVudGF0aW9uIG1vZGUsIGVpdGhlciAxIG9yIDIuXG4gICAqICAgTW9kZSAxIHdpbGwgc3RydWN0dXJlIGl0cyBvcGVyYXRpb25zIGFzIGEgbGFyZ2VyIG51bWJlciBvZlxuICAgKiAgIHNtYWxsZXIgZG90IHByb2R1Y3RzIGFuZCBhZGRpdGlvbnMsIHdoZXJlYXMgbW9kZSAyIHdpbGxcbiAgICogICBiYXRjaCB0aGVtIGludG8gZmV3ZXIsIGxhcmdlciBvcGVyYXRpb25zLiBUaGVzZSBtb2RlcyB3aWxsXG4gICAqICAgaGF2ZSBkaWZmZXJlbnQgcGVyZm9ybWFuY2UgcHJvZmlsZXMgb24gZGlmZmVyZW50IGhhcmR3YXJlIGFuZFxuICAgKiAgIGZvciBkaWZmZXJlbnQgYXBwbGljYXRpb25zLlxuICAgKlxuICAgKiBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXMgaW1wbGVtZW50YXRpb25cbiAgICogMiwgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mIHRoaXMgY29uZmlnIGZpZWxkLlxuICAgKi9cbiAgaW1wbGVtZW50YXRpb24/OiBudW1iZXI7XG59XG5cbmV4cG9ydCBjbGFzcyBMU1RNIGV4dGVuZHMgUk5OIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTFNUTSc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IExTVE1MYXllckFyZ3MpIHtcbiAgICBpZiAoYXJncy5pbXBsZW1lbnRhdGlvbiA9PT0gMCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdgaW1wbGVtZW50YXRpb249MGAgaGFzIGJlZW4gZGVwcmVjYXRlZCwgYW5kIG5vdyBkZWZhdWx0cyB0byAnICtcbiAgICAgICAgICAnYGltcGxlbWVudGF0aW9uPTFgLiBQbGVhc2UgdXBkYXRlIHlvdXIgbGF5ZXIgY2FsbC4nKTtcbiAgICB9XG4gICAgYXJncy5jZWxsID0gbmV3IExTVE1DZWxsKGFyZ3MpO1xuICAgIHN1cGVyKGFyZ3MgYXMgUk5OTGF5ZXJBcmdzKTtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgYWN0aXZpdHlSZWd1bGFyaXplci5cbiAgfVxuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpZiAodGhpcy5jZWxsLmRyb3BvdXRNYXNrICE9IG51bGwpIHtcbiAgICAgICAgdGZjLmRpc3Bvc2UodGhpcy5jZWxsLmRyb3BvdXRNYXNrKTtcbiAgICAgICAgdGhpcy5jZWxsLmRyb3BvdXRNYXNrID0gbnVsbDtcbiAgICAgIH1cbiAgICAgIGlmICh0aGlzLmNlbGwucmVjdXJyZW50RHJvcG91dE1hc2sgIT0gbnVsbCkge1xuICAgICAgICB0ZmMuZGlzcG9zZSh0aGlzLmNlbGwucmVjdXJyZW50RHJvcG91dE1hc2spO1xuICAgICAgICB0aGlzLmNlbGwucmVjdXJyZW50RHJvcG91dE1hc2sgPSBudWxsO1xuICAgICAgfVxuICAgICAgY29uc3QgbWFzayA9IGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snbWFzayddO1xuICAgICAgY29uc3QgdHJhaW5pbmcgPSBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ3RyYWluaW5nJ107XG4gICAgICBjb25zdCBpbml0aWFsU3RhdGU6IFRlbnNvcltdID1cbiAgICAgICAgICBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ2luaXRpYWxTdGF0ZSddO1xuICAgICAgcmV0dXJuIHN1cGVyLmNhbGwoaW5wdXRzLCB7bWFzaywgdHJhaW5pbmcsIGluaXRpYWxTdGF0ZX0pO1xuICAgIH0pO1xuICB9XG5cbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBmcm9tQ29uZmlnPFQgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sXG4gICAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCk6IFQge1xuICAgIGlmIChjb25maWdbJ2ltcGxtZW50YXRpb24nXSA9PT0gMCkge1xuICAgICAgY29uZmlnWydpbXBsZW1lbnRhdGlvbiddID0gMTtcbiAgICB9XG4gICAgcmV0dXJuIG5ldyBjbHMoY29uZmlnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKExTVE0pO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgU3RhY2tlZFJOTkNlbGxzQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBBIGBBcnJheWAgb2YgYFJOTkNlbGxgIGluc3RhbmNlcy5cbiAgICovXG4gIGNlbGxzOiBSTk5DZWxsW107XG59XG5cbmV4cG9ydCBjbGFzcyBTdGFja2VkUk5OQ2VsbHMgZXh0ZW5kcyBSTk5DZWxsIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnU3RhY2tlZFJOTkNlbGxzJztcbiAgcHJvdGVjdGVkIGNlbGxzOiBSTk5DZWxsW107XG5cbiAgY29uc3RydWN0b3IoYXJnczogU3RhY2tlZFJOTkNlbGxzQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMuY2VsbHMgPSBhcmdzLmNlbGxzO1xuICB9XG5cbiAgZ2V0IHN0YXRlU2l6ZSgpOiBudW1iZXJbXSB7XG4gICAgLy8gU3RhdGVzIGFyZSBhIGZsYXQgbGlzdCBpbiByZXZlcnNlIG9yZGVyIG9mIHRoZSBjZWxsIHN0YWNrLlxuICAgIC8vIFRoaXMgYWxsb3dzIHBlcnNlcnZpbmcgdGhlIHJlcXVpcmVtZW50IGBzdGFjay5zdGF0ZXNpemVbMF0gPT09XG4gICAgLy8gb3V0cHV0RGltYC4gRS5nLiwgc3RhdGVzIG9mIGEgMi1sYXllciBMU1RNIHdvdWxkIGJlIGBbaDIsIGMyLCBoMSwgYzFdYCxcbiAgICAvLyBhc3N1bWluZyBvbmUgTFNUTSBoYXMgc3RhdGVzIGBbaCwgY11gLlxuICAgIGNvbnN0IHN0YXRlU2l6ZTogbnVtYmVyW10gPSBbXTtcbiAgICBmb3IgKGNvbnN0IGNlbGwgb2YgdGhpcy5jZWxscy5zbGljZSgpLnJldmVyc2UoKSkge1xuICAgICAgaWYgKEFycmF5LmlzQXJyYXkoY2VsbC5zdGF0ZVNpemUpKSB7XG4gICAgICAgIHN0YXRlU2l6ZS5wdXNoKC4uLmNlbGwuc3RhdGVTaXplKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHN0YXRlU2l6ZS5wdXNoKGNlbGwuc3RhdGVTaXplKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHN0YXRlU2l6ZTtcbiAgfVxuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpbnB1dHMgPSBpbnB1dHMgYXMgVGVuc29yW107XG4gICAgICBsZXQgc3RhdGVzID0gaW5wdXRzLnNsaWNlKDEpO1xuXG4gICAgICAvLyBSZWNvdmVyIHBlci1jZWxsIHN0YXRlcy5cbiAgICAgIGNvbnN0IG5lc3RlZFN0YXRlczogVGVuc29yW11bXSA9IFtdO1xuICAgICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMuc2xpY2UoKS5yZXZlcnNlKCkpIHtcbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkoY2VsbC5zdGF0ZVNpemUpKSB7XG4gICAgICAgICAgbmVzdGVkU3RhdGVzLnB1c2goc3RhdGVzLnNwbGljZSgwLCBjZWxsLnN0YXRlU2l6ZS5sZW5ndGgpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBuZXN0ZWRTdGF0ZXMucHVzaChzdGF0ZXMuc3BsaWNlKDAsIDEpKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgbmVzdGVkU3RhdGVzLnJldmVyc2UoKTtcblxuICAgICAgLy8gQ2FsbCB0aGUgY2VsbHMgaW4gb3JkZXIgYW5kIHN0b3JlIHRoZSByZXR1cm5lZCBzdGF0ZXMuXG4gICAgICBjb25zdCBuZXdOZXN0ZWRTdGF0ZXM6IFRlbnNvcltdW10gPSBbXTtcbiAgICAgIGxldCBjYWxsSW5wdXRzOiBUZW5zb3JbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5jZWxscy5sZW5ndGg7ICsraSkge1xuICAgICAgICBjb25zdCBjZWxsID0gdGhpcy5jZWxsc1tpXTtcbiAgICAgICAgc3RhdGVzID0gbmVzdGVkU3RhdGVzW2ldO1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgY29uc3RhbnRzLlxuICAgICAgICBpZiAoaSA9PT0gMCkge1xuICAgICAgICAgIGNhbGxJbnB1dHMgPSBbaW5wdXRzWzBdXS5jb25jYXQoc3RhdGVzKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBjYWxsSW5wdXRzID0gW2NhbGxJbnB1dHNbMF1dLmNvbmNhdChzdGF0ZXMpO1xuICAgICAgICB9XG4gICAgICAgIGNhbGxJbnB1dHMgPSBjZWxsLmNhbGwoY2FsbElucHV0cywga3dhcmdzKSBhcyBUZW5zb3JbXTtcbiAgICAgICAgbmV3TmVzdGVkU3RhdGVzLnB1c2goY2FsbElucHV0cy5zbGljZSgxKSk7XG4gICAgICB9XG5cbiAgICAgIC8vIEZvcm1hdCB0aGUgbmV3IHN0YXRlcyBhcyBhIGZsYXQgbGlzdCBpbiByZXZlcnNlIGNlbGwgb3JkZXIuXG4gICAgICBzdGF0ZXMgPSBbXTtcbiAgICAgIGZvciAoY29uc3QgY2VsbFN0YXRlcyBvZiBuZXdOZXN0ZWRTdGF0ZXMuc2xpY2UoKS5yZXZlcnNlKCkpIHtcbiAgICAgICAgc3RhdGVzLnB1c2goLi4uY2VsbFN0YXRlcyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gW2NhbGxJbnB1dHNbMF1dLmNvbmNhdChzdGF0ZXMpO1xuICAgIH0pO1xuICB9XG5cbiAgcHVibGljIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICBpZiAoaXNBcnJheU9mU2hhcGVzKGlucHV0U2hhcGUpKSB7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgaW5wdXQgY29uc3RhbnRzLlxuICAgICAgLy8gY29uc3QgY29uc3RhbnRTaGFwZSA9IGlucHV0U2hhcGUuc2xpY2UoMSk7XG4gICAgICBpbnB1dFNoYXBlID0gKGlucHV0U2hhcGUgYXMgU2hhcGVbXSlbMF07XG4gICAgfVxuICAgIGlucHV0U2hhcGUgPSBpbnB1dFNoYXBlIGFzIFNoYXBlO1xuICAgIGxldCBvdXRwdXREaW06IG51bWJlcjtcbiAgICB0aGlzLmNlbGxzLmZvckVhY2goKGNlbGwsIGkpID0+IHtcbiAgICAgIG5hbWVTY29wZShgUk5OQ2VsbF8ke2l9YCwgKCkgPT4ge1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgaW5wdXQgY29uc3RhbnRzLlxuXG4gICAgICAgIGNlbGwuYnVpbGQoaW5wdXRTaGFwZSk7XG4gICAgICAgIGlmIChBcnJheS5pc0FycmF5KGNlbGwuc3RhdGVTaXplKSkge1xuICAgICAgICAgIG91dHB1dERpbSA9IGNlbGwuc3RhdGVTaXplWzBdO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIG91dHB1dERpbSA9IGNlbGwuc3RhdGVTaXplO1xuICAgICAgICB9XG4gICAgICAgIGlucHV0U2hhcGUgPSBbaW5wdXRTaGFwZVswXSwgb3V0cHV0RGltXSBhcyBTaGFwZTtcbiAgICAgIH0pO1xuICAgIH0pO1xuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuXG4gICAgY29uc3QgZ2V0Q2VsbENvbmZpZyA9IChjZWxsOiBSTk5DZWxsKSA9PiB7XG4gICAgICByZXR1cm4ge1xuICAgICAgICAnY2xhc3NOYW1lJzogY2VsbC5nZXRDbGFzc05hbWUoKSxcbiAgICAgICAgJ2NvbmZpZyc6IGNlbGwuZ2V0Q29uZmlnKCksXG4gICAgICB9O1xuICAgIH07XG5cbiAgICBjb25zdCBjZWxsQ29uZmlncyA9IHRoaXMuY2VsbHMubWFwKGdldENlbGxDb25maWcpO1xuXG4gICAgY29uc3QgY29uZmlnID0geydjZWxscyc6IGNlbGxDb25maWdzfTtcblxuICAgIHJldHVybiB7Li4uYmFzZUNvbmZpZywgLi4uY29uZmlnfTtcbiAgfVxuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgZnJvbUNvbmZpZzxUIGV4dGVuZHMgc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGU+KFxuICAgICAgY2xzOiBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yPFQ+LFxuICAgICAgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QsXG4gICAgICBjdXN0b21PYmplY3RzID0ge30gYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0KTogVCB7XG4gICAgY29uc3QgY2VsbHM6IFJOTkNlbGxbXSA9IFtdO1xuICAgIGZvciAoY29uc3QgY2VsbENvbmZpZyBvZiAoY29uZmlnWydjZWxscyddIGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdFtdKSkge1xuICAgICAgY2VsbHMucHVzaChkZXNlcmlhbGl6ZShjZWxsQ29uZmlnLCBjdXN0b21PYmplY3RzKSBhcyBSTk5DZWxsKTtcbiAgICB9XG4gICAgcmV0dXJuIG5ldyBjbHMoe2NlbGxzfSk7XG4gIH1cblxuICBnZXQgdHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIGlmICghdGhpcy50cmFpbmFibGUpIHtcbiAgICAgIHJldHVybiBbXTtcbiAgICB9XG4gICAgY29uc3Qgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMpIHtcbiAgICAgIHdlaWdodHMucHVzaCguLi5jZWxsLnRyYWluYWJsZVdlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gd2VpZ2h0cztcbiAgfVxuXG4gIGdldCBub25UcmFpbmFibGVXZWlnaHRzKCk6IExheWVyVmFyaWFibGVbXSB7XG4gICAgY29uc3Qgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMpIHtcbiAgICAgIHdlaWdodHMucHVzaCguLi5jZWxsLm5vblRyYWluYWJsZVdlaWdodHMpO1xuICAgIH1cbiAgICBpZiAoIXRoaXMudHJhaW5hYmxlKSB7XG4gICAgICBjb25zdCB0cmFpbmFibGVXZWlnaHRzOiBMYXllclZhcmlhYmxlW10gPSBbXTtcbiAgICAgIGZvciAoY29uc3QgY2VsbCBvZiB0aGlzLmNlbGxzKSB7XG4gICAgICAgIHRyYWluYWJsZVdlaWdodHMucHVzaCguLi5jZWxsLnRyYWluYWJsZVdlaWdodHMpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHRyYWluYWJsZVdlaWdodHMuY29uY2F0KHdlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gd2VpZ2h0cztcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXRyaWV2ZSB0aGUgd2VpZ2h0cyBvZiBhIHRoZSBtb2RlbC5cbiAgICpcbiAgICogQHJldHVybnMgQSBmbGF0IGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzLlxuICAgKi9cbiAgZ2V0V2VpZ2h0cygpOiBUZW5zb3JbXSB7XG4gICAgY29uc3Qgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMpIHtcbiAgICAgIHdlaWdodHMucHVzaCguLi5jZWxsLndlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gYmF0Y2hHZXRWYWx1ZSh3ZWlnaHRzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXQgdGhlIHdlaWdodHMgb2YgdGhlIG1vZGVsLlxuICAgKlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBBbiBgQXJyYXlgIG9mIGB0Zi5UZW5zb3JgcyB3aXRoIHNoYXBlcyBhbmQgdHlwZXMgbWF0Y2hpbmdcbiAgICogICAgIHRoZSBvdXRwdXQgb2YgYGdldFdlaWdodHMoKWAuXG4gICAqL1xuICBzZXRXZWlnaHRzKHdlaWdodHM6IFRlbnNvcltdKTogdm9pZCB7XG4gICAgY29uc3QgdHVwbGVzOiBBcnJheTxbTGF5ZXJWYXJpYWJsZSwgVGVuc29yXT4gPSBbXTtcbiAgICBmb3IgKGNvbnN0IGNlbGwgb2YgdGhpcy5jZWxscykge1xuICAgICAgY29uc3QgbnVtUGFyYW1zID0gY2VsbC53ZWlnaHRzLmxlbmd0aDtcbiAgICAgIGNvbnN0IGlucHV0V2VpZ2h0cyA9IHdlaWdodHMuc3BsaWNlKG51bVBhcmFtcyk7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IGNlbGwud2VpZ2h0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICB0dXBsZXMucHVzaChbY2VsbC53ZWlnaHRzW2ldLCBpbnB1dFdlaWdodHNbaV1dKTtcbiAgICAgIH1cbiAgICB9XG4gICAgYmF0Y2hTZXRWYWx1ZSh0dXBsZXMpO1xuICB9XG5cbiAgLy8gVE9ETyhjYWlzKTogTWF5YmUgaW1wbGVtbnQgYGxvc3Nlc2AgYW5kIGBnZXRMb3NzZXNGb3JgLlxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFN0YWNrZWRSTk5DZWxscyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZW5lcmF0ZURyb3BvdXRNYXNrKGFyZ3M6IHtcbiAgb25lczogKCkgPT4gdGZjLlRlbnNvcixcbiAgcmF0ZTogbnVtYmVyLFxuICB0cmFpbmluZz86IGJvb2xlYW4sXG4gIGNvdW50PzogbnVtYmVyLFxuICBkcm9wb3V0RnVuYz86IEZ1bmN0aW9uLFxufSk6IHRmYy5UZW5zb3J8dGZjLlRlbnNvcltdIHtcbiAgY29uc3Qge29uZXMsIHJhdGUsIHRyYWluaW5nID0gZmFsc2UsIGNvdW50ID0gMSwgZHJvcG91dEZ1bmN9ID0gYXJncztcblxuICBjb25zdCBkcm9wcGVkSW5wdXRzID0gKCkgPT5cbiAgICAgIGRyb3BvdXRGdW5jICE9IG51bGwgPyBkcm9wb3V0RnVuYyhvbmVzKCksIHJhdGUpIDogSy5kcm9wb3V0KG9uZXMoKSwgcmF0ZSk7XG5cbiAgY29uc3QgY3JlYXRlTWFzayA9ICgpID0+IEsuaW5UcmFpblBoYXNlKGRyb3BwZWRJbnB1dHMsIG9uZXMsIHRyYWluaW5nKTtcblxuICAvLyBqdXN0IGluIGNhc2UgY291bnQgaXMgcHJvdmlkZWQgd2l0aCBudWxsIG9yIHVuZGVmaW5lZFxuICBpZiAoIWNvdW50IHx8IGNvdW50IDw9IDEpIHtcbiAgICByZXR1cm4gdGZjLmtlZXAoY3JlYXRlTWFzaygpLmNsb25lKCkpO1xuICB9XG5cbiAgY29uc3QgbWFza3MgPSBBcnJheShjb3VudCkuZmlsbCh1bmRlZmluZWQpLm1hcChjcmVhdGVNYXNrKTtcblxuICByZXR1cm4gbWFza3MubWFwKG0gPT4gdGZjLmtlZXAobS5jbG9uZSgpKSk7XG59XG4iXX0=
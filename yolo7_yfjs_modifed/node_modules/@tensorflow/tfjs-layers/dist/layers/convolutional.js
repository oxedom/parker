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
 * TensorFlow.js Layers: Convolutional Layers
 */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import { getActivation, serializeActivation } from '../activations';
import { imageDataFormat } from '../backend/common';
import * as K from '../backend/tfjs_backend';
import { checkDataFormat, checkInterpolationFormat, checkPaddingMode } from '../common';
import { getConstraint, serializeConstraint } from '../constraints';
import { InputSpec, Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { getInitializer, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import { convOutputLength, deconvLength, normalizeArray } from '../utils/conv_utils';
import * as generic_utils from '../utils/generic_utils';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
/**
 * Transpose and cast the input before the conv2d.
 * @param x Input image tensor.
 * @param dataFormat
 */
export function preprocessConv2DInput(x, dataFormat) {
    // TODO(cais): Cast type to float32 if not.
    return tidy(() => {
        checkDataFormat(dataFormat);
        if (dataFormat === 'channelsFirst') {
            return tfc.transpose(x, [0, 2, 3, 1]); // NCHW -> NHWC.
        }
        else {
            return x;
        }
    });
}
/**
 * Transpose and cast the input before the conv3d.
 * @param x Input image tensor.
 * @param dataFormat
 */
export function preprocessConv3DInput(x, dataFormat) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        if (dataFormat === 'channelsFirst') {
            return tfc.transpose(x, [0, 2, 3, 4, 1]); // NCDHW -> NDHWC.
        }
        else {
            return x;
        }
    });
}
/**
 * 1D-convolution with bias added.
 *
 * Porting Note: This function does not exist in the Python Keras backend.
 *   It is exactly the same as `conv2d`, except the added `bias`.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.
 * @param bias Bias, rank-3, of shape `[outDepth]`.
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
export function conv1dWithBias(x, kernel, bias, strides = 1, padding = 'valid', dataFormat, dilationRate = 1) {
    return tidy(() => {
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        checkDataFormat(dataFormat);
        // Check the ranks of x, kernel and bias.
        if (x.shape.length !== 3) {
            throw new ValueError(`The input of a conv1dWithBias operation should be 3, but is ` +
                `${x.shape.length} instead.`);
        }
        if (kernel.shape.length !== 3) {
            throw new ValueError(`The kernel for a conv1dWithBias operation should be 3, but is ` +
                `${kernel.shape.length} instead`);
        }
        if (bias != null && bias.shape.length !== 1) {
            throw new ValueError(`The bias for a conv1dWithBias operation should be 1, but is ` +
                `${kernel.shape.length} instead`);
        }
        // TODO(cais): Support CAUSAL padding mode.
        if (dataFormat === 'channelsFirst') {
            x = tfc.transpose(x, [0, 2, 1]); // NCW -> NWC.
        }
        if (padding === 'causal') {
            throw new NotImplementedError('The support for CAUSAL padding mode in conv1dWithBias is not ' +
                'implemented yet.');
        }
        let y = tfc.conv1d(x, kernel, strides, padding === 'same' ? 'same' : 'valid', 'NWC', dilationRate);
        if (bias != null) {
            y = K.biasAdd(y, bias);
        }
        return y;
    });
}
/**
 * 1D-convolution.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.s
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
export function conv1d(x, kernel, strides = 1, padding = 'valid', dataFormat, dilationRate = 1) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        return conv1dWithBias(x, kernel, null, strides, padding, dataFormat, dilationRate);
    });
}
/**
 * 2D Convolution
 * @param x
 * @param kernel kernel of the convolution.
 * @param strides strides array.
 * @param padding padding mode. Default to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param dilationRate dilation rate array.
 * @returns Result of the 2D pooling.
 */
export function conv2d(x, kernel, strides = [1, 1], padding = 'valid', dataFormat, dilationRate) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        return conv2dWithBiasActivation(x, kernel, null, strides, padding, dataFormat, dilationRate);
    });
}
/**
 * 2D Convolution with an added bias and optional activation.
 * Note: This function does not exist in the Python Keras Backend. This function
 * is exactly the same as `conv2d`, except the added `bias`.
 */
export function conv2dWithBiasActivation(x, kernel, bias, strides = [1, 1], padding = 'valid', dataFormat, dilationRate, activation = null) {
    return tidy(() => {
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        checkDataFormat(dataFormat);
        if (x.rank !== 3 && x.rank !== 4) {
            throw new ValueError(`conv2dWithBiasActivation expects input to be of rank 3 or 4, ` +
                `but received ${x.rank}.`);
        }
        if (kernel.rank !== 3 && kernel.rank !== 4) {
            throw new ValueError(`conv2dWithBiasActivation expects kernel to be of rank 3 or 4, ` +
                `but received ${x.rank}.`);
        }
        let y = preprocessConv2DInput(x, dataFormat);
        if (padding === 'causal') {
            throw new NotImplementedError('The support for CAUSAL padding mode in conv1dWithBias is not ' +
                'implemented yet.');
        }
        y = tfc.fused.conv2d({
            x: y,
            filter: kernel,
            strides: strides,
            pad: padding === 'same' ? 'same' : 'valid',
            dilations: dilationRate,
            dataFormat: 'NHWC',
            bias,
            activation
        });
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 3, 1, 2]);
        }
        return y;
    });
}
/**
 * 3D Convolution.
 * @param x
 * @param kernel kernel of the convolution.
 * @param strides strides array.
 * @param padding padding mode. Default to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param dilationRate dilation rate array.
 * @returns Result of the 3D convolution.
 */
export function conv3d(x, kernel, strides = [1, 1, 1], padding = 'valid', dataFormat, dilationRate) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        return conv3dWithBias(x, kernel, null, strides, padding, dataFormat, dilationRate);
    });
}
/**
 * 3D Convolution with an added bias.
 * Note: This function does not exist in the Python Keras Backend. This function
 * is exactly the same as `conv3d`, except the added `bias`.
 */
export function conv3dWithBias(x, kernel, bias, strides = [1, 1, 1], padding = 'valid', dataFormat, dilationRate) {
    return tidy(() => {
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        checkDataFormat(dataFormat);
        if (x.rank !== 4 && x.rank !== 5) {
            throw new ValueError(`conv3dWithBias expects input to be of rank 4 or 5, but received ` +
                `${x.rank}.`);
        }
        if (kernel.rank !== 4 && kernel.rank !== 5) {
            throw new ValueError(`conv3dWithBias expects kernel to be of rank 4 or 5, but received ` +
                `${x.rank}.`);
        }
        let y = preprocessConv3DInput(x, dataFormat);
        if (padding === 'causal') {
            throw new NotImplementedError('The support for CAUSAL padding mode in conv3dWithBias is not ' +
                'implemented yet.');
        }
        y = tfc.conv3d(y, kernel, strides, padding === 'same' ? 'same' : 'valid', 'NDHWC', dilationRate);
        if (bias != null) {
            y = K.biasAdd(y, bias);
        }
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 4, 1, 2, 3]);
        }
        return y;
    });
}
/**
 * Abstract convolution layer.
 */
export class BaseConv extends Layer {
    constructor(rank, args) {
        super(args);
        this.bias = null;
        this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        BaseConv.verifyArgs(args);
        this.rank = rank;
        generic_utils.assertPositiveInteger(this.rank, 'rank');
        if (this.rank !== 1 && this.rank !== 2 && this.rank !== 3) {
            throw new NotImplementedError(`Convolution layer for rank other than 1, 2, or 3 (${this.rank}) is ` +
                `not implemented yet.`);
        }
        this.kernelSize = normalizeArray(args.kernelSize, rank, 'kernelSize');
        this.strides = normalizeArray(args.strides == null ? 1 : args.strides, rank, 'strides');
        this.padding = args.padding == null ? 'valid' : args.padding;
        checkPaddingMode(this.padding);
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        this.activation = getActivation(args.activation);
        this.useBias = args.useBias == null ? true : args.useBias;
        this.biasInitializer =
            getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
        this.biasConstraint = getConstraint(args.biasConstraint);
        this.biasRegularizer = getRegularizer(args.biasRegularizer);
        this.activityRegularizer = getRegularizer(args.activityRegularizer);
        this.dilationRate = normalizeArray(args.dilationRate == null ? 1 : args.dilationRate, rank, 'dilationRate');
        if (this.rank === 1 &&
            (Array.isArray(this.dilationRate) && this.dilationRate.length !== 1)) {
            throw new ValueError(`dilationRate must be a number or an array of a single number ` +
                `for 1D convolution, but received ` +
                `${JSON.stringify(this.dilationRate)}`);
        }
        else if (this.rank === 2) {
            if (typeof this.dilationRate === 'number') {
                this.dilationRate = [this.dilationRate, this.dilationRate];
            }
            else if (this.dilationRate.length !== 2) {
                throw new ValueError(`dilationRate must be a number or array of two numbers for 2D ` +
                    `convolution, but received ${JSON.stringify(this.dilationRate)}`);
            }
        }
        else if (this.rank === 3) {
            if (typeof this.dilationRate === 'number') {
                this.dilationRate =
                    [this.dilationRate, this.dilationRate, this.dilationRate];
            }
            else if (this.dilationRate.length !== 3) {
                throw new ValueError(`dilationRate must be a number or array of three numbers for 3D ` +
                    `convolution, but received ${JSON.stringify(this.dilationRate)}`);
            }
        }
    }
    static verifyArgs(args) {
        // Check config.kernelSize type and shape.
        generic_utils.assert('kernelSize' in args, `required key 'kernelSize' not in config`);
        if (typeof args.kernelSize !== 'number' &&
            !generic_utils.checkArrayTypeAndLength(args.kernelSize, 'number', 1, 3)) {
            throw new ValueError(`BaseConv expects config.kernelSize to be number or number[] with ` +
                `length 1, 2, or 3, but received ${JSON.stringify(args.kernelSize)}.`);
        }
    }
    getConfig() {
        const config = {
            kernelSize: this.kernelSize,
            strides: this.strides,
            padding: this.padding,
            dataFormat: this.dataFormat,
            dilationRate: this.dilationRate,
            activation: serializeActivation(this.activation),
            useBias: this.useBias,
            biasInitializer: serializeInitializer(this.biasInitializer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            biasConstraint: serializeConstraint(this.biasConstraint)
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/**
 * Abstract nD convolution layer.  Ancestor of convolution layers which reduce
 * across channels, i.e., Conv1D and Conv2D, but not DepthwiseConv2D.
 */
export class Conv extends BaseConv {
    constructor(rank, args) {
        super(rank, args);
        this.kernel = null;
        Conv.verifyArgs(args);
        this.filters = args.filters;
        generic_utils.assertPositiveInteger(this.filters, 'filters');
        this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
        this.kernelConstraint = getConstraint(args.kernelConstraint);
        this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const channelAxis = this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;
        if (inputShape[channelAxis] == null) {
            throw new ValueError(`The channel dimension of the input should be defined. ` +
                `Found ${inputShape[channelAxis]}`);
        }
        const inputDim = inputShape[channelAxis];
        const kernelShape = this.kernelSize.concat([inputDim, this.filters]);
        this.kernel = this.addWeight('kernel', kernelShape, null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.filters], null, this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        this.inputSpec = [{ ndim: this.rank + 2, axes: { [channelAxis]: inputDim } }];
        this.built = true;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = getExactlyOneTensor(inputs);
            let outputs;
            const biasValue = this.bias == null ? null : this.bias.read();
            const fusedActivationName = generic_utils.mapActivationToFusedKernel(this.activation.getClassName());
            if (fusedActivationName != null && this.rank === 2) {
                outputs = conv2dWithBiasActivation(inputs, this.kernel.read(), biasValue, this.strides, this.padding, this.dataFormat, this.dilationRate, fusedActivationName);
            }
            else {
                if (this.rank === 1) {
                    outputs = conv1dWithBias(inputs, this.kernel.read(), biasValue, this.strides[0], this.padding, this.dataFormat, this.dilationRate[0]);
                }
                else if (this.rank === 2) {
                    // TODO(cais): Move up to constructor.
                    outputs = conv2dWithBiasActivation(inputs, this.kernel.read(), biasValue, this.strides, this.padding, this.dataFormat, this.dilationRate);
                }
                else if (this.rank === 3) {
                    outputs = conv3dWithBias(inputs, this.kernel.read(), biasValue, this.strides, this.padding, this.dataFormat, this.dilationRate);
                }
                else {
                    throw new NotImplementedError('convolutions greater than 3D are not implemented yet.');
                }
                if (this.activation != null) {
                    outputs = this.activation.apply(outputs);
                }
            }
            return outputs;
        });
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const newSpace = [];
        const space = (this.dataFormat === 'channelsLast') ?
            inputShape.slice(1, inputShape.length - 1) :
            inputShape.slice(2);
        for (let i = 0; i < space.length; ++i) {
            const newDim = convOutputLength(space[i], this.kernelSize[i], this.padding, this.strides[i], typeof this.dilationRate === 'number' ? this.dilationRate :
                this.dilationRate[i]);
            newSpace.push(newDim);
        }
        let outputShape = [inputShape[0]];
        if (this.dataFormat === 'channelsLast') {
            outputShape = outputShape.concat(newSpace);
            outputShape.push(this.filters);
        }
        else {
            outputShape.push(this.filters);
            outputShape = outputShape.concat(newSpace);
        }
        return outputShape;
    }
    getConfig() {
        const config = {
            filters: this.filters,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint)
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    static verifyArgs(args) {
        // Check config.filters type, shape, and value.
        if (!('filters' in args) || typeof args.filters !== 'number' ||
            args.filters < 1) {
            throw new ValueError(`Convolution layer expected config.filters to be a 'number' > 0 ` +
                `but got ${JSON.stringify(args.filters)}`);
        }
    }
}
export class Conv2D extends Conv {
    constructor(args) {
        super(2, args);
        Conv2D.verifyArgs(args);
    }
    getConfig() {
        const config = super.getConfig();
        delete config['rank'];
        return config;
    }
    static verifyArgs(args) {
        // config.kernelSize must be a number or array of numbers.
        if ((typeof args.kernelSize !== 'number') &&
            !generic_utils.checkArrayTypeAndLength(args.kernelSize, 'number', 1, 2)) {
            throw new ValueError(`Conv2D expects config.kernelSize to be number or number[] with ` +
                `length 1 or 2, but received ${JSON.stringify(args.kernelSize)}.`);
        }
    }
}
/** @nocollapse */
Conv2D.className = 'Conv2D';
serialization.registerClass(Conv2D);
export class Conv3D extends Conv {
    constructor(args) {
        super(3, args);
        Conv3D.verifyArgs(args);
    }
    getConfig() {
        const config = super.getConfig();
        delete config['rank'];
        return config;
    }
    static verifyArgs(args) {
        // config.kernelSize must be a number or array of numbers.
        if (typeof args.kernelSize !== 'number') {
            if (!(Array.isArray(args.kernelSize) &&
                (args.kernelSize.length === 1 || args.kernelSize.length === 3))) {
                throw new ValueError(`Conv3D expects config.kernelSize to be number or` +
                    ` [number, number, number], but received ${JSON.stringify(args.kernelSize)}.`);
            }
        }
    }
}
/** @nocollapse */
Conv3D.className = 'Conv3D';
serialization.registerClass(Conv3D);
export class Conv2DTranspose extends Conv2D {
    constructor(args) {
        super(args);
        this.inputSpec = [new InputSpec({ ndim: 4 })];
        if (this.padding !== 'same' && this.padding !== 'valid') {
            throw new ValueError(`Conv2DTranspose currently supports only padding modes 'same' ` +
                `and 'valid', but received padding mode ${this.padding}`);
        }
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        if (inputShape.length !== 4) {
            throw new ValueError('Input should have rank 4; Received input shape: ' +
                JSON.stringify(inputShape));
        }
        const channelAxis = this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;
        if (inputShape[channelAxis] == null) {
            throw new ValueError('The channel dimension of the inputs should be defined. ' +
                'Found `None`.');
        }
        const inputDim = inputShape[channelAxis];
        const kernelShape = this.kernelSize.concat([this.filters, inputDim]);
        this.kernel = this.addWeight('kernel', kernelShape, 'float32', this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.filters], 'float32', this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        // Set input spec.
        this.inputSpec =
            [new InputSpec({ ndim: 4, axes: { [channelAxis]: inputDim } })];
        this.built = true;
    }
    call(inputs, kwargs) {
        return tfc.tidy(() => {
            let input = getExactlyOneTensor(inputs);
            if (input.shape.length !== 4) {
                throw new ValueError(`Conv2DTranspose.call() expects input tensor to be rank-4, but ` +
                    `received a tensor of rank-${input.shape.length}`);
            }
            const inputShape = input.shape;
            const batchSize = inputShape[0];
            let hAxis;
            let wAxis;
            if (this.dataFormat === 'channelsFirst') {
                hAxis = 2;
                wAxis = 3;
            }
            else {
                hAxis = 1;
                wAxis = 2;
            }
            const height = inputShape[hAxis];
            const width = inputShape[wAxis];
            const kernelH = this.kernelSize[0];
            const kernelW = this.kernelSize[1];
            const strideH = this.strides[0];
            const strideW = this.strides[1];
            // Infer the dynamic output shape.
            const outHeight = deconvLength(height, strideH, kernelH, this.padding);
            const outWidth = deconvLength(width, strideW, kernelW, this.padding);
            // Porting Note: We don't branch based on `this.dataFormat` here,
            // because
            //   the tjfs-core function `conv2dTranspose` called below always
            //   assumes channelsLast.
            const outputShape = [batchSize, outHeight, outWidth, this.filters];
            if (this.dataFormat !== 'channelsLast') {
                input = tfc.transpose(input, [0, 2, 3, 1]);
            }
            let outputs = tfc.conv2dTranspose(input, this.kernel.read(), outputShape, this.strides, this.padding);
            if (this.dataFormat !== 'channelsLast') {
                outputs = tfc.transpose(outputs, [0, 3, 1, 2]);
            }
            if (this.bias != null) {
                outputs =
                    K.biasAdd(outputs, this.bias.read(), this.dataFormat);
            }
            if (this.activation != null) {
                outputs = this.activation.apply(outputs);
            }
            return outputs;
        });
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const outputShape = inputShape.slice();
        let channelAxis;
        let heightAxis;
        let widthAxis;
        if (this.dataFormat === 'channelsFirst') {
            channelAxis = 1;
            heightAxis = 2;
            widthAxis = 3;
        }
        else {
            channelAxis = 3;
            heightAxis = 1;
            widthAxis = 2;
        }
        const kernelH = this.kernelSize[0];
        const kernelW = this.kernelSize[1];
        const strideH = this.strides[0];
        const strideW = this.strides[1];
        outputShape[channelAxis] = this.filters;
        outputShape[heightAxis] =
            deconvLength(outputShape[heightAxis], strideH, kernelH, this.padding);
        outputShape[widthAxis] =
            deconvLength(outputShape[widthAxis], strideW, kernelW, this.padding);
        return outputShape;
    }
    getConfig() {
        const config = super.getConfig();
        delete config['dilationRate'];
        return config;
    }
}
/** @nocollapse */
Conv2DTranspose.className = 'Conv2DTranspose';
serialization.registerClass(Conv2DTranspose);
export class Conv3DTranspose extends Conv3D {
    constructor(args) {
        super(args);
        this.inputSpec = [new InputSpec({ ndim: 5 })];
        if (this.padding !== 'same' && this.padding !== 'valid') {
            throw new ValueError(`Conv3DTranspose currently supports only padding modes 'same' ` +
                `and 'valid', but received padding mode ${this.padding}`);
        }
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        if (inputShape.length !== 5) {
            throw new ValueError('Input should have rank 5; Received input shape: ' +
                JSON.stringify(inputShape));
        }
        const channelAxis = this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;
        if (inputShape[channelAxis] == null) {
            throw new ValueError('The channel dimension of the inputs should be defined. ' +
                'Found `None`.');
        }
        const inputDim = inputShape[channelAxis];
        const kernelShape = this.kernelSize.concat([this.filters, inputDim]);
        this.kernel = this.addWeight('kernel', kernelShape, 'float32', this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.filters], 'float32', this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        // Set input spec.
        this.inputSpec =
            [new InputSpec({ ndim: 5, axes: { [channelAxis]: inputDim } })];
        this.built = true;
    }
    call(inputs, kwargs) {
        return tfc.tidy(() => {
            let input = getExactlyOneTensor(inputs);
            if (input.shape.length !== 5) {
                throw new ValueError(`Conv3DTranspose.call() expects input tensor to be rank-4, but ` +
                    `received a tensor of rank-${input.shape.length}`);
            }
            const inputShape = input.shape;
            const batchSize = inputShape[0];
            let hAxis;
            let wAxis;
            let dAxis;
            if (this.dataFormat === 'channelsFirst') {
                dAxis = 2;
                hAxis = 3;
                wAxis = 4;
            }
            else {
                dAxis = 1;
                hAxis = 2;
                wAxis = 3;
            }
            const depth = inputShape[dAxis];
            const height = inputShape[hAxis];
            const width = inputShape[wAxis];
            const kernelD = this.kernelSize[0];
            const kernelH = this.kernelSize[1];
            const kernelW = this.kernelSize[2];
            const strideD = this.strides[0];
            const strideH = this.strides[1];
            const strideW = this.strides[2];
            // Infer the dynamic output shape.
            const outDepth = deconvLength(depth, strideD, kernelD, this.padding);
            const outHeight = deconvLength(height, strideH, kernelH, this.padding);
            const outWidth = deconvLength(width, strideW, kernelW, this.padding);
            // Same as `conv2dTranspose`. We always assumes channelsLast.
            const outputShape = [batchSize, outDepth, outHeight, outWidth, this.filters];
            if (this.dataFormat !== 'channelsLast') {
                input = tfc.transpose(input, [0, 2, 3, 4, 1]);
            }
            let outputs = tfc.conv3dTranspose(input, this.kernel.read(), outputShape, this.strides, this.padding);
            if (this.dataFormat !== 'channelsLast') {
                outputs = tfc.transpose(outputs, [0, 4, 1, 2, 3]);
            }
            if (this.bias !== null) {
                outputs =
                    K.biasAdd(outputs, this.bias.read(), this.dataFormat);
            }
            if (this.activation !== null) {
                outputs = this.activation.apply(outputs);
            }
            return outputs;
        });
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const outputShape = inputShape.slice();
        let channelAxis;
        let depthAxis;
        let heightAxis;
        let widthAxis;
        if (this.dataFormat === 'channelsFirst') {
            channelAxis = 1;
            depthAxis = 2;
            heightAxis = 3;
            widthAxis = 4;
        }
        else {
            channelAxis = 4;
            depthAxis = 1;
            heightAxis = 2;
            widthAxis = 3;
        }
        const kernelD = this.kernelSize[0];
        const kernelH = this.kernelSize[1];
        const kernelW = this.kernelSize[2];
        const strideD = this.strides[0];
        const strideH = this.strides[1];
        const strideW = this.strides[2];
        outputShape[channelAxis] = this.filters;
        outputShape[depthAxis] =
            deconvLength(outputShape[depthAxis], strideD, kernelD, this.padding);
        outputShape[heightAxis] =
            deconvLength(outputShape[heightAxis], strideH, kernelH, this.padding);
        outputShape[widthAxis] =
            deconvLength(outputShape[widthAxis], strideW, kernelW, this.padding);
        return outputShape;
    }
    getConfig() {
        const config = super.getConfig();
        delete config['dilationRate'];
        return config;
    }
}
/** @nocollapse */
Conv3DTranspose.className = 'Conv3DTranspose';
serialization.registerClass(Conv3DTranspose);
export class SeparableConv extends Conv {
    constructor(rank, config) {
        super(rank, config);
        this.DEFAULT_DEPTHWISE_INITIALIZER = 'glorotUniform';
        this.DEFAULT_POINTWISE_INITIALIZER = 'glorotUniform';
        this.depthwiseKernel = null;
        this.pointwiseKernel = null;
        if (config.filters == null) {
            throw new ValueError('The `filters` configuration field is required by SeparableConv, ' +
                'but is unspecified.');
        }
        if (config.kernelInitializer != null || config.kernelRegularizer != null ||
            config.kernelConstraint != null) {
            throw new ValueError('Fields kernelInitializer, kernelRegularizer and kernelConstraint ' +
                'are invalid for SeparableConv2D. Use depthwiseInitializer, ' +
                'depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, ' +
                'pointwiseRegularizer and pointwiseConstraint instead.');
        }
        if (config.padding != null && config.padding !== 'same' &&
            config.padding !== 'valid') {
            throw new ValueError(`SeparableConv${this.rank}D supports only padding modes: ` +
                `'same' and 'valid', but received ${JSON.stringify(config.padding)}`);
        }
        this.depthMultiplier =
            config.depthMultiplier == null ? 1 : config.depthMultiplier;
        this.depthwiseInitializer = getInitializer(config.depthwiseInitializer || this.DEFAULT_DEPTHWISE_INITIALIZER);
        this.depthwiseRegularizer = getRegularizer(config.depthwiseRegularizer);
        this.depthwiseConstraint = getConstraint(config.depthwiseConstraint);
        this.pointwiseInitializer = getInitializer(config.depthwiseInitializer || this.DEFAULT_POINTWISE_INITIALIZER);
        this.pointwiseRegularizer = getRegularizer(config.pointwiseRegularizer);
        this.pointwiseConstraint = getConstraint(config.pointwiseConstraint);
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        if (inputShape.length < this.rank + 2) {
            throw new ValueError(`Inputs to SeparableConv${this.rank}D should have rank ` +
                `${this.rank + 2}, but received input shape: ` +
                `${JSON.stringify(inputShape)}`);
        }
        const channelAxis = this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;
        if (inputShape[channelAxis] == null || inputShape[channelAxis] < 0) {
            throw new ValueError(`The channel dimension of the inputs should be defined, ` +
                `but found ${JSON.stringify(inputShape[channelAxis])}`);
        }
        const inputDim = inputShape[channelAxis];
        const depthwiseKernelShape = this.kernelSize.concat([inputDim, this.depthMultiplier]);
        const pointwiseKernelShape = [];
        for (let i = 0; i < this.rank; ++i) {
            pointwiseKernelShape.push(1);
        }
        pointwiseKernelShape.push(inputDim * this.depthMultiplier, this.filters);
        const trainable = true;
        this.depthwiseKernel = this.addWeight('depthwise_kernel', depthwiseKernelShape, 'float32', this.depthwiseInitializer, this.depthwiseRegularizer, trainable, this.depthwiseConstraint);
        this.pointwiseKernel = this.addWeight('pointwise_kernel', pointwiseKernelShape, 'float32', this.pointwiseInitializer, this.pointwiseRegularizer, trainable, this.pointwiseConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.filters], 'float32', this.biasInitializer, this.biasRegularizer, trainable, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        this.inputSpec =
            [new InputSpec({ ndim: this.rank + 2, axes: { [channelAxis]: inputDim } })];
        this.built = true;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = getExactlyOneTensor(inputs);
            let output;
            if (this.rank === 1) {
                throw new NotImplementedError('1D separable convolution is not implemented yet.');
            }
            else if (this.rank === 2) {
                if (this.dataFormat === 'channelsFirst') {
                    inputs = tfc.transpose(inputs, [0, 2, 3, 1]); // NCHW -> NHWC.
                }
                output = tfc.separableConv2d(inputs, this.depthwiseKernel.read(), this.pointwiseKernel.read(), this.strides, this.padding, this.dilationRate, 'NHWC');
            }
            if (this.useBias) {
                output = K.biasAdd(output, this.bias.read(), this.dataFormat);
            }
            if (this.activation != null) {
                output = this.activation.apply(output);
            }
            if (this.dataFormat === 'channelsFirst') {
                output = tfc.transpose(output, [0, 3, 1, 2]); // NHWC -> NCHW.
            }
            return output;
        });
    }
    getConfig() {
        const config = super.getConfig();
        delete config['rank'];
        delete config['kernelInitializer'];
        delete config['kernelRegularizer'];
        delete config['kernelConstraint'];
        config['depthwiseInitializer'] =
            serializeInitializer(this.depthwiseInitializer);
        config['pointwiseInitializer'] =
            serializeInitializer(this.pointwiseInitializer);
        config['depthwiseRegularizer'] =
            serializeRegularizer(this.depthwiseRegularizer);
        config['pointwiseRegularizer'] =
            serializeRegularizer(this.pointwiseRegularizer);
        config['depthwiseConstraint'] =
            serializeConstraint(this.depthwiseConstraint);
        config['pointwiseConstraint'] =
            serializeConstraint(this.pointwiseConstraint);
        return config;
    }
}
/** @nocollapse */
SeparableConv.className = 'SeparableConv';
export class SeparableConv2D extends SeparableConv {
    constructor(args) {
        super(2, args);
    }
}
/** @nocollapse */
SeparableConv2D.className = 'SeparableConv2D';
serialization.registerClass(SeparableConv2D);
export class Conv1D extends Conv {
    constructor(args) {
        super(1, args);
        Conv1D.verifyArgs(args);
        this.inputSpec = [{ ndim: 3 }];
    }
    getConfig() {
        const config = super.getConfig();
        delete config['rank'];
        delete config['dataFormat'];
        return config;
    }
    static verifyArgs(args) {
        // config.kernelSize must be a number or array of numbers.
        if (typeof args.kernelSize !== 'number' &&
            !generic_utils.checkArrayTypeAndLength(args.kernelSize, 'number', 1, 1)) {
            throw new ValueError(`Conv1D expects config.kernelSize to be number or number[] with ` +
                `length 1, but received ${JSON.stringify(args.kernelSize)}.`);
        }
    }
}
/** @nocollapse */
Conv1D.className = 'Conv1D';
serialization.registerClass(Conv1D);
export class Cropping2D extends Layer {
    constructor(args) {
        super(args);
        if (typeof args.cropping === 'number') {
            this.cropping =
                [[args.cropping, args.cropping], [args.cropping, args.cropping]];
        }
        else if (typeof args.cropping[0] === 'number') {
            this.cropping = [
                [args.cropping[0], args.cropping[0]],
                [args.cropping[1], args.cropping[1]]
            ];
        }
        else {
            this.cropping = args.cropping;
        }
        this.dataFormat =
            args.dataFormat === undefined ? 'channelsLast' : args.dataFormat;
        this.inputSpec = [{ ndim: 4 }];
    }
    computeOutputShape(inputShape) {
        if (this.dataFormat === 'channelsFirst') {
            return [
                inputShape[0], inputShape[1],
                inputShape[2] - this.cropping[0][0] - this.cropping[0][1],
                inputShape[3] - this.cropping[1][0] - this.cropping[1][1]
            ];
        }
        else {
            return [
                inputShape[0],
                inputShape[1] - this.cropping[0][0] - this.cropping[0][1],
                inputShape[2] - this.cropping[1][0] - this.cropping[1][1], inputShape[3]
            ];
        }
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = getExactlyOneTensor(inputs);
            if (this.dataFormat === 'channelsLast') {
                const hSliced = K.sliceAlongAxis(inputs, this.cropping[0][0], inputs.shape[1] - this.cropping[0][0] - this.cropping[0][1], 2);
                return K.sliceAlongAxis(hSliced, this.cropping[1][0], inputs.shape[2] - this.cropping[1][1] - this.cropping[1][0], 3);
            }
            else {
                const hSliced = K.sliceAlongAxis(inputs, this.cropping[0][0], inputs.shape[2] - this.cropping[0][0] - this.cropping[0][1], 3);
                return K.sliceAlongAxis(hSliced, this.cropping[1][0], inputs.shape[3] - this.cropping[1][1] - this.cropping[1][0], 4);
            }
        });
    }
    getConfig() {
        const config = { cropping: this.cropping, dataFormat: this.dataFormat };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
Cropping2D.className = 'Cropping2D';
serialization.registerClass(Cropping2D);
export class UpSampling2D extends Layer {
    constructor(args) {
        super(args);
        this.DEFAULT_SIZE = [2, 2];
        this.inputSpec = [{ ndim: 4 }];
        this.size = args.size == null ? this.DEFAULT_SIZE : args.size;
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        this.interpolation =
            args.interpolation == null ? 'nearest' : args.interpolation;
        checkInterpolationFormat(this.interpolation);
    }
    computeOutputShape(inputShape) {
        if (this.dataFormat === 'channelsFirst') {
            const height = inputShape[2] == null ? null : this.size[0] * inputShape[2];
            const width = inputShape[3] == null ? null : this.size[1] * inputShape[3];
            return [inputShape[0], inputShape[1], height, width];
        }
        else {
            const height = inputShape[1] == null ? null : this.size[0] * inputShape[1];
            const width = inputShape[2] == null ? null : this.size[1] * inputShape[2];
            return [inputShape[0], height, width, inputShape[3]];
        }
    }
    call(inputs, kwargs) {
        return tfc.tidy(() => {
            let input = getExactlyOneTensor(inputs);
            const inputShape = input.shape;
            if (this.dataFormat === 'channelsFirst') {
                input = tfc.transpose(input, [0, 2, 3, 1]);
                const height = this.size[0] * inputShape[2];
                const width = this.size[1] * inputShape[3];
                const resized = this.interpolation === 'nearest' ?
                    tfc.image.resizeNearestNeighbor(input, [height, width]) :
                    tfc.image.resizeBilinear(input, [height, width]);
                return tfc.transpose(resized, [0, 3, 1, 2]);
            }
            else {
                const height = this.size[0] * inputShape[1];
                const width = this.size[1] * inputShape[2];
                return this.interpolation === 'nearest' ?
                    tfc.image.resizeNearestNeighbor(input, [height, width]) :
                    tfc.image.resizeBilinear(input, [height, width]);
            }
        });
    }
    getConfig() {
        const config = { size: this.size, dataFormat: this.dataFormat };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
UpSampling2D.className = 'UpSampling2D';
serialization.registerClass(UpSampling2D);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udm9sdXRpb25hbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvY29udm9sdXRpb25hbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVIOztHQUVHO0FBRUgsT0FBTyxLQUFLLEdBQUcsTUFBTSx1QkFBdUIsQ0FBQztBQUM3QyxPQUFPLEVBQVEsYUFBYSxFQUE0RCxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUUzSCxPQUFPLEVBQWEsYUFBYSxFQUFFLG1CQUFtQixFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDOUUsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQ2xELE9BQU8sS0FBSyxDQUFDLE1BQU0seUJBQXlCLENBQUM7QUFDN0MsT0FBTyxFQUFDLGVBQWUsRUFBRSx3QkFBd0IsRUFBRSxnQkFBZ0IsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUN0RixPQUFPLEVBQW1DLGFBQWEsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQ3BHLE9BQU8sRUFBQyxTQUFTLEVBQUUsS0FBSyxFQUFZLE1BQU0sb0JBQW9CLENBQUM7QUFDL0QsT0FBTyxFQUFDLG1CQUFtQixFQUFFLFVBQVUsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUMxRCxPQUFPLEVBQUMsY0FBYyxFQUFzQyxvQkFBb0IsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBR3pHLE9BQU8sRUFBQyxjQUFjLEVBQXNDLG9CQUFvQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFekcsT0FBTyxFQUFDLGdCQUFnQixFQUFFLFlBQVksRUFBRSxjQUFjLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUNuRixPQUFPLEtBQUssYUFBYSxNQUFNLHdCQUF3QixDQUFDO0FBQ3hELE9BQU8sRUFBQyxrQkFBa0IsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLHNCQUFzQixD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QUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsS0FBSyxFQUFFLENBQUMsTUFBTSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUM7YUFDdEQ7UUFDSCxDQUFDLENBQUMsQ0FBQztJQUNMLENBQUM7SUFFRCxTQUFTO1FBQ1AsTUFBTSxNQUFNLEdBQUcsRUFBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLElBQUksRUFBRSxVQUFVLEVBQUUsSUFBSSxDQUFDLFVBQVUsRUFBQyxDQUFDO1FBQzlELE1BQU0sVUFBVSxHQUFHLEtBQUssQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNyQyxNQUFNLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNsQyxPQUFPLE1BQU0sQ0FBQztJQUNoQixDQUFDOztBQTlERCxrQkFBa0I7QUFDWCxzQkFBUyxHQUFHLGNBQWMsQ0FBQztBQStEcEMsYUFBYSxDQUFDLGFBQWEsQ0FBQyxZQUFZLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogVGVuc29yRmxvdy5qcyBMYXllcnM6IENvbnZvbHV0aW9uYWwgTGF5ZXJzXG4gKi9cblxuaW1wb3J0ICogYXMgdGZjIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2Z1c2VkLCBzZXJpYWxpemF0aW9uLCBUZW5zb3IsIFRlbnNvcjFELCBUZW5zb3IyRCwgVGVuc29yM0QsIFRlbnNvcjRELCBUZW5zb3I1RCwgdGlkeX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtBY3RpdmF0aW9uLCBnZXRBY3RpdmF0aW9uLCBzZXJpYWxpemVBY3RpdmF0aW9ufSBmcm9tICcuLi9hY3RpdmF0aW9ucyc7XG5pbXBvcnQge2ltYWdlRGF0YUZvcm1hdH0gZnJvbSAnLi4vYmFja2VuZC9jb21tb24nO1xuaW1wb3J0ICogYXMgSyBmcm9tICcuLi9iYWNrZW5kL3RmanNfYmFja2VuZCc7XG5pbXBvcnQge2NoZWNrRGF0YUZvcm1hdCwgY2hlY2tJbnRlcnBvbGF0aW9uRm9ybWF0LCBjaGVja1BhZGRpbmdNb2RlfSBmcm9tICcuLi9jb21tb24nO1xuaW1wb3J0IHtDb25zdHJhaW50LCBDb25zdHJhaW50SWRlbnRpZmllciwgZ2V0Q29uc3RyYWludCwgc2VyaWFsaXplQ29uc3RyYWludH0gZnJvbSAnLi4vY29uc3RyYWludHMnO1xuaW1wb3J0IHtJbnB1dFNwZWMsIExheWVyLCBMYXllckFyZ3N9IGZyb20gJy4uL2VuZ2luZS90b3BvbG9neSc7XG5pbXBvcnQge05vdEltcGxlbWVudGVkRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge2dldEluaXRpYWxpemVyLCBJbml0aWFsaXplciwgSW5pdGlhbGl6ZXJJZGVudGlmaWVyLCBzZXJpYWxpemVJbml0aWFsaXplcn0gZnJvbSAnLi4vaW5pdGlhbGl6ZXJzJztcbmltcG9ydCB7QWN0aXZhdGlvbklkZW50aWZpZXJ9IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9hY3RpdmF0aW9uX2NvbmZpZyc7XG5pbXBvcnQge0RhdGFGb3JtYXQsIEludGVycG9sYXRpb25Gb3JtYXQsIFBhZGRpbmdNb2RlLCBTaGFwZX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge2dldFJlZ3VsYXJpemVyLCBSZWd1bGFyaXplciwgUmVndWxhcml6ZXJJZGVudGlmaWVyLCBzZXJpYWxpemVSZWd1bGFyaXplcn0gZnJvbSAnLi4vcmVndWxhcml6ZXJzJztcbmltcG9ydCB7S3dhcmdzfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2NvbnZPdXRwdXRMZW5ndGgsIGRlY29udkxlbmd0aCwgbm9ybWFsaXplQXJyYXl9IGZyb20gJy4uL3V0aWxzL2NvbnZfdXRpbHMnO1xuaW1wb3J0ICogYXMgZ2VuZXJpY191dGlscyBmcm9tICcuLi91dGlscy9nZW5lcmljX3V0aWxzJztcbmltcG9ydCB7Z2V0RXhhY3RseU9uZVNoYXBlLCBnZXRFeGFjdGx5T25lVGVuc29yfSBmcm9tICcuLi91dGlscy90eXBlc191dGlscyc7XG5pbXBvcnQge0xheWVyVmFyaWFibGV9IGZyb20gJy4uL3ZhcmlhYmxlcyc7XG5cbi8qKlxuICogVHJhbnNwb3NlIGFuZCBjYXN0IHRoZSBpbnB1dCBiZWZvcmUgdGhlIGNvbnYyZC5cbiAqIEBwYXJhbSB4IElucHV0IGltYWdlIHRlbnNvci5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVwcm9jZXNzQ29udjJESW5wdXQoXG4gICAgeDogVGVuc29yLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0KTogVGVuc29yIHtcbiAgLy8gVE9ETyhjYWlzKTogQ2FzdCB0eXBlIHRvIGZsb2F0MzIgaWYgbm90LlxuICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIGlmIChkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICAgIHJldHVybiB0ZmMudHJhbnNwb3NlKHgsIFswLCAyLCAzLCAxXSk7ICAvLyBOQ0hXIC0+IE5IV0MuXG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiB4O1xuICAgIH1cbiAgfSk7XG59XG5cbi8qKlxuICogVHJhbnNwb3NlIGFuZCBjYXN0IHRoZSBpbnB1dCBiZWZvcmUgdGhlIGNvbnYzZC5cbiAqIEBwYXJhbSB4IElucHV0IGltYWdlIHRlbnNvci5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVwcm9jZXNzQ29udjNESW5wdXQoXG4gICAgeDogVGVuc29yLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0KTogVGVuc29yIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICByZXR1cm4gdGZjLnRyYW5zcG9zZSh4LCBbMCwgMiwgMywgNCwgMV0pOyAgLy8gTkNESFcgLT4gTkRIV0MuXG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiB4O1xuICAgIH1cbiAgfSk7XG59XG5cbi8qKlxuICogMUQtY29udm9sdXRpb24gd2l0aCBiaWFzIGFkZGVkLlxuICpcbiAqIFBvcnRpbmcgTm90ZTogVGhpcyBmdW5jdGlvbiBkb2VzIG5vdCBleGlzdCBpbiB0aGUgUHl0aG9uIEtlcmFzIGJhY2tlbmQuXG4gKiAgIEl0IGlzIGV4YWN0bHkgdGhlIHNhbWUgYXMgYGNvbnYyZGAsIGV4Y2VwdCB0aGUgYWRkZWQgYGJpYXNgLlxuICpcbiAqIEBwYXJhbSB4IElucHV0IHRlbnNvciwgcmFuay0zLCBvZiBzaGFwZSBgW2JhdGNoU2l6ZSwgd2lkdGgsIGluQ2hhbm5lbHNdYC5cbiAqIEBwYXJhbSBrZXJuZWwgS2VybmVsLCByYW5rLTMsIG9mIHNoYXBlIGBbZmlsdGVyV2lkdGgsIGluRGVwdGgsIG91dERlcHRoXWAuXG4gKiBAcGFyYW0gYmlhcyBCaWFzLCByYW5rLTMsIG9mIHNoYXBlIGBbb3V0RGVwdGhdYC5cbiAqIEBwYXJhbSBzdHJpZGVzXG4gKiBAcGFyYW0gcGFkZGluZyBQYWRkaW5nIG1vZGUuXG4gKiBAcGFyYW0gZGF0YUZvcm1hdCBEYXRhIGZvcm1hdC5cbiAqIEBwYXJhbSBkaWxhdGlvblJhdGVcbiAqIEByZXR1cm5zIFRoZSByZXN1bHQgb2YgdGhlIDFEIGNvbnZvbHV0aW9uLlxuICogQHRocm93cyBWYWx1ZUVycm9yLCBpZiBgeGAsIGBrZXJuZWxgIG9yIGBiaWFzYCBpcyBub3Qgb2YgdGhlIGNvcnJlY3QgcmFuay5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbnYxZFdpdGhCaWFzKFxuICAgIHg6IFRlbnNvciwga2VybmVsOiBUZW5zb3IsIGJpYXM6IFRlbnNvciwgc3RyaWRlcyA9IDEsIHBhZGRpbmcgPSAndmFsaWQnLFxuICAgIGRhdGFGb3JtYXQ/OiBEYXRhRm9ybWF0LCBkaWxhdGlvblJhdGUgPSAxKTogVGVuc29yIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgIGlmIChkYXRhRm9ybWF0ID09IG51bGwpIHtcbiAgICAgIGRhdGFGb3JtYXQgPSBpbWFnZURhdGFGb3JtYXQoKTtcbiAgICB9XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIC8vIENoZWNrIHRoZSByYW5rcyBvZiB4LCBrZXJuZWwgYW5kIGJpYXMuXG4gICAgaWYgKHguc2hhcGUubGVuZ3RoICE9PSAzKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgVGhlIGlucHV0IG9mIGEgY29udjFkV2l0aEJpYXMgb3BlcmF0aW9uIHNob3VsZCBiZSAzLCBidXQgaXMgYCArXG4gICAgICAgICAgYCR7eC5zaGFwZS5sZW5ndGh9IGluc3RlYWQuYCk7XG4gICAgfVxuICAgIGlmIChrZXJuZWwuc2hhcGUubGVuZ3RoICE9PSAzKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgVGhlIGtlcm5lbCBmb3IgYSBjb252MWRXaXRoQmlhcyBvcGVyYXRpb24gc2hvdWxkIGJlIDMsIGJ1dCBpcyBgICtcbiAgICAgICAgICBgJHtrZXJuZWwuc2hhcGUubGVuZ3RofSBpbnN0ZWFkYCk7XG4gICAgfVxuICAgIGlmIChiaWFzICE9IG51bGwgJiYgYmlhcy5zaGFwZS5sZW5ndGggIT09IDEpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBUaGUgYmlhcyBmb3IgYSBjb252MWRXaXRoQmlhcyBvcGVyYXRpb24gc2hvdWxkIGJlIDEsIGJ1dCBpcyBgICtcbiAgICAgICAgICBgJHtrZXJuZWwuc2hhcGUubGVuZ3RofSBpbnN0ZWFkYCk7XG4gICAgfVxuICAgIC8vIFRPRE8oY2Fpcyk6IFN1cHBvcnQgQ0FVU0FMIHBhZGRpbmcgbW9kZS5cbiAgICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICB4ID0gdGZjLnRyYW5zcG9zZSh4LCBbMCwgMiwgMV0pOyAgLy8gTkNXIC0+IE5XQy5cbiAgICB9XG4gICAgaWYgKHBhZGRpbmcgPT09ICdjYXVzYWwnKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAnVGhlIHN1cHBvcnQgZm9yIENBVVNBTCBwYWRkaW5nIG1vZGUgaW4gY29udjFkV2l0aEJpYXMgaXMgbm90ICcgK1xuICAgICAgICAgICdpbXBsZW1lbnRlZCB5ZXQuJyk7XG4gICAgfVxuICAgIGxldCB5OiBUZW5zb3IgPSB0ZmMuY29udjFkKFxuICAgICAgICB4IGFzIFRlbnNvcjJEIHwgVGVuc29yM0QsIGtlcm5lbCBhcyBUZW5zb3IzRCwgc3RyaWRlcyxcbiAgICAgICAgcGFkZGluZyA9PT0gJ3NhbWUnID8gJ3NhbWUnIDogJ3ZhbGlkJywgJ05XQycsIGRpbGF0aW9uUmF0ZSk7XG4gICAgaWYgKGJpYXMgIT0gbnVsbCkge1xuICAgICAgeSA9IEsuYmlhc0FkZCh5LCBiaWFzKTtcbiAgICB9XG4gICAgcmV0dXJuIHk7XG4gIH0pO1xufVxuXG4vKipcbiAqIDFELWNvbnZvbHV0aW9uLlxuICpcbiAqIEBwYXJhbSB4IElucHV0IHRlbnNvciwgcmFuay0zLCBvZiBzaGFwZSBgW2JhdGNoU2l6ZSwgd2lkdGgsIGluQ2hhbm5lbHNdYC5cbiAqIEBwYXJhbSBrZXJuZWwgS2VybmVsLCByYW5rLTMsIG9mIHNoYXBlIGBbZmlsdGVyV2lkdGgsIGluRGVwdGgsIG91dERlcHRoXWAuc1xuICogQHBhcmFtIHN0cmlkZXNcbiAqIEBwYXJhbSBwYWRkaW5nIFBhZGRpbmcgbW9kZS5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0IERhdGEgZm9ybWF0LlxuICogQHBhcmFtIGRpbGF0aW9uUmF0ZVxuICogQHJldHVybnMgVGhlIHJlc3VsdCBvZiB0aGUgMUQgY29udm9sdXRpb24uXG4gKiBAdGhyb3dzIFZhbHVlRXJyb3IsIGlmIGB4YCwgYGtlcm5lbGAgb3IgYGJpYXNgIGlzIG5vdCBvZiB0aGUgY29ycmVjdCByYW5rLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY29udjFkKFxuICAgIHg6IFRlbnNvciwga2VybmVsOiBUZW5zb3IsIHN0cmlkZXMgPSAxLCBwYWRkaW5nID0gJ3ZhbGlkJyxcbiAgICBkYXRhRm9ybWF0PzogRGF0YUZvcm1hdCwgZGlsYXRpb25SYXRlID0gMSk6IFRlbnNvciB7XG4gIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgcmV0dXJuIGNvbnYxZFdpdGhCaWFzKFxuICAgICAgICB4LCBrZXJuZWwsIG51bGwsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsIGRpbGF0aW9uUmF0ZSk7XG4gIH0pO1xufVxuXG4vKipcbiAqIDJEIENvbnZvbHV0aW9uXG4gKiBAcGFyYW0geFxuICogQHBhcmFtIGtlcm5lbCBrZXJuZWwgb2YgdGhlIGNvbnZvbHV0aW9uLlxuICogQHBhcmFtIHN0cmlkZXMgc3RyaWRlcyBhcnJheS5cbiAqIEBwYXJhbSBwYWRkaW5nIHBhZGRpbmcgbW9kZS4gRGVmYXVsdCB0byAndmFsaWQnLlxuICogQHBhcmFtIGRhdGFGb3JtYXQgZGF0YSBmb3JtYXQuIERlZmF1bHRzIHRvICdjaGFubmVsc0xhc3QnLlxuICogQHBhcmFtIGRpbGF0aW9uUmF0ZSBkaWxhdGlvbiByYXRlIGFycmF5LlxuICogQHJldHVybnMgUmVzdWx0IG9mIHRoZSAyRCBwb29saW5nLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY29udjJkKFxuICAgIHg6IFRlbnNvciwga2VybmVsOiBUZW5zb3IsIHN0cmlkZXMgPSBbMSwgMV0sIHBhZGRpbmcgPSAndmFsaWQnLFxuICAgIGRhdGFGb3JtYXQ/OiBEYXRhRm9ybWF0LCBkaWxhdGlvblJhdGU/OiBbbnVtYmVyLCBudW1iZXJdKTogVGVuc29yIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICByZXR1cm4gY29udjJkV2l0aEJpYXNBY3RpdmF0aW9uKFxuICAgICAgICB4LCBrZXJuZWwsIG51bGwsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsIGRpbGF0aW9uUmF0ZSk7XG4gIH0pO1xufVxuXG4vKipcbiAqIDJEIENvbnZvbHV0aW9uIHdpdGggYW4gYWRkZWQgYmlhcyBhbmQgb3B0aW9uYWwgYWN0aXZhdGlvbi5cbiAqIE5vdGU6IFRoaXMgZnVuY3Rpb24gZG9lcyBub3QgZXhpc3QgaW4gdGhlIFB5dGhvbiBLZXJhcyBCYWNrZW5kLiBUaGlzIGZ1bmN0aW9uXG4gKiBpcyBleGFjdGx5IHRoZSBzYW1lIGFzIGBjb252MmRgLCBleGNlcHQgdGhlIGFkZGVkIGBiaWFzYC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbnYyZFdpdGhCaWFzQWN0aXZhdGlvbihcbiAgICB4OiBUZW5zb3IsIGtlcm5lbDogVGVuc29yLCBiaWFzOiBUZW5zb3IsIHN0cmlkZXMgPSBbMSwgMV0sXG4gICAgcGFkZGluZyA9ICd2YWxpZCcsIGRhdGFGb3JtYXQ/OiBEYXRhRm9ybWF0LCBkaWxhdGlvblJhdGU/OiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIGFjdGl2YXRpb246IGZ1c2VkLkFjdGl2YXRpb24gPSBudWxsKTogVGVuc29yIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgIGlmIChkYXRhRm9ybWF0ID09IG51bGwpIHtcbiAgICAgIGRhdGFGb3JtYXQgPSBpbWFnZURhdGFGb3JtYXQoKTtcbiAgICB9XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIGlmICh4LnJhbmsgIT09IDMgJiYgeC5yYW5rICE9PSA0KSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgY29udjJkV2l0aEJpYXNBY3RpdmF0aW9uIGV4cGVjdHMgaW5wdXQgdG8gYmUgb2YgcmFuayAzIG9yIDQsIGAgK1xuICAgICAgICAgIGBidXQgcmVjZWl2ZWQgJHt4LnJhbmt9LmApO1xuICAgIH1cbiAgICBpZiAoa2VybmVsLnJhbmsgIT09IDMgJiYga2VybmVsLnJhbmsgIT09IDQpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBjb252MmRXaXRoQmlhc0FjdGl2YXRpb24gZXhwZWN0cyBrZXJuZWwgdG8gYmUgb2YgcmFuayAzIG9yIDQsIGAgK1xuICAgICAgICAgIGBidXQgcmVjZWl2ZWQgJHt4LnJhbmt9LmApO1xuICAgIH1cbiAgICBsZXQgeSA9IHByZXByb2Nlc3NDb252MkRJbnB1dCh4LCBkYXRhRm9ybWF0KTtcbiAgICBpZiAocGFkZGluZyA9PT0gJ2NhdXNhbCcpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICdUaGUgc3VwcG9ydCBmb3IgQ0FVU0FMIHBhZGRpbmcgbW9kZSBpbiBjb252MWRXaXRoQmlhcyBpcyBub3QgJyArXG4gICAgICAgICAgJ2ltcGxlbWVudGVkIHlldC4nKTtcbiAgICB9XG4gICAgeSA9IHRmYy5mdXNlZC5jb252MmQoe1xuICAgICAgeDogeSBhcyBUZW5zb3IzRCB8IFRlbnNvcjRELFxuICAgICAgZmlsdGVyOiBrZXJuZWwgYXMgVGVuc29yNEQsXG4gICAgICBzdHJpZGVzOiBzdHJpZGVzIGFzIFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBwYWQ6IHBhZGRpbmcgPT09ICdzYW1lJyA/ICdzYW1lJyA6ICd2YWxpZCcsXG4gICAgICBkaWxhdGlvbnM6IGRpbGF0aW9uUmF0ZSxcbiAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgIGJpYXMsXG4gICAgICBhY3RpdmF0aW9uXG4gICAgfSk7XG4gICAgaWYgKGRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgICAgeSA9IHRmYy50cmFuc3Bvc2UoeSwgWzAsIDMsIDEsIDJdKTtcbiAgICB9XG4gICAgcmV0dXJuIHk7XG4gIH0pO1xufVxuXG4vKipcbiAqIDNEIENvbnZvbHV0aW9uLlxuICogQHBhcmFtIHhcbiAqIEBwYXJhbSBrZXJuZWwga2VybmVsIG9mIHRoZSBjb252b2x1dGlvbi5cbiAqIEBwYXJhbSBzdHJpZGVzIHN0cmlkZXMgYXJyYXkuXG4gKiBAcGFyYW0gcGFkZGluZyBwYWRkaW5nIG1vZGUuIERlZmF1bHQgdG8gJ3ZhbGlkJy5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0IGRhdGEgZm9ybWF0LiBEZWZhdWx0cyB0byAnY2hhbm5lbHNMYXN0Jy5cbiAqIEBwYXJhbSBkaWxhdGlvblJhdGUgZGlsYXRpb24gcmF0ZSBhcnJheS5cbiAqIEByZXR1cm5zIFJlc3VsdCBvZiB0aGUgM0QgY29udm9sdXRpb24uXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb252M2QoXG4gICAgeDogVGVuc29yLCBrZXJuZWw6IFRlbnNvciwgc3RyaWRlcyA9IFsxLCAxLCAxXSwgcGFkZGluZyA9ICd2YWxpZCcsXG4gICAgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQsIGRpbGF0aW9uUmF0ZT86IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSk6IFRlbnNvciB7XG4gIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgcmV0dXJuIGNvbnYzZFdpdGhCaWFzKFxuICAgICAgICB4LCBrZXJuZWwsIG51bGwsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsIGRpbGF0aW9uUmF0ZSk7XG4gIH0pO1xufVxuXG4vKipcbiAqIDNEIENvbnZvbHV0aW9uIHdpdGggYW4gYWRkZWQgYmlhcy5cbiAqIE5vdGU6IFRoaXMgZnVuY3Rpb24gZG9lcyBub3QgZXhpc3QgaW4gdGhlIFB5dGhvbiBLZXJhcyBCYWNrZW5kLiBUaGlzIGZ1bmN0aW9uXG4gKiBpcyBleGFjdGx5IHRoZSBzYW1lIGFzIGBjb252M2RgLCBleGNlcHQgdGhlIGFkZGVkIGBiaWFzYC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbnYzZFdpdGhCaWFzKFxuICAgIHg6IFRlbnNvciwga2VybmVsOiBUZW5zb3IsIGJpYXM6IFRlbnNvciwgc3RyaWRlcyA9IFsxLCAxLCAxXSxcbiAgICBwYWRkaW5nID0gJ3ZhbGlkJywgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQsXG4gICAgZGlsYXRpb25SYXRlPzogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTogVGVuc29yIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgIGlmIChkYXRhRm9ybWF0ID09IG51bGwpIHtcbiAgICAgIGRhdGFGb3JtYXQgPSBpbWFnZURhdGFGb3JtYXQoKTtcbiAgICB9XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIGlmICh4LnJhbmsgIT09IDQgJiYgeC5yYW5rICE9PSA1KSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgY29udjNkV2l0aEJpYXMgZXhwZWN0cyBpbnB1dCB0byBiZSBvZiByYW5rIDQgb3IgNSwgYnV0IHJlY2VpdmVkIGAgK1xuICAgICAgICAgIGAke3gucmFua30uYCk7XG4gICAgfVxuICAgIGlmIChrZXJuZWwucmFuayAhPT0gNCAmJiBrZXJuZWwucmFuayAhPT0gNSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYGNvbnYzZFdpdGhCaWFzIGV4cGVjdHMga2VybmVsIHRvIGJlIG9mIHJhbmsgNCBvciA1LCBidXQgcmVjZWl2ZWQgYCArXG4gICAgICAgICAgYCR7eC5yYW5rfS5gKTtcbiAgICB9XG4gICAgbGV0IHkgPSBwcmVwcm9jZXNzQ29udjNESW5wdXQoeCwgZGF0YUZvcm1hdCk7XG4gICAgaWYgKHBhZGRpbmcgPT09ICdjYXVzYWwnKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAnVGhlIHN1cHBvcnQgZm9yIENBVVNBTCBwYWRkaW5nIG1vZGUgaW4gY29udjNkV2l0aEJpYXMgaXMgbm90ICcgK1xuICAgICAgICAgICdpbXBsZW1lbnRlZCB5ZXQuJyk7XG4gICAgfVxuICAgIHkgPSB0ZmMuY29udjNkKFxuICAgICAgICB5IGFzIFRlbnNvcjREIHwgdGZjLlRlbnNvcjx0ZmMuUmFuay5SNT4sXG4gICAgICAgIGtlcm5lbCBhcyB0ZmMuVGVuc29yPHRmYy5SYW5rLlI1Piwgc3RyaWRlcyBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICAgIHBhZGRpbmcgPT09ICdzYW1lJyA/ICdzYW1lJyA6ICd2YWxpZCcsICdOREhXQycsIGRpbGF0aW9uUmF0ZSk7XG4gICAgaWYgKGJpYXMgIT0gbnVsbCkge1xuICAgICAgeSA9IEsuYmlhc0FkZCh5LCBiaWFzIGFzIFRlbnNvcjFEKTtcbiAgICB9XG4gICAgaWYgKGRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgICAgeSA9IHRmYy50cmFuc3Bvc2UoeSwgWzAsIDQsIDEsIDIsIDNdKTtcbiAgICB9XG4gICAgcmV0dXJuIHk7XG4gIH0pO1xufVxuXG4vKipcbiAqIEJhc2UgTGF5ZXJDb25maWcgZm9yIGRlcHRod2lzZSBhbmQgbm9uLWRlcHRod2lzZSBjb252b2x1dGlvbmFsIGxheWVycy5cbiAqL1xuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIEJhc2VDb252TGF5ZXJBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIFRoZSBkaW1lbnNpb25zIG9mIHRoZSBjb252b2x1dGlvbiB3aW5kb3cuIElmIGtlcm5lbFNpemUgaXMgYSBudW1iZXIsIHRoZVxuICAgKiBjb252b2x1dGlvbmFsIHdpbmRvdyB3aWxsIGJlIHNxdWFyZS5cbiAgICovXG4gIGtlcm5lbFNpemU6IG51bWJlcnxudW1iZXJbXTtcblxuICAvKipcbiAgICogVGhlIHN0cmlkZXMgb2YgdGhlIGNvbnZvbHV0aW9uIGluIGVhY2ggZGltZW5zaW9uLiBJZiBzdHJpZGVzIGlzIGEgbnVtYmVyLFxuICAgKiBzdHJpZGVzIGluIGJvdGggZGltZW5zaW9ucyBhcmUgZXF1YWwuXG4gICAqXG4gICAqIFNwZWNpZnlpbmcgYW55IHN0cmlkZSB2YWx1ZSAhPSAxIGlzIGluY29tcGF0aWJsZSB3aXRoIHNwZWNpZnlpbmcgYW55XG4gICAqIGBkaWxhdGlvblJhdGVgIHZhbHVlICE9IDEuXG4gICAqL1xuICBzdHJpZGVzPzogbnVtYmVyfG51bWJlcltdO1xuXG4gIC8qKlxuICAgKiBQYWRkaW5nIG1vZGUuXG4gICAqL1xuICBwYWRkaW5nPzogUGFkZGluZ01vZGU7XG5cbiAgLyoqXG4gICAqIEZvcm1hdCBvZiB0aGUgZGF0YSwgd2hpY2ggZGV0ZXJtaW5lcyB0aGUgb3JkZXJpbmcgb2YgdGhlIGRpbWVuc2lvbnMgaW5cbiAgICogdGhlIGlucHV0cy5cbiAgICpcbiAgICogYGNoYW5uZWxzX2xhc3RgIGNvcnJlc3BvbmRzIHRvIGlucHV0cyB3aXRoIHNoYXBlXG4gICAqICAgYChiYXRjaCwgLi4uLCBjaGFubmVscylgXG4gICAqXG4gICAqICBgY2hhbm5lbHNfZmlyc3RgIGNvcnJlc3BvbmRzIHRvIGlucHV0cyB3aXRoIHNoYXBlIGAoYmF0Y2gsIGNoYW5uZWxzLFxuICAgKiAuLi4pYC5cbiAgICpcbiAgICogRGVmYXVsdHMgdG8gYGNoYW5uZWxzX2xhc3RgLlxuICAgKi9cbiAgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQ7XG5cbiAgLyoqXG4gICAqIFRoZSBkaWxhdGlvbiByYXRlIHRvIHVzZSBmb3IgdGhlIGRpbGF0ZWQgY29udm9sdXRpb24gaW4gZWFjaCBkaW1lbnNpb24uXG4gICAqIFNob3VsZCBiZSBhbiBpbnRlZ2VyIG9yIGFycmF5IG9mIHR3byBvciB0aHJlZSBpbnRlZ2Vycy5cbiAgICpcbiAgICogQ3VycmVudGx5LCBzcGVjaWZ5aW5nIGFueSBgZGlsYXRpb25SYXRlYCB2YWx1ZSAhPSAxIGlzIGluY29tcGF0aWJsZSB3aXRoXG4gICAqIHNwZWNpZnlpbmcgYW55IGBzdHJpZGVzYCB2YWx1ZSAhPSAxLlxuICAgKi9cbiAgZGlsYXRpb25SYXRlPzogbnVtYmVyfFtudW1iZXJdfFtudW1iZXIsIG51bWJlcl18W251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuXG4gIC8qKlxuICAgKiBBY3RpdmF0aW9uIGZ1bmN0aW9uIG9mIHRoZSBsYXllci5cbiAgICpcbiAgICogSWYgeW91IGRvbid0IHNwZWNpZnkgdGhlIGFjdGl2YXRpb24sIG5vbmUgaXMgYXBwbGllZC5cbiAgICovXG4gIGFjdGl2YXRpb24/OiBBY3RpdmF0aW9uSWRlbnRpZmllcjtcblxuICAvKipcbiAgICogV2hldGhlciB0aGUgbGF5ZXIgdXNlcyBhIGJpYXMgdmVjdG9yLiBEZWZhdWx0cyB0byBgdHJ1ZWAuXG4gICAqL1xuICB1c2VCaWFzPzogYm9vbGVhbjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBjb252b2x1dGlvbmFsIGtlcm5lbCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIGtlcm5lbEluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBJbml0aWFsaXplciBmb3IgdGhlIGJpYXMgdmVjdG9yLlxuICAgKi9cbiAgYmlhc0luaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZvciB0aGUgY29udm9sdXRpb25hbCBrZXJuZWwgd2VpZ2h0cy5cbiAgICovXG4gIGtlcm5lbENvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZvciB0aGUgYmlhcyB2ZWN0b3IuXG4gICAqL1xuICBiaWFzQ29uc3RyYWludD86IENvbnN0cmFpbnRJZGVudGlmaWVyfENvbnN0cmFpbnQ7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGtlcm5lbCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIGtlcm5lbFJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJJZGVudGlmaWVyfFJlZ3VsYXJpemVyO1xuXG4gIC8qKlxuICAgKiBSZWd1bGFyaXplciBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIGJpYXNSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcblxuICAvKipcbiAgICogUmVndWxhcml6ZXIgZnVuY3Rpb24gYXBwbGllZCB0byB0aGUgYWN0aXZhdGlvbi5cbiAgICovXG4gIGFjdGl2aXR5UmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG59XG5cbi8qKlxuICogTGF5ZXJDb25maWcgZm9yIG5vbi1kZXB0aHdpc2UgY29udm9sdXRpb25hbCBsYXllcnMuXG4gKiBBcHBsaWVzIHRvIG5vbi1kZXB0aHdpc2UgY29udm9sdXRpb24gb2YgYWxsIHJhbmtzIChlLmcsIENvbnYxRCwgQ29udjJELFxuICogQ29udjNEKS5cbiAqL1xuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIENvbnZMYXllckFyZ3MgZXh0ZW5kcyBCYXNlQ29udkxheWVyQXJncyB7XG4gIC8qKlxuICAgKiBUaGUgZGltZW5zaW9uYWxpdHkgb2YgdGhlIG91dHB1dCBzcGFjZSAoaS5lLiB0aGUgbnVtYmVyIG9mIGZpbHRlcnMgaW4gdGhlXG4gICAqIGNvbnZvbHV0aW9uKS5cbiAgICovXG4gIGZpbHRlcnM6IG51bWJlcjtcbn1cblxuLyoqXG4gKiBBYnN0cmFjdCBjb252b2x1dGlvbiBsYXllci5cbiAqL1xuZXhwb3J0IGFic3RyYWN0IGNsYXNzIEJhc2VDb252IGV4dGVuZHMgTGF5ZXIge1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcmFuazogbnVtYmVyO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkga2VybmVsU2l6ZTogbnVtYmVyW107XG4gIHByb3RlY3RlZCByZWFkb25seSBzdHJpZGVzOiBudW1iZXJbXTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHBhZGRpbmc6IFBhZGRpbmdNb2RlO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgZGF0YUZvcm1hdDogRGF0YUZvcm1hdDtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGFjdGl2YXRpb246IEFjdGl2YXRpb247XG4gIHByb3RlY3RlZCByZWFkb25seSB1c2VCaWFzOiBib29sZWFuO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgZGlsYXRpb25SYXRlOiBudW1iZXJbXTtcblxuICAvLyBCaWFzLXJlbGF0ZWQgbWVtYmVycyBhcmUgaGVyZSBiZWNhdXNlIGFsbCBjb252b2x1dGlvbiBzdWJjbGFzc2VzIHVzZSB0aGVcbiAgLy8gc2FtZSBjb25maWd1cmF0aW9uIHBhcm1ldGVycyB0byBjb250cm9sIGJpYXMuICBLZXJuZWwtcmVsYXRlZCBtZW1iZXJzXG4gIC8vIGFyZSBpbiBzdWJjbGFzcyBgQ29udmAgYmVjYXVzZSBzb21lIHN1YmNsYXNzZXMgdXNlIGRpZmZlcmVudCBwYXJhbWV0ZXJzIHRvXG4gIC8vIGNvbnRyb2wga2VybmVsIHByb3BlcnRpZXMsIGZvciBpbnN0YW5jZSwgYERlcHRod2lzZUNvbnYyRGAgdXNlc1xuICAvLyBgZGVwdGh3aXNlSW5pdGlhbGl6ZXJgIGluc3RlYWQgb2YgYGtlcm5lbEluaXRpYWxpemVyYC5cbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGJpYXNJbml0aWFsaXplcj86IEluaXRpYWxpemVyO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgYmlhc0NvbnN0cmFpbnQ/OiBDb25zdHJhaW50O1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgYmlhc1JlZ3VsYXJpemVyPzogUmVndWxhcml6ZXI7XG5cbiAgcHJvdGVjdGVkIGJpYXM6IExheWVyVmFyaWFibGUgPSBudWxsO1xuXG4gIHJlYWRvbmx5IERFRkFVTFRfS0VSTkVMX0lOSVRJQUxJWkVSOiBJbml0aWFsaXplcklkZW50aWZpZXIgPSAnZ2xvcm90Tm9ybWFsJztcbiAgcmVhZG9ubHkgREVGQVVMVF9CSUFTX0lOSVRJQUxJWkVSOiBJbml0aWFsaXplcklkZW50aWZpZXIgPSAnemVyb3MnO1xuXG4gIGNvbnN0cnVjdG9yKHJhbms6IG51bWJlciwgYXJnczogQmFzZUNvbnZMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzIGFzIExheWVyQXJncyk7XG4gICAgQmFzZUNvbnYudmVyaWZ5QXJncyhhcmdzKTtcbiAgICB0aGlzLnJhbmsgPSByYW5rO1xuICAgIGdlbmVyaWNfdXRpbHMuYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMucmFuaywgJ3JhbmsnKTtcbiAgICBpZiAodGhpcy5yYW5rICE9PSAxICYmIHRoaXMucmFuayAhPT0gMiAmJiB0aGlzLnJhbmsgIT09IDMpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgIGBDb252b2x1dGlvbiBsYXllciBmb3IgcmFuayBvdGhlciB0aGFuIDEsIDIsIG9yIDMgKCR7XG4gICAgICAgICAgICAgIHRoaXMucmFua30pIGlzIGAgK1xuICAgICAgICAgIGBub3QgaW1wbGVtZW50ZWQgeWV0LmApO1xuICAgIH1cbiAgICB0aGlzLmtlcm5lbFNpemUgPSBub3JtYWxpemVBcnJheShhcmdzLmtlcm5lbFNpemUsIHJhbmssICdrZXJuZWxTaXplJyk7XG4gICAgdGhpcy5zdHJpZGVzID0gbm9ybWFsaXplQXJyYXkoXG4gICAgICAgIGFyZ3Muc3RyaWRlcyA9PSBudWxsID8gMSA6IGFyZ3Muc3RyaWRlcywgcmFuaywgJ3N0cmlkZXMnKTtcbiAgICB0aGlzLnBhZGRpbmcgPSBhcmdzLnBhZGRpbmcgPT0gbnVsbCA/ICd2YWxpZCcgOiBhcmdzLnBhZGRpbmc7XG4gICAgY2hlY2tQYWRkaW5nTW9kZSh0aGlzLnBhZGRpbmcpO1xuICAgIHRoaXMuZGF0YUZvcm1hdCA9XG4gICAgICAgIGFyZ3MuZGF0YUZvcm1hdCA9PSBudWxsID8gJ2NoYW5uZWxzTGFzdCcgOiBhcmdzLmRhdGFGb3JtYXQ7XG4gICAgY2hlY2tEYXRhRm9ybWF0KHRoaXMuZGF0YUZvcm1hdCk7XG4gICAgdGhpcy5hY3RpdmF0aW9uID0gZ2V0QWN0aXZhdGlvbihhcmdzLmFjdGl2YXRpb24pO1xuICAgIHRoaXMudXNlQmlhcyA9IGFyZ3MudXNlQmlhcyA9PSBudWxsID8gdHJ1ZSA6IGFyZ3MudXNlQmlhcztcbiAgICB0aGlzLmJpYXNJbml0aWFsaXplciA9XG4gICAgICAgIGdldEluaXRpYWxpemVyKGFyZ3MuYmlhc0luaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9CSUFTX0lOSVRJQUxJWkVSKTtcbiAgICB0aGlzLmJpYXNDb25zdHJhaW50ID0gZ2V0Q29uc3RyYWludChhcmdzLmJpYXNDb25zdHJhaW50KTtcbiAgICB0aGlzLmJpYXNSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MuYmlhc1JlZ3VsYXJpemVyKTtcbiAgICB0aGlzLmFjdGl2aXR5UmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLmFjdGl2aXR5UmVndWxhcml6ZXIpO1xuICAgIHRoaXMuZGlsYXRpb25SYXRlID0gbm9ybWFsaXplQXJyYXkoXG4gICAgICAgIGFyZ3MuZGlsYXRpb25SYXRlID09IG51bGwgPyAxIDogYXJncy5kaWxhdGlvblJhdGUsIHJhbmssXG4gICAgICAgICdkaWxhdGlvblJhdGUnKTtcbiAgICBpZiAodGhpcy5yYW5rID09PSAxICYmXG4gICAgICAgIChBcnJheS5pc0FycmF5KHRoaXMuZGlsYXRpb25SYXRlKSAmJiB0aGlzLmRpbGF0aW9uUmF0ZS5sZW5ndGggIT09IDEpKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgZGlsYXRpb25SYXRlIG11c3QgYmUgYSBudW1iZXIgb3IgYW4gYXJyYXkgb2YgYSBzaW5nbGUgbnVtYmVyIGAgK1xuICAgICAgICAgIGBmb3IgMUQgY29udm9sdXRpb24sIGJ1dCByZWNlaXZlZCBgICtcbiAgICAgICAgICBgJHtKU09OLnN0cmluZ2lmeSh0aGlzLmRpbGF0aW9uUmF0ZSl9YCk7XG4gICAgfSBlbHNlIGlmICh0aGlzLnJhbmsgPT09IDIpIHtcbiAgICAgIGlmICh0eXBlb2YgdGhpcy5kaWxhdGlvblJhdGUgPT09ICdudW1iZXInKSB7XG4gICAgICAgIHRoaXMuZGlsYXRpb25SYXRlID0gW3RoaXMuZGlsYXRpb25SYXRlLCB0aGlzLmRpbGF0aW9uUmF0ZV07XG4gICAgICB9IGVsc2UgaWYgKHRoaXMuZGlsYXRpb25SYXRlLmxlbmd0aCAhPT0gMikge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBkaWxhdGlvblJhdGUgbXVzdCBiZSBhIG51bWJlciBvciBhcnJheSBvZiB0d28gbnVtYmVycyBmb3IgMkQgYCArXG4gICAgICAgICAgICBgY29udm9sdXRpb24sIGJ1dCByZWNlaXZlZCAke0pTT04uc3RyaW5naWZ5KHRoaXMuZGlsYXRpb25SYXRlKX1gKTtcbiAgICAgIH1cbiAgICB9IGVsc2UgaWYgKHRoaXMucmFuayA9PT0gMykge1xuICAgICAgaWYgKHR5cGVvZiB0aGlzLmRpbGF0aW9uUmF0ZSA9PT0gJ251bWJlcicpIHtcbiAgICAgICAgdGhpcy5kaWxhdGlvblJhdGUgPVxuICAgICAgICAgICAgW3RoaXMuZGlsYXRpb25SYXRlLCB0aGlzLmRpbGF0aW9uUmF0ZSwgdGhpcy5kaWxhdGlvblJhdGVdO1xuICAgICAgfSBlbHNlIGlmICh0aGlzLmRpbGF0aW9uUmF0ZS5sZW5ndGggIT09IDMpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgZGlsYXRpb25SYXRlIG11c3QgYmUgYSBudW1iZXIgb3IgYXJyYXkgb2YgdGhyZWUgbnVtYmVycyBmb3IgM0QgYCArXG4gICAgICAgICAgICBgY29udm9sdXRpb24sIGJ1dCByZWNlaXZlZCAke0pTT04uc3RyaW5naWZ5KHRoaXMuZGlsYXRpb25SYXRlKX1gKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBwcm90ZWN0ZWQgc3RhdGljIHZlcmlmeUFyZ3MoYXJnczogQmFzZUNvbnZMYXllckFyZ3MpIHtcbiAgICAvLyBDaGVjayBjb25maWcua2VybmVsU2l6ZSB0eXBlIGFuZCBzaGFwZS5cbiAgICBnZW5lcmljX3V0aWxzLmFzc2VydChcbiAgICAgICAgJ2tlcm5lbFNpemUnIGluIGFyZ3MsIGByZXF1aXJlZCBrZXkgJ2tlcm5lbFNpemUnIG5vdCBpbiBjb25maWdgKTtcbiAgICBpZiAodHlwZW9mIGFyZ3Mua2VybmVsU2l6ZSAhPT0gJ251bWJlcicgJiZcbiAgICAgICAgIWdlbmVyaWNfdXRpbHMuY2hlY2tBcnJheVR5cGVBbmRMZW5ndGgoXG4gICAgICAgICAgICBhcmdzLmtlcm5lbFNpemUsICdudW1iZXInLCAxLCAzKSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYEJhc2VDb252IGV4cGVjdHMgY29uZmlnLmtlcm5lbFNpemUgdG8gYmUgbnVtYmVyIG9yIG51bWJlcltdIHdpdGggYCArXG4gICAgICAgICAgYGxlbmd0aCAxLCAyLCBvciAzLCBidXQgcmVjZWl2ZWQgJHtcbiAgICAgICAgICAgICAgSlNPTi5zdHJpbmdpZnkoYXJncy5rZXJuZWxTaXplKX0uYCk7XG4gICAgfVxuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7XG4gICAgICBrZXJuZWxTaXplOiB0aGlzLmtlcm5lbFNpemUsXG4gICAgICBzdHJpZGVzOiB0aGlzLnN0cmlkZXMsXG4gICAgICBwYWRkaW5nOiB0aGlzLnBhZGRpbmcsXG4gICAgICBkYXRhRm9ybWF0OiB0aGlzLmRhdGFGb3JtYXQsXG4gICAgICBkaWxhdGlvblJhdGU6IHRoaXMuZGlsYXRpb25SYXRlLFxuICAgICAgYWN0aXZhdGlvbjogc2VyaWFsaXplQWN0aXZhdGlvbih0aGlzLmFjdGl2YXRpb24pLFxuICAgICAgdXNlQmlhczogdGhpcy51c2VCaWFzLFxuICAgICAgYmlhc0luaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmJpYXNJbml0aWFsaXplciksXG4gICAgICBiaWFzUmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMuYmlhc1JlZ3VsYXJpemVyKSxcbiAgICAgIGFjdGl2aXR5UmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMuYWN0aXZpdHlSZWd1bGFyaXplciksXG4gICAgICBiaWFzQ29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLmJpYXNDb25zdHJhaW50KVxuICAgIH07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5cbi8qKlxuICogQWJzdHJhY3QgbkQgY29udm9sdXRpb24gbGF5ZXIuICBBbmNlc3RvciBvZiBjb252b2x1dGlvbiBsYXllcnMgd2hpY2ggcmVkdWNlXG4gKiBhY3Jvc3MgY2hhbm5lbHMsIGkuZS4sIENvbnYxRCBhbmQgQ29udjJELCBidXQgbm90IERlcHRod2lzZUNvbnYyRC5cbiAqL1xuZXhwb3J0IGFic3RyYWN0IGNsYXNzIENvbnYgZXh0ZW5kcyBCYXNlQ29udiB7XG4gIHByb3RlY3RlZCByZWFkb25seSBmaWx0ZXJzOiBudW1iZXI7XG5cbiAgcHJvdGVjdGVkIGtlcm5lbDogTGF5ZXJWYXJpYWJsZSA9IG51bGw7XG5cbiAgLy8gQmlhcy1yZWxhdGVkIHByb3BlcnRpZXMgYXJlIHN0b3JlZCBpbiB0aGUgc3VwZXJjbGFzcyBgQmFzZUNvbnZgIGJlY2F1c2UgYWxsXG4gIC8vIGNvbnZvbHV0aW9uIHN1YmNsYXNzZXMgdXNlIHRoZSBzYW1lIGNvbmZpZ3VyYXRpb24gcGFyYW1ldGVycyB0byBjb250cm9sXG4gIC8vIGJpYXMuIEtlcm5lbC1yZWxhdGVkIHByb3BlcnRpZXMgYXJlIGRlZmluZWQgaGVyZSByYXRoZXIgdGhhbiBpbiB0aGVcbiAgLy8gc3VwZXJjbGFzcyBiZWNhdXNlIHNvbWUgY29udm9sdXRpb24gc3ViY2xhc3NlcyB1c2UgZGlmZmVyZW50IG5hbWVzIGFuZFxuICAvLyBjb25maWd1cmF0aW9uIHBhcmFtZXRlcnMgZm9yIHRoZWlyIGludGVybmFsIGtlcm5lbCBzdGF0ZS5cbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGtlcm5lbEluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXI7XG4gIHByb3RlY3RlZCByZWFkb25seSBrZXJuZWxDb25zdHJhaW50PzogQ29uc3RyYWludDtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGtlcm5lbFJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXI7XG5cbiAgY29uc3RydWN0b3IocmFuazogbnVtYmVyLCBhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIocmFuaywgYXJncyBhcyBCYXNlQ29udkxheWVyQXJncyk7XG4gICAgQ29udi52ZXJpZnlBcmdzKGFyZ3MpO1xuICAgIHRoaXMuZmlsdGVycyA9IGFyZ3MuZmlsdGVycztcbiAgICBnZW5lcmljX3V0aWxzLmFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLmZpbHRlcnMsICdmaWx0ZXJzJyk7XG4gICAgdGhpcy5rZXJuZWxJbml0aWFsaXplciA9IGdldEluaXRpYWxpemVyKFxuICAgICAgICBhcmdzLmtlcm5lbEluaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9LRVJORUxfSU5JVElBTElaRVIpO1xuICAgIHRoaXMua2VybmVsQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5rZXJuZWxDb25zdHJhaW50KTtcbiAgICB0aGlzLmtlcm5lbFJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5rZXJuZWxSZWd1bGFyaXplcik7XG4gIH1cblxuICBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBjaGFubmVsQXhpcyA9XG4gICAgICAgIHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnID8gMSA6IGlucHV0U2hhcGUubGVuZ3RoIC0gMTtcbiAgICBpZiAoaW5wdXRTaGFwZVtjaGFubmVsQXhpc10gPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYFRoZSBjaGFubmVsIGRpbWVuc2lvbiBvZiB0aGUgaW5wdXQgc2hvdWxkIGJlIGRlZmluZWQuIGAgK1xuICAgICAgICAgIGBGb3VuZCAke2lucHV0U2hhcGVbY2hhbm5lbEF4aXNdfWApO1xuICAgIH1cbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGVbY2hhbm5lbEF4aXNdO1xuXG4gICAgY29uc3Qga2VybmVsU2hhcGUgPSB0aGlzLmtlcm5lbFNpemUuY29uY2F0KFtpbnB1dERpbSwgdGhpcy5maWx0ZXJzXSk7XG5cbiAgICB0aGlzLmtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAna2VybmVsJywga2VybmVsU2hhcGUsIG51bGwsIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIsXG4gICAgICAgIHRoaXMua2VybmVsUmVndWxhcml6ZXIsIHRydWUsIHRoaXMua2VybmVsQ29uc3RyYWludCk7XG4gICAgaWYgKHRoaXMudXNlQmlhcykge1xuICAgICAgdGhpcy5iaWFzID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2JpYXMnLCBbdGhpcy5maWx0ZXJzXSwgbnVsbCwgdGhpcy5iaWFzSW5pdGlhbGl6ZXIsXG4gICAgICAgICAgdGhpcy5iaWFzUmVndWxhcml6ZXIsIHRydWUsIHRoaXMuYmlhc0NvbnN0cmFpbnQpO1xuICAgIH1cblxuICAgIHRoaXMuaW5wdXRTcGVjID0gW3tuZGltOiB0aGlzLnJhbmsgKyAyLCBheGVzOiB7W2NoYW5uZWxBeGlzXTogaW5wdXREaW19fV07XG4gICAgdGhpcy5idWlsdCA9IHRydWU7XG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaW5wdXRzID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgICAgbGV0IG91dHB1dHM6IFRlbnNvcjtcbiAgICAgIGNvbnN0IGJpYXNWYWx1ZSA9IHRoaXMuYmlhcyA9PSBudWxsID8gbnVsbCA6IHRoaXMuYmlhcy5yZWFkKCk7XG4gICAgICBjb25zdCBmdXNlZEFjdGl2YXRpb25OYW1lID0gZ2VuZXJpY191dGlscy5tYXBBY3RpdmF0aW9uVG9GdXNlZEtlcm5lbChcbiAgICAgICAgICB0aGlzLmFjdGl2YXRpb24uZ2V0Q2xhc3NOYW1lKCkpO1xuXG4gICAgICBpZiAoZnVzZWRBY3RpdmF0aW9uTmFtZSAhPSBudWxsICYmIHRoaXMucmFuayA9PT0gMikge1xuICAgICAgICBvdXRwdXRzID0gY29udjJkV2l0aEJpYXNBY3RpdmF0aW9uKFxuICAgICAgICAgICAgaW5wdXRzLCB0aGlzLmtlcm5lbC5yZWFkKCksIGJpYXNWYWx1ZSwgdGhpcy5zdHJpZGVzLCB0aGlzLnBhZGRpbmcsXG4gICAgICAgICAgICB0aGlzLmRhdGFGb3JtYXQsIHRoaXMuZGlsYXRpb25SYXRlIGFzIFtudW1iZXIsIG51bWJlcl0sXG4gICAgICAgICAgICBmdXNlZEFjdGl2YXRpb25OYW1lKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGlmICh0aGlzLnJhbmsgPT09IDEpIHtcbiAgICAgICAgICBvdXRwdXRzID0gY29udjFkV2l0aEJpYXMoXG4gICAgICAgICAgICAgIGlucHV0cywgdGhpcy5rZXJuZWwucmVhZCgpLCBiaWFzVmFsdWUsIHRoaXMuc3RyaWRlc1swXSxcbiAgICAgICAgICAgICAgdGhpcy5wYWRkaW5nLCB0aGlzLmRhdGFGb3JtYXQsIHRoaXMuZGlsYXRpb25SYXRlWzBdKTtcbiAgICAgICAgfSBlbHNlIGlmICh0aGlzLnJhbmsgPT09IDIpIHtcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBNb3ZlIHVwIHRvIGNvbnN0cnVjdG9yLlxuICAgICAgICAgIG91dHB1dHMgPSBjb252MmRXaXRoQmlhc0FjdGl2YXRpb24oXG4gICAgICAgICAgICAgIGlucHV0cywgdGhpcy5rZXJuZWwucmVhZCgpLCBiaWFzVmFsdWUsIHRoaXMuc3RyaWRlcywgdGhpcy5wYWRkaW5nLFxuICAgICAgICAgICAgICB0aGlzLmRhdGFGb3JtYXQsIHRoaXMuZGlsYXRpb25SYXRlIGFzIFtudW1iZXIsIG51bWJlcl0pO1xuICAgICAgICB9IGVsc2UgaWYgKHRoaXMucmFuayA9PT0gMykge1xuICAgICAgICAgIG91dHB1dHMgPSBjb252M2RXaXRoQmlhcyhcbiAgICAgICAgICAgICAgaW5wdXRzLCB0aGlzLmtlcm5lbC5yZWFkKCksIGJpYXNWYWx1ZSwgdGhpcy5zdHJpZGVzLCB0aGlzLnBhZGRpbmcsXG4gICAgICAgICAgICAgIHRoaXMuZGF0YUZvcm1hdCwgdGhpcy5kaWxhdGlvblJhdGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAgICAgJ2NvbnZvbHV0aW9ucyBncmVhdGVyIHRoYW4gM0QgYXJlIG5vdCBpbXBsZW1lbnRlZCB5ZXQuJyk7XG4gICAgICAgIH1cblxuICAgICAgICBpZiAodGhpcy5hY3RpdmF0aW9uICE9IG51bGwpIHtcbiAgICAgICAgICBvdXRwdXRzID0gdGhpcy5hY3RpdmF0aW9uLmFwcGx5KG91dHB1dHMpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIHJldHVybiBvdXRwdXRzO1xuICAgIH0pO1xuICB9XG5cbiAgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IG5ld1NwYWNlOiBudW1iZXJbXSA9IFtdO1xuICAgIGNvbnN0IHNwYWNlID0gKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpID9cbiAgICAgICAgaW5wdXRTaGFwZS5zbGljZSgxLCBpbnB1dFNoYXBlLmxlbmd0aCAtIDEpIDpcbiAgICAgICAgaW5wdXRTaGFwZS5zbGljZSgyKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNwYWNlLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCBuZXdEaW0gPSBjb252T3V0cHV0TGVuZ3RoKFxuICAgICAgICAgIHNwYWNlW2ldLCB0aGlzLmtlcm5lbFNpemVbaV0sIHRoaXMucGFkZGluZywgdGhpcy5zdHJpZGVzW2ldLFxuICAgICAgICAgIHR5cGVvZiB0aGlzLmRpbGF0aW9uUmF0ZSA9PT0gJ251bWJlcicgPyB0aGlzLmRpbGF0aW9uUmF0ZSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuZGlsYXRpb25SYXRlW2ldKTtcbiAgICAgIG5ld1NwYWNlLnB1c2gobmV3RGltKTtcbiAgICB9XG5cbiAgICBsZXQgb3V0cHV0U2hhcGUgPSBbaW5wdXRTaGFwZVswXV07XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICAgIG91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGUuY29uY2F0KG5ld1NwYWNlKTtcbiAgICAgIG91dHB1dFNoYXBlLnB1c2godGhpcy5maWx0ZXJzKTtcbiAgICB9IGVsc2Uge1xuICAgICAgb3V0cHV0U2hhcGUucHVzaCh0aGlzLmZpbHRlcnMpO1xuICAgICAgb3V0cHV0U2hhcGUgPSBvdXRwdXRTaGFwZS5jb25jYXQobmV3U3BhY2UpO1xuICAgIH1cbiAgICByZXR1cm4gb3V0cHV0U2hhcGU7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSB7XG4gICAgICBmaWx0ZXJzOiB0aGlzLmZpbHRlcnMsXG4gICAgICBrZXJuZWxJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5rZXJuZWxJbml0aWFsaXplciksXG4gICAgICBrZXJuZWxSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5rZXJuZWxSZWd1bGFyaXplciksXG4gICAgICBrZXJuZWxDb25zdHJhaW50OiBzZXJpYWxpemVDb25zdHJhaW50KHRoaXMua2VybmVsQ29uc3RyYWludClcbiAgICB9O1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxuXG4gIHByb3RlY3RlZCBzdGF0aWMgdmVyaWZ5QXJncyhhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gICAgLy8gQ2hlY2sgY29uZmlnLmZpbHRlcnMgdHlwZSwgc2hhcGUsIGFuZCB2YWx1ZS5cbiAgICBpZiAoISgnZmlsdGVycycgaW4gYXJncykgfHwgdHlwZW9mIGFyZ3MuZmlsdGVycyAhPT0gJ251bWJlcicgfHxcbiAgICAgICAgYXJncy5maWx0ZXJzIDwgMSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYENvbnZvbHV0aW9uIGxheWVyIGV4cGVjdGVkIGNvbmZpZy5maWx0ZXJzIHRvIGJlIGEgJ251bWJlcicgPiAwIGAgK1xuICAgICAgICAgIGBidXQgZ290ICR7SlNPTi5zdHJpbmdpZnkoYXJncy5maWx0ZXJzKX1gKTtcbiAgICB9XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIENvbnYyRCBleHRlbmRzIENvbnYge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdDb252MkQnO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoMiwgYXJncyk7XG4gICAgQ29udjJELnZlcmlmeUFyZ3MoYXJncyk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBkZWxldGUgY29uZmlnWydyYW5rJ107XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxuXG4gIHByb3RlY3RlZCBzdGF0aWMgdmVyaWZ5QXJncyhhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gICAgLy8gY29uZmlnLmtlcm5lbFNpemUgbXVzdCBiZSBhIG51bWJlciBvciBhcnJheSBvZiBudW1iZXJzLlxuICAgIGlmICgodHlwZW9mIGFyZ3Mua2VybmVsU2l6ZSAhPT0gJ251bWJlcicpICYmXG4gICAgICAgICFnZW5lcmljX3V0aWxzLmNoZWNrQXJyYXlUeXBlQW5kTGVuZ3RoKFxuICAgICAgICAgICAgYXJncy5rZXJuZWxTaXplLCAnbnVtYmVyJywgMSwgMikpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBDb252MkQgZXhwZWN0cyBjb25maWcua2VybmVsU2l6ZSB0byBiZSBudW1iZXIgb3IgbnVtYmVyW10gd2l0aCBgICtcbiAgICAgICAgICBgbGVuZ3RoIDEgb3IgMiwgYnV0IHJlY2VpdmVkICR7SlNPTi5zdHJpbmdpZnkoYXJncy5rZXJuZWxTaXplKX0uYCk7XG4gICAgfVxuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQ29udjJEKTtcblxuZXhwb3J0IGNsYXNzIENvbnYzRCBleHRlbmRzIENvbnYge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdDb252M0QnO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoMywgYXJncyk7XG4gICAgQ29udjNELnZlcmlmeUFyZ3MoYXJncyk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBkZWxldGUgY29uZmlnWydyYW5rJ107XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxuXG4gIHByb3RlY3RlZCBzdGF0aWMgdmVyaWZ5QXJncyhhcmdzOiBDb252TGF5ZXJBcmdzKSB7XG4gICAgLy8gY29uZmlnLmtlcm5lbFNpemUgbXVzdCBiZSBhIG51bWJlciBvciBhcnJheSBvZiBudW1iZXJzLlxuICAgIGlmICh0eXBlb2YgYXJncy5rZXJuZWxTaXplICE9PSAnbnVtYmVyJykge1xuICAgICAgaWYgKCEoQXJyYXkuaXNBcnJheShhcmdzLmtlcm5lbFNpemUpICYmXG4gICAgICAgICAgICAoYXJncy5rZXJuZWxTaXplLmxlbmd0aCA9PT0gMSB8fCBhcmdzLmtlcm5lbFNpemUubGVuZ3RoID09PSAzKSkpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgQ29udjNEIGV4cGVjdHMgY29uZmlnLmtlcm5lbFNpemUgdG8gYmUgbnVtYmVyIG9yYCArXG4gICAgICAgICAgICBgIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgYnV0IHJlY2VpdmVkICR7XG4gICAgICAgICAgICAgICAgSlNPTi5zdHJpbmdpZnkoYXJncy5rZXJuZWxTaXplKX0uYCk7XG4gICAgICB9XG4gICAgfVxuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQ29udjNEKTtcblxuZXhwb3J0IGNsYXNzIENvbnYyRFRyYW5zcG9zZSBleHRlbmRzIENvbnYyRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0NvbnYyRFRyYW5zcG9zZSc7XG4gIGlucHV0U3BlYzogSW5wdXRTcGVjW107XG5cbiAgY29uc3RydWN0b3IoYXJnczogQ29udkxheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDR9KV07XG5cbiAgICBpZiAodGhpcy5wYWRkaW5nICE9PSAnc2FtZScgJiYgdGhpcy5wYWRkaW5nICE9PSAndmFsaWQnKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgQ29udjJEVHJhbnNwb3NlIGN1cnJlbnRseSBzdXBwb3J0cyBvbmx5IHBhZGRpbmcgbW9kZXMgJ3NhbWUnIGAgK1xuICAgICAgICAgIGBhbmQgJ3ZhbGlkJywgYnV0IHJlY2VpdmVkIHBhZGRpbmcgbW9kZSAke3RoaXMucGFkZGluZ31gKTtcbiAgICB9XG4gIH1cblxuICBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcblxuICAgIGlmIChpbnB1dFNoYXBlLmxlbmd0aCAhPT0gNCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ0lucHV0IHNob3VsZCBoYXZlIHJhbmsgNDsgUmVjZWl2ZWQgaW5wdXQgc2hhcGU6ICcgK1xuICAgICAgICAgIEpTT04uc3RyaW5naWZ5KGlucHV0U2hhcGUpKTtcbiAgICB9XG5cbiAgICBjb25zdCBjaGFubmVsQXhpcyA9XG4gICAgICAgIHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnID8gMSA6IGlucHV0U2hhcGUubGVuZ3RoIC0gMTtcbiAgICBpZiAoaW5wdXRTaGFwZVtjaGFubmVsQXhpc10gPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ1RoZSBjaGFubmVsIGRpbWVuc2lvbiBvZiB0aGUgaW5wdXRzIHNob3VsZCBiZSBkZWZpbmVkLiAnICtcbiAgICAgICAgICAnRm91bmQgYE5vbmVgLicpO1xuICAgIH1cbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGVbY2hhbm5lbEF4aXNdO1xuICAgIGNvbnN0IGtlcm5lbFNoYXBlID0gdGhpcy5rZXJuZWxTaXplLmNvbmNhdChbdGhpcy5maWx0ZXJzLCBpbnB1dERpbV0pO1xuXG4gICAgdGhpcy5rZXJuZWwgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ2tlcm5lbCcsIGtlcm5lbFNoYXBlLCAnZmxvYXQzMicsIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIsXG4gICAgICAgIHRoaXMua2VybmVsUmVndWxhcml6ZXIsIHRydWUsIHRoaXMua2VybmVsQ29uc3RyYWludCk7XG4gICAgaWYgKHRoaXMudXNlQmlhcykge1xuICAgICAgdGhpcy5iaWFzID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2JpYXMnLCBbdGhpcy5maWx0ZXJzXSwgJ2Zsb2F0MzInLCB0aGlzLmJpYXNJbml0aWFsaXplcixcbiAgICAgICAgICB0aGlzLmJpYXNSZWd1bGFyaXplciwgdHJ1ZSwgdGhpcy5iaWFzQ29uc3RyYWludCk7XG4gICAgfVxuXG4gICAgLy8gU2V0IGlucHV0IHNwZWMuXG4gICAgdGhpcy5pbnB1dFNwZWMgPVxuICAgICAgICBbbmV3IElucHV0U3BlYyh7bmRpbTogNCwgYXhlczoge1tjaGFubmVsQXhpc106IGlucHV0RGltfX0pXTtcbiAgICB0aGlzLmJ1aWx0ID0gdHJ1ZTtcbiAgfVxuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGZjLnRpZHkoKCkgPT4ge1xuICAgICAgbGV0IGlucHV0ID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgICAgaWYgKGlucHV0LnNoYXBlLmxlbmd0aCAhPT0gNCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBDb252MkRUcmFuc3Bvc2UuY2FsbCgpIGV4cGVjdHMgaW5wdXQgdGVuc29yIHRvIGJlIHJhbmstNCwgYnV0IGAgK1xuICAgICAgICAgICAgYHJlY2VpdmVkIGEgdGVuc29yIG9mIHJhbmstJHtpbnB1dC5zaGFwZS5sZW5ndGh9YCk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IGlucHV0U2hhcGUgPSBpbnB1dC5zaGFwZTtcbiAgICAgIGNvbnN0IGJhdGNoU2l6ZSA9IGlucHV0U2hhcGVbMF07XG5cbiAgICAgIGxldCBoQXhpczogbnVtYmVyO1xuICAgICAgbGV0IHdBeGlzOiBudW1iZXI7XG4gICAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICAgICAgaEF4aXMgPSAyO1xuICAgICAgICB3QXhpcyA9IDM7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBoQXhpcyA9IDE7XG4gICAgICAgIHdBeGlzID0gMjtcbiAgICAgIH1cblxuICAgICAgY29uc3QgaGVpZ2h0ID0gaW5wdXRTaGFwZVtoQXhpc107XG4gICAgICBjb25zdCB3aWR0aCA9IGlucHV0U2hhcGVbd0F4aXNdO1xuICAgICAgY29uc3Qga2VybmVsSCA9IHRoaXMua2VybmVsU2l6ZVswXTtcbiAgICAgIGNvbnN0IGtlcm5lbFcgPSB0aGlzLmtlcm5lbFNpemVbMV07XG4gICAgICBjb25zdCBzdHJpZGVIID0gdGhpcy5zdHJpZGVzWzBdO1xuICAgICAgY29uc3Qgc3RyaWRlVyA9IHRoaXMuc3RyaWRlc1sxXTtcblxuICAgICAgLy8gSW5mZXIgdGhlIGR5bmFtaWMgb3V0cHV0IHNoYXBlLlxuICAgICAgY29uc3Qgb3V0SGVpZ2h0ID0gZGVjb252TGVuZ3RoKGhlaWdodCwgc3RyaWRlSCwga2VybmVsSCwgdGhpcy5wYWRkaW5nKTtcbiAgICAgIGNvbnN0IG91dFdpZHRoID0gZGVjb252TGVuZ3RoKHdpZHRoLCBzdHJpZGVXLCBrZXJuZWxXLCB0aGlzLnBhZGRpbmcpO1xuXG4gICAgICAvLyBQb3J0aW5nIE5vdGU6IFdlIGRvbid0IGJyYW5jaCBiYXNlZCBvbiBgdGhpcy5kYXRhRm9ybWF0YCBoZXJlLFxuICAgICAgLy8gYmVjYXVzZVxuICAgICAgLy8gICB0aGUgdGpmcy1jb3JlIGZ1bmN0aW9uIGBjb252MmRUcmFuc3Bvc2VgIGNhbGxlZCBiZWxvdyBhbHdheXNcbiAgICAgIC8vICAgYXNzdW1lcyBjaGFubmVsc0xhc3QuXG4gICAgICBjb25zdCBvdXRwdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICAgIFtiYXRjaFNpemUsIG91dEhlaWdodCwgb3V0V2lkdGgsIHRoaXMuZmlsdGVyc107XG5cbiAgICAgIGlmICh0aGlzLmRhdGFGb3JtYXQgIT09ICdjaGFubmVsc0xhc3QnKSB7XG4gICAgICAgIGlucHV0ID0gdGZjLnRyYW5zcG9zZShpbnB1dCwgWzAsIDIsIDMsIDFdKTtcbiAgICAgIH1cbiAgICAgIGxldCBvdXRwdXRzID0gdGZjLmNvbnYyZFRyYW5zcG9zZShcbiAgICAgICAgICBpbnB1dCBhcyBUZW5zb3I0RCwgdGhpcy5rZXJuZWwucmVhZCgpIGFzIFRlbnNvcjRELCBvdXRwdXRTaGFwZSxcbiAgICAgICAgICB0aGlzLnN0cmlkZXMgYXMgW251bWJlciwgbnVtYmVyXSwgdGhpcy5wYWRkaW5nIGFzICdzYW1lJyB8ICd2YWxpZCcpO1xuICAgICAgaWYgKHRoaXMuZGF0YUZvcm1hdCAhPT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICAgICAgb3V0cHV0cyA9IHRmYy50cmFuc3Bvc2Uob3V0cHV0cywgWzAsIDMsIDEsIDJdKTtcbiAgICAgIH1cblxuICAgICAgaWYgKHRoaXMuYmlhcyAhPSBudWxsKSB7XG4gICAgICAgIG91dHB1dHMgPVxuICAgICAgICAgICAgSy5iaWFzQWRkKG91dHB1dHMsIHRoaXMuYmlhcy5yZWFkKCksIHRoaXMuZGF0YUZvcm1hdCkgYXMgVGVuc29yNEQ7XG4gICAgICB9XG4gICAgICBpZiAodGhpcy5hY3RpdmF0aW9uICE9IG51bGwpIHtcbiAgICAgICAgb3V0cHV0cyA9IHRoaXMuYWN0aXZhdGlvbi5hcHBseShvdXRwdXRzKSBhcyBUZW5zb3I0RDtcbiAgICAgIH1cbiAgICAgIHJldHVybiBvdXRwdXRzO1xuICAgIH0pO1xuICB9XG5cbiAgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gaW5wdXRTaGFwZS5zbGljZSgpO1xuXG4gICAgbGV0IGNoYW5uZWxBeGlzOiBudW1iZXI7XG4gICAgbGV0IGhlaWdodEF4aXM6IG51bWJlcjtcbiAgICBsZXQgd2lkdGhBeGlzOiBudW1iZXI7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICBjaGFubmVsQXhpcyA9IDE7XG4gICAgICBoZWlnaHRBeGlzID0gMjtcbiAgICAgIHdpZHRoQXhpcyA9IDM7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNoYW5uZWxBeGlzID0gMztcbiAgICAgIGhlaWdodEF4aXMgPSAxO1xuICAgICAgd2lkdGhBeGlzID0gMjtcbiAgICB9XG5cbiAgICBjb25zdCBrZXJuZWxIID0gdGhpcy5rZXJuZWxTaXplWzBdO1xuICAgIGNvbnN0IGtlcm5lbFcgPSB0aGlzLmtlcm5lbFNpemVbMV07XG4gICAgY29uc3Qgc3RyaWRlSCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgICBjb25zdCBzdHJpZGVXID0gdGhpcy5zdHJpZGVzWzFdO1xuXG4gICAgb3V0cHV0U2hhcGVbY2hhbm5lbEF4aXNdID0gdGhpcy5maWx0ZXJzO1xuICAgIG91dHB1dFNoYXBlW2hlaWdodEF4aXNdID1cbiAgICAgICAgZGVjb252TGVuZ3RoKG91dHB1dFNoYXBlW2hlaWdodEF4aXNdLCBzdHJpZGVILCBrZXJuZWxILCB0aGlzLnBhZGRpbmcpO1xuICAgIG91dHB1dFNoYXBlW3dpZHRoQXhpc10gPVxuICAgICAgICBkZWNvbnZMZW5ndGgob3V0cHV0U2hhcGVbd2lkdGhBeGlzXSwgc3RyaWRlVywga2VybmVsVywgdGhpcy5wYWRkaW5nKTtcbiAgICByZXR1cm4gb3V0cHV0U2hhcGU7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBkZWxldGUgY29uZmlnWydkaWxhdGlvblJhdGUnXTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQ29udjJEVHJhbnNwb3NlKTtcblxuZXhwb3J0IGNsYXNzIENvbnYzRFRyYW5zcG9zZSBleHRlbmRzIENvbnYzRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0NvbnYzRFRyYW5zcG9zZSc7XG4gIGlucHV0U3BlYzogSW5wdXRTcGVjW107XG5cbiAgY29uc3RydWN0b3IoYXJnczogQ29udkxheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDV9KV07XG5cbiAgICBpZiAodGhpcy5wYWRkaW5nICE9PSAnc2FtZScgJiYgdGhpcy5wYWRkaW5nICE9PSAndmFsaWQnKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgQ29udjNEVHJhbnNwb3NlIGN1cnJlbnRseSBzdXBwb3J0cyBvbmx5IHBhZGRpbmcgbW9kZXMgJ3NhbWUnIGAgK1xuICAgICAgICAgIGBhbmQgJ3ZhbGlkJywgYnV0IHJlY2VpdmVkIHBhZGRpbmcgbW9kZSAke3RoaXMucGFkZGluZ31gKTtcbiAgICB9XG4gIH1cblxuICBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcblxuICAgIGlmIChpbnB1dFNoYXBlLmxlbmd0aCAhPT0gNSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ0lucHV0IHNob3VsZCBoYXZlIHJhbmsgNTsgUmVjZWl2ZWQgaW5wdXQgc2hhcGU6ICcgK1xuICAgICAgICAgIEpTT04uc3RyaW5naWZ5KGlucHV0U2hhcGUpKTtcbiAgICB9XG5cbiAgICBjb25zdCBjaGFubmVsQXhpcyA9XG4gICAgICAgIHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnID8gMSA6IGlucHV0U2hhcGUubGVuZ3RoIC0gMTtcbiAgICBpZiAoaW5wdXRTaGFwZVtjaGFubmVsQXhpc10gPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ1RoZSBjaGFubmVsIGRpbWVuc2lvbiBvZiB0aGUgaW5wdXRzIHNob3VsZCBiZSBkZWZpbmVkLiAnICtcbiAgICAgICAgICAnRm91bmQgYE5vbmVgLicpO1xuICAgIH1cbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGVbY2hhbm5lbEF4aXNdO1xuICAgIGNvbnN0IGtlcm5lbFNoYXBlID0gdGhpcy5rZXJuZWxTaXplLmNvbmNhdChbdGhpcy5maWx0ZXJzLCBpbnB1dERpbV0pO1xuXG4gICAgdGhpcy5rZXJuZWwgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ2tlcm5lbCcsIGtlcm5lbFNoYXBlLCAnZmxvYXQzMicsIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIsXG4gICAgICAgIHRoaXMua2VybmVsUmVndWxhcml6ZXIsIHRydWUsIHRoaXMua2VybmVsQ29uc3RyYWludCk7XG4gICAgaWYgKHRoaXMudXNlQmlhcykge1xuICAgICAgdGhpcy5iaWFzID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2JpYXMnLCBbdGhpcy5maWx0ZXJzXSwgJ2Zsb2F0MzInLCB0aGlzLmJpYXNJbml0aWFsaXplcixcbiAgICAgICAgICB0aGlzLmJpYXNSZWd1bGFyaXplciwgdHJ1ZSwgdGhpcy5iaWFzQ29uc3RyYWludCk7XG4gICAgfVxuXG4gICAgLy8gU2V0IGlucHV0IHNwZWMuXG4gICAgdGhpcy5pbnB1dFNwZWMgPVxuICAgICAgICBbbmV3IElucHV0U3BlYyh7bmRpbTogNSwgYXhlczoge1tjaGFubmVsQXhpc106IGlucHV0RGltfX0pXTtcbiAgICB0aGlzLmJ1aWx0ID0gdHJ1ZTtcbiAgfVxuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGZjLnRpZHk8dGZjLlRlbnNvcjVEPigoKSA9PiB7XG4gICAgICBsZXQgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICBpZiAoaW5wdXQuc2hhcGUubGVuZ3RoICE9PSA1KSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYENvbnYzRFRyYW5zcG9zZS5jYWxsKCkgZXhwZWN0cyBpbnB1dCB0ZW5zb3IgdG8gYmUgcmFuay00LCBidXQgYCArXG4gICAgICAgICAgICBgcmVjZWl2ZWQgYSB0ZW5zb3Igb2YgcmFuay0ke2lucHV0LnNoYXBlLmxlbmd0aH1gKTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgaW5wdXRTaGFwZSA9IGlucHV0LnNoYXBlO1xuICAgICAgY29uc3QgYmF0Y2hTaXplID0gaW5wdXRTaGFwZVswXTtcblxuICAgICAgbGV0IGhBeGlzOiBudW1iZXI7XG4gICAgICBsZXQgd0F4aXM6IG51bWJlcjtcbiAgICAgIGxldCBkQXhpczogbnVtYmVyO1xuXG4gICAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICAgICAgZEF4aXMgPSAyO1xuICAgICAgICBoQXhpcyA9IDM7XG4gICAgICAgIHdBeGlzID0gNDtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGRBeGlzID0gMTtcbiAgICAgICAgaEF4aXMgPSAyO1xuICAgICAgICB3QXhpcyA9IDM7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IGRlcHRoID0gaW5wdXRTaGFwZVtkQXhpc107XG4gICAgICBjb25zdCBoZWlnaHQgPSBpbnB1dFNoYXBlW2hBeGlzXTtcbiAgICAgIGNvbnN0IHdpZHRoID0gaW5wdXRTaGFwZVt3QXhpc107XG4gICAgICBjb25zdCBrZXJuZWxEID0gdGhpcy5rZXJuZWxTaXplWzBdO1xuICAgICAgY29uc3Qga2VybmVsSCA9IHRoaXMua2VybmVsU2l6ZVsxXTtcbiAgICAgIGNvbnN0IGtlcm5lbFcgPSB0aGlzLmtlcm5lbFNpemVbMl07XG4gICAgICBjb25zdCBzdHJpZGVEID0gdGhpcy5zdHJpZGVzWzBdO1xuICAgICAgY29uc3Qgc3RyaWRlSCA9IHRoaXMuc3RyaWRlc1sxXTtcbiAgICAgIGNvbnN0IHN0cmlkZVcgPSB0aGlzLnN0cmlkZXNbMl07XG5cbiAgICAgIC8vIEluZmVyIHRoZSBkeW5hbWljIG91dHB1dCBzaGFwZS5cbiAgICAgIGNvbnN0IG91dERlcHRoID0gZGVjb252TGVuZ3RoKGRlcHRoLCBzdHJpZGVELCBrZXJuZWxELCB0aGlzLnBhZGRpbmcpO1xuICAgICAgY29uc3Qgb3V0SGVpZ2h0ID0gZGVjb252TGVuZ3RoKGhlaWdodCwgc3RyaWRlSCwga2VybmVsSCwgdGhpcy5wYWRkaW5nKTtcbiAgICAgIGNvbnN0IG91dFdpZHRoID0gZGVjb252TGVuZ3RoKHdpZHRoLCBzdHJpZGVXLCBrZXJuZWxXLCB0aGlzLnBhZGRpbmcpO1xuXG4gICAgICAvLyBTYW1lIGFzIGBjb252MmRUcmFuc3Bvc2VgLiBXZSBhbHdheXMgYXNzdW1lcyBjaGFubmVsc0xhc3QuXG4gICAgICBjb25zdCBvdXRwdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgICAgW2JhdGNoU2l6ZSwgb3V0RGVwdGgsIG91dEhlaWdodCwgb3V0V2lkdGgsIHRoaXMuZmlsdGVyc107XG4gICAgICBpZiAodGhpcy5kYXRhRm9ybWF0ICE9PSAnY2hhbm5lbHNMYXN0Jykge1xuICAgICAgICBpbnB1dCA9IHRmYy50cmFuc3Bvc2UoaW5wdXQsIFswLCAyLCAzLCA0LCAxXSk7XG4gICAgICB9XG4gICAgICBsZXQgb3V0cHV0cyA9IHRmYy5jb252M2RUcmFuc3Bvc2UoXG4gICAgICAgICAgaW5wdXQgYXMgVGVuc29yNUQsIHRoaXMua2VybmVsLnJlYWQoKSBhcyBUZW5zb3I1RCwgb3V0cHV0U2hhcGUsXG4gICAgICAgICAgdGhpcy5zdHJpZGVzIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgICAgICB0aGlzLnBhZGRpbmcgYXMgJ3NhbWUnIHwgJ3ZhbGlkJyk7XG4gICAgICBpZiAodGhpcy5kYXRhRm9ybWF0ICE9PSAnY2hhbm5lbHNMYXN0Jykge1xuICAgICAgICBvdXRwdXRzID0gdGZjLnRyYW5zcG9zZShvdXRwdXRzLCBbMCwgNCwgMSwgMiwgM10pO1xuICAgICAgfVxuXG4gICAgICBpZiAodGhpcy5iaWFzICE9PSBudWxsKSB7XG4gICAgICAgIG91dHB1dHMgPVxuICAgICAgICAgICAgSy5iaWFzQWRkKG91dHB1dHMsIHRoaXMuYmlhcy5yZWFkKCksIHRoaXMuZGF0YUZvcm1hdCkgYXMgVGVuc29yNUQ7XG4gICAgICB9XG4gICAgICBpZiAodGhpcy5hY3RpdmF0aW9uICE9PSBudWxsKSB7XG4gICAgICAgIG91dHB1dHMgPSB0aGlzLmFjdGl2YXRpb24uYXBwbHkob3V0cHV0cykgYXMgVGVuc29yNUQ7XG4gICAgICB9XG4gICAgICByZXR1cm4gb3V0cHV0cztcbiAgICB9KTtcbiAgfVxuXG4gIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGlucHV0U2hhcGUuc2xpY2UoKTtcblxuICAgIGxldCBjaGFubmVsQXhpczogbnVtYmVyO1xuICAgIGxldCBkZXB0aEF4aXM6IG51bWJlcjtcbiAgICBsZXQgaGVpZ2h0QXhpczogbnVtYmVyO1xuICAgIGxldCB3aWR0aEF4aXM6IG51bWJlcjtcbiAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICAgIGNoYW5uZWxBeGlzID0gMTtcbiAgICAgIGRlcHRoQXhpcyA9IDI7XG4gICAgICBoZWlnaHRBeGlzID0gMztcbiAgICAgIHdpZHRoQXhpcyA9IDQ7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNoYW5uZWxBeGlzID0gNDtcbiAgICAgIGRlcHRoQXhpcyA9IDE7XG4gICAgICBoZWlnaHRBeGlzID0gMjtcbiAgICAgIHdpZHRoQXhpcyA9IDM7XG4gICAgfVxuXG4gICAgY29uc3Qga2VybmVsRCA9IHRoaXMua2VybmVsU2l6ZVswXTtcbiAgICBjb25zdCBrZXJuZWxIID0gdGhpcy5rZXJuZWxTaXplWzFdO1xuICAgIGNvbnN0IGtlcm5lbFcgPSB0aGlzLmtlcm5lbFNpemVbMl07XG4gICAgY29uc3Qgc3RyaWRlRCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgICBjb25zdCBzdHJpZGVIID0gdGhpcy5zdHJpZGVzWzFdO1xuICAgIGNvbnN0IHN0cmlkZVcgPSB0aGlzLnN0cmlkZXNbMl07XG5cbiAgICBvdXRwdXRTaGFwZVtjaGFubmVsQXhpc10gPSB0aGlzLmZpbHRlcnM7XG4gICAgb3V0cHV0U2hhcGVbZGVwdGhBeGlzXSA9XG4gICAgICAgIGRlY29udkxlbmd0aChvdXRwdXRTaGFwZVtkZXB0aEF4aXNdLCBzdHJpZGVELCBrZXJuZWxELCB0aGlzLnBhZGRpbmcpO1xuICAgIG91dHB1dFNoYXBlW2hlaWdodEF4aXNdID1cbiAgICAgICAgZGVjb252TGVuZ3RoKG91dHB1dFNoYXBlW2hlaWdodEF4aXNdLCBzdHJpZGVILCBrZXJuZWxILCB0aGlzLnBhZGRpbmcpO1xuICAgIG91dHB1dFNoYXBlW3dpZHRoQXhpc10gPVxuICAgICAgICBkZWNvbnZMZW5ndGgob3V0cHV0U2hhcGVbd2lkdGhBeGlzXSwgc3RyaWRlVywga2VybmVsVywgdGhpcy5wYWRkaW5nKTtcbiAgICByZXR1cm4gb3V0cHV0U2hhcGU7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBkZWxldGUgY29uZmlnWydkaWxhdGlvblJhdGUnXTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQ29udjNEVHJhbnNwb3NlKTtcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFNlcGFyYWJsZUNvbnZMYXllckFyZ3MgZXh0ZW5kcyBDb252TGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIFRoZSBudW1iZXIgb2YgZGVwdGh3aXNlIGNvbnZvbHV0aW9uIG91dHB1dCBjaGFubmVscyBmb3IgZWFjaCBpbnB1dFxuICAgKiBjaGFubmVsLlxuICAgKiBUaGUgdG90YWwgbnVtYmVyIG9mIGRlcHRod2lzZSBjb252b2x1dGlvbiBvdXRwdXQgY2hhbm5lbHMgd2lsbCBiZSBlcXVhbFxuICAgKiB0byBgZmlsdGVyc0luICogZGVwdGhNdWx0aXBsaWVyYC4gRGVmYXVsdDogMS5cbiAgICovXG4gIGRlcHRoTXVsdGlwbGllcj86IG51bWJlcjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBkZXB0aHdpc2Uga2VybmVsIG1hdHJpeC5cbiAgICovXG4gIGRlcHRod2lzZUluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBJbml0aWFsaXplciBmb3IgdGhlIHBvaW50d2lzZSBrZXJuZWwgbWF0cml4LlxuICAgKi9cbiAgcG9pbnR3aXNlSW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGRlcHRod2lzZSBrZXJuZWwgbWF0cml4LlxuICAgKi9cbiAgZGVwdGh3aXNlUmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIHBvaW50d2lzZSBrZXJuZWwgbWF0cml4LlxuICAgKi9cbiAgcG9pbnR3aXNlUmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG5cbiAgLyoqXG4gICAqIENvbnN0cmFpbnQgZnVuY3Rpb24gYXBwbGllZCB0byB0aGUgZGVwdGh3aXNlIGtlcm5lbCBtYXRyaXguXG4gICAqL1xuICBkZXB0aHdpc2VDb25zdHJhaW50PzogQ29uc3RyYWludElkZW50aWZpZXJ8Q29uc3RyYWludDtcblxuICAvKipcbiAgICogQ29uc3RyYWludCBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBwb2ludHdpc2Uga2VybmVsIG1hdHJpeC5cbiAgICovXG4gIHBvaW50d2lzZUNvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xufVxuXG5leHBvcnQgY2xhc3MgU2VwYXJhYmxlQ29udiBleHRlbmRzIENvbnYge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdTZXBhcmFibGVDb252JztcblxuICByZWFkb25seSBkZXB0aE11bHRpcGxpZXI6IG51bWJlcjtcblxuICBwcm90ZWN0ZWQgcmVhZG9ubHkgZGVwdGh3aXNlSW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcjtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGRlcHRod2lzZVJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXI7XG4gIHByb3RlY3RlZCByZWFkb25seSBkZXB0aHdpc2VDb25zdHJhaW50PzogQ29uc3RyYWludDtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHBvaW50d2lzZUluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXI7XG4gIHByb3RlY3RlZCByZWFkb25seSBwb2ludHdpc2VSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVyO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcG9pbnR3aXNlQ29uc3RyYWludD86IENvbnN0cmFpbnQ7XG5cbiAgcmVhZG9ubHkgREVGQVVMVF9ERVBUSFdJU0VfSU5JVElBTElaRVI6IEluaXRpYWxpemVySWRlbnRpZmllciA9XG4gICAgICAnZ2xvcm90VW5pZm9ybSc7XG4gIHJlYWRvbmx5IERFRkFVTFRfUE9JTlRXSVNFX0lOSVRJQUxJWkVSOiBJbml0aWFsaXplcklkZW50aWZpZXIgPVxuICAgICAgJ2dsb3JvdFVuaWZvcm0nO1xuXG4gIHByb3RlY3RlZCBkZXB0aHdpc2VLZXJuZWw6IExheWVyVmFyaWFibGUgPSBudWxsO1xuICBwcm90ZWN0ZWQgcG9pbnR3aXNlS2VybmVsOiBMYXllclZhcmlhYmxlID0gbnVsbDtcblxuICBjb25zdHJ1Y3RvcihyYW5rOiBudW1iZXIsIGNvbmZpZz86IFNlcGFyYWJsZUNvbnZMYXllckFyZ3MpIHtcbiAgICBzdXBlcihyYW5rLCBjb25maWcpO1xuXG4gICAgaWYgKGNvbmZpZy5maWx0ZXJzID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdUaGUgYGZpbHRlcnNgIGNvbmZpZ3VyYXRpb24gZmllbGQgaXMgcmVxdWlyZWQgYnkgU2VwYXJhYmxlQ29udiwgJyArXG4gICAgICAgICAgJ2J1dCBpcyB1bnNwZWNpZmllZC4nKTtcbiAgICB9XG4gICAgaWYgKGNvbmZpZy5rZXJuZWxJbml0aWFsaXplciAhPSBudWxsIHx8IGNvbmZpZy5rZXJuZWxSZWd1bGFyaXplciAhPSBudWxsIHx8XG4gICAgICAgIGNvbmZpZy5rZXJuZWxDb25zdHJhaW50ICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdGaWVsZHMga2VybmVsSW5pdGlhbGl6ZXIsIGtlcm5lbFJlZ3VsYXJpemVyIGFuZCBrZXJuZWxDb25zdHJhaW50ICcgK1xuICAgICAgICAgICdhcmUgaW52YWxpZCBmb3IgU2VwYXJhYmxlQ29udjJELiBVc2UgZGVwdGh3aXNlSW5pdGlhbGl6ZXIsICcgK1xuICAgICAgICAgICdkZXB0aHdpc2VSZWd1bGFyaXplciwgZGVwdGh3aXNlQ29uc3RyYWludCwgcG9pbnR3aXNlSW5pdGlhbGl6ZXIsICcgK1xuICAgICAgICAgICdwb2ludHdpc2VSZWd1bGFyaXplciBhbmQgcG9pbnR3aXNlQ29uc3RyYWludCBpbnN0ZWFkLicpO1xuICAgIH1cbiAgICBpZiAoY29uZmlnLnBhZGRpbmcgIT0gbnVsbCAmJiBjb25maWcucGFkZGluZyAhPT0gJ3NhbWUnICYmXG4gICAgICAgIGNvbmZpZy5wYWRkaW5nICE9PSAndmFsaWQnKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgU2VwYXJhYmxlQ29udiR7dGhpcy5yYW5rfUQgc3VwcG9ydHMgb25seSBwYWRkaW5nIG1vZGVzOiBgICtcbiAgICAgICAgICBgJ3NhbWUnIGFuZCAndmFsaWQnLCBidXQgcmVjZWl2ZWQgJHtKU09OLnN0cmluZ2lmeShjb25maWcucGFkZGluZyl9YCk7XG4gICAgfVxuXG4gICAgdGhpcy5kZXB0aE11bHRpcGxpZXIgPVxuICAgICAgICBjb25maWcuZGVwdGhNdWx0aXBsaWVyID09IG51bGwgPyAxIDogY29uZmlnLmRlcHRoTXVsdGlwbGllcjtcbiAgICB0aGlzLmRlcHRod2lzZUluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXIoXG4gICAgICAgIGNvbmZpZy5kZXB0aHdpc2VJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfREVQVEhXSVNFX0lOSVRJQUxJWkVSKTtcbiAgICB0aGlzLmRlcHRod2lzZVJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoY29uZmlnLmRlcHRod2lzZVJlZ3VsYXJpemVyKTtcbiAgICB0aGlzLmRlcHRod2lzZUNvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGNvbmZpZy5kZXB0aHdpc2VDb25zdHJhaW50KTtcbiAgICB0aGlzLnBvaW50d2lzZUluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXIoXG4gICAgICAgIGNvbmZpZy5kZXB0aHdpc2VJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfUE9JTlRXSVNFX0lOSVRJQUxJWkVSKTtcbiAgICB0aGlzLnBvaW50d2lzZVJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoY29uZmlnLnBvaW50d2lzZVJlZ3VsYXJpemVyKTtcbiAgICB0aGlzLnBvaW50d2lzZUNvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGNvbmZpZy5wb2ludHdpc2VDb25zdHJhaW50KTtcbiAgfVxuXG4gIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuICAgIGlmIChpbnB1dFNoYXBlLmxlbmd0aCA8IHRoaXMucmFuayArIDIpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBJbnB1dHMgdG8gU2VwYXJhYmxlQ29udiR7dGhpcy5yYW5rfUQgc2hvdWxkIGhhdmUgcmFuayBgICtcbiAgICAgICAgICBgJHt0aGlzLnJhbmsgKyAyfSwgYnV0IHJlY2VpdmVkIGlucHV0IHNoYXBlOiBgICtcbiAgICAgICAgICBgJHtKU09OLnN0cmluZ2lmeShpbnB1dFNoYXBlKX1gKTtcbiAgICB9XG4gICAgY29uc3QgY2hhbm5lbEF4aXMgPVxuICAgICAgICB0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0JyA/IDEgOiBpbnB1dFNoYXBlLmxlbmd0aCAtIDE7XG4gICAgaWYgKGlucHV0U2hhcGVbY2hhbm5lbEF4aXNdID09IG51bGwgfHwgaW5wdXRTaGFwZVtjaGFubmVsQXhpc10gPCAwKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgVGhlIGNoYW5uZWwgZGltZW5zaW9uIG9mIHRoZSBpbnB1dHMgc2hvdWxkIGJlIGRlZmluZWQsIGAgK1xuICAgICAgICAgIGBidXQgZm91bmQgJHtKU09OLnN0cmluZ2lmeShpbnB1dFNoYXBlW2NoYW5uZWxBeGlzXSl9YCk7XG4gICAgfVxuXG4gICAgY29uc3QgaW5wdXREaW0gPSBpbnB1dFNoYXBlW2NoYW5uZWxBeGlzXTtcbiAgICBjb25zdCBkZXB0aHdpc2VLZXJuZWxTaGFwZSA9XG4gICAgICAgIHRoaXMua2VybmVsU2l6ZS5jb25jYXQoW2lucHV0RGltLCB0aGlzLmRlcHRoTXVsdGlwbGllcl0pO1xuICAgIGNvbnN0IHBvaW50d2lzZUtlcm5lbFNoYXBlID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLnJhbms7ICsraSkge1xuICAgICAgcG9pbnR3aXNlS2VybmVsU2hhcGUucHVzaCgxKTtcbiAgICB9XG4gICAgcG9pbnR3aXNlS2VybmVsU2hhcGUucHVzaChpbnB1dERpbSAqIHRoaXMuZGVwdGhNdWx0aXBsaWVyLCB0aGlzLmZpbHRlcnMpO1xuXG4gICAgY29uc3QgdHJhaW5hYmxlID0gdHJ1ZTtcbiAgICB0aGlzLmRlcHRod2lzZUtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAnZGVwdGh3aXNlX2tlcm5lbCcsIGRlcHRod2lzZUtlcm5lbFNoYXBlLCAnZmxvYXQzMicsXG4gICAgICAgIHRoaXMuZGVwdGh3aXNlSW5pdGlhbGl6ZXIsIHRoaXMuZGVwdGh3aXNlUmVndWxhcml6ZXIsIHRyYWluYWJsZSxcbiAgICAgICAgdGhpcy5kZXB0aHdpc2VDb25zdHJhaW50KTtcbiAgICB0aGlzLnBvaW50d2lzZUtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAncG9pbnR3aXNlX2tlcm5lbCcsIHBvaW50d2lzZUtlcm5lbFNoYXBlLCAnZmxvYXQzMicsXG4gICAgICAgIHRoaXMucG9pbnR3aXNlSW5pdGlhbGl6ZXIsIHRoaXMucG9pbnR3aXNlUmVndWxhcml6ZXIsIHRyYWluYWJsZSxcbiAgICAgICAgdGhpcy5wb2ludHdpc2VDb25zdHJhaW50KTtcbiAgICBpZiAodGhpcy51c2VCaWFzKSB7XG4gICAgICB0aGlzLmJpYXMgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgICAnYmlhcycsIFt0aGlzLmZpbHRlcnNdLCAnZmxvYXQzMicsIHRoaXMuYmlhc0luaXRpYWxpemVyLFxuICAgICAgICAgIHRoaXMuYmlhc1JlZ3VsYXJpemVyLCB0cmFpbmFibGUsIHRoaXMuYmlhc0NvbnN0cmFpbnQpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmJpYXMgPSBudWxsO1xuICAgIH1cblxuICAgIHRoaXMuaW5wdXRTcGVjID1cbiAgICAgICAgW25ldyBJbnB1dFNwZWMoe25kaW06IHRoaXMucmFuayArIDIsIGF4ZXM6IHtbY2hhbm5lbEF4aXNdOiBpbnB1dERpbX19KV07XG4gICAgdGhpcy5idWlsdCA9IHRydWU7XG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaW5wdXRzID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuXG4gICAgICBsZXQgb3V0cHV0OiBUZW5zb3I7XG4gICAgICBpZiAodGhpcy5yYW5rID09PSAxKSB7XG4gICAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICAgJzFEIHNlcGFyYWJsZSBjb252b2x1dGlvbiBpcyBub3QgaW1wbGVtZW50ZWQgeWV0LicpO1xuICAgICAgfSBlbHNlIGlmICh0aGlzLnJhbmsgPT09IDIpIHtcbiAgICAgICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICAgICAgaW5wdXRzID0gdGZjLnRyYW5zcG9zZShpbnB1dHMsIFswLCAyLCAzLCAxXSk7ICAvLyBOQ0hXIC0+IE5IV0MuXG4gICAgICAgIH1cblxuICAgICAgICBvdXRwdXQgPSB0ZmMuc2VwYXJhYmxlQ29udjJkKFxuICAgICAgICAgICAgaW5wdXRzIGFzIFRlbnNvcjRELCB0aGlzLmRlcHRod2lzZUtlcm5lbC5yZWFkKCkgYXMgVGVuc29yNEQsXG4gICAgICAgICAgICB0aGlzLnBvaW50d2lzZUtlcm5lbC5yZWFkKCkgYXMgVGVuc29yNEQsXG4gICAgICAgICAgICB0aGlzLnN0cmlkZXMgYXMgW251bWJlciwgbnVtYmVyXSwgdGhpcy5wYWRkaW5nIGFzICdzYW1lJyB8ICd2YWxpZCcsXG4gICAgICAgICAgICB0aGlzLmRpbGF0aW9uUmF0ZSBhcyBbbnVtYmVyLCBudW1iZXJdLCAnTkhXQycpO1xuICAgICAgfVxuXG4gICAgICBpZiAodGhpcy51c2VCaWFzKSB7XG4gICAgICAgIG91dHB1dCA9IEsuYmlhc0FkZChvdXRwdXQsIHRoaXMuYmlhcy5yZWFkKCksIHRoaXMuZGF0YUZvcm1hdCk7XG4gICAgICB9XG4gICAgICBpZiAodGhpcy5hY3RpdmF0aW9uICE9IG51bGwpIHtcbiAgICAgICAgb3V0cHV0ID0gdGhpcy5hY3RpdmF0aW9uLmFwcGx5KG91dHB1dCk7XG4gICAgICB9XG5cbiAgICAgIGlmICh0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgICAgICBvdXRwdXQgPSB0ZmMudHJhbnNwb3NlKG91dHB1dCwgWzAsIDMsIDEsIDJdKTsgIC8vIE5IV0MgLT4gTkNIVy5cbiAgICAgIH1cbiAgICAgIHJldHVybiBvdXRwdXQ7XG4gICAgfSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBkZWxldGUgY29uZmlnWydyYW5rJ107XG4gICAgZGVsZXRlIGNvbmZpZ1sna2VybmVsSW5pdGlhbGl6ZXInXTtcbiAgICBkZWxldGUgY29uZmlnWydrZXJuZWxSZWd1bGFyaXplciddO1xuICAgIGRlbGV0ZSBjb25maWdbJ2tlcm5lbENvbnN0cmFpbnQnXTtcbiAgICBjb25maWdbJ2RlcHRod2lzZUluaXRpYWxpemVyJ10gPVxuICAgICAgICBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmRlcHRod2lzZUluaXRpYWxpemVyKTtcbiAgICBjb25maWdbJ3BvaW50d2lzZUluaXRpYWxpemVyJ10gPVxuICAgICAgICBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLnBvaW50d2lzZUluaXRpYWxpemVyKTtcbiAgICBjb25maWdbJ2RlcHRod2lzZVJlZ3VsYXJpemVyJ10gPVxuICAgICAgICBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmRlcHRod2lzZVJlZ3VsYXJpemVyKTtcbiAgICBjb25maWdbJ3BvaW50d2lzZVJlZ3VsYXJpemVyJ10gPVxuICAgICAgICBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLnBvaW50d2lzZVJlZ3VsYXJpemVyKTtcbiAgICBjb25maWdbJ2RlcHRod2lzZUNvbnN0cmFpbnQnXSA9XG4gICAgICAgIHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5kZXB0aHdpc2VDb25zdHJhaW50KTtcbiAgICBjb25maWdbJ3BvaW50d2lzZUNvbnN0cmFpbnQnXSA9XG4gICAgICAgIHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5wb2ludHdpc2VDb25zdHJhaW50KTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBTZXBhcmFibGVDb252MkQgZXh0ZW5kcyBTZXBhcmFibGVDb252IHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnU2VwYXJhYmxlQ29udjJEJztcbiAgY29uc3RydWN0b3IoYXJncz86IFNlcGFyYWJsZUNvbnZMYXllckFyZ3MpIHtcbiAgICBzdXBlcigyLCBhcmdzKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFNlcGFyYWJsZUNvbnYyRCk7XG5cbmV4cG9ydCBjbGFzcyBDb252MUQgZXh0ZW5kcyBDb252IHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnQ29udjFEJztcbiAgY29uc3RydWN0b3IoYXJnczogQ29udkxheWVyQXJncykge1xuICAgIHN1cGVyKDEsIGFyZ3MpO1xuICAgIENvbnYxRC52ZXJpZnlBcmdzKGFyZ3MpO1xuICAgIHRoaXMuaW5wdXRTcGVjID0gW3tuZGltOiAzfV07XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBkZWxldGUgY29uZmlnWydyYW5rJ107XG4gICAgZGVsZXRlIGNvbmZpZ1snZGF0YUZvcm1hdCddO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3RhdGljIHZlcmlmeUFyZ3MoYXJnczogQ29udkxheWVyQXJncykge1xuICAgIC8vIGNvbmZpZy5rZXJuZWxTaXplIG11c3QgYmUgYSBudW1iZXIgb3IgYXJyYXkgb2YgbnVtYmVycy5cbiAgICBpZiAodHlwZW9mIGFyZ3Mua2VybmVsU2l6ZSAhPT0gJ251bWJlcicgJiZcbiAgICAgICAgIWdlbmVyaWNfdXRpbHMuY2hlY2tBcnJheVR5cGVBbmRMZW5ndGgoXG4gICAgICAgICAgICBhcmdzLmtlcm5lbFNpemUsICdudW1iZXInLCAxLCAxKSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYENvbnYxRCBleHBlY3RzIGNvbmZpZy5rZXJuZWxTaXplIHRvIGJlIG51bWJlciBvciBudW1iZXJbXSB3aXRoIGAgK1xuICAgICAgICAgIGBsZW5ndGggMSwgYnV0IHJlY2VpdmVkICR7SlNPTi5zdHJpbmdpZnkoYXJncy5rZXJuZWxTaXplKX0uYCk7XG4gICAgfVxuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQ29udjFEKTtcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIENyb3BwaW5nMkRMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogRGltZW5zaW9uIG9mIHRoZSBjcm9wcGluZyBhbG9uZyB0aGUgd2lkdGggYW5kIHRoZSBoZWlnaHQuXG4gICAqIC0gSWYgaW50ZWdlcjogdGhlIHNhbWUgc3ltbWV0cmljIGNyb3BwaW5nXG4gICAqICBpcyBhcHBsaWVkIHRvIHdpZHRoIGFuZCBoZWlnaHQuXG4gICAqIC0gSWYgbGlzdCBvZiAyIGludGVnZXJzOlxuICAgKiAgIGludGVycHJldGVkIGFzIHR3byBkaWZmZXJlbnRcbiAgICogICBzeW1tZXRyaWMgY3JvcHBpbmcgdmFsdWVzIGZvciBoZWlnaHQgYW5kIHdpZHRoOlxuICAgKiAgIGBbc3ltbWV0cmljX2hlaWdodF9jcm9wLCBzeW1tZXRyaWNfd2lkdGhfY3JvcF1gLlxuICAgKiAtIElmIGEgbGlzdCBvZiAyIGxpc3Qgb2YgMiBpbnRlZ2VyczpcbiAgICogICBpbnRlcnByZXRlZCBhc1xuICAgKiAgIGBbW3RvcF9jcm9wLCBib3R0b21fY3JvcF0sIFtsZWZ0X2Nyb3AsIHJpZ2h0X2Nyb3BdXWBcbiAgICovXG4gIGNyb3BwaW5nOiBudW1iZXJ8W251bWJlciwgbnVtYmVyXXxbW251bWJlciwgbnVtYmVyXSwgW251bWJlciwgbnVtYmVyXV07XG5cbiAgLyoqXG4gICAqIEZvcm1hdCBvZiB0aGUgZGF0YSwgd2hpY2ggZGV0ZXJtaW5lcyB0aGUgb3JkZXJpbmcgb2YgdGhlIGRpbWVuc2lvbnMgaW5cbiAgICogdGhlIGlucHV0cy5cbiAgICpcbiAgICogYGNoYW5uZWxzX2xhc3RgIGNvcnJlc3BvbmRzIHRvIGlucHV0cyB3aXRoIHNoYXBlXG4gICAqICAgYChiYXRjaCwgLi4uLCBjaGFubmVscylgXG4gICAqXG4gICAqIGBjaGFubmVsc19maXJzdGAgY29ycmVzcG9uZHMgdG8gaW5wdXRzIHdpdGggc2hhcGVcbiAgICogICBgKGJhdGNoLCBjaGFubmVscywgLi4uKWBcbiAgICpcbiAgICogRGVmYXVsdHMgdG8gYGNoYW5uZWxzX2xhc3RgLlxuICAgKi9cbiAgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQ7XG59XG5cbmV4cG9ydCBjbGFzcyBDcm9wcGluZzJEIGV4dGVuZHMgTGF5ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdDcm9wcGluZzJEJztcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGNyb3BwaW5nOiBbW251bWJlciwgbnVtYmVyXSwgW251bWJlciwgbnVtYmVyXV07XG4gIHByb3RlY3RlZCByZWFkb25seSBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0O1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IENyb3BwaW5nMkRMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgICBpZiAodHlwZW9mIGFyZ3MuY3JvcHBpbmcgPT09ICdudW1iZXInKSB7XG4gICAgICB0aGlzLmNyb3BwaW5nID1cbiAgICAgICAgICBbW2FyZ3MuY3JvcHBpbmcsIGFyZ3MuY3JvcHBpbmddLCBbYXJncy5jcm9wcGluZywgYXJncy5jcm9wcGluZ11dO1xuICAgIH0gZWxzZSBpZiAodHlwZW9mIGFyZ3MuY3JvcHBpbmdbMF0gPT09ICdudW1iZXInKSB7XG4gICAgICB0aGlzLmNyb3BwaW5nID0gW1xuICAgICAgICBbYXJncy5jcm9wcGluZ1swXSwgYXJncy5jcm9wcGluZ1swXV0sXG4gICAgICAgIFthcmdzLmNyb3BwaW5nWzFdIGFzIG51bWJlciwgYXJncy5jcm9wcGluZ1sxXSBhcyBudW1iZXJdXG4gICAgICBdO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmNyb3BwaW5nID0gYXJncy5jcm9wcGluZyBhcyBbW251bWJlciwgbnVtYmVyXSwgW251bWJlciwgbnVtYmVyXV07XG4gICAgfVxuICAgIHRoaXMuZGF0YUZvcm1hdCA9XG4gICAgICAgIGFyZ3MuZGF0YUZvcm1hdCA9PT0gdW5kZWZpbmVkID8gJ2NoYW5uZWxzTGFzdCcgOiBhcmdzLmRhdGFGb3JtYXQ7XG4gICAgdGhpcy5pbnB1dFNwZWMgPSBbe25kaW06IDR9XTtcbiAgfVxuXG4gIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZSk6IFNoYXBlIHtcbiAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICAgIHJldHVybiBbXG4gICAgICAgIGlucHV0U2hhcGVbMF0sIGlucHV0U2hhcGVbMV0sXG4gICAgICAgIGlucHV0U2hhcGVbMl0gLSB0aGlzLmNyb3BwaW5nWzBdWzBdIC0gdGhpcy5jcm9wcGluZ1swXVsxXSxcbiAgICAgICAgaW5wdXRTaGFwZVszXSAtIHRoaXMuY3JvcHBpbmdbMV1bMF0gLSB0aGlzLmNyb3BwaW5nWzFdWzFdXG4gICAgICBdO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gW1xuICAgICAgICBpbnB1dFNoYXBlWzBdLFxuICAgICAgICBpbnB1dFNoYXBlWzFdIC0gdGhpcy5jcm9wcGluZ1swXVswXSAtIHRoaXMuY3JvcHBpbmdbMF1bMV0sXG4gICAgICAgIGlucHV0U2hhcGVbMl0gLSB0aGlzLmNyb3BwaW5nWzFdWzBdIC0gdGhpcy5jcm9wcGluZ1sxXVsxXSwgaW5wdXRTaGFwZVszXVxuICAgICAgXTtcbiAgICB9XG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaW5wdXRzID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuXG4gICAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0Jykge1xuICAgICAgICBjb25zdCBoU2xpY2VkID0gSy5zbGljZUFsb25nQXhpcyhcbiAgICAgICAgICAgIGlucHV0cywgdGhpcy5jcm9wcGluZ1swXVswXSxcbiAgICAgICAgICAgIGlucHV0cy5zaGFwZVsxXSAtIHRoaXMuY3JvcHBpbmdbMF1bMF0gLSB0aGlzLmNyb3BwaW5nWzBdWzFdLCAyKTtcbiAgICAgICAgcmV0dXJuIEsuc2xpY2VBbG9uZ0F4aXMoXG4gICAgICAgICAgICBoU2xpY2VkLCB0aGlzLmNyb3BwaW5nWzFdWzBdLFxuICAgICAgICAgICAgaW5wdXRzLnNoYXBlWzJdIC0gdGhpcy5jcm9wcGluZ1sxXVsxXSAtIHRoaXMuY3JvcHBpbmdbMV1bMF0sIDMpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgY29uc3QgaFNsaWNlZCA9IEsuc2xpY2VBbG9uZ0F4aXMoXG4gICAgICAgICAgICBpbnB1dHMsIHRoaXMuY3JvcHBpbmdbMF1bMF0sXG4gICAgICAgICAgICBpbnB1dHMuc2hhcGVbMl0gLSB0aGlzLmNyb3BwaW5nWzBdWzBdIC0gdGhpcy5jcm9wcGluZ1swXVsxXSwgMyk7XG4gICAgICAgIHJldHVybiBLLnNsaWNlQWxvbmdBeGlzKFxuICAgICAgICAgICAgaFNsaWNlZCwgdGhpcy5jcm9wcGluZ1sxXVswXSxcbiAgICAgICAgICAgIGlucHV0cy5zaGFwZVszXSAtIHRoaXMuY3JvcHBpbmdbMV1bMV0gLSB0aGlzLmNyb3BwaW5nWzFdWzBdLCA0KTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGNvbmZpZyA9IHtjcm9wcGluZzogdGhpcy5jcm9wcGluZywgZGF0YUZvcm1hdDogdGhpcy5kYXRhRm9ybWF0fTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhDcm9wcGluZzJEKTtcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFVwU2FtcGxpbmcyRExheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBUaGUgdXBzYW1wbGluZyBmYWN0b3JzIGZvciByb3dzIGFuZCBjb2x1bW5zLlxuICAgKlxuICAgKiBEZWZhdWx0cyB0byBgWzIsIDJdYC5cbiAgICovXG4gIHNpemU/OiBudW1iZXJbXTtcbiAgLyoqXG4gICAqIEZvcm1hdCBvZiB0aGUgZGF0YSwgd2hpY2ggZGV0ZXJtaW5lcyB0aGUgb3JkZXJpbmcgb2YgdGhlIGRpbWVuc2lvbnMgaW5cbiAgICogdGhlIGlucHV0cy5cbiAgICpcbiAgICogYFwiY2hhbm5lbHNMYXN0XCJgIGNvcnJlc3BvbmRzIHRvIGlucHV0cyB3aXRoIHNoYXBlXG4gICAqICAgYFtiYXRjaCwgLi4uLCBjaGFubmVsc11gXG4gICAqXG4gICAqICBgXCJjaGFubmVsc0ZpcnN0XCJgIGNvcnJlc3BvbmRzIHRvIGlucHV0cyB3aXRoIHNoYXBlIGBbYmF0Y2gsIGNoYW5uZWxzLFxuICAgKiAuLi5dYC5cbiAgICpcbiAgICogRGVmYXVsdHMgdG8gYFwiY2hhbm5lbHNMYXN0XCJgLlxuICAgKi9cbiAgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQ7XG4gIC8qKlxuICAgKiBUaGUgaW50ZXJwb2xhdGlvbiBtZWNoYW5pc20sIG9uZSBvZiBgXCJuZWFyZXN0XCJgIG9yIGBcImJpbGluZWFyXCJgLCBkZWZhdWx0XG4gICAqIHRvIGBcIm5lYXJlc3RcImAuXG4gICAqL1xuICBpbnRlcnBvbGF0aW9uPzogSW50ZXJwb2xhdGlvbkZvcm1hdDtcbn1cblxuZXhwb3J0IGNsYXNzIFVwU2FtcGxpbmcyRCBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnVXBTYW1wbGluZzJEJztcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IERFRkFVTFRfU0laRSA9IFsyLCAyXTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHNpemU6IG51bWJlcltdO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgZGF0YUZvcm1hdDogRGF0YUZvcm1hdDtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGludGVycG9sYXRpb246IEludGVycG9sYXRpb25Gb3JtYXQ7XG5cbiAgY29uc3RydWN0b3IoYXJnczogVXBTYW1wbGluZzJETGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gICAgdGhpcy5pbnB1dFNwZWMgPSBbe25kaW06IDR9XTtcbiAgICB0aGlzLnNpemUgPSBhcmdzLnNpemUgPT0gbnVsbCA/IHRoaXMuREVGQVVMVF9TSVpFIDogYXJncy5zaXplO1xuICAgIHRoaXMuZGF0YUZvcm1hdCA9XG4gICAgICAgIGFyZ3MuZGF0YUZvcm1hdCA9PSBudWxsID8gJ2NoYW5uZWxzTGFzdCcgOiBhcmdzLmRhdGFGb3JtYXQ7XG4gICAgY2hlY2tEYXRhRm9ybWF0KHRoaXMuZGF0YUZvcm1hdCk7XG4gICAgdGhpcy5pbnRlcnBvbGF0aW9uID1cbiAgICAgICAgYXJncy5pbnRlcnBvbGF0aW9uID09IG51bGwgPyAnbmVhcmVzdCcgOiBhcmdzLmludGVycG9sYXRpb247XG4gICAgY2hlY2tJbnRlcnBvbGF0aW9uRm9ybWF0KHRoaXMuaW50ZXJwb2xhdGlvbik7XG4gIH1cblxuICBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGUpOiBTaGFwZSB7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICBjb25zdCBoZWlnaHQgPVxuICAgICAgICAgIGlucHV0U2hhcGVbMl0gPT0gbnVsbCA/IG51bGwgOiB0aGlzLnNpemVbMF0gKiBpbnB1dFNoYXBlWzJdO1xuICAgICAgY29uc3Qgd2lkdGggPSBpbnB1dFNoYXBlWzNdID09IG51bGwgPyBudWxsIDogdGhpcy5zaXplWzFdICogaW5wdXRTaGFwZVszXTtcbiAgICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVsxXSwgaGVpZ2h0LCB3aWR0aF07XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGhlaWdodCA9XG4gICAgICAgICAgaW5wdXRTaGFwZVsxXSA9PSBudWxsID8gbnVsbCA6IHRoaXMuc2l6ZVswXSAqIGlucHV0U2hhcGVbMV07XG4gICAgICBjb25zdCB3aWR0aCA9IGlucHV0U2hhcGVbMl0gPT0gbnVsbCA/IG51bGwgOiB0aGlzLnNpemVbMV0gKiBpbnB1dFNoYXBlWzJdO1xuICAgICAgcmV0dXJuIFtpbnB1dFNoYXBlWzBdLCBoZWlnaHQsIHdpZHRoLCBpbnB1dFNoYXBlWzNdXTtcbiAgICB9XG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICAgIGxldCBpbnB1dCA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKSBhcyBUZW5zb3I0RDtcbiAgICAgIGNvbnN0IGlucHV0U2hhcGUgPSBpbnB1dC5zaGFwZTtcblxuICAgICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICAgIGlucHV0ID0gdGZjLnRyYW5zcG9zZShpbnB1dCwgWzAsIDIsIDMsIDFdKTtcbiAgICAgICAgY29uc3QgaGVpZ2h0ID0gdGhpcy5zaXplWzBdICogaW5wdXRTaGFwZVsyXTtcbiAgICAgICAgY29uc3Qgd2lkdGggPSB0aGlzLnNpemVbMV0gKiBpbnB1dFNoYXBlWzNdO1xuXG4gICAgICAgIGNvbnN0IHJlc2l6ZWQgPSB0aGlzLmludGVycG9sYXRpb24gPT09ICduZWFyZXN0JyA/XG4gICAgICAgICAgICB0ZmMuaW1hZ2UucmVzaXplTmVhcmVzdE5laWdoYm9yKGlucHV0LCBbaGVpZ2h0LCB3aWR0aF0pIDpcbiAgICAgICAgICAgIHRmYy5pbWFnZS5yZXNpemVCaWxpbmVhcihpbnB1dCwgW2hlaWdodCwgd2lkdGhdKTtcbiAgICAgICAgcmV0dXJuIHRmYy50cmFuc3Bvc2UocmVzaXplZCwgWzAsIDMsIDEsIDJdKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbnN0IGhlaWdodCA9IHRoaXMuc2l6ZVswXSAqIGlucHV0U2hhcGVbMV07XG4gICAgICAgIGNvbnN0IHdpZHRoID0gdGhpcy5zaXplWzFdICogaW5wdXRTaGFwZVsyXTtcbiAgICAgICAgcmV0dXJuIHRoaXMuaW50ZXJwb2xhdGlvbiA9PT0gJ25lYXJlc3QnID9cbiAgICAgICAgICAgIHRmYy5pbWFnZS5yZXNpemVOZWFyZXN0TmVpZ2hib3IoaW5wdXQsIFtoZWlnaHQsIHdpZHRoXSkgOlxuICAgICAgICAgICAgdGZjLmltYWdlLnJlc2l6ZUJpbGluZWFyKGlucHV0LCBbaGVpZ2h0LCB3aWR0aF0pO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnID0ge3NpemU6IHRoaXMuc2l6ZSwgZGF0YUZvcm1hdDogdGhpcy5kYXRhRm9ybWF0fTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhVcFNhbXBsaW5nMkQpO1xuIl19
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
 * TensorFlow.js Layers: Pooling Layers.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import { imageDataFormat } from '../backend/common';
import * as K from '../backend/tfjs_backend';
import { checkDataFormat, checkPaddingMode, checkPoolMode } from '../common';
import { InputSpec } from '../engine/topology';
import { Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { convOutputLength } from '../utils/conv_utils';
import { assertPositiveInteger } from '../utils/generic_utils';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
import { preprocessConv2DInput, preprocessConv3DInput } from './convolutional';
/**
 * 2D pooling.
 * @param x
 * @param poolSize
 * @param stridesdes strides. Defaults to [1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 2D pooling.
 */
export function pool2d(x, poolSize, strides, padding, dataFormat, poolMode) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        checkPoolMode(poolMode);
        checkPaddingMode(padding);
        if (strides == null) {
            strides = [1, 1];
        }
        if (padding == null) {
            padding = 'valid';
        }
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        if (poolMode == null) {
            poolMode = 'max';
        }
        // TODO(cais): Remove the preprocessing step once deeplearn.js supports
        // dataFormat as an input argument.
        x = preprocessConv2DInput(x, dataFormat); // x is NHWC after preprocessing.
        let y;
        const paddingString = (padding === 'same') ? 'same' : 'valid';
        if (poolMode === 'max') {
            // TODO(cais): Rank check?
            y = tfc.maxPool(x, poolSize, strides, paddingString);
        }
        else { // 'avg'
            // TODO(cais): Check the dtype and rank of x and give clear error message
            //   if those are incorrect.
            y = tfc.avgPool(
            // TODO(cais): Rank check?
            x, poolSize, strides, paddingString);
        }
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 3, 1, 2]); // NHWC -> NCHW.
        }
        return y;
    });
}
/**
 * 3D pooling.
 * @param x
 * @param poolSize. Default to [1, 1, 1].
 * @param strides strides. Defaults to [1, 1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 3D pooling.
 */
export function pool3d(x, poolSize, strides, padding, dataFormat, poolMode) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        checkPoolMode(poolMode);
        checkPaddingMode(padding);
        if (strides == null) {
            strides = [1, 1, 1];
        }
        if (padding == null) {
            padding = 'valid';
        }
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        if (poolMode == null) {
            poolMode = 'max';
        }
        // x is NDHWC after preprocessing.
        x = preprocessConv3DInput(x, dataFormat);
        let y;
        const paddingString = (padding === 'same') ? 'same' : 'valid';
        if (poolMode === 'max') {
            y = tfc.maxPool3d(x, poolSize, strides, paddingString);
        }
        else { // 'avg'
            y = tfc.avgPool3d(x, poolSize, strides, paddingString);
        }
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 4, 1, 2, 3]); // NDHWC -> NCDHW.
        }
        return y;
    });
}
/**
 * Abstract class for different pooling 1D layers.
 */
export class Pooling1D extends Layer {
    /**
     *
     * @param args Parameters for the Pooling layer.
     *
     * config.poolSize defaults to 2.
     */
    constructor(args) {
        if (args.poolSize == null) {
            args.poolSize = 2;
        }
        super(args);
        if (typeof args.poolSize === 'number') {
            this.poolSize = [args.poolSize];
        }
        else if (Array.isArray(args.poolSize) &&
            args.poolSize.length === 1 &&
            typeof args.poolSize[0] === 'number') {
            this.poolSize = args.poolSize;
        }
        else {
            throw new ValueError(`poolSize for 1D convolutional layer must be a number or an ` +
                `Array of a single number, but received ` +
                `${JSON.stringify(args.poolSize)}`);
        }
        assertPositiveInteger(this.poolSize, 'poolSize');
        if (args.strides == null) {
            this.strides = this.poolSize;
        }
        else {
            if (typeof args.strides === 'number') {
                this.strides = [args.strides];
            }
            else if (Array.isArray(args.strides) &&
                args.strides.length === 1 &&
                typeof args.strides[0] === 'number') {
                this.strides = args.strides;
            }
            else {
                throw new ValueError(`strides for 1D convolutional layer must be a number or an ` +
                    `Array of a single number, but received ` +
                    `${JSON.stringify(args.strides)}`);
            }
        }
        assertPositiveInteger(this.strides, 'strides');
        this.padding = args.padding == null ? 'valid' : args.padding;
        checkPaddingMode(this.padding);
        this.inputSpec = [new InputSpec({ ndim: 3 })];
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const length = convOutputLength(inputShape[1], this.poolSize[0], this.padding, this.strides[0]);
        return [inputShape[0], length, inputShape[2]];
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            // Add dummy last dimension.
            inputs = K.expandDims(getExactlyOneTensor(inputs), 2);
            const output = this.poolingFunction(getExactlyOneTensor(inputs), [this.poolSize[0], 1], [this.strides[0], 1], this.padding, 'channelsLast');
            // Remove dummy last dimension.
            return tfc.squeeze(output, [2]);
        });
    }
    getConfig() {
        const config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class MaxPooling1D extends Pooling1D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
    }
}
/** @nocollapse */
MaxPooling1D.className = 'MaxPooling1D';
serialization.registerClass(MaxPooling1D);
export class AveragePooling1D extends Pooling1D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    }
}
/** @nocollapse */
AveragePooling1D.className = 'AveragePooling1D';
serialization.registerClass(AveragePooling1D);
/**
 * Abstract class for different pooling 2D layers.
 */
export class Pooling2D extends Layer {
    constructor(args) {
        if (args.poolSize == null) {
            args.poolSize = [2, 2];
        }
        super(args);
        this.poolSize = Array.isArray(args.poolSize) ?
            args.poolSize :
            [args.poolSize, args.poolSize];
        if (args.strides == null) {
            this.strides = this.poolSize;
        }
        else if (Array.isArray(args.strides)) {
            if (args.strides.length !== 2) {
                throw new ValueError(`If the strides property of a 2D pooling layer is an Array, ` +
                    `it is expected to have a length of 2, but received length ` +
                    `${args.strides.length}.`);
            }
            this.strides = args.strides;
        }
        else {
            // `config.strides` is a number.
            this.strides = [args.strides, args.strides];
        }
        assertPositiveInteger(this.poolSize, 'poolSize');
        assertPositiveInteger(this.strides, 'strides');
        this.padding = args.padding == null ? 'valid' : args.padding;
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        checkPaddingMode(this.padding);
        this.inputSpec = [new InputSpec({ ndim: 4 })];
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        let rows = this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
        let cols = this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
        rows =
            convOutputLength(rows, this.poolSize[0], this.padding, this.strides[0]);
        cols =
            convOutputLength(cols, this.poolSize[1], this.padding, this.strides[1]);
        if (this.dataFormat === 'channelsFirst') {
            return [inputShape[0], inputShape[1], rows, cols];
        }
        else {
            return [inputShape[0], rows, cols, inputShape[3]];
        }
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            return this.poolingFunction(getExactlyOneTensor(inputs), this.poolSize, this.strides, this.padding, this.dataFormat);
        });
    }
    getConfig() {
        const config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
            dataFormat: this.dataFormat
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class MaxPooling2D extends Pooling2D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
    }
}
/** @nocollapse */
MaxPooling2D.className = 'MaxPooling2D';
serialization.registerClass(MaxPooling2D);
export class AveragePooling2D extends Pooling2D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    }
}
/** @nocollapse */
AveragePooling2D.className = 'AveragePooling2D';
serialization.registerClass(AveragePooling2D);
/**
 * Abstract class for different pooling 3D layers.
 */
export class Pooling3D extends Layer {
    constructor(args) {
        if (args.poolSize == null) {
            args.poolSize = [2, 2, 2];
        }
        super(args);
        this.poolSize = Array.isArray(args.poolSize) ?
            args.poolSize :
            [args.poolSize, args.poolSize, args.poolSize];
        if (args.strides == null) {
            this.strides = this.poolSize;
        }
        else if (Array.isArray(args.strides)) {
            if (args.strides.length !== 3) {
                throw new ValueError(`If the strides property of a 3D pooling layer is an Array, ` +
                    `it is expected to have a length of 3, but received length ` +
                    `${args.strides.length}.`);
            }
            this.strides = args.strides;
        }
        else {
            // `config.strides` is a number.
            this.strides = [args.strides, args.strides, args.strides];
        }
        assertPositiveInteger(this.poolSize, 'poolSize');
        assertPositiveInteger(this.strides, 'strides');
        this.padding = args.padding == null ? 'valid' : args.padding;
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        checkPaddingMode(this.padding);
        this.inputSpec = [new InputSpec({ ndim: 5 })];
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        let depths = this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
        let rows = this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
        let cols = this.dataFormat === 'channelsFirst' ? inputShape[4] : inputShape[3];
        depths = convOutputLength(depths, this.poolSize[0], this.padding, this.strides[0]);
        rows =
            convOutputLength(rows, this.poolSize[1], this.padding, this.strides[1]);
        cols =
            convOutputLength(cols, this.poolSize[2], this.padding, this.strides[2]);
        if (this.dataFormat === 'channelsFirst') {
            return [inputShape[0], inputShape[1], depths, rows, cols];
        }
        else {
            return [inputShape[0], depths, rows, cols, inputShape[4]];
        }
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            return this.poolingFunction(getExactlyOneTensor(inputs), this.poolSize, this.strides, this.padding, this.dataFormat);
        });
    }
    getConfig() {
        const config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
            dataFormat: this.dataFormat
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class MaxPooling3D extends Pooling3D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool3d(inputs, poolSize, strides, padding, dataFormat, 'max');
    }
}
/** @nocollapse */
MaxPooling3D.className = 'MaxPooling3D';
serialization.registerClass(MaxPooling3D);
export class AveragePooling3D extends Pooling3D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool3d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    }
}
/** @nocollapse */
AveragePooling3D.className = 'AveragePooling3D';
serialization.registerClass(AveragePooling3D);
/**
 * Abstract class for different global pooling 1D layers.
 */
export class GlobalPooling1D extends Layer {
    constructor(args) {
        super(args);
        this.inputSpec = [new InputSpec({ ndim: 3 })];
    }
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[2]];
    }
    call(inputs, kwargs) {
        throw new NotImplementedError();
    }
}
export class GlobalAveragePooling1D extends GlobalPooling1D {
    constructor(args) {
        super(args || {});
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            return tfc.mean(input, 1);
        });
    }
}
/** @nocollapse */
GlobalAveragePooling1D.className = 'GlobalAveragePooling1D';
serialization.registerClass(GlobalAveragePooling1D);
export class GlobalMaxPooling1D extends GlobalPooling1D {
    constructor(args) {
        super(args || {});
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            return tfc.max(input, 1);
        });
    }
}
/** @nocollapse */
GlobalMaxPooling1D.className = 'GlobalMaxPooling1D';
serialization.registerClass(GlobalMaxPooling1D);
/**
 * Abstract class for different global pooling 2D layers.
 */
export class GlobalPooling2D extends Layer {
    constructor(args) {
        super(args);
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        this.inputSpec = [new InputSpec({ ndim: 4 })];
    }
    computeOutputShape(inputShape) {
        inputShape = inputShape;
        if (this.dataFormat === 'channelsLast') {
            return [inputShape[0], inputShape[3]];
        }
        else {
            return [inputShape[0], inputShape[1]];
        }
    }
    call(inputs, kwargs) {
        throw new NotImplementedError();
    }
    getConfig() {
        const config = { dataFormat: this.dataFormat };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class GlobalAveragePooling2D extends GlobalPooling2D {
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            if (this.dataFormat === 'channelsLast') {
                return tfc.mean(input, [1, 2]);
            }
            else {
                return tfc.mean(input, [2, 3]);
            }
        });
    }
}
/** @nocollapse */
GlobalAveragePooling2D.className = 'GlobalAveragePooling2D';
serialization.registerClass(GlobalAveragePooling2D);
export class GlobalMaxPooling2D extends GlobalPooling2D {
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            if (this.dataFormat === 'channelsLast') {
                return tfc.max(input, [1, 2]);
            }
            else {
                return tfc.max(input, [2, 3]);
            }
        });
    }
}
/** @nocollapse */
GlobalMaxPooling2D.className = 'GlobalMaxPooling2D';
serialization.registerClass(GlobalMaxPooling2D);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbGluZy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvcG9vbGluZy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVIOztHQUVHO0FBRUgsT0FBTyxLQUFLLEdBQUcsTUFBTSx1QkFBdUIsQ0FBQztBQUM3QyxPQUFPLEVBQUMsYUFBYSxFQUF3QyxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUVoRyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFDbEQsT0FBTyxLQUFLLENBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUM3QyxPQUFPLEVBQUMsZUFBZSxFQUFFLGdCQUFnQixFQUFFLGFBQWEsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUMzRSxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDN0MsT0FBTyxFQUFDLEtBQUssRUFBWSxNQUFNLG9CQUFvQixDQUFDO0FBQ3BELE9BQU8sRUFBQyxtQkFBbUIsRUFBRSxVQUFVLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFHMUQsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFDckQsT0FBTyxFQUFDLHFCQUFxQixFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDN0QsT0FBTyxFQUFDLGtCQUFrQixFQUFFLG1CQUFtQixFQUFDLE1BQU0sc0JBQXNCLENBQUM7QUFFN0UsT0FBTyxFQUFDLHFCQUFxQixFQUFFLHFCQUFxQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFN0U7Ozs7Ozs7OztHQVNHO0FBQ0gsTUFBTSxVQUFVLE1BQU0sQ0FDbEIsQ0FBUyxFQUFFLFFBQTBCLEVBQUUsT0FBMEIsRUFDakUsT0FBcUIsRUFBRSxVQUF1QixFQUM5QyxRQUFtQjtJQUNyQixPQUFPLElBQUksQ0FBQyxHQUFHLEVBQUU7UUFDZixlQUFlLENBQUMsVUFB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TCxDQUFDOztBQVpELGtCQUFrQjtBQUNYLGdDQUFTLEdBQUcsd0JBQXdCLENBQUM7QUFhOUMsYUFBYSxDQUFDLGFBQWEsQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDO0FBRXBELE1BQU0sT0FBTyxrQkFBbUIsU0FBUSxlQUFlO0lBSXJELElBQUksQ0FBQyxNQUF1QixFQUFFLE1BQWM7UUFDMUMsT0FBTyxJQUFJLENBQUMsR0FBRyxFQUFFO1lBQ2YsTUFBTSxLQUFLLEdBQUcsbUJBQW1CLENBQUMsTUFBTSxDQUFDLENBQUM7WUFDMUMsSUFBSSxJQUFJLENBQUMsVUFBVSxLQUFLLGNBQWMsRUFBRTtnQkFDdEMsT0FBTyxHQUFHLENBQUMsR0FBRyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQy9CO2lCQUFNO2dCQUNMLE9BQU8sR0FBRyxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMvQjtRQUNILENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQzs7QUFaRCxrQkFBa0I7QUFDWCw0QkFBUyxHQUFHLG9CQUFvQixDQUFDO0FBYTFDLGFBQWEsQ0FBQyxhQUFhLENBQUMsa0JBQWtCLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogVGVuc29yRmxvdy5qcyBMYXllcnM6IFBvb2xpbmcgTGF5ZXJzLlxuICovXG5cbmltcG9ydCAqIGFzIHRmYyBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtzZXJpYWxpemF0aW9uLCBUZW5zb3IsIFRlbnNvcjNELCBUZW5zb3I0RCwgVGVuc29yNUQsIHRpZHl9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7aW1hZ2VEYXRhRm9ybWF0fSBmcm9tICcuLi9iYWNrZW5kL2NvbW1vbic7XG5pbXBvcnQgKiBhcyBLIGZyb20gJy4uL2JhY2tlbmQvdGZqc19iYWNrZW5kJztcbmltcG9ydCB7Y2hlY2tEYXRhRm9ybWF0LCBjaGVja1BhZGRpbmdNb2RlLCBjaGVja1Bvb2xNb2RlfSBmcm9tICcuLi9jb21tb24nO1xuaW1wb3J0IHtJbnB1dFNwZWN9IGZyb20gJy4uL2VuZ2luZS90b3BvbG9neSc7XG5pbXBvcnQge0xheWVyLCBMYXllckFyZ3N9IGZyb20gJy4uL2VuZ2luZS90b3BvbG9neSc7XG5pbXBvcnQge05vdEltcGxlbWVudGVkRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge0RhdGFGb3JtYXQsIFBhZGRpbmdNb2RlLCBQb29sTW9kZSwgU2hhcGV9IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9jb21tb24nO1xuaW1wb3J0IHtLd2FyZ3N9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7Y29udk91dHB1dExlbmd0aH0gZnJvbSAnLi4vdXRpbHMvY29udl91dGlscyc7XG5pbXBvcnQge2Fzc2VydFBvc2l0aXZlSW50ZWdlcn0gZnJvbSAnLi4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5pbXBvcnQge2dldEV4YWN0bHlPbmVTaGFwZSwgZ2V0RXhhY3RseU9uZVRlbnNvcn0gZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xuXG5pbXBvcnQge3ByZXByb2Nlc3NDb252MkRJbnB1dCwgcHJlcHJvY2Vzc0NvbnYzRElucHV0fSBmcm9tICcuL2NvbnZvbHV0aW9uYWwnO1xuXG4vKipcbiAqIDJEIHBvb2xpbmcuXG4gKiBAcGFyYW0geFxuICogQHBhcmFtIHBvb2xTaXplXG4gKiBAcGFyYW0gc3RyaWRlc2RlcyBzdHJpZGVzLiBEZWZhdWx0cyB0byBbMSwgMV0uXG4gKiBAcGFyYW0gcGFkZGluZyBwYWRkaW5nLiBEZWZhdWx0cyB0byAndmFsaWQnLlxuICogQHBhcmFtIGRhdGFGb3JtYXQgZGF0YSBmb3JtYXQuIERlZmF1bHRzIHRvICdjaGFubmVsc0xhc3QnLlxuICogQHBhcmFtIHBvb2xNb2RlIE1vZGUgb2YgcG9vbGluZy4gRGVmYXVsdHMgdG8gJ21heCcuXG4gKiBAcmV0dXJucyBSZXN1bHQgb2YgdGhlIDJEIHBvb2xpbmcuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwb29sMmQoXG4gICAgeDogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyXSwgc3RyaWRlcz86IFtudW1iZXIsIG51bWJlcl0sXG4gICAgcGFkZGluZz86IFBhZGRpbmdNb2RlLCBkYXRhRm9ybWF0PzogRGF0YUZvcm1hdCxcbiAgICBwb29sTW9kZT86IFBvb2xNb2RlKTogVGVuc29yIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICBjaGVja1Bvb2xNb2RlKHBvb2xNb2RlKTtcbiAgICBjaGVja1BhZGRpbmdNb2RlKHBhZGRpbmcpO1xuICAgIGlmIChzdHJpZGVzID09IG51bGwpIHtcbiAgICAgIHN0cmlkZXMgPSBbMSwgMV07XG4gICAgfVxuICAgIGlmIChwYWRkaW5nID09IG51bGwpIHtcbiAgICAgIHBhZGRpbmcgPSAndmFsaWQnO1xuICAgIH1cbiAgICBpZiAoZGF0YUZvcm1hdCA9PSBudWxsKSB7XG4gICAgICBkYXRhRm9ybWF0ID0gaW1hZ2VEYXRhRm9ybWF0KCk7XG4gICAgfVxuICAgIGlmIChwb29sTW9kZSA9PSBudWxsKSB7XG4gICAgICBwb29sTW9kZSA9ICdtYXgnO1xuICAgIH1cblxuICAgIC8vIFRPRE8oY2Fpcyk6IFJlbW92ZSB0aGUgcHJlcHJvY2Vzc2luZyBzdGVwIG9uY2UgZGVlcGxlYXJuLmpzIHN1cHBvcnRzXG4gICAgLy8gZGF0YUZvcm1hdCBhcyBhbiBpbnB1dCBhcmd1bWVudC5cbiAgICB4ID0gcHJlcHJvY2Vzc0NvbnYyRElucHV0KHgsIGRhdGFGb3JtYXQpOyAgLy8geCBpcyBOSFdDIGFmdGVyIHByZXByb2Nlc3NpbmcuXG4gICAgbGV0IHk6IFRlbnNvcjtcbiAgICBjb25zdCBwYWRkaW5nU3RyaW5nID0gKHBhZGRpbmcgPT09ICdzYW1lJykgPyAnc2FtZScgOiAndmFsaWQnO1xuICAgIGlmIChwb29sTW9kZSA9PT0gJ21heCcpIHtcbiAgICAgIC8vIFRPRE8oY2Fpcyk6IFJhbmsgY2hlY2s/XG4gICAgICB5ID0gdGZjLm1heFBvb2woeCBhcyBUZW5zb3I0RCwgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmdTdHJpbmcpO1xuICAgIH0gZWxzZSB7ICAvLyAnYXZnJ1xuICAgICAgLy8gVE9ETyhjYWlzKTogQ2hlY2sgdGhlIGR0eXBlIGFuZCByYW5rIG9mIHggYW5kIGdpdmUgY2xlYXIgZXJyb3IgbWVzc2FnZVxuICAgICAgLy8gICBpZiB0aG9zZSBhcmUgaW5jb3JyZWN0LlxuICAgICAgeSA9IHRmYy5hdmdQb29sKFxuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IFJhbmsgY2hlY2s/XG4gICAgICAgICAgeCBhcyBUZW5zb3IzRCB8IFRlbnNvcjRELCBwb29sU2l6ZSwgc3RyaWRlcywgcGFkZGluZ1N0cmluZyk7XG4gICAgfVxuICAgIGlmIChkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICAgIHkgPSB0ZmMudHJhbnNwb3NlKHksIFswLCAzLCAxLCAyXSk7ICAvLyBOSFdDIC0+IE5DSFcuXG4gICAgfVxuICAgIHJldHVybiB5O1xuICB9KTtcbn1cblxuLyoqXG4gKiAzRCBwb29saW5nLlxuICogQHBhcmFtIHhcbiAqIEBwYXJhbSBwb29sU2l6ZS4gRGVmYXVsdCB0byBbMSwgMSwgMV0uXG4gKiBAcGFyYW0gc3RyaWRlcyBzdHJpZGVzLiBEZWZhdWx0cyB0byBbMSwgMSwgMV0uXG4gKiBAcGFyYW0gcGFkZGluZyBwYWRkaW5nLiBEZWZhdWx0cyB0byAndmFsaWQnLlxuICogQHBhcmFtIGRhdGFGb3JtYXQgZGF0YSBmb3JtYXQuIERlZmF1bHRzIHRvICdjaGFubmVsc0xhc3QnLlxuICogQHBhcmFtIHBvb2xNb2RlIE1vZGUgb2YgcG9vbGluZy4gRGVmYXVsdHMgdG8gJ21heCcuXG4gKiBAcmV0dXJucyBSZXN1bHQgb2YgdGhlIDNEIHBvb2xpbmcuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwb29sM2QoXG4gICAgeDogVGVuc29yNUQsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgc3RyaWRlcz86IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgcGFkZGluZz86IFBhZGRpbmdNb2RlLFxuICAgIGRhdGFGb3JtYXQ/OiBEYXRhRm9ybWF0LCBwb29sTW9kZT86IFBvb2xNb2RlKTogVGVuc29yIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICBjaGVja1Bvb2xNb2RlKHBvb2xNb2RlKTtcbiAgICBjaGVja1BhZGRpbmdNb2RlKHBhZGRpbmcpO1xuICAgIGlmIChzdHJpZGVzID09IG51bGwpIHtcbiAgICAgIHN0cmlkZXMgPSBbMSwgMSwgMV07XG4gICAgfVxuICAgIGlmIChwYWRkaW5nID09IG51bGwpIHtcbiAgICAgIHBhZGRpbmcgPSAndmFsaWQnO1xuICAgIH1cbiAgICBpZiAoZGF0YUZvcm1hdCA9PSBudWxsKSB7XG4gICAgICBkYXRhRm9ybWF0ID0gaW1hZ2VEYXRhRm9ybWF0KCk7XG4gICAgfVxuICAgIGlmIChwb29sTW9kZSA9PSBudWxsKSB7XG4gICAgICBwb29sTW9kZSA9ICdtYXgnO1xuICAgIH1cblxuICAgIC8vIHggaXMgTkRIV0MgYWZ0ZXIgcHJlcHJvY2Vzc2luZy5cbiAgICB4ID0gcHJlcHJvY2Vzc0NvbnYzRElucHV0KHggYXMgVGVuc29yLCBkYXRhRm9ybWF0KSBhcyBUZW5zb3I1RDtcbiAgICBsZXQgeTogVGVuc29yO1xuICAgIGNvbnN0IHBhZGRpbmdTdHJpbmcgPSAocGFkZGluZyA9PT0gJ3NhbWUnKSA/ICdzYW1lJyA6ICd2YWxpZCc7XG4gICAgaWYgKHBvb2xNb2RlID09PSAnbWF4Jykge1xuICAgICAgeSA9IHRmYy5tYXhQb29sM2QoeCwgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmdTdHJpbmcpO1xuICAgIH0gZWxzZSB7ICAvLyAnYXZnJ1xuICAgICAgeSA9IHRmYy5hdmdQb29sM2QoeCwgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmdTdHJpbmcpO1xuICAgIH1cbiAgICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICB5ID0gdGZjLnRyYW5zcG9zZSh5LCBbMCwgNCwgMSwgMiwgM10pOyAgLy8gTkRIV0MgLT4gTkNESFcuXG4gICAgfVxuICAgIHJldHVybiB5O1xuICB9KTtcbn1cblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFBvb2xpbmcxRExheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBTaXplIG9mIHRoZSB3aW5kb3cgdG8gcG9vbCBvdmVyLCBzaG91bGQgYmUgYW4gaW50ZWdlci5cbiAgICovXG4gIHBvb2xTaXplPzogbnVtYmVyfFtudW1iZXJdO1xuICAvKipcbiAgICogUGVyaW9kIGF0IHdoaWNoIHRvIHNhbXBsZSB0aGUgcG9vbGVkIHZhbHVlcy5cbiAgICpcbiAgICogSWYgYG51bGxgLCBkZWZhdWx0cyB0byBgcG9vbFNpemVgLlxuICAgKi9cbiAgc3RyaWRlcz86IG51bWJlcnxbbnVtYmVyXTtcbiAgLyoqIEhvdyB0byBmaWxsIGluIGRhdGEgdGhhdCdzIG5vdCBhbiBpbnRlZ2VyIG11bHRpcGxlIG9mIHBvb2xTaXplLiAqL1xuICBwYWRkaW5nPzogUGFkZGluZ01vZGU7XG59XG5cbi8qKlxuICogQWJzdHJhY3QgY2xhc3MgZm9yIGRpZmZlcmVudCBwb29saW5nIDFEIGxheWVycy5cbiAqL1xuZXhwb3J0IGFic3RyYWN0IGNsYXNzIFBvb2xpbmcxRCBleHRlbmRzIExheWVyIHtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHBvb2xTaXplOiBbbnVtYmVyXTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHN0cmlkZXM6IFtudW1iZXJdO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcGFkZGluZzogUGFkZGluZ01vZGU7XG5cbiAgLyoqXG4gICAqXG4gICAqIEBwYXJhbSBhcmdzIFBhcmFtZXRlcnMgZm9yIHRoZSBQb29saW5nIGxheWVyLlxuICAgKlxuICAgKiBjb25maWcucG9vbFNpemUgZGVmYXVsdHMgdG8gMi5cbiAgICovXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmcxRExheWVyQXJncykge1xuICAgIGlmIChhcmdzLnBvb2xTaXplID09IG51bGwpIHtcbiAgICAgIGFyZ3MucG9vbFNpemUgPSAyO1xuICAgIH1cbiAgICBzdXBlcihhcmdzKTtcbiAgICBpZiAodHlwZW9mIGFyZ3MucG9vbFNpemUgPT09ICdudW1iZXInKSB7XG4gICAgICB0aGlzLnBvb2xTaXplID0gW2FyZ3MucG9vbFNpemVdO1xuICAgIH0gZWxzZSBpZiAoXG4gICAgICAgIEFycmF5LmlzQXJyYXkoYXJncy5wb29sU2l6ZSkgJiZcbiAgICAgICAgKGFyZ3MucG9vbFNpemUgYXMgbnVtYmVyW10pLmxlbmd0aCA9PT0gMSAmJlxuICAgICAgICB0eXBlb2YgKGFyZ3MucG9vbFNpemUgYXMgbnVtYmVyW10pWzBdID09PSAnbnVtYmVyJykge1xuICAgICAgdGhpcy5wb29sU2l6ZSA9IGFyZ3MucG9vbFNpemU7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBwb29sU2l6ZSBmb3IgMUQgY29udm9sdXRpb25hbCBsYXllciBtdXN0IGJlIGEgbnVtYmVyIG9yIGFuIGAgK1xuICAgICAgICAgIGBBcnJheSBvZiBhIHNpbmdsZSBudW1iZXIsIGJ1dCByZWNlaXZlZCBgICtcbiAgICAgICAgICBgJHtKU09OLnN0cmluZ2lmeShhcmdzLnBvb2xTaXplKX1gKTtcbiAgICB9XG4gICAgYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMucG9vbFNpemUsICdwb29sU2l6ZScpO1xuICAgIGlmIChhcmdzLnN0cmlkZXMgPT0gbnVsbCkge1xuICAgICAgdGhpcy5zdHJpZGVzID0gdGhpcy5wb29sU2l6ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgaWYgKHR5cGVvZiBhcmdzLnN0cmlkZXMgPT09ICdudW1iZXInKSB7XG4gICAgICAgIHRoaXMuc3RyaWRlcyA9IFthcmdzLnN0cmlkZXNdO1xuICAgICAgfSBlbHNlIGlmIChcbiAgICAgICAgICBBcnJheS5pc0FycmF5KGFyZ3Muc3RyaWRlcykgJiZcbiAgICAgICAgICAoYXJncy5zdHJpZGVzIGFzIG51bWJlcltdKS5sZW5ndGggPT09IDEgJiZcbiAgICAgICAgICB0eXBlb2YgKGFyZ3Muc3RyaWRlcyBhcyBudW1iZXJbXSlbMF0gPT09ICdudW1iZXInKSB7XG4gICAgICAgIHRoaXMuc3RyaWRlcyA9IGFyZ3Muc3RyaWRlcztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYHN0cmlkZXMgZm9yIDFEIGNvbnZvbHV0aW9uYWwgbGF5ZXIgbXVzdCBiZSBhIG51bWJlciBvciBhbiBgICtcbiAgICAgICAgICAgIGBBcnJheSBvZiBhIHNpbmdsZSBudW1iZXIsIGJ1dCByZWNlaXZlZCBgICtcbiAgICAgICAgICAgIGAke0pTT04uc3RyaW5naWZ5KGFyZ3Muc3RyaWRlcyl9YCk7XG4gICAgICB9XG4gICAgfVxuICAgIGFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLnN0cmlkZXMsICdzdHJpZGVzJyk7XG5cbiAgICB0aGlzLnBhZGRpbmcgPSBhcmdzLnBhZGRpbmcgPT0gbnVsbCA/ICd2YWxpZCcgOiBhcmdzLnBhZGRpbmc7XG4gICAgY2hlY2tQYWRkaW5nTW9kZSh0aGlzLnBhZGRpbmcpO1xuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDN9KV07XG4gIH1cblxuICBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgbGVuZ3RoID0gY29udk91dHB1dExlbmd0aChcbiAgICAgICAgaW5wdXRTaGFwZVsxXSwgdGhpcy5wb29sU2l6ZVswXSwgdGhpcy5wYWRkaW5nLCB0aGlzLnN0cmlkZXNbMF0pO1xuICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgbGVuZ3RoLCBpbnB1dFNoYXBlWzJdXTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhYnN0cmFjdCBwb29saW5nRnVuY3Rpb24oXG4gICAgICBpbnB1dHM6IFRlbnNvciwgcG9vbFNpemU6IFtudW1iZXIsIG51bWJlcl0sIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBwYWRkaW5nOiBQYWRkaW5nTW9kZSwgZGF0YUZvcm1hdDogRGF0YUZvcm1hdCk6IFRlbnNvcjtcblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgdGhpcy5pbnZva2VDYWxsSG9vayhpbnB1dHMsIGt3YXJncyk7XG4gICAgICAvLyBBZGQgZHVtbXkgbGFzdCBkaW1lbnNpb24uXG4gICAgICBpbnB1dHMgPSBLLmV4cGFuZERpbXMoZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpLCAyKTtcbiAgICAgIGNvbnN0IG91dHB1dCA9IHRoaXMucG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgICAgIGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKSwgW3RoaXMucG9vbFNpemVbMF0sIDFdLFxuICAgICAgICAgIFt0aGlzLnN0cmlkZXNbMF0sIDFdLCB0aGlzLnBhZGRpbmcsICdjaGFubmVsc0xhc3QnKTtcbiAgICAgIC8vIFJlbW92ZSBkdW1teSBsYXN0IGRpbWVuc2lvbi5cbiAgICAgIHJldHVybiB0ZmMuc3F1ZWV6ZShvdXRwdXQsIFsyXSk7XG4gICAgfSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSB7XG4gICAgICBwb29sU2l6ZTogdGhpcy5wb29sU2l6ZSxcbiAgICAgIHBhZGRpbmc6IHRoaXMucGFkZGluZyxcbiAgICAgIHN0cmlkZXM6IHRoaXMuc3RyaWRlcyxcbiAgICB9O1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgTWF4UG9vbGluZzFEIGV4dGVuZHMgUG9vbGluZzFEIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTWF4UG9vbGluZzFEJztcbiAgY29uc3RydWN0b3IoYXJnczogUG9vbGluZzFETGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgcG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgaW5wdXRzOiBUZW5zb3IsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXJdLCBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgcGFkZGluZzogUGFkZGluZ01vZGUsIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQpOiBUZW5zb3Ige1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICBjaGVja1BhZGRpbmdNb2RlKHBhZGRpbmcpO1xuICAgIHJldHVybiBwb29sMmQoaW5wdXRzLCBwb29sU2l6ZSwgc3RyaWRlcywgcGFkZGluZywgZGF0YUZvcm1hdCwgJ21heCcpO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoTWF4UG9vbGluZzFEKTtcblxuZXhwb3J0IGNsYXNzIEF2ZXJhZ2VQb29saW5nMUQgZXh0ZW5kcyBQb29saW5nMUQge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdBdmVyYWdlUG9vbGluZzFEJztcbiAgY29uc3RydWN0b3IoYXJnczogUG9vbGluZzFETGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgcG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgaW5wdXRzOiBUZW5zb3IsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXJdLCBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgcGFkZGluZzogUGFkZGluZ01vZGUsIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQpOiBUZW5zb3Ige1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICBjaGVja1BhZGRpbmdNb2RlKHBhZGRpbmcpO1xuICAgIHJldHVybiBwb29sMmQoaW5wdXRzLCBwb29sU2l6ZSwgc3RyaWRlcywgcGFkZGluZywgZGF0YUZvcm1hdCwgJ2F2ZycpO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQXZlcmFnZVBvb2xpbmcxRCk7XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBQb29saW5nMkRMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogRmFjdG9ycyBieSB3aGljaCB0byBkb3duc2NhbGUgaW4gZWFjaCBkaW1lbnNpb24gW3ZlcnRpY2FsLCBob3Jpem9udGFsXS5cbiAgICogRXhwZWN0cyBhbiBpbnRlZ2VyIG9yIGFuIGFycmF5IG9mIDIgaW50ZWdlcnMuXG4gICAqXG4gICAqIEZvciBleGFtcGxlLCBgWzIsIDJdYCB3aWxsIGhhbHZlIHRoZSBpbnB1dCBpbiBib3RoIHNwYXRpYWwgZGltZW5zaW9uLlxuICAgKiBJZiBvbmx5IG9uZSBpbnRlZ2VyIGlzIHNwZWNpZmllZCwgdGhlIHNhbWUgd2luZG93IGxlbmd0aFxuICAgKiB3aWxsIGJlIHVzZWQgZm9yIGJvdGggZGltZW5zaW9ucy5cbiAgICovXG4gIHBvb2xTaXplPzogbnVtYmVyfFtudW1iZXIsIG51bWJlcl07XG5cbiAgLyoqXG4gICAqIFRoZSBzaXplIG9mIHRoZSBzdHJpZGUgaW4gZWFjaCBkaW1lbnNpb24gb2YgdGhlIHBvb2xpbmcgd2luZG93LiBFeHBlY3RzXG4gICAqIGFuIGludGVnZXIgb3IgYW4gYXJyYXkgb2YgMiBpbnRlZ2Vycy4gSW50ZWdlciwgdHVwbGUgb2YgMiBpbnRlZ2Vycywgb3JcbiAgICogTm9uZS5cbiAgICpcbiAgICogSWYgYG51bGxgLCBkZWZhdWx0cyB0byBgcG9vbFNpemVgLlxuICAgKi9cbiAgc3RyaWRlcz86IG51bWJlcnxbbnVtYmVyLCBudW1iZXJdO1xuXG4gIC8qKiBUaGUgcGFkZGluZyB0eXBlIHRvIHVzZSBmb3IgdGhlIHBvb2xpbmcgbGF5ZXIuICovXG4gIHBhZGRpbmc/OiBQYWRkaW5nTW9kZTtcbiAgLyoqIFRoZSBkYXRhIGZvcm1hdCB0byB1c2UgZm9yIHRoZSBwb29saW5nIGxheWVyLiAqL1xuICBkYXRhRm9ybWF0PzogRGF0YUZvcm1hdDtcbn1cblxuLyoqXG4gKiBBYnN0cmFjdCBjbGFzcyBmb3IgZGlmZmVyZW50IHBvb2xpbmcgMkQgbGF5ZXJzLlxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgUG9vbGluZzJEIGV4dGVuZHMgTGF5ZXIge1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcG9vbFNpemU6IFtudW1iZXIsIG51bWJlcl07XG4gIHByb3RlY3RlZCByZWFkb25seSBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcGFkZGluZzogUGFkZGluZ01vZGU7XG4gIHByb3RlY3RlZCByZWFkb25seSBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0O1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmcyRExheWVyQXJncykge1xuICAgIGlmIChhcmdzLnBvb2xTaXplID09IG51bGwpIHtcbiAgICAgIGFyZ3MucG9vbFNpemUgPSBbMiwgMl07XG4gICAgfVxuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMucG9vbFNpemUgPSBBcnJheS5pc0FycmF5KGFyZ3MucG9vbFNpemUpID9cbiAgICAgICAgYXJncy5wb29sU2l6ZSA6XG4gICAgICAgIFthcmdzLnBvb2xTaXplLCBhcmdzLnBvb2xTaXplXTtcbiAgICBpZiAoYXJncy5zdHJpZGVzID09IG51bGwpIHtcbiAgICAgIHRoaXMuc3RyaWRlcyA9IHRoaXMucG9vbFNpemU7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KGFyZ3Muc3RyaWRlcykpIHtcbiAgICAgIGlmIChhcmdzLnN0cmlkZXMubGVuZ3RoICE9PSAyKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYElmIHRoZSBzdHJpZGVzIHByb3BlcnR5IG9mIGEgMkQgcG9vbGluZyBsYXllciBpcyBhbiBBcnJheSwgYCArXG4gICAgICAgICAgICBgaXQgaXMgZXhwZWN0ZWQgdG8gaGF2ZSBhIGxlbmd0aCBvZiAyLCBidXQgcmVjZWl2ZWQgbGVuZ3RoIGAgK1xuICAgICAgICAgICAgYCR7YXJncy5zdHJpZGVzLmxlbmd0aH0uYCk7XG4gICAgICB9XG4gICAgICB0aGlzLnN0cmlkZXMgPSBhcmdzLnN0cmlkZXM7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIGBjb25maWcuc3RyaWRlc2AgaXMgYSBudW1iZXIuXG4gICAgICB0aGlzLnN0cmlkZXMgPSBbYXJncy5zdHJpZGVzLCBhcmdzLnN0cmlkZXNdO1xuICAgIH1cbiAgICBhc3NlcnRQb3NpdGl2ZUludGVnZXIodGhpcy5wb29sU2l6ZSwgJ3Bvb2xTaXplJyk7XG4gICAgYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMuc3RyaWRlcywgJ3N0cmlkZXMnKTtcbiAgICB0aGlzLnBhZGRpbmcgPSBhcmdzLnBhZGRpbmcgPT0gbnVsbCA/ICd2YWxpZCcgOiBhcmdzLnBhZGRpbmc7XG4gICAgdGhpcy5kYXRhRm9ybWF0ID1cbiAgICAgICAgYXJncy5kYXRhRm9ybWF0ID09IG51bGwgPyAnY2hhbm5lbHNMYXN0JyA6IGFyZ3MuZGF0YUZvcm1hdDtcbiAgICBjaGVja0RhdGFGb3JtYXQodGhpcy5kYXRhRm9ybWF0KTtcbiAgICBjaGVja1BhZGRpbmdNb2RlKHRoaXMucGFkZGluZyk7XG5cbiAgICB0aGlzLmlucHV0U3BlYyA9IFtuZXcgSW5wdXRTcGVjKHtuZGltOiA0fSldO1xuICB9XG5cbiAgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuICAgIGxldCByb3dzID1cbiAgICAgICAgdGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcgPyBpbnB1dFNoYXBlWzJdIDogaW5wdXRTaGFwZVsxXTtcbiAgICBsZXQgY29scyA9XG4gICAgICAgIHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnID8gaW5wdXRTaGFwZVszXSA6IGlucHV0U2hhcGVbMl07XG4gICAgcm93cyA9XG4gICAgICAgIGNvbnZPdXRwdXRMZW5ndGgocm93cywgdGhpcy5wb29sU2l6ZVswXSwgdGhpcy5wYWRkaW5nLCB0aGlzLnN0cmlkZXNbMF0pO1xuICAgIGNvbHMgPVxuICAgICAgICBjb252T3V0cHV0TGVuZ3RoKGNvbHMsIHRoaXMucG9vbFNpemVbMV0sIHRoaXMucGFkZGluZywgdGhpcy5zdHJpZGVzWzFdKTtcbiAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVsxXSwgcm93cywgY29sc107XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgcm93cywgY29scywgaW5wdXRTaGFwZVszXV07XG4gICAgfVxuICB9XG5cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyXSwgc3RyaWRlczogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHBhZGRpbmc6IFBhZGRpbmdNb2RlLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0KTogVGVuc29yO1xuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICB0aGlzLmludm9rZUNhbGxIb29rKGlucHV0cywga3dhcmdzKTtcbiAgICAgIHJldHVybiB0aGlzLnBvb2xpbmdGdW5jdGlvbihcbiAgICAgICAgICBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyksIHRoaXMucG9vbFNpemUsIHRoaXMuc3RyaWRlcyxcbiAgICAgICAgICB0aGlzLnBhZGRpbmcsIHRoaXMuZGF0YUZvcm1hdCk7XG4gICAgfSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSB7XG4gICAgICBwb29sU2l6ZTogdGhpcy5wb29sU2l6ZSxcbiAgICAgIHBhZGRpbmc6IHRoaXMucGFkZGluZyxcbiAgICAgIHN0cmlkZXM6IHRoaXMuc3RyaWRlcyxcbiAgICAgIGRhdGFGb3JtYXQ6IHRoaXMuZGF0YUZvcm1hdFxuICAgIH07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBNYXhQb29saW5nMkQgZXh0ZW5kcyBQb29saW5nMkQge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdNYXhQb29saW5nMkQnO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBQb29saW5nMkRMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBwb29saW5nRnVuY3Rpb24oXG4gICAgICBpbnB1dHM6IFRlbnNvciwgcG9vbFNpemU6IFtudW1iZXIsIG51bWJlcl0sIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBwYWRkaW5nOiBQYWRkaW5nTW9kZSwgZGF0YUZvcm1hdDogRGF0YUZvcm1hdCk6IFRlbnNvciB7XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIGNoZWNrUGFkZGluZ01vZGUocGFkZGluZyk7XG4gICAgcmV0dXJuIHBvb2wyZChpbnB1dHMsIHBvb2xTaXplLCBzdHJpZGVzLCBwYWRkaW5nLCBkYXRhRm9ybWF0LCAnbWF4Jyk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhNYXhQb29saW5nMkQpO1xuXG5leHBvcnQgY2xhc3MgQXZlcmFnZVBvb2xpbmcyRCBleHRlbmRzIFBvb2xpbmcyRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0F2ZXJhZ2VQb29saW5nMkQnO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBQb29saW5nMkRMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBwb29saW5nRnVuY3Rpb24oXG4gICAgICBpbnB1dHM6IFRlbnNvciwgcG9vbFNpemU6IFtudW1iZXIsIG51bWJlcl0sIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBwYWRkaW5nOiBQYWRkaW5nTW9kZSwgZGF0YUZvcm1hdDogRGF0YUZvcm1hdCk6IFRlbnNvciB7XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIGNoZWNrUGFkZGluZ01vZGUocGFkZGluZyk7XG4gICAgcmV0dXJuIHBvb2wyZChpbnB1dHMsIHBvb2xTaXplLCBzdHJpZGVzLCBwYWRkaW5nLCBkYXRhRm9ybWF0LCAnYXZnJyk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhBdmVyYWdlUG9vbGluZzJEKTtcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFBvb2xpbmczRExheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBGYWN0b3JzIGJ5IHdoaWNoIHRvIGRvd25zY2FsZSBpbiBlYWNoIGRpbWVuc2lvbiBbZGVwdGgsIGhlaWdodCwgd2lkdGhdLlxuICAgKiBFeHBlY3RzIGFuIGludGVnZXIgb3IgYW4gYXJyYXkgb2YgMyBpbnRlZ2Vycy5cbiAgICpcbiAgICogRm9yIGV4YW1wbGUsIGBbMiwgMiwgMl1gIHdpbGwgaGFsdmUgdGhlIGlucHV0IGluIHRocmVlIGRpbWVuc2lvbnMuXG4gICAqIElmIG9ubHkgb25lIGludGVnZXIgaXMgc3BlY2lmaWVkLCB0aGUgc2FtZSB3aW5kb3cgbGVuZ3RoXG4gICAqIHdpbGwgYmUgdXNlZCBmb3IgYWxsIGRpbWVuc2lvbnMuXG4gICAqL1xuICBwb29sU2l6ZT86IG51bWJlcnxbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG5cbiAgLyoqXG4gICAqIFRoZSBzaXplIG9mIHRoZSBzdHJpZGUgaW4gZWFjaCBkaW1lbnNpb24gb2YgdGhlIHBvb2xpbmcgd2luZG93LiBFeHBlY3RzXG4gICAqIGFuIGludGVnZXIgb3IgYW4gYXJyYXkgb2YgMyBpbnRlZ2Vycy4gSW50ZWdlciwgdHVwbGUgb2YgMyBpbnRlZ2Vycywgb3JcbiAgICogTm9uZS5cbiAgICpcbiAgICogSWYgYG51bGxgLCBkZWZhdWx0cyB0byBgcG9vbFNpemVgLlxuICAgKi9cbiAgc3RyaWRlcz86IG51bWJlcnxbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG5cbiAgLyoqIFRoZSBwYWRkaW5nIHR5cGUgdG8gdXNlIGZvciB0aGUgcG9vbGluZyBsYXllci4gKi9cbiAgcGFkZGluZz86IFBhZGRpbmdNb2RlO1xuICAvKiogVGhlIGRhdGEgZm9ybWF0IHRvIHVzZSBmb3IgdGhlIHBvb2xpbmcgbGF5ZXIuICovXG4gIGRhdGFGb3JtYXQ/OiBEYXRhRm9ybWF0O1xufVxuXG4vKipcbiAqIEFic3RyYWN0IGNsYXNzIGZvciBkaWZmZXJlbnQgcG9vbGluZyAzRCBsYXllcnMuXG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBQb29saW5nM0QgZXh0ZW5kcyBMYXllciB7XG4gIHByb3RlY3RlZCByZWFkb25seSBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgc3RyaWRlczogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcGFkZGluZzogUGFkZGluZ01vZGU7XG4gIHByb3RlY3RlZCByZWFkb25seSBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0O1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmczRExheWVyQXJncykge1xuICAgIGlmIChhcmdzLnBvb2xTaXplID09IG51bGwpIHtcbiAgICAgIGFyZ3MucG9vbFNpemUgPSBbMiwgMiwgMl07XG4gICAgfVxuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMucG9vbFNpemUgPSBBcnJheS5pc0FycmF5KGFyZ3MucG9vbFNpemUpID9cbiAgICAgICAgYXJncy5wb29sU2l6ZSA6XG4gICAgICAgIFthcmdzLnBvb2xTaXplLCBhcmdzLnBvb2xTaXplLCBhcmdzLnBvb2xTaXplXTtcbiAgICBpZiAoYXJncy5zdHJpZGVzID09IG51bGwpIHtcbiAgICAgIHRoaXMuc3RyaWRlcyA9IHRoaXMucG9vbFNpemU7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KGFyZ3Muc3RyaWRlcykpIHtcbiAgICAgIGlmIChhcmdzLnN0cmlkZXMubGVuZ3RoICE9PSAzKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYElmIHRoZSBzdHJpZGVzIHByb3BlcnR5IG9mIGEgM0QgcG9vbGluZyBsYXllciBpcyBhbiBBcnJheSwgYCArXG4gICAgICAgICAgICBgaXQgaXMgZXhwZWN0ZWQgdG8gaGF2ZSBhIGxlbmd0aCBvZiAzLCBidXQgcmVjZWl2ZWQgbGVuZ3RoIGAgK1xuICAgICAgICAgICAgYCR7YXJncy5zdHJpZGVzLmxlbmd0aH0uYCk7XG4gICAgICB9XG4gICAgICB0aGlzLnN0cmlkZXMgPSBhcmdzLnN0cmlkZXM7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIGBjb25maWcuc3RyaWRlc2AgaXMgYSBudW1iZXIuXG4gICAgICB0aGlzLnN0cmlkZXMgPSBbYXJncy5zdHJpZGVzLCBhcmdzLnN0cmlkZXMsIGFyZ3Muc3RyaWRlc107XG4gICAgfVxuICAgIGFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLnBvb2xTaXplLCAncG9vbFNpemUnKTtcbiAgICBhc3NlcnRQb3NpdGl2ZUludGVnZXIodGhpcy5zdHJpZGVzLCAnc3RyaWRlcycpO1xuICAgIHRoaXMucGFkZGluZyA9IGFyZ3MucGFkZGluZyA9PSBudWxsID8gJ3ZhbGlkJyA6IGFyZ3MucGFkZGluZztcbiAgICB0aGlzLmRhdGFGb3JtYXQgPVxuICAgICAgICBhcmdzLmRhdGFGb3JtYXQgPT0gbnVsbCA/ICdjaGFubmVsc0xhc3QnIDogYXJncy5kYXRhRm9ybWF0O1xuICAgIGNoZWNrRGF0YUZvcm1hdCh0aGlzLmRhdGFGb3JtYXQpO1xuICAgIGNoZWNrUGFkZGluZ01vZGUodGhpcy5wYWRkaW5nKTtcblxuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDV9KV07XG4gIH1cblxuICBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgbGV0IGRlcHRocyA9XG4gICAgICAgIHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnID8gaW5wdXRTaGFwZVsyXSA6IGlucHV0U2hhcGVbMV07XG4gICAgbGV0IHJvd3MgPVxuICAgICAgICB0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0JyA/IGlucHV0U2hhcGVbM10gOiBpbnB1dFNoYXBlWzJdO1xuICAgIGxldCBjb2xzID1cbiAgICAgICAgdGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcgPyBpbnB1dFNoYXBlWzRdIDogaW5wdXRTaGFwZVszXTtcbiAgICBkZXB0aHMgPSBjb252T3V0cHV0TGVuZ3RoKFxuICAgICAgICBkZXB0aHMsIHRoaXMucG9vbFNpemVbMF0sIHRoaXMucGFkZGluZywgdGhpcy5zdHJpZGVzWzBdKTtcbiAgICByb3dzID1cbiAgICAgICAgY29udk91dHB1dExlbmd0aChyb3dzLCB0aGlzLnBvb2xTaXplWzFdLCB0aGlzLnBhZGRpbmcsIHRoaXMuc3RyaWRlc1sxXSk7XG4gICAgY29scyA9XG4gICAgICAgIGNvbnZPdXRwdXRMZW5ndGgoY29scywgdGhpcy5wb29sU2l6ZVsyXSwgdGhpcy5wYWRkaW5nLCB0aGlzLnN0cmlkZXNbMl0pO1xuICAgIGlmICh0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgICAgcmV0dXJuIFtpbnB1dFNoYXBlWzBdLCBpbnB1dFNoYXBlWzFdLCBkZXB0aHMsIHJvd3MsIGNvbHNdO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gW2lucHV0U2hhcGVbMF0sIGRlcHRocywgcm93cywgY29scywgaW5wdXRTaGFwZVs0XV07XG4gICAgfVxuICB9XG5cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBwYWRkaW5nOiBQYWRkaW5nTW9kZSxcbiAgICAgIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQpOiBUZW5zb3I7XG5cbiAgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIHRoaXMuaW52b2tlQ2FsbEhvb2soaW5wdXRzLCBrd2FyZ3MpO1xuICAgICAgcmV0dXJuIHRoaXMucG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgICAgIGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKSwgdGhpcy5wb29sU2l6ZSwgdGhpcy5zdHJpZGVzLFxuICAgICAgICAgIHRoaXMucGFkZGluZywgdGhpcy5kYXRhRm9ybWF0KTtcbiAgICB9KTtcbiAgfVxuXG4gIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGNvbmZpZyA9IHtcbiAgICAgIHBvb2xTaXplOiB0aGlzLnBvb2xTaXplLFxuICAgICAgcGFkZGluZzogdGhpcy5wYWRkaW5nLFxuICAgICAgc3RyaWRlczogdGhpcy5zdHJpZGVzLFxuICAgICAgZGF0YUZvcm1hdDogdGhpcy5kYXRhRm9ybWF0XG4gICAgfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIE1heFBvb2xpbmczRCBleHRlbmRzIFBvb2xpbmczRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ01heFBvb2xpbmczRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmczRExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBwYWRkaW5nOiBQYWRkaW5nTW9kZSxcbiAgICAgIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQpOiBUZW5zb3Ige1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICBjaGVja1BhZGRpbmdNb2RlKHBhZGRpbmcpO1xuICAgIHJldHVybiBwb29sM2QoXG4gICAgICAgIGlucHV0cyBhcyBUZW5zb3I1RCwgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsICdtYXgnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE1heFBvb2xpbmczRCk7XG5cbmV4cG9ydCBjbGFzcyBBdmVyYWdlUG9vbGluZzNEIGV4dGVuZHMgUG9vbGluZzNEIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnQXZlcmFnZVBvb2xpbmczRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmczRExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBwYWRkaW5nOiBQYWRkaW5nTW9kZSxcbiAgICAgIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQpOiBUZW5zb3Ige1xuICAgIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgICBjaGVja1BhZGRpbmdNb2RlKHBhZGRpbmcpO1xuICAgIHJldHVybiBwb29sM2QoXG4gICAgICAgIGlucHV0cyBhcyBUZW5zb3I1RCwgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsICdhdmcnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEF2ZXJhZ2VQb29saW5nM0QpO1xuXG4vKipcbiAqIEFic3RyYWN0IGNsYXNzIGZvciBkaWZmZXJlbnQgZ2xvYmFsIHBvb2xpbmcgMUQgbGF5ZXJzLlxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgR2xvYmFsUG9vbGluZzFEIGV4dGVuZHMgTGF5ZXIge1xuICBjb25zdHJ1Y3RvcihhcmdzOiBMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgICB0aGlzLmlucHV0U3BlYyA9IFtuZXcgSW5wdXRTcGVjKHtuZGltOiAzfSldO1xuICB9XG5cbiAgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlKTogU2hhcGUge1xuICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVsyXV07XG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoKTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgR2xvYmFsQXZlcmFnZVBvb2xpbmcxRCBleHRlbmRzIEdsb2JhbFBvb2xpbmcxRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0dsb2JhbEF2ZXJhZ2VQb29saW5nMUQnO1xuICBjb25zdHJ1Y3RvcihhcmdzPzogTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyB8fCB7fSk7XG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICByZXR1cm4gdGZjLm1lYW4oaW5wdXQsIDEpO1xuICAgIH0pO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoR2xvYmFsQXZlcmFnZVBvb2xpbmcxRCk7XG5cbmV4cG9ydCBjbGFzcyBHbG9iYWxNYXhQb29saW5nMUQgZXh0ZW5kcyBHbG9iYWxQb29saW5nMUQge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdHbG9iYWxNYXhQb29saW5nMUQnO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzIHx8IHt9KTtcbiAgfVxuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBpbnB1dCA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKTtcbiAgICAgIHJldHVybiB0ZmMubWF4KGlucHV0LCAxKTtcbiAgICB9KTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEdsb2JhbE1heFBvb2xpbmcxRCk7XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBHbG9iYWxQb29saW5nMkRMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogT25lIG9mIGBDSEFOTkVMX0xBU1RgIChkZWZhdWx0KSBvciBgQ0hBTk5FTF9GSVJTVGAuXG4gICAqXG4gICAqIFRoZSBvcmRlcmluZyBvZiB0aGUgZGltZW5zaW9ucyBpbiB0aGUgaW5wdXRzLiBgQ0hBTk5FTF9MQVNUYCBjb3JyZXNwb25kc1xuICAgKiB0byBpbnB1dHMgd2l0aCBzaGFwZSBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc1tgIHdoaWxlXG4gICAqIGBDSEFOTkVMX0ZJUlNUYCBjb3JyZXNwb25kcyB0byBpbnB1dHMgd2l0aCBzaGFwZVxuICAgKiBgW2JhdGNoLCBjaGFubmVscywgaGVpZ2h0LCB3aWR0aF1gLlxuICAgKi9cbiAgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQ7XG59XG5cbi8qKlxuICogQWJzdHJhY3QgY2xhc3MgZm9yIGRpZmZlcmVudCBnbG9iYWwgcG9vbGluZyAyRCBsYXllcnMuXG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBHbG9iYWxQb29saW5nMkQgZXh0ZW5kcyBMYXllciB7XG4gIHByb3RlY3RlZCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0O1xuICBjb25zdHJ1Y3RvcihhcmdzOiBHbG9iYWxQb29saW5nMkRMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgICB0aGlzLmRhdGFGb3JtYXQgPVxuICAgICAgICBhcmdzLmRhdGFGb3JtYXQgPT0gbnVsbCA/ICdjaGFubmVsc0xhc3QnIDogYXJncy5kYXRhRm9ybWF0O1xuICAgIGNoZWNrRGF0YUZvcm1hdCh0aGlzLmRhdGFGb3JtYXQpO1xuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDR9KV07XG4gIH1cblxuICBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlucHV0U2hhcGUgPSBpbnB1dFNoYXBlIGFzIFNoYXBlO1xuICAgIGlmICh0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0xhc3QnKSB7XG4gICAgICByZXR1cm4gW2lucHV0U2hhcGVbMF0sIGlucHV0U2hhcGVbM11dO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gW2lucHV0U2hhcGVbMF0sIGlucHV0U2hhcGVbMV1dO1xuICAgIH1cbiAgfVxuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcigpO1xuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnID0ge2RhdGFGb3JtYXQ6IHRoaXMuZGF0YUZvcm1hdH07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBHbG9iYWxBdmVyYWdlUG9vbGluZzJEIGV4dGVuZHMgR2xvYmFsUG9vbGluZzJEIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnR2xvYmFsQXZlcmFnZVBvb2xpbmcyRCc7XG5cbiAgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IGlucHV0ID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICAgICAgcmV0dXJuIHRmYy5tZWFuKGlucHV0LCBbMSwgMl0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIHRmYy5tZWFuKGlucHV0LCBbMiwgM10pO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoR2xvYmFsQXZlcmFnZVBvb2xpbmcyRCk7XG5cbmV4cG9ydCBjbGFzcyBHbG9iYWxNYXhQb29saW5nMkQgZXh0ZW5kcyBHbG9iYWxQb29saW5nMkQge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdHbG9iYWxNYXhQb29saW5nMkQnO1xuXG4gIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBpbnB1dCA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKTtcbiAgICAgIGlmICh0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0xhc3QnKSB7XG4gICAgICAgIHJldHVybiB0ZmMubWF4KGlucHV0LCBbMSwgMl0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIHRmYy5tYXgoaW5wdXQsIFsyLCAzXSk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhHbG9iYWxNYXhQb29saW5nMkQpO1xuIl19
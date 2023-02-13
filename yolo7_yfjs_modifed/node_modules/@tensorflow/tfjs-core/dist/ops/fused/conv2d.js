/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { ENGINE } from '../../engine';
import { customGrad } from '../../gradients';
import { FusedConv2D } from '../../kernel_names';
import { makeTypesMatch } from '../../tensor_util';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { add } from '../add';
import * as broadcast_util from '../broadcast_util';
import { conv2d as unfusedConv2d } from '../conv2d';
import { conv2DBackpropFilter } from '../conv2d_backprop_filter';
import { conv2DBackpropInput } from '../conv2d_backprop_input';
import * as conv_util from '../conv_util';
import { applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse } from '../fused_util';
import { op } from '../operation';
import { reshape } from '../reshape';
/**
 * Computes a 2D convolution over the input x, optionally fused with adding a
 * bias and applying an activation.
 *
 * ```js
 * const inputDepth = 2;
 * const inShape = [2, 2, 2, inputDepth];
 * const outputDepth = 2;
 * const fSize = 1;
 * const pad = 0;
 * const strides = 1;
 *
 * const x = tf.tensor4d( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
 * 16], inShape);
 * const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth,
 * outputDepth]);
 *
 * tf.fused.conv2d({ x, filter: w, strides, pad, dataFormat: 'NHWC',
 * dilations: [1, 1], bias: tf.scalar(5), activation: 'relu' }).print();
 * ```
 *
 * @param obj An object with the following properties:
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid` output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`) to be
 *     applied
 *      after biasAdd.
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 * @param leakyreluAlpha Optional. Alpha to be applied as part of a `leakyrelu`
 *     activation.
 */
function fusedConv2d_({ x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode, bias, activation = 'linear', preluActivationWeights, leakyreluAlpha }) {
    activation = activation || 'linear';
    if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
        let result = unfusedConv2d(x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
        if (bias != null) {
            result = add(result, bias);
        }
        return applyActivation(result, activation, preluActivationWeights, leakyreluAlpha);
    }
    const $x = convertToTensor(x, 'x', 'conv2d', 'float32');
    const $filter = convertToTensor(filter, 'filter', 'conv2d', 'float32');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    util.assert(x4D.rank === 4, () => `Error in fused conv2d: input must be rank 4, but got rank ` +
        `${x4D.rank}.`);
    util.assert($filter.rank === 4, () => `Error in fused conv2d: filter must be rank 4, but got rank ` +
        `${$filter.rank}.`);
    conv_util.checkPadOnDimRoundingMode('fused conv2d', pad, dimRoundingMode);
    util.assert(x4D.shape[3] === $filter.shape[2], () => `Error in conv2d: depth of input (${x4D.shape[3]}) must match ` +
        `input depth for filter ${$filter.shape[2]}.`);
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in conv2D: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    util.assert(dataFormat === 'NHWC', () => `Error in conv2d: got dataFormat of ${dataFormat} but only NHWC is currently supported.`);
    const convInfo = conv_util.computeConv2DInfo(x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode);
    let $bias;
    if (bias != null) {
        $bias = convertToTensor(bias, 'bias', 'fused conv2d');
        [$bias] = makeTypesMatch($bias, $x);
        broadcast_util.assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
    }
    let $preluActivationWeights;
    if (preluActivationWeights != null) {
        $preluActivationWeights = convertToTensor(preluActivationWeights, 'prelu weights', 'fused conv2d');
    }
    const grad = (dy, saved) => {
        const [$filter, x4D, y, $bias] = saved;
        const dyActivation = getFusedDyActivation(dy, y, activation);
        util.assert(conv_util.tupleValuesAreOne(dilations), () => 'Error in gradient of fused conv2D: ' +
            `dilation rates greater than 1 ` +
            `are not yet supported in gradients. Got dilations '${dilations}'`);
        const xDer = conv2DBackpropInput(x4D.shape, dyActivation, $filter, strides, pad);
        const filterDer = conv2DBackpropFilter(x4D, dyActivation, $filter.shape, strides, pad);
        const der = [xDer, filterDer];
        if ($bias != null) {
            const biasDer = getFusedBiasGradient($bias, dyActivation);
            der.push(biasDer);
        }
        return der;
    };
    const inputs = {
        x: x4D,
        filter: $filter,
        bias: $bias,
        preluActivationWeights: $preluActivationWeights
    };
    const attrs = {
        strides,
        pad,
        dataFormat,
        dilations,
        dimRoundingMode,
        activation,
        leakyreluAlpha
    };
    // Depending on the the params passed in we will have different number of
    // inputs and thus a a different number of elements in the gradient.
    if (bias == null) {
        const customOp = customGrad((x4D, filter, save) => {
            let res = 
            // tslint:disable-next-line: no-unnecessary-type-assertion
            ENGINE.runKernel(FusedConv2D, inputs, attrs);
            save([filter, x4D, res]);
            if (reshapedTo4D) {
                // tslint:disable-next-line: no-unnecessary-type-assertion
                res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
            }
            return { value: res, gradFunc: grad };
        });
        return customOp(x4D, $filter);
    }
    else {
        const customOpWithBias = customGrad((x4D, filter, bias, save) => {
            let res = ENGINE.runKernel(FusedConv2D, inputs, attrs);
            save([filter, x4D, res, bias]);
            if (reshapedTo4D) {
                // tslint:disable-next-line: no-unnecessary-type-assertion
                res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
            }
            return { value: res, gradFunc: grad };
        });
        return customOpWithBias(x4D, $filter, $bias);
    }
}
export const conv2d = op({ fusedConv2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjJkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZnVzZWQvY29udjJkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBQzNDLE9BQU8sRUFBQyxXQUFXLEVBQXNDLE1BQU0sb0JBQW9CLENBQUM7QUFJcEYsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQ2pELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEtBQUssSUFBSSxNQUFNLFlBQVksQ0FBQztBQUNuQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sS0FBSyxjQUFjLE1BQU0sbUJBQW1CLENBQUM7QUFDcEQsT0FBTyxFQUFDLE1BQU0sSUFBSSxhQUFhLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEQsT0FBTyxFQUFDLG9CQUFvQixFQUFDLE1BQU0sMkJBQTJCLENBQUM7QUFDL0QsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0sMEJBQTBCLENBQUM7QUFDN0QsT0FBTyxLQUFLLFNBQVMsTUFBTSxjQUFjLENBQUM7QUFFMUMsT0FBTyxFQUFDLGVBQWUsRUFBRSxvQkFBb0IsRUFBRSxvQkFBb0IsRUFBRSxVQUFVLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDdEcsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRW5DOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXdERztBQUNILFNB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InNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uLy4uL2VuZ2luZSc7XG5pbXBvcnQge2N1c3RvbUdyYWR9IGZyb20gJy4uLy4uL2dyYWRpZW50cyc7XG5pbXBvcnQge0Z1c2VkQ29udjJELCBGdXNlZENvbnYyREF0dHJzLCBGdXNlZENvbnYyRElucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi8uLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3IsIFRlbnNvcjNELCBUZW5zb3I0RH0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7R3JhZFNhdmVGdW5jLCBOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7bWFrZVR5cGVzTWF0Y2h9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuaW1wb3J0IHthZGR9IGZyb20gJy4uL2FkZCc7XG5pbXBvcnQgKiBhcyBicm9hZGNhc3RfdXRpbCBmcm9tICcuLi9icm9hZGNhc3RfdXRpbCc7XG5pbXBvcnQge2NvbnYyZCBhcyB1bmZ1c2VkQ29udjJkfSBmcm9tICcuLi9jb252MmQnO1xuaW1wb3J0IHtjb252MkRCYWNrcHJvcEZpbHRlcn0gZnJvbSAnLi4vY29udjJkX2JhY2twcm9wX2ZpbHRlcic7XG5pbXBvcnQge2NvbnYyREJhY2twcm9wSW5wdXR9IGZyb20gJy4uL2NvbnYyZF9iYWNrcHJvcF9pbnB1dCc7XG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi4vY29udl91dGlsJztcbmltcG9ydCB7QWN0aXZhdGlvbn0gZnJvbSAnLi4vZnVzZWRfdHlwZXMnO1xuaW1wb3J0IHthcHBseUFjdGl2YXRpb24sIGdldEZ1c2VkQmlhc0dyYWRpZW50LCBnZXRGdXNlZER5QWN0aXZhdGlvbiwgc2hvdWxkRnVzZX0gZnJvbSAnLi4vZnVzZWRfdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuLi9yZXNoYXBlJztcblxuLyoqXG4gKiBDb21wdXRlcyBhIDJEIGNvbnZvbHV0aW9uIG92ZXIgdGhlIGlucHV0IHgsIG9wdGlvbmFsbHkgZnVzZWQgd2l0aCBhZGRpbmcgYVxuICogYmlhcyBhbmQgYXBwbHlpbmcgYW4gYWN0aXZhdGlvbi5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW5wdXREZXB0aCA9IDI7XG4gKiBjb25zdCBpblNoYXBlID0gWzIsIDIsIDIsIGlucHV0RGVwdGhdO1xuICogY29uc3Qgb3V0cHV0RGVwdGggPSAyO1xuICogY29uc3QgZlNpemUgPSAxO1xuICogY29uc3QgcGFkID0gMDtcbiAqIGNvbnN0IHN0cmlkZXMgPSAxO1xuICpcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZCggWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxMSwgMTIsIDEzLCAxNCwgMTUsXG4gKiAxNl0sIGluU2hhcGUpO1xuICogY29uc3QgdyA9IHRmLnRlbnNvcjRkKFstMSwgMSwgLTIsIDAuNV0sIFtmU2l6ZSwgZlNpemUsIGlucHV0RGVwdGgsXG4gKiBvdXRwdXREZXB0aF0pO1xuICpcbiAqIHRmLmZ1c2VkLmNvbnYyZCh7IHgsIGZpbHRlcjogdywgc3RyaWRlcywgcGFkLCBkYXRhRm9ybWF0OiAnTkhXQycsXG4gKiBkaWxhdGlvbnM6IFsxLCAxXSwgYmlhczogdGYuc2NhbGFyKDUpLCBhY3RpdmF0aW9uOiAncmVsdScgfSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBvYmogQW4gb2JqZWN0IHdpdGggdGhlIGZvbGxvd2luZyBwcm9wZXJ0aWVzOlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvciwgb2YgcmFuayA0IG9yIHJhbmsgMywgb2Ygc2hhcGVcbiAqICAgICBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXWAuIElmIHJhbmsgMywgYmF0Y2ggb2YgMSBpc1xuICogYXNzdW1lZC5cbiAqIEBwYXJhbSBmaWx0ZXIgVGhlIGZpbHRlciwgcmFuayA0LCBvZiBzaGFwZVxuICogICAgIGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgaW5EZXB0aCwgb3V0RGVwdGhdYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBjb252b2x1dGlvbjogYFtzdHJpZGVIZWlnaHQsXG4gKiBzdHJpZGVXaWR0aF1gLlxuICogQHBhcmFtIHBhZCBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobS5cbiAqICAgLSBgc2FtZWAgYW5kIHN0cmlkZSAxOiBvdXRwdXQgd2lsbCBiZSBvZiBzYW1lIHNpemUgYXMgaW5wdXQsXG4gKiAgICAgICByZWdhcmRsZXNzIG9mIGZpbHRlciBzaXplLlxuICogICAtIGB2YWxpZGAgb3V0cHV0IHdpbGwgYmUgc21hbGxlciB0aGFuIGlucHV0IGlmIGZpbHRlciBpcyBsYXJnZXJcbiAqICAgICAgIHRoYW4gMXgxLlxuICogICAtIEZvciBtb3JlIGluZm8sIHNlZSB0aGlzIGd1aWRlOlxuICogICAgIFtodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb25dKFxuICogICAgICAgICAgaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uKVxuICogQHBhcmFtIGRhdGFGb3JtYXQgQW4gb3B0aW9uYWwgc3RyaW5nIGZyb206IFwiTkhXQ1wiLCBcIk5DSFdcIi4gRGVmYXVsdHMgdG9cbiAqICAgICBcIk5IV0NcIi4gU3BlY2lmeSB0aGUgZGF0YSBmb3JtYXQgb2YgdGhlIGlucHV0IGFuZCBvdXRwdXQgZGF0YS4gV2l0aCB0aGVcbiAqICAgICBkZWZhdWx0IGZvcm1hdCBcIk5IV0NcIiwgdGhlIGRhdGEgaXMgc3RvcmVkIGluIHRoZSBvcmRlciBvZjogW2JhdGNoLFxuICogICAgIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXS4gT25seSBcIk5IV0NcIiBpcyBjdXJyZW50bHkgc3VwcG9ydGVkLlxuICogQHBhcmFtIGRpbGF0aW9ucyBUaGUgZGlsYXRpb24gcmF0ZXM6IGBbZGlsYXRpb25IZWlnaHQsIGRpbGF0aW9uV2lkdGhdYFxuICogICAgIGluIHdoaWNoIHdlIHNhbXBsZSBpbnB1dCB2YWx1ZXMgYWNyb3NzIHRoZSBoZWlnaHQgYW5kIHdpZHRoIGRpbWVuc2lvbnNcbiAqICAgICBpbiBhdHJvdXMgY29udm9sdXRpb24uIERlZmF1bHRzIHRvIGBbMSwgMV1gLiBJZiBgZGlsYXRpb25zYCBpcyBhIHNpbmdsZVxuICogICAgIG51bWJlciwgdGhlbiBgZGlsYXRpb25IZWlnaHQgPT0gZGlsYXRpb25XaWR0aGAuIElmIGl0IGlzIGdyZWF0ZXIgdGhhblxuICogICAgIDEsIHRoZW4gYWxsIHZhbHVlcyBvZiBgc3RyaWRlc2AgbXVzdCBiZSAxLlxuICogQHBhcmFtIGRpbVJvdW5kaW5nTW9kZSBBIHN0cmluZyBmcm9tOiAnY2VpbCcsICdyb3VuZCcsICdmbG9vcicuIElmIG5vbmUgaXNcbiAqICAgICBwcm92aWRlZCwgaXQgd2lsbCBkZWZhdWx0IHRvIHRydW5jYXRlLlxuICogQHBhcmFtIGJpYXMgVGVuc29yIHRvIGJlIGFkZGVkIHRvIHRoZSByZXN1bHQuXG4gKiBAcGFyYW0gYWN0aXZhdGlvbiBOYW1lIG9mIGFjdGl2YXRpb24ga2VybmVsIChkZWZhdWx0cyB0byBgbGluZWFyYCkgdG8gYmVcbiAqICAgICBhcHBsaWVkXG4gKiAgICAgIGFmdGVyIGJpYXNBZGQuXG4gKiBAcGFyYW0gcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyBUZW5zb3Igb2YgcHJlbHUgd2VpZ2h0cyB0byBiZSBhcHBsaWVkIGFzIHBhcnRcbiAqICAgICBvZiBhIGBwcmVsdWAgYWN0aXZhdGlvbiwgdHlwaWNhbGx5IHRoZSBzYW1lIHNoYXBlIGFzIGB4YC5cbiAqIEBwYXJhbSBsZWFreXJlbHVBbHBoYSBPcHRpb25hbC4gQWxwaGEgdG8gYmUgYXBwbGllZCBhcyBwYXJ0IG9mIGEgYGxlYWt5cmVsdWBcbiAqICAgICBhY3RpdmF0aW9uLlxuICovXG5mdW5jdGlvbiBmdXNlZENvbnYyZF88VCBleHRlbmRzIFRlbnNvcjNEfFRlbnNvcjREPih7XG4gIHgsXG4gIGZpbHRlcixcbiAgc3RyaWRlcyxcbiAgcGFkLFxuICBkYXRhRm9ybWF0ID0gJ05IV0MnLFxuICBkaWxhdGlvbnMgPSBbMSwgMV0sXG4gIGRpbVJvdW5kaW5nTW9kZSxcbiAgYmlhcyxcbiAgYWN0aXZhdGlvbiA9ICdsaW5lYXInLFxuICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLFxuICBsZWFreXJlbHVBbHBoYVxufToge1xuICB4OiBUfFRlbnNvckxpa2UsXG4gIGZpbHRlcjogVGVuc29yNER8VGVuc29yTGlrZSxcbiAgc3RyaWRlczogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyfGNvbnZfdXRpbC5FeHBsaWNpdFBhZGRpbmcsXG4gIGRhdGFGb3JtYXQ/OiAnTkhXQyd8J05DSFcnLFxuICBkaWxhdGlvbnM/OiBbbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgZGltUm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJyxcbiAgYmlhcz86IFRlbnNvcnxUZW5zb3JMaWtlLFxuICBhY3RpdmF0aW9uPzogQWN0aXZhdGlvbixcbiAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cz86IFRlbnNvcixcbiAgbGVha3lyZWx1QWxwaGE/OiBudW1iZXJcbn0pOiBUIHtcbiAgYWN0aXZhdGlvbiA9IGFjdGl2YXRpb24gfHwgJ2xpbmVhcic7XG5cbiAgaWYgKHNob3VsZEZ1c2UoRU5HSU5FLnN0YXRlLmdyYWRpZW50RGVwdGgsIGFjdGl2YXRpb24pID09PSBmYWxzZSkge1xuICAgIGxldCByZXN1bHQgPSB1bmZ1c2VkQ29udjJkKFxuICAgICAgICB4LCBmaWx0ZXIsIHN0cmlkZXMsIHBhZCwgZGF0YUZvcm1hdCwgZGlsYXRpb25zLCBkaW1Sb3VuZGluZ01vZGUpO1xuICAgIGlmIChiaWFzICE9IG51bGwpIHtcbiAgICAgIHJlc3VsdCA9IGFkZChyZXN1bHQsIGJpYXMpO1xuICAgIH1cblxuICAgIHJldHVybiBhcHBseUFjdGl2YXRpb24oXG4gICAgICAgICAgICAgICByZXN1bHQsIGFjdGl2YXRpb24sIHByZWx1QWN0aXZhdGlvbldlaWdodHMsIGxlYWt5cmVsdUFscGhhKSBhcyBUO1xuICB9XG5cbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnY29udjJkJywgJ2Zsb2F0MzInKTtcbiAgY29uc3QgJGZpbHRlciA9IGNvbnZlcnRUb1RlbnNvcihmaWx0ZXIsICdmaWx0ZXInLCAnY29udjJkJywgJ2Zsb2F0MzInKTtcblxuICBsZXQgeDREID0gJHggYXMgVGVuc29yNEQ7XG4gIGxldCByZXNoYXBlZFRvNEQgPSBmYWxzZTtcblxuICBpZiAoJHgucmFuayA9PT0gMykge1xuICAgIHJlc2hhcGVkVG80RCA9IHRydWU7XG4gICAgeDREID0gcmVzaGFwZSgkeCwgWzEsICR4LnNoYXBlWzBdLCAkeC5zaGFwZVsxXSwgJHguc2hhcGVbMl1dKTtcbiAgfVxuICB1dGlsLmFzc2VydChcbiAgICAgIHg0RC5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGZ1c2VkIGNvbnYyZDogaW5wdXQgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICBgJHt4NEQucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGZpbHRlci5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGZ1c2VkIGNvbnYyZDogZmlsdGVyIG11c3QgYmUgcmFuayA0LCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgYCR7JGZpbHRlci5yYW5rfS5gKTtcbiAgY29udl91dGlsLmNoZWNrUGFkT25EaW1Sb3VuZGluZ01vZGUoJ2Z1c2VkIGNvbnYyZCcsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4NEQuc2hhcGVbM10gPT09ICRmaWx0ZXIuc2hhcGVbMl0sXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjJkOiBkZXB0aCBvZiBpbnB1dCAoJHt4NEQuc2hhcGVbM119KSBtdXN0IG1hdGNoIGAgK1xuICAgICAgICAgIGBpbnB1dCBkZXB0aCBmb3IgZmlsdGVyICR7JGZpbHRlci5zaGFwZVsyXX0uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgY29udl91dGlsLmVpdGhlclN0cmlkZXNPckRpbGF0aW9uc0FyZU9uZShzdHJpZGVzLCBkaWxhdGlvbnMpLFxuICAgICAgKCkgPT4gJ0Vycm9yIGluIGNvbnYyRDogRWl0aGVyIHN0cmlkZXMgb3IgZGlsYXRpb25zIG11c3QgYmUgMS4gJyArXG4gICAgICAgICAgYEdvdCBzdHJpZGVzICR7c3RyaWRlc30gYW5kIGRpbGF0aW9ucyAnJHtkaWxhdGlvbnN9J2ApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGRhdGFGb3JtYXQgPT09ICdOSFdDJyxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252MmQ6IGdvdCBkYXRhRm9ybWF0IG9mICR7XG4gICAgICAgICAgZGF0YUZvcm1hdH0gYnV0IG9ubHkgTkhXQyBpcyBjdXJyZW50bHkgc3VwcG9ydGVkLmApO1xuXG4gIGNvbnN0IGNvbnZJbmZvID0gY29udl91dGlsLmNvbXB1dGVDb252MkRJbmZvKFxuICAgICAgeDRELnNoYXBlLCAkZmlsdGVyLnNoYXBlLCBzdHJpZGVzLCBkaWxhdGlvbnMsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcblxuICBsZXQgJGJpYXM6IFRlbnNvcjtcbiAgaWYgKGJpYXMgIT0gbnVsbCkge1xuICAgICRiaWFzID0gY29udmVydFRvVGVuc29yKGJpYXMsICdiaWFzJywgJ2Z1c2VkIGNvbnYyZCcpO1xuICAgIFskYmlhc10gPSBtYWtlVHlwZXNNYXRjaCgkYmlhcywgJHgpO1xuXG4gICAgYnJvYWRjYXN0X3V0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0U2hhcGUoY29udkluZm8ub3V0U2hhcGUsICRiaWFzLnNoYXBlKTtcbiAgfVxuXG4gIGxldCAkcHJlbHVBY3RpdmF0aW9uV2VpZ2h0czogVGVuc29yO1xuICBpZiAocHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyAhPSBudWxsKSB7XG4gICAgJHByZWx1QWN0aXZhdGlvbldlaWdodHMgPSBjb252ZXJ0VG9UZW5zb3IoXG4gICAgICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHMsICdwcmVsdSB3ZWlnaHRzJywgJ2Z1c2VkIGNvbnYyZCcpO1xuICB9XG5cbiAgY29uc3QgZ3JhZCA9IChkeTogVGVuc29yNEQsIHNhdmVkOiBUZW5zb3JbXSkgPT4ge1xuICAgIGNvbnN0IFskZmlsdGVyLCB4NEQsIHksICRiaWFzXSA9XG4gICAgICAgIHNhdmVkIGFzIFtUZW5zb3I0RCwgVGVuc29yNEQsIFRlbnNvcjRELCBUZW5zb3JdO1xuXG4gICAgY29uc3QgZHlBY3RpdmF0aW9uID0gZ2V0RnVzZWREeUFjdGl2YXRpb24oZHksIHksIGFjdGl2YXRpb24pIGFzIFRlbnNvcjREO1xuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGNvbnZfdXRpbC50dXBsZVZhbHVlc0FyZU9uZShkaWxhdGlvbnMpLFxuICAgICAgICAoKSA9PiAnRXJyb3IgaW4gZ3JhZGllbnQgb2YgZnVzZWQgY29udjJEOiAnICtcbiAgICAgICAgICAgIGBkaWxhdGlvbiByYXRlcyBncmVhdGVyIHRoYW4gMSBgICtcbiAgICAgICAgICAgIGBhcmUgbm90IHlldCBzdXBwb3J0ZWQgaW4gZ3JhZGllbnRzLiBHb3QgZGlsYXRpb25zICcke2RpbGF0aW9uc30nYCk7XG5cbiAgICBjb25zdCB4RGVyID1cbiAgICAgICAgY29udjJEQmFja3Byb3BJbnB1dCh4NEQuc2hhcGUsIGR5QWN0aXZhdGlvbiwgJGZpbHRlciwgc3RyaWRlcywgcGFkKTtcbiAgICBjb25zdCBmaWx0ZXJEZXIgPVxuICAgICAgICBjb252MkRCYWNrcHJvcEZpbHRlcih4NEQsIGR5QWN0aXZhdGlvbiwgJGZpbHRlci5zaGFwZSwgc3RyaWRlcywgcGFkKTtcbiAgICBjb25zdCBkZXI6IFRlbnNvcltdID0gW3hEZXIsIGZpbHRlckRlcl07XG5cbiAgICBpZiAoJGJpYXMgIT0gbnVsbCkge1xuICAgICAgY29uc3QgYmlhc0RlciA9IGdldEZ1c2VkQmlhc0dyYWRpZW50KCRiaWFzLCBkeUFjdGl2YXRpb24pO1xuICAgICAgZGVyLnB1c2goYmlhc0Rlcik7XG4gICAgfVxuICAgIHJldHVybiBkZXI7XG4gIH07XG5cbiAgY29uc3QgaW5wdXRzOiBGdXNlZENvbnYyRElucHV0cyA9IHtcbiAgICB4OiB4NEQsXG4gICAgZmlsdGVyOiAkZmlsdGVyLFxuICAgIGJpYXM6ICRiaWFzLFxuICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHM6ICRwcmVsdUFjdGl2YXRpb25XZWlnaHRzXG4gIH07XG5cbiAgY29uc3QgYXR0cnM6IEZ1c2VkQ29udjJEQXR0cnMgPSB7XG4gICAgc3RyaWRlcyxcbiAgICBwYWQsXG4gICAgZGF0YUZvcm1hdCxcbiAgICBkaWxhdGlvbnMsXG4gICAgZGltUm91bmRpbmdNb2RlLFxuICAgIGFjdGl2YXRpb24sXG4gICAgbGVha3lyZWx1QWxwaGFcbiAgfTtcblxuICAvLyBEZXBlbmRpbmcgb24gdGhlIHRoZSBwYXJhbXMgcGFzc2VkIGluIHdlIHdpbGwgaGF2ZSBkaWZmZXJlbnQgbnVtYmVyIG9mXG4gIC8vIGlucHV0cyBhbmQgdGh1cyBhIGEgZGlmZmVyZW50IG51bWJlciBvZiBlbGVtZW50cyBpbiB0aGUgZ3JhZGllbnQuXG4gIGlmIChiaWFzID09IG51bGwpIHtcbiAgICBjb25zdCBjdXN0b21PcCA9XG4gICAgICAgIGN1c3RvbUdyYWQoKHg0RDogVGVuc29yNEQsIGZpbHRlcjogVGVuc29yNEQsIHNhdmU6IEdyYWRTYXZlRnVuYykgPT4ge1xuICAgICAgICAgIGxldCByZXM6IFRlbnNvcjREfFRlbnNvcjNEID1cbiAgICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICAgICAgICAgICAgICBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgICAgICAgRnVzZWRDb252MkQsIGlucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICAgICAgICAgIGF0dHJzIGFzIHt9IGFzIE5hbWVkQXR0ck1hcCk7XG5cbiAgICAgICAgICBzYXZlKFtmaWx0ZXIsIHg0RCwgcmVzXSk7XG5cbiAgICAgICAgICBpZiAocmVzaGFwZWRUbzREKSB7XG4gICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICAgICAgICByZXMgPSByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhc1xuICAgICAgICAgICAgICAgIFRlbnNvcjNEO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIHJldHVybiB7dmFsdWU6IHJlcywgZ3JhZEZ1bmM6IGdyYWR9O1xuICAgICAgICB9KTtcbiAgICByZXR1cm4gY3VzdG9tT3AoeDRELCAkZmlsdGVyKSBhcyBUO1xuICB9IGVsc2Uge1xuICAgIGNvbnN0IGN1c3RvbU9wV2l0aEJpYXMgPSBjdXN0b21HcmFkKFxuICAgICAgICAoeDREOiBUZW5zb3I0RCwgZmlsdGVyOiBUZW5zb3I0RCwgYmlhczogVGVuc29yLCBzYXZlOiBHcmFkU2F2ZUZ1bmMpID0+IHtcbiAgICAgICAgICBsZXQgcmVzOiBUZW5zb3I0RHxUZW5zb3IzRCA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgIEZ1c2VkQ29udjJELCBpbnB1dHMgYXMge30gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgIGF0dHJzIGFzIHt9IGFzIE5hbWVkQXR0ck1hcCk7XG5cbiAgICAgICAgICBzYXZlKFtmaWx0ZXIsIHg0RCwgcmVzLCBiaWFzXSk7XG5cbiAgICAgICAgICBpZiAocmVzaGFwZWRUbzREKSB7XG4gICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICAgICAgICByZXMgPSByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhc1xuICAgICAgICAgICAgICAgIFRlbnNvcjNEO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIHJldHVybiB7dmFsdWU6IHJlcywgZ3JhZEZ1bmM6IGdyYWR9O1xuICAgICAgICB9KTtcblxuICAgIHJldHVybiBjdXN0b21PcFdpdGhCaWFzKHg0RCwgJGZpbHRlciwgJGJpYXMpIGFzIFQ7XG4gIH1cbn1cbmV4cG9ydCBjb25zdCBjb252MmQgPSBvcCh7ZnVzZWRDb252MmRffSk7XG4iXX0=
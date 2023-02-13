/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { util } from '@tensorflow/tfjs-core';
import { Im2ColPackedProgram } from '../im2col_packed_gpu';
import { mapActivationToShaderProgram } from '../kernel_utils/kernel_funcs_utils';
import { MatMulPackedProgram } from '../mulmat_packed_gpu';
import * as webgl_util from '../webgl_util';
import { batchMatMulImpl, MATMUL_SHARED_DIM_THRESHOLD } from './BatchMatMul_impl';
import { identity } from './Identity';
import { reshape } from './Reshape';
// For 1x1 kernels that iterate through every point in the input, convolution
// can be expressed as matrix multiplication (without need for memory
// remapping).
export function conv2dByMatMul({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    // Reshapes conv2D input to 2D tensors, uses matMul and then reshape the
    // result from 2D to 4D.
    const xShape = x.shape;
    const xTexData = backend.texData.get(x.dataId);
    const sharedMatMulDim = convInfo.inChannels;
    const outerShapeX = xShape[0] * xShape[1] * xShape[2];
    const outerShapeFilter = convInfo.outChannels;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const transposeA = false;
    const transposeB = false;
    let out;
    const intermediates = [];
    // TODO: Once reduction ops are packed, batchMatMul will always be packed
    // and we can remove this condition.
    const batchMatMulWillBeUnpacked = (outerShapeX === 1 || outerShapeFilter === 1) &&
        sharedMatMulDim > MATMUL_SHARED_DIM_THRESHOLD;
    // The algorithm in the if condition assumes (1) the output will be packed,
    // (2) x is packed, (3) x isChannelsLast, (4)  x's packed texture is already
    // on GPU, (5) col is odd, (6) the width, height and inChannels are the same
    // for xTexData.shape and xShape.
    const canOptimize = !batchMatMulWillBeUnpacked && xTexData.isPacked &&
        isChannelsLast && xTexData.texture != null && xShape[2] % 2 !== 0 &&
        util.arraysEqual(xTexData.shape.slice(-3), xShape.slice(-3));
    if (canOptimize) {
        // We avoid expensive packed 2x2 reshape by padding col count to next,
        // even number. When col is odd, the result of packed batchMatMul is
        // the same (has the same texture layout and and values in the texture) as
        // it is for next even col. We make the odd-cols tensor to look like
        // even-cols tensor before the operation and, after the batchMatMul,
        // fix the even-cols result to have odd number of cols.
        const targetShape = xShape[0] * xShape[1] * (xShape[2] + 1);
        const xReshaped = {
            dataId: x.dataId,
            shape: [1, targetShape, convInfo.inChannels],
            dtype: x.dtype
        };
        // xTexData.shape gets referenced from GPGPUBinary.inShapeInfos.
        // Decrementing col count, after batchMatMul->...->compileProgram leads to
        // invalid col count within the reference in GPGPUBinary.inShapeInfos.
        // Alternative fix would be to provide a copy to GPGPUBinary.inShapeInfos
        // in compileProgram method, but that would affect compilation of all
        // programs - instead, provide a copy here, with even col count, before
        // calling batchMatMul->...->compileProgram and after that, the original
        // xTexData.shape is restored.
        const originalXTexDataShape = xTexData.shape;
        xTexData.shape = xTexData.shape.slice();
        xTexData.shape[xTexData.shape.length - 2]++;
        util.assert(webgl_util.isReshapeFree(xTexData.shape, xReshaped.shape), () => `packed reshape ${xTexData.shape} to ${xReshaped.shape} isn't free`);
        const filterReshaped = reshape({
            inputs: { x: filter },
            backend,
            attrs: { shape: [1, convInfo.inChannels, convInfo.outChannels] }
        });
        intermediates.push(filterReshaped);
        const pointwiseConv = batchMatMulImpl({
            a: xReshaped,
            b: filterReshaped,
            backend,
            transposeA,
            transposeB,
            bias,
            activation,
            preluActivationWeights,
            leakyreluAlpha
        });
        const pointwiseConvTexData = backend.texData.get(pointwiseConv.dataId);
        util.assert(pointwiseConvTexData.isPacked, () => 'batchMatMul result is expected to be packed');
        // Restore the input shape to original.
        xTexData.shape = originalXTexDataShape;
        // Set the output shape - there is no need for expensive reshape as data
        // layout is already correct.
        pointwiseConvTexData.shape = convInfo.outShape;
        out = identity({ inputs: { x: pointwiseConv }, backend });
        out.shape = convInfo.outShape;
        intermediates.push(pointwiseConv);
    }
    else {
        const targetShape = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
            xShape[0] * xShape[2] * xShape[3];
        const xReshaped = reshape({
            inputs: { x },
            backend,
            attrs: { shape: [1, targetShape, convInfo.inChannels] }
        });
        const filterReshaped = reshape({
            inputs: { x: filter },
            backend,
            attrs: { shape: [1, convInfo.inChannels, convInfo.outChannels] }
        });
        const result = batchMatMulImpl({
            a: xReshaped,
            b: filterReshaped,
            transposeA,
            transposeB,
            backend,
            bias,
            activation,
            preluActivationWeights,
            leakyreluAlpha
        });
        out = reshape({ inputs: { x: result }, backend, attrs: { shape: convInfo.outShape } });
        intermediates.push(xReshaped);
        intermediates.push(filterReshaped);
        intermediates.push(result);
    }
    for (const i of intermediates) {
        backend.disposeIntermediateTensorInfo(i);
    }
    return out;
}
// Implements the im2row algorithm as outlined in "High Performance
// Convolutional Neural Networks for Document Processing" (Suvisoft, 2006)
export function conv2dWithIm2Row({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    // Rearranges conv2d input so each block to be convolved over forms the
    // column of a new matrix with shape [filterWidth * filterHeight *
    // inChannels, outHeight * outWidth]. The filter is also rearranged so each
    // output channel forms a row of a new matrix with shape [outChannels,
    // filterWidth * filterHeight * inChannels]. The convolution is then
    // computed by multiplying these matrices and reshaping the result.
    const { filterWidth, filterHeight, inChannels, outWidth, outHeight, dataFormat } = convInfo;
    const isChannelsLast = dataFormat === 'channelsLast';
    const sharedDim = filterWidth * filterHeight * inChannels;
    const numCols = outHeight * outWidth;
    const x2ColShape = [sharedDim, numCols];
    const transposeA = true;
    const transposeB = false;
    const intermediates = [];
    const xSqueezed = reshape({ inputs: { x }, backend, attrs: { shape: x.shape.slice(1) } });
    const w2Row = reshape({
        inputs: { x: filter },
        backend,
        attrs: { shape: [1, sharedDim, util.sizeFromShape(filter.shape) / sharedDim] }
    });
    intermediates.push(xSqueezed);
    intermediates.push(w2Row);
    const im2ColProgram = new Im2ColPackedProgram(x2ColShape, convInfo);
    const customValues = [
        xSqueezed.shape, [convInfo.padInfo.top, convInfo.padInfo.left],
        [convInfo.strideHeight, convInfo.strideWidth],
        [convInfo.dilationHeight, convInfo.dilationWidth], [convInfo.inChannels],
        [convInfo.filterWidth * convInfo.inChannels], [convInfo.outWidth]
    ];
    const im2Col = backend.runWebGLProgram(im2ColProgram, [xSqueezed], 'float32', customValues);
    const im2ColReshaped = reshape({
        inputs: { x: im2Col },
        backend,
        attrs: { shape: [1, x2ColShape[0], x2ColShape[1]] }
    });
    intermediates.push(im2Col);
    intermediates.push(im2ColReshaped);
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const hasLeakyreluAlpha = activation === 'leakyrelu';
    const fusedActivation = activation ? mapActivationToShaderProgram(activation, true) : null;
    const matmulProgram = new MatMulPackedProgram(im2ColReshaped.shape, w2Row.shape, [1, numCols, convInfo.outChannels], transposeA, transposeB, hasBias, fusedActivation, hasPreluActivationWeights, hasLeakyreluAlpha);
    const inputs = [im2ColReshaped, w2Row];
    if (bias) {
        inputs.push(bias);
    }
    if (hasPreluActivationWeights) {
        inputs.push(preluActivationWeights);
    }
    if (hasLeakyreluAlpha) {
        const $leakyreluAlpha = backend.makeTensorInfo([], 'float32', util.createScalarValue(leakyreluAlpha, 'float32'));
        inputs.push($leakyreluAlpha);
        intermediates.push($leakyreluAlpha);
    }
    const product = backend.runWebGLProgram(matmulProgram, inputs, 'float32');
    const outShape = isChannelsLast ?
        [1, outHeight, outWidth, convInfo.outChannels] :
        [1, convInfo.outChannels, outHeight, outWidth];
    const out = reshape({ inputs: { x: product }, backend, attrs: { shape: outShape } });
    intermediates.push(product);
    for (const i of intermediates) {
        backend.disposeIntermediateTensorInfo(i);
    }
    return out;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ29udjJEX2ltcGwuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ2wvc3JjL2tlcm5lbHMvQ29udjJEX2ltcGwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUEyQixJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUdyRSxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUN6RCxPQUFPLEVBQUMsNEJBQTRCLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUNoRixPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUN6RCxPQUFPLEtBQUssVUFBVSxNQUFNLGVBQWUsQ0FBQztBQUU1QyxPQUFPLEVBQUMsZUFBZSxFQUFFLDJCQUEyQixFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDaEYsT0FBTyxFQUFDLFFBQVEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNwQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBYWxDLDZFQUE2RTtBQUM3RSxxRUFBcUU7QUFDckUsY0FBYztBQUNkLE1BQU0sVUFBVSxjQUFjLENBQUMsRUFDN0IsQ0FBQyxFQUNELE1BQU0sRUFDTixRQUFRLEVBQ1IsT0FBTyxFQUNQLElBQUksR0FBRyxJQUFJLEVBQ1gsc0JBQXNCLEdBQUcsSUFBSSxFQUM3QixjQUFjLEdBQUcsQ0FBQyxFQUNsQixVQUFVLEdBQUcsSUFBSSxFQUNKO0lBQ2Isd0VBQXdFO0lBQ3hFLHdCQUF3QjtJQUN4QixNQUFNLE1BQU0sR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDO0lBQ3ZCLE1BQU0sUUFBUSxHQUFHLE9BQU8sQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUMvQyxNQUFNLGVBQWUsR0FBRyxRQUFRLENBQUMsVUFBVSxD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QUFDO0lBRTFFLE1BQU0sUUFBUSxHQUFHLGNBQWMsQ0FBQyxDQUFDO1FBQzdCLENBQUMsQ0FBQyxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsUUFBUSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUM7UUFDaEQsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLFdBQVcsRUFBRSxTQUFTLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDbkQsTUFBTSxHQUFHLEdBQ0wsT0FBTyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxLQUFLLEVBQUUsUUFBUSxFQUFDLEVBQUMsQ0FBQyxDQUFDO0lBRXZFLGFBQWEsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDNUIsS0FBSyxNQUFNLENBQUMsSUFBSSxhQUFhLEVBQUU7UUFDN0IsT0FBTyxDQUFDLDZCQUE2QixDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQzFDO0lBRUQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgVGVuc29ySW5mbywgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtNYXRoQmFja2VuZFdlYkdMfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdsJztcbmltcG9ydCB7SW0yQ29sUGFja2VkUHJvZ3JhbX0gZnJvbSAnLi4vaW0yY29sX3BhY2tlZF9ncHUnO1xuaW1wb3J0IHttYXBBY3RpdmF0aW9uVG9TaGFkZXJQcm9ncmFtfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMva2VybmVsX2Z1bmNzX3V0aWxzJztcbmltcG9ydCB7TWF0TXVsUGFja2VkUHJvZ3JhbX0gZnJvbSAnLi4vbXVsbWF0X3BhY2tlZF9ncHUnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuLi93ZWJnbF91dGlsJztcblxuaW1wb3J0IHtiYXRjaE1hdE11bEltcGwsIE1BVE1VTF9TSEFSRURfRElNX1RIUkVTSE9MRH0gZnJvbSAnLi9CYXRjaE1hdE11bF9pbXBsJztcbmltcG9ydCB7aWRlbnRpdHl9IGZyb20gJy4vSWRlbnRpdHknO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL1Jlc2hhcGUnO1xuXG50eXBlIENvbnYyRENvbmZpZyA9IHtcbiAgeDogVGVuc29ySW5mbyxcbiAgZmlsdGVyOiBUZW5zb3JJbmZvLFxuICBjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYyREluZm8sXG4gIGJhY2tlbmQ6IE1hdGhCYWNrZW5kV2ViR0wsXG4gIGJpYXM/OiBUZW5zb3JJbmZvLFxuICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzPzogVGVuc29ySW5mbyxcbiAgbGVha3lyZWx1QWxwaGE/OiBudW1iZXIsXG4gIGFjdGl2YXRpb24/OiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvblxufTtcblxuLy8gRm9yIDF4MSBrZXJuZWxzIHRoYXQgaXRlcmF0ZSB0aHJvdWdoIGV2ZXJ5IHBvaW50IGluIHRoZSBpbnB1dCwgY29udm9sdXRpb25cbi8vIGNhbiBiZSBleHByZXNzZWQgYXMgbWF0cml4IG11bHRpcGxpY2F0aW9uICh3aXRob3V0IG5lZWQgZm9yIG1lbW9yeVxuLy8gcmVtYXBwaW5nKS5cbmV4cG9ydCBmdW5jdGlvbiBjb252MmRCeU1hdE11bCh7XG4gIHgsXG4gIGZpbHRlcixcbiAgY29udkluZm8sXG4gIGJhY2tlbmQsXG4gIGJpYXMgPSBudWxsLFxuICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gbnVsbCxcbiAgbGVha3lyZWx1QWxwaGEgPSAwLFxuICBhY3RpdmF0aW9uID0gbnVsbFxufTogQ29udjJEQ29uZmlnKSB7XG4gIC8vIFJlc2hhcGVzIGNvbnYyRCBpbnB1dCB0byAyRCB0ZW5zb3JzLCB1c2VzIG1hdE11bCBhbmQgdGhlbiByZXNoYXBlIHRoZVxuICAvLyByZXN1bHQgZnJvbSAyRCB0byA0RC5cbiAgY29uc3QgeFNoYXBlID0geC5zaGFwZTtcbiAgY29uc3QgeFRleERhdGEgPSBiYWNrZW5kLnRleERhdGEuZ2V0KHguZGF0YUlkKTtcbiAgY29uc3Qgc2hhcmVkTWF0TXVsRGltID0gY29udkluZm8uaW5DaGFubmVscztcbiAgY29uc3Qgb3V0ZXJTaGFwZVggPSB4U2hhcGVbMF0gKiB4U2hhcGVbMV0gKiB4U2hhcGVbMl07XG4gIGNvbnN0IG91dGVyU2hhcGVGaWx0ZXIgPSBjb252SW5mby5vdXRDaGFubmVscztcbiAgY29uc3QgaXNDaGFubmVsc0xhc3QgPSBjb252SW5mby5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0JztcbiAgY29uc3QgdHJhbnNwb3NlQSA9IGZhbHNlO1xuICBjb25zdCB0cmFuc3Bvc2VCID0gZmFsc2U7XG5cbiAgbGV0IG91dDogVGVuc29ySW5mbztcbiAgY29uc3QgaW50ZXJtZWRpYXRlczogVGVuc29ySW5mb1tdID0gW107XG5cbiAgLy8gVE9ETzogT25jZSByZWR1Y3Rpb24gb3BzIGFyZSBwYWNrZWQsIGJhdGNoTWF0TXVsIHdpbGwgYWx3YXlzIGJlIHBhY2tlZFxuICAvLyBhbmQgd2UgY2FuIHJlbW92ZSB0aGlzIGNvbmRpdGlvbi5cbiAgY29uc3QgYmF0Y2hNYXRNdWxXaWxsQmVVbnBhY2tlZCA9XG4gICAgICAob3V0ZXJTaGFwZVggPT09IDEgfHwgb3V0ZXJTaGFwZUZpbHRlciA9PT0gMSkgJiZcbiAgICAgIHNoYXJlZE1hdE11bERpbSA+IE1BVE1VTF9TSEFSRURfRElNX1RIUkVTSE9MRDtcblxuICAvLyBUaGUgYWxnb3JpdGhtIGluIHRoZSBpZiBjb25kaXRpb24gYXNzdW1lcyAoMSkgdGhlIG91dHB1dCB3aWxsIGJlIHBhY2tlZCxcbiAgLy8gKDIpIHggaXMgcGFja2VkLCAoMykgeCBpc0NoYW5uZWxzTGFzdCwgKDQpICB4J3MgcGFja2VkIHRleHR1cmUgaXMgYWxyZWFkeVxuICAvLyBvbiBHUFUsICg1KSBjb2wgaXMgb2RkLCAoNikgdGhlIHdpZHRoLCBoZWlnaHQgYW5kIGluQ2hhbm5lbHMgYXJlIHRoZSBzYW1lXG4gIC8vIGZvciB4VGV4RGF0YS5zaGFwZSBhbmQgeFNoYXBlLlxuICBjb25zdCBjYW5PcHRpbWl6ZSA9ICFiYXRjaE1hdE11bFdpbGxCZVVucGFja2VkICYmIHhUZXhEYXRhLmlzUGFja2VkICYmXG4gICAgICBpc0NoYW5uZWxzTGFzdCAmJiB4VGV4RGF0YS50ZXh0dXJlICE9IG51bGwgJiYgeFNoYXBlWzJdICUgMiAhPT0gMCAmJlxuICAgICAgdXRpbC5hcnJheXNFcXVhbCh4VGV4RGF0YS5zaGFwZS5zbGljZSgtMyksIHhTaGFwZS5zbGljZSgtMykpO1xuXG4gIGlmIChjYW5PcHRpbWl6ZSkge1xuICAgIC8vIFdlIGF2b2lkIGV4cGVuc2l2ZSBwYWNrZWQgMngyIHJlc2hhcGUgYnkgcGFkZGluZyBjb2wgY291bnQgdG8gbmV4dCxcbiAgICAvLyBldmVuIG51bWJlci4gV2hlbiBjb2wgaXMgb2RkLCB0aGUgcmVzdWx0IG9mIHBhY2tlZCBiYXRjaE1hdE11bCBpc1xuICAgIC8vIHRoZSBzYW1lIChoYXMgdGhlIHNhbWUgdGV4dHVyZSBsYXlvdXQgYW5kIGFuZCB2YWx1ZXMgaW4gdGhlIHRleHR1cmUpIGFzXG4gICAgLy8gaXQgaXMgZm9yIG5leHQgZXZlbiBjb2wuIFdlIG1ha2UgdGhlIG9kZC1jb2xzIHRlbnNvciB0byBsb29rIGxpa2VcbiAgICAvLyBldmVuLWNvbHMgdGVuc29yIGJlZm9yZSB0aGUgb3BlcmF0aW9uIGFuZCwgYWZ0ZXIgdGhlIGJhdGNoTWF0TXVsLFxuICAgIC8vIGZpeCB0aGUgZXZlbi1jb2xzIHJlc3VsdCB0byBoYXZlIG9kZCBudW1iZXIgb2YgY29scy5cbiAgICBjb25zdCB0YXJnZXRTaGFwZSA9IHhTaGFwZVswXSAqIHhTaGFwZVsxXSAqICh4U2hhcGVbMl0gKyAxKTtcbiAgICBjb25zdCB4UmVzaGFwZWQ6IFRlbnNvckluZm8gPSB7XG4gICAgICBkYXRhSWQ6IHguZGF0YUlkLFxuICAgICAgc2hhcGU6IFsxLCB0YXJnZXRTaGFwZSwgY29udkluZm8uaW5DaGFubmVsc10sXG4gICAgICBkdHlwZTogeC5kdHlwZVxuICAgIH07XG4gICAgLy8geFRleERhdGEuc2hhcGUgZ2V0cyByZWZlcmVuY2VkIGZyb20gR1BHUFVCaW5hcnkuaW5TaGFwZUluZm9zLlxuICAgIC8vIERlY3JlbWVudGluZyBjb2wgY291bnQsIGFmdGVyIGJhdGNoTWF0TXVsLT4uLi4tPmNvbXBpbGVQcm9ncmFtIGxlYWRzIHRvXG4gICAgLy8gaW52YWxpZCBjb2wgY291bnQgd2l0aGluIHRoZSByZWZlcmVuY2UgaW4gR1BHUFVCaW5hcnkuaW5TaGFwZUluZm9zLlxuICAgIC8vIEFsdGVybmF0aXZlIGZpeCB3b3VsZCBiZSB0byBwcm92aWRlIGEgY29weSB0byBHUEdQVUJpbmFyeS5pblNoYXBlSW5mb3NcbiAgICAvLyBpbiBjb21waWxlUHJvZ3JhbSBtZXRob2QsIGJ1dCB0aGF0IHdvdWxkIGFmZmVjdCBjb21waWxhdGlvbiBvZiBhbGxcbiAgICAvLyBwcm9ncmFtcyAtIGluc3RlYWQsIHByb3ZpZGUgYSBjb3B5IGhlcmUsIHdpdGggZXZlbiBjb2wgY291bnQsIGJlZm9yZVxuICAgIC8vIGNhbGxpbmcgYmF0Y2hNYXRNdWwtPi4uLi0+Y29tcGlsZVByb2dyYW0gYW5kIGFmdGVyIHRoYXQsIHRoZSBvcmlnaW5hbFxuICAgIC8vIHhUZXhEYXRhLnNoYXBlIGlzIHJlc3RvcmVkLlxuICAgIGNvbnN0IG9yaWdpbmFsWFRleERhdGFTaGFwZSA9IHhUZXhEYXRhLnNoYXBlO1xuICAgIHhUZXhEYXRhLnNoYXBlID0geFRleERhdGEuc2hhcGUuc2xpY2UoKTtcbiAgICB4VGV4RGF0YS5zaGFwZVt4VGV4RGF0YS5zaGFwZS5sZW5ndGggLSAyXSsrO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB3ZWJnbF91dGlsLmlzUmVzaGFwZUZyZWUoeFRleERhdGEuc2hhcGUsIHhSZXNoYXBlZC5zaGFwZSksXG4gICAgICAgICgpID0+IGBwYWNrZWQgcmVzaGFwZSAke3hUZXhEYXRhLnNoYXBlfSB0byAke1xuICAgICAgICAgICAgeFJlc2hhcGVkLnNoYXBlfSBpc24ndCBmcmVlYCk7XG4gICAgY29uc3QgZmlsdGVyUmVzaGFwZWQgPSByZXNoYXBlKHtcbiAgICAgIGlucHV0czoge3g6IGZpbHRlcn0sXG4gICAgICBiYWNrZW5kLFxuICAgICAgYXR0cnM6IHtzaGFwZTogWzEsIGNvbnZJbmZvLmluQ2hhbm5lbHMsIGNvbnZJbmZvLm91dENoYW5uZWxzXX1cbiAgICB9KTtcbiAgICBpbnRlcm1lZGlhdGVzLnB1c2goZmlsdGVyUmVzaGFwZWQpO1xuICAgIGNvbnN0IHBvaW50d2lzZUNvbnYgPSBiYXRjaE1hdE11bEltcGwoe1xuICAgICAgYTogeFJlc2hhcGVkLFxuICAgICAgYjogZmlsdGVyUmVzaGFwZWQsXG4gICAgICBiYWNrZW5kLFxuICAgICAgdHJhbnNwb3NlQSxcbiAgICAgIHRyYW5zcG9zZUIsXG4gICAgICBiaWFzLFxuICAgICAgYWN0aXZhdGlvbixcbiAgICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHMsXG4gICAgICBsZWFreXJlbHVBbHBoYVxuICAgIH0pO1xuXG4gICAgY29uc3QgcG9pbnR3aXNlQ29udlRleERhdGEgPSBiYWNrZW5kLnRleERhdGEuZ2V0KHBvaW50d2lzZUNvbnYuZGF0YUlkKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgcG9pbnR3aXNlQ29udlRleERhdGEuaXNQYWNrZWQsXG4gICAgICAgICgpID0+ICdiYXRjaE1hdE11bCByZXN1bHQgaXMgZXhwZWN0ZWQgdG8gYmUgcGFja2VkJyk7XG4gICAgLy8gUmVzdG9yZSB0aGUgaW5wdXQgc2hhcGUgdG8gb3JpZ2luYWwuXG4gICAgeFRleERhdGEuc2hhcGUgPSBvcmlnaW5hbFhUZXhEYXRhU2hhcGU7XG4gICAgLy8gU2V0IHRoZSBvdXRwdXQgc2hhcGUgLSB0aGVyZSBpcyBubyBuZWVkIGZvciBleHBlbnNpdmUgcmVzaGFwZSBhcyBkYXRhXG4gICAgLy8gbGF5b3V0IGlzIGFscmVhZHkgY29ycmVjdC5cbiAgICBwb2ludHdpc2VDb252VGV4RGF0YS5zaGFwZSA9IGNvbnZJbmZvLm91dFNoYXBlO1xuXG4gICAgb3V0ID0gaWRlbnRpdHkoe2lucHV0czoge3g6IHBvaW50d2lzZUNvbnZ9LCBiYWNrZW5kfSk7XG4gICAgb3V0LnNoYXBlID0gY29udkluZm8ub3V0U2hhcGU7XG5cbiAgICBpbnRlcm1lZGlhdGVzLnB1c2gocG9pbnR3aXNlQ29udik7XG4gIH0gZWxzZSB7XG4gICAgY29uc3QgdGFyZ2V0U2hhcGUgPSBpc0NoYW5uZWxzTGFzdCA/IHhTaGFwZVswXSAqIHhTaGFwZVsxXSAqIHhTaGFwZVsyXSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHhTaGFwZVswXSAqIHhTaGFwZVsyXSAqIHhTaGFwZVszXTtcbiAgICBjb25zdCB4UmVzaGFwZWQgPSByZXNoYXBlKHtcbiAgICAgIGlucHV0czoge3h9LFxuICAgICAgYmFja2VuZCxcbiAgICAgIGF0dHJzOiB7c2hhcGU6IFsxLCB0YXJnZXRTaGFwZSwgY29udkluZm8uaW5DaGFubmVsc119XG4gICAgfSk7XG4gICAgY29uc3QgZmlsdGVyUmVzaGFwZWQgPSByZXNoYXBlKHtcbiAgICAgIGlucHV0czoge3g6IGZpbHRlcn0sXG4gICAgICBiYWNrZW5kLFxuICAgICAgYXR0cnM6IHtzaGFwZTogWzEsIGNvbnZJbmZvLmluQ2hhbm5lbHMsIGNvbnZJbmZvLm91dENoYW5uZWxzXX1cbiAgICB9KTtcbiAgICBjb25zdCByZXN1bHQgPSBiYXRjaE1hdE11bEltcGwoe1xuICAgICAgYTogeFJlc2hhcGVkLFxuICAgICAgYjogZmlsdGVyUmVzaGFwZWQsXG4gICAgICB0cmFuc3Bvc2VBLFxuICAgICAgdHJhbnNwb3NlQixcbiAgICAgIGJhY2tlbmQsXG4gICAgICBiaWFzLFxuICAgICAgYWN0aXZhdGlvbixcbiAgICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHMsXG4gICAgICBsZWFreXJlbHVBbHBoYVxuICAgIH0pO1xuXG4gICAgb3V0ID0gcmVzaGFwZShcbiAgICAgICAge2lucHV0czoge3g6IHJlc3VsdH0sIGJhY2tlbmQsIGF0dHJzOiB7c2hhcGU6IGNvbnZJbmZvLm91dFNoYXBlfX0pO1xuXG4gICAgaW50ZXJtZWRpYXRlcy5wdXNoKHhSZXNoYXBlZCk7XG4gICAgaW50ZXJtZWRpYXRlcy5wdXNoKGZpbHRlclJlc2hhcGVkKTtcbiAgICBpbnRlcm1lZGlhdGVzLnB1c2gocmVzdWx0KTtcbiAgfVxuXG4gIGZvciAoY29uc3QgaSBvZiBpbnRlcm1lZGlhdGVzKSB7XG4gICAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhpKTtcbiAgfVxuXG4gIHJldHVybiBvdXQ7XG59XG5cbi8vIEltcGxlbWVudHMgdGhlIGltMnJvdyBhbGdvcml0aG0gYXMgb3V0bGluZWQgaW4gXCJIaWdoIFBlcmZvcm1hbmNlXG4vLyBDb252b2x1dGlvbmFsIE5ldXJhbCBOZXR3b3JrcyBmb3IgRG9jdW1lbnQgUHJvY2Vzc2luZ1wiIChTdXZpc29mdCwgMjAwNilcbmV4cG9ydCBmdW5jdGlvbiBjb252MmRXaXRoSW0yUm93KHtcbiAgeCxcbiAgZmlsdGVyLFxuICBjb252SW5mbyxcbiAgYmFja2VuZCxcbiAgYmlhcyA9IG51bGwsXG4gIHByZWx1QWN0aXZhdGlvbldlaWdodHMgPSBudWxsLFxuICBsZWFreXJlbHVBbHBoYSA9IDAsXG4gIGFjdGl2YXRpb24gPSBudWxsXG59OiBDb252MkRDb25maWcpIHtcbiAgLy8gUmVhcnJhbmdlcyBjb252MmQgaW5wdXQgc28gZWFjaCBibG9jayB0byBiZSBjb252b2x2ZWQgb3ZlciBmb3JtcyB0aGVcbiAgLy8gY29sdW1uIG9mIGEgbmV3IG1hdHJpeCB3aXRoIHNoYXBlIFtmaWx0ZXJXaWR0aCAqIGZpbHRlckhlaWdodCAqXG4gIC8vIGluQ2hhbm5lbHMsIG91dEhlaWdodCAqIG91dFdpZHRoXS4gVGhlIGZpbHRlciBpcyBhbHNvIHJlYXJyYW5nZWQgc28gZWFjaFxuICAvLyBvdXRwdXQgY2hhbm5lbCBmb3JtcyBhIHJvdyBvZiBhIG5ldyBtYXRyaXggd2l0aCBzaGFwZSBbb3V0Q2hhbm5lbHMsXG4gIC8vIGZpbHRlcldpZHRoICogZmlsdGVySGVpZ2h0ICogaW5DaGFubmVsc10uIFRoZSBjb252b2x1dGlvbiBpcyB0aGVuXG4gIC8vIGNvbXB1dGVkIGJ5IG11bHRpcGx5aW5nIHRoZXNlIG1hdHJpY2VzIGFuZCByZXNoYXBpbmcgdGhlIHJlc3VsdC5cbiAgY29uc3Qge1xuICAgIGZpbHRlcldpZHRoLFxuICAgIGZpbHRlckhlaWdodCxcbiAgICBpbkNoYW5uZWxzLFxuICAgIG91dFdpZHRoLFxuICAgIG91dEhlaWdodCxcbiAgICBkYXRhRm9ybWF0XG4gIH0gPSBjb252SW5mbztcblxuICBjb25zdCBpc0NoYW5uZWxzTGFzdCA9IGRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0xhc3QnO1xuXG4gIGNvbnN0IHNoYXJlZERpbSA9IGZpbHRlcldpZHRoICogZmlsdGVySGVpZ2h0ICogaW5DaGFubmVscztcbiAgY29uc3QgbnVtQ29scyA9IG91dEhlaWdodCAqIG91dFdpZHRoO1xuICBjb25zdCB4MkNvbFNoYXBlID0gW3NoYXJlZERpbSwgbnVtQ29sc107XG4gIGNvbnN0IHRyYW5zcG9zZUEgPSB0cnVlO1xuICBjb25zdCB0cmFuc3Bvc2VCID0gZmFsc2U7XG5cbiAgY29uc3QgaW50ZXJtZWRpYXRlczogVGVuc29ySW5mb1tdID0gW107XG5cbiAgY29uc3QgeFNxdWVlemVkID1cbiAgICAgIHJlc2hhcGUoe2lucHV0czoge3h9LCBiYWNrZW5kLCBhdHRyczoge3NoYXBlOiB4LnNoYXBlLnNsaWNlKDEpfX0pO1xuICBjb25zdCB3MlJvdyA9IHJlc2hhcGUoe1xuICAgIGlucHV0czoge3g6IGZpbHRlcn0sXG4gICAgYmFja2VuZCxcbiAgICBhdHRyczoge3NoYXBlOiBbMSwgc2hhcmVkRGltLCB1dGlsLnNpemVGcm9tU2hhcGUoZmlsdGVyLnNoYXBlKSAvIHNoYXJlZERpbV19XG4gIH0pO1xuXG4gIGludGVybWVkaWF0ZXMucHVzaCh4U3F1ZWV6ZWQpO1xuICBpbnRlcm1lZGlhdGVzLnB1c2godzJSb3cpO1xuXG4gIGNvbnN0IGltMkNvbFByb2dyYW0gPSBuZXcgSW0yQ29sUGFja2VkUHJvZ3JhbSh4MkNvbFNoYXBlLCBjb252SW5mbyk7XG4gIGNvbnN0IGN1c3RvbVZhbHVlcyA9IFtcbiAgICB4U3F1ZWV6ZWQuc2hhcGUsIFtjb252SW5mby5wYWRJbmZvLnRvcCwgY29udkluZm8ucGFkSW5mby5sZWZ0XSxcbiAgICBbY29udkluZm8uc3RyaWRlSGVpZ2h0LCBjb252SW5mby5zdHJpZGVXaWR0aF0sXG4gICAgW2NvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0LCBjb252SW5mby5kaWxhdGlvbldpZHRoXSwgW2NvbnZJbmZvLmluQ2hhbm5lbHNdLFxuICAgIFtjb252SW5mby5maWx0ZXJXaWR0aCAqIGNvbnZJbmZvLmluQ2hhbm5lbHNdLCBbY29udkluZm8ub3V0V2lkdGhdXG4gIF07XG4gIGNvbnN0IGltMkNvbCA9IGJhY2tlbmQucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgaW0yQ29sUHJvZ3JhbSwgW3hTcXVlZXplZF0sICdmbG9hdDMyJywgY3VzdG9tVmFsdWVzKTtcbiAgY29uc3QgaW0yQ29sUmVzaGFwZWQgPSByZXNoYXBlKHtcbiAgICBpbnB1dHM6IHt4OiBpbTJDb2x9LFxuICAgIGJhY2tlbmQsXG4gICAgYXR0cnM6IHtzaGFwZTogWzEsIHgyQ29sU2hhcGVbMF0sIHgyQ29sU2hhcGVbMV1dfVxuICB9KTtcblxuICBpbnRlcm1lZGlhdGVzLnB1c2goaW0yQ29sKTtcbiAgaW50ZXJtZWRpYXRlcy5wdXNoKGltMkNvbFJlc2hhcGVkKTtcblxuICBjb25zdCBoYXNCaWFzID0gYmlhcyAhPSBudWxsO1xuICBjb25zdCBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyAhPSBudWxsO1xuICBjb25zdCBoYXNMZWFreXJlbHVBbHBoYSA9IGFjdGl2YXRpb24gPT09ICdsZWFreXJlbHUnO1xuICBjb25zdCBmdXNlZEFjdGl2YXRpb24gPVxuICAgICAgYWN0aXZhdGlvbiA/IG1hcEFjdGl2YXRpb25Ub1NoYWRlclByb2dyYW0oYWN0aXZhdGlvbiwgdHJ1ZSkgOiBudWxsO1xuICBjb25zdCBtYXRtdWxQcm9ncmFtID0gbmV3IE1hdE11bFBhY2tlZFByb2dyYW0oXG4gICAgICBpbTJDb2xSZXNoYXBlZC5zaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICB3MlJvdy5zaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICBbMSwgbnVtQ29scywgY29udkluZm8ub3V0Q2hhbm5lbHNdLCB0cmFuc3Bvc2VBLCB0cmFuc3Bvc2VCLCBoYXNCaWFzLFxuICAgICAgZnVzZWRBY3RpdmF0aW9uLCBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzLCBoYXNMZWFreXJlbHVBbHBoYSk7XG4gIGNvbnN0IGlucHV0czogVGVuc29ySW5mb1tdID0gW2ltMkNvbFJlc2hhcGVkLCB3MlJvd107XG4gIGlmIChiaWFzKSB7XG4gICAgaW5wdXRzLnB1c2goYmlhcyk7XG4gIH1cbiAgaWYgKGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMpIHtcbiAgICBpbnB1dHMucHVzaChwcmVsdUFjdGl2YXRpb25XZWlnaHRzKTtcbiAgfVxuICBpZiAoaGFzTGVha3lyZWx1QWxwaGEpIHtcbiAgICBjb25zdCAkbGVha3lyZWx1QWxwaGEgPSBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKFxuICAgICAgICBbXSwgJ2Zsb2F0MzInLFxuICAgICAgICB1dGlsLmNyZWF0ZVNjYWxhclZhbHVlKGxlYWt5cmVsdUFscGhhIGFzIHt9IGFzICdmbG9hdDMyJywgJ2Zsb2F0MzInKSk7XG4gICAgaW5wdXRzLnB1c2goJGxlYWt5cmVsdUFscGhhKTtcbiAgICBpbnRlcm1lZGlhdGVzLnB1c2goJGxlYWt5cmVsdUFscGhhKTtcbiAgfVxuICBjb25zdCBwcm9kdWN0ID0gYmFja2VuZC5ydW5XZWJHTFByb2dyYW0obWF0bXVsUHJvZ3JhbSwgaW5wdXRzLCAnZmxvYXQzMicpO1xuXG4gIGNvbnN0IG91dFNoYXBlID0gaXNDaGFubmVsc0xhc3QgP1xuICAgICAgWzEsIG91dEhlaWdodCwgb3V0V2lkdGgsIGNvbnZJbmZvLm91dENoYW5uZWxzXSA6XG4gICAgICBbMSwgY29udkluZm8ub3V0Q2hhbm5lbHMsIG91dEhlaWdodCwgb3V0V2lkdGhdO1xuICBjb25zdCBvdXQgPVxuICAgICAgcmVzaGFwZSh7aW5wdXRzOiB7eDogcHJvZHVjdH0sIGJhY2tlbmQsIGF0dHJzOiB7c2hhcGU6IG91dFNoYXBlfX0pO1xuXG4gIGludGVybWVkaWF0ZXMucHVzaChwcm9kdWN0KTtcbiAgZm9yIChjb25zdCBpIG9mIGludGVybWVkaWF0ZXMpIHtcbiAgICBiYWNrZW5kLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKGkpO1xuICB9XG5cbiAgcmV0dXJuIG91dDtcbn1cbiJdfQ==
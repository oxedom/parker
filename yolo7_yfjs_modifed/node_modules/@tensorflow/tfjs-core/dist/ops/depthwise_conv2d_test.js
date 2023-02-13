/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import * as tf from '../index';
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { expectArraysClose } from '../test_util';
describeWithFlags('depthwiseConv2D', ALL_ENVS, () => {
    it('input=1x3x3x1,f=2,s=1,d=1,p=valid,chMul=1', async () => {
        const fSize = 2;
        const pad = 'valid';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 2, 2, 1]);
        const expected = [1.07022, 1.03167, 0.67041, 0.778863];
        expectArraysClose(await result.data(), expected);
    });
    it('input=1x3x3x1,f=2,s=1,d=1,p=explicit,chMul=1', async () => {
        const fSize = 2;
        const pad = [[0, 0], [1, 2], [0, 1], [0, 0]];
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 5, 3, 1]);
        const expected = [
            0.826533, 0.197560, 0.0098898, 1.070216, 1.031675, 0.126422, 0.6704096,
            0.778863, 0.273041, 0.116357, 0.204908, 0.106774, 0, 0, 0
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=1x5x5x1,f=3,s=1,d=1,p=valid,chMul=1', async () => {
        const fSize = 3;
        const pad = 'valid';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
            0.211859, 0.633501, 0.186427, 0.777034, 0.50001, 0.607341, 0.95303,
            0.696479, 0.050387, 0.62045, 0.728049, 0.028043, 0.437009, 0.712881,
            0.741935, 0.974474, 0.621102, 0.171411
        ], [1, 5, 5, inDepth]);
        const w = tf.tensor4d([
            0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
            0.180695, 0.691992
        ], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 3, 3, 1]);
        const expected = [
            2.540022, 2.505885, 2.454062, 2.351701, 2.459601, 3.076421, 3.29848,
            3.437421, 2.93419
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=1x3x3x1,f=2,s=1,d=2,p=valid,chMul=1', async () => {
        const fSize = 2;
        const pad = 'valid';
        const stride = 1;
        const dilation = 2;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        // adding a dilation rate is equivalent to using a filter
        // with 0s for the dilation rate
        const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
        const wDilated = tf.tensor4d([0.303873, 0, 0.229223, 0, 0, 0, 0.144333, 0, 0.803373], [fSizeDilated, fSizeDilated, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);
        const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);
        expect(result.shape).toEqual(expectedResult.shape);
        expectArraysClose(await result.data(), await expectedResult.data());
    });
    it('input=1x5x5x1,f=3,s=1,d=2,p=valid,chMul=1', async () => {
        const fSize = 3;
        const pad = 'valid';
        const stride = 1;
        const dilation = 2;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
            0.211859, 0.633501, 0.186427, 0.777034, 0.50001, 0.607341, 0.95303,
            0.696479, 0.050387, 0.62045, 0.728049, 0.028043, 0.437009, 0.712881,
            0.741935, 0.974474, 0.621102, 0.171411
        ], [1, 5, 5, inDepth]);
        const w = tf.tensor4d([
            0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
            0.180695, 0.691992
        ], [fSize, fSize, inDepth, chMul]);
        // adding a dilation rate is equivalent to using a filter
        // with 0s for the dilation rate
        const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
        const wDilated = tf.tensor4d([
            0.125386, 0, 0.975199, 0, 0.640437, 0, 0, 0, 0, 0,
            0.281895, 0, 0.990968, 0, 0.347208, 0, 0, 0, 0, 0,
            0.889702, 0, 0.180695, 0, 0.691992
        ], [fSizeDilated, fSizeDilated, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);
        const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);
        expect(result.shape).toEqual(expectedResult.shape);
        expectArraysClose(await result.data(), await expectedResult.data());
    });
    it('input=1x5x5x1,f=2,s=1,d=4,p=valid,chMul=1', async () => {
        const fSize = 2;
        const pad = 'valid';
        const stride = 1;
        const dilation = 4;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
            0.211859, 0.633501, 0.186427, 0.777034, 0.50001, 0.607341, 0.95303,
            0.696479, 0.050387, 0.62045, 0.728049, 0.028043, 0.437009, 0.712881,
            0.741935, 0.974474, 0.621102, 0.171411
        ], [1, 5, 5, inDepth]);
        const w = tf.tensor4d([0.125386, 0.975199, 0.640437, 0.281895], [fSize, fSize, inDepth, chMul]);
        // adding a dilation rate is equivalent to using a filter
        // with 0s for the dilation rate
        const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
        const wDilated = tf.tensor4d([
            0.125386, 0, 0, 0, 0.975199, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0.640437, 0, 0, 0, 0.281895
        ], [fSizeDilated, fSizeDilated, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);
        const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);
        expect(result.shape).toEqual(expectedResult.shape);
        expectArraysClose(await result.data(), await expectedResult.data());
    });
    it('input=1x3x3x2,f=2,s=1,d=1,p=same,chMul=1', async () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const chMul = 1;
        const inDepth = 2;
        const x = tf.tensor4d([
            0.111057, 0.661818, 0.701979, 0.424362, 0.992854, 0.417599, 0.423036,
            0.500499, 0.368484, 0.714135, 0.456693, 0.531058, 0.636636, 0.345024,
            0.0506303, 0.789682, 0.177473, 0.793569
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([
            0.614293, 0.0648011, 0.101113, 0.452887, 0.0582746, 0.426481,
            0.872743, 0.765767
        ], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 3, 3, 2]);
        const expected = [
            0.485445, 0.995389, 0.95166, 0.927856, 0.636516, 0.253547, 0.378414,
            1.10771, 0.430373, 1.23126, 0.290885, 0.372855, 0.3962, 0.379995,
            0.0490466, 0.410569, 0.10902, 0.0514242
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=1x5x5x1,f=3,s=1,d=1,p=same,chMul=1', async () => {
        const fSize = 3;
        const pad = 'same';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
            0.211859, 0.633501, 0.186427, 0.777034, 0.50001, 0.607341, 0.95303,
            0.696479, 0.050387, 0.62045, 0.728049, 0.028043, 0.437009, 0.712881,
            0.741935, 0.974474, 0.621102, 0.171411
        ], [1, 5, 5, inDepth]);
        const w = tf.tensor4d([
            0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
            0.180695, 0.691992
        ], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 5, 5, 1]);
        const expected = [
            0.684796, 1.179251, 1.680593, 0.885615, 1.152995, 1.52291, 2.540022,
            2.505885, 2.454062, 1.871258, 2.371015, 2.351701, 2.459601, 3.076421,
            1.323994, 1.985572, 3.29848, 3.437421, 2.93419, 1.823238, 1.410545,
            2.352186, 2.19622, 1.348218, 0.774635
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=1x3x3x2,f=2,s=1,d=2,p=same,chMul=1', async () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const dilation = 2;
        const inDepth = 2;
        const x = tf.tensor4d([
            0.111057, 0.661818, 0.701979, 0.424362, 0.992854, 0.417599, 0.423036,
            0.500499, 0.368484, 0.714135, 0.456693, 0.531058, 0.636636, 0.345024,
            0.0506303, 0.789682, 0.177473, 0.793569
        ], [1, 3, 3, inDepth]);
        const w = tf.stack([
            tf.tensor2d([0.614293, 0.0648011, 0.101113, 0.452887], [fSize, fSize]),
            tf.tensor2d([0.0582746, 0.426481, 0.872743, 0.765767], [fSize, fSize])
        ], 2)
            .expandDims(3);
        // adding a dilation rate is equivalent to using a filter
        // with 0s for the dilation rate
        const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
        const wDilated = tf.stack([
            tf.tensor2d([0.614293, 0, 0.0648011, 0, 0, 0, 0.101113, 0, 0.452887], [fSizeDilated, fSizeDilated]),
            tf.tensor2d([0.0582746, 0, 0.426481, 0, 0, 0, 0.872743, 0, 0.765767], [fSizeDilated, fSizeDilated])
        ], 2)
            .expandDims(3);
        expect(wDilated.shape).toEqual([fSizeDilated, fSizeDilated, inDepth, 1]);
        const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);
        const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);
        expect(result.shape).toEqual(expectedResult.shape);
        expectArraysClose(await result.data(), await expectedResult.data());
    });
    it('input=1x5x5x1,f=3,s=1,d=2,p=same,chMul=1', async () => {
        const fSize = 3;
        const pad = 'valid';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
            0.211859, 0.633501, 0.186427, 0.777034, 0.50001, 0.607341, 0.95303,
            0.696479, 0.050387, 0.62045, 0.728049, 0.028043, 0.437009, 0.712881,
            0.741935, 0.974474, 0.621102, 0.171411
        ], [1, 5, 5, inDepth]);
        const w = tf.tensor4d([
            0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
            0.180695, 0.691992
        ], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 3, 3, 1]);
        const expected = [
            2.540022, 2.505885, 2.454062, 2.351701, 2.459601, 3.076421, 3.29848,
            3.437421, 2.93419
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=1x5x5x1,f=3,s=1,d=2,p=explicit,chMul=1', async () => {
        const fSize = 3;
        const pad = [[0, 0], [0, 0], [0, 1], [0, 1]];
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
            0.211859, 0.633501, 0.186427, 0.777034, 0.50001, 0.607341, 0.95303,
            0.696479, 0.050387, 0.62045, 0.728049, 0.028043, 0.437009, 0.712881,
            0.741935, 0.974474, 0.621102, 0.171411
        ], [1, 5, 5, inDepth]);
        const w = tf.tensor4d([
            0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
            0.180695, 0.691992
        ], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 3, 4, 1]);
        const expected = [
            2.540022, 2.505885, 2.454062, 1.871258, 2.35170, 2.459601, 3.076421,
            1.32399, 3.298480, 3.437421, 2.93419, 1.823238
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=1x3x3x2,f=2,s=1,p=same,chMul=2', async () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const chMul = 2;
        const inDepth = 2;
        const x = tf.tensor4d([
            0.675707, 0.758567, 0.413529, 0.963967, 0.217291, 0.101335, 0.804231,
            0.329673, 0.924503, 0.728742, 0.180217, 0.210459, 0.133869, 0.650827,
            0.047613, 0.554795, 0.653365, 0.442196
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([
            0.347154, 0.386692, 0.327191, 0.483784, 0.591807, 0.24263, 0.95182,
            0.174353, 0.592136, 0.623469, 0.988244, 0.660731, 0.946534, 0.0801365,
            0.864889, 0.874602
        ], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 3, 3, 4]);
        const expected = [
            1.83059, 0.937125, 2.1218, 1.39024, 0.990167, 0.803472,
            1.31405, 1.14959, 0.182147, 0.196385, 0.241141, 0.188081,
            0.950656, 0.622581, 1.92451, 1.20179, 1.07422, 0.483268,
            1.36948, 1.14256, 0.449444, 0.477042, 0.505857, 0.393989,
            0.0746509, 0.0633184, 0.74101, 0.41159, 0.403195, 0.176938,
            0.602415, 0.345499, 0.226819, 0.252651, 0.144682, 0.213927
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=2x3x3x2,f=2,s=1,p=same,chMul=2', async () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const chMul = 2;
        const inDepth = 2;
        const x = tf.tensor4d([
            0.261945, 0.0528113, 0.656698, 0.127345, 0.610039, 0.169131,
            0.458647, 0.0988288, 0.966109, 0.0421747, 0.82035, 0.274711,
            0.359377, 0.512113, 0.689682, 0.941571, 0.31961, 0.743826,
            0.858147, 0.984766, 0.926973, 0.579597, 0.444104, 0.505969,
            0.241437, 0.937999, 0.0957074, 0.773611, 0.46023, 0.469379,
            0.363789, 0.269745, 0.486136, 0.894215, 0.794299, 0.724615
        ], [2, 3, 3, inDepth]);
        const w = tf.tensor4d([
            0.240347, 0.906352, 0.478657, 0.825918, 0.380769, 0.184705, 0.238241,
            0.201907, 0.294087, 0.181165, 0.191303, 0.7225, 0.430064, 0.900622,
            0.670338, 0.33478
        ], [fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([2, 3, 3, 4]);
        const expected = [
            0.863379, 1.3119, 0.102795, 0.154853, 1.02704, 1.62173, 0.293466,
            0.261764, 0.387876, 0.701529, 0.133508, 0.338167, 0.880395, 1.28039,
            0.786492, 0.775361, 0.884845, 1.43995, 0.764374, 1.0196, 0.291162,
            0.801428, 0.273788, 0.764303, 0.348985, 0.45311, 0.469447, 0.613073,
            0.287461, 0.684128, 0.627899, 0.927844, 0.0768174, 0.28968, 0.356037,
            0.614339, 0.67138, 1.07894, 1.30747, 1.86705, 0.617971, 1.35402,
            0.860607, 1.29693, 0.242087, 0.485892, 0.331979, 0.757015, 0.410527,
            0.740235, 1.28431, 1.42516, 0.68281, 0.975185, 1.13892, 1.62237,
            0.344208, 0.561029, 0.363292, 0.911203, 0.272541, 0.419513, 0.342154,
            0.403335, 0.419286, 0.587321, 0.600655, 0.884853, 0.190907, 0.719914,
            0.346842, 0.598472
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('input=2x3x3x2,f=2,s=1,d=2,p=same,chMul=2', async () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const inDepth = 2;
        const dilation = 2;
        const noDilation = 1;
        const x = tf.tensor4d([
            0.261945, 0.0528113, 0.656698, 0.127345, 0.610039, 0.169131,
            0.458647, 0.0988288, 0.966109, 0.0421747, 0.82035, 0.274711,
            0.359377, 0.512113, 0.689682, 0.941571, 0.31961, 0.743826,
            0.858147, 0.984766, 0.926973, 0.579597, 0.444104, 0.505969,
            0.241437, 0.937999, 0.0957074, 0.773611, 0.46023, 0.469379,
            0.363789, 0.269745, 0.486136, 0.894215, 0.794299, 0.724615
        ], [2, 3, 3, inDepth]);
        const w = tf.stack([
            tf.stack([
                tf.tensor2d([0.240347, 0.906352, 0.478657, 0.825918], [fSize, fSize]),
                tf.tensor2d([0.380769, 0.184705, 0.238241, 0.201907], [fSize, fSize])
            ], 2),
            tf.stack([
                tf.tensor2d([0.294087, 0.181165, 0.191303, 0.7225], [fSize, fSize]),
                tf.tensor2d([0.430064, 0.900622, 0.670338, 0.33478], [fSize, fSize])
            ], 2)
        ], 3);
        const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
        const wDilated = tf.stack([
            tf.stack([
                tf.tensor2d([0.240347, 0, 0.906352, 0, 0, 0, 0.478657, 0, 0.825918], [fSizeDilated, fSizeDilated]),
                tf.tensor2d([0.380769, 0, 0.184705, 0, 0, 0, 0.238241, 0, 0.201907], [fSizeDilated, fSizeDilated])
            ], 2),
            tf.stack([
                tf.tensor2d([0.294087, 0, 0.181165, 0, 0, 0, 0.191303, 0, 0.7225], [fSizeDilated, fSizeDilated]),
                tf.tensor2d([0.430064, 0, 0.900622, 0, 0, 0, 0.670338, 0, 0.33478], [fSizeDilated, fSizeDilated])
            ], 2)
        ], 3);
        const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);
        const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad, 'NHWC', noDilation);
        expect(result.shape).toEqual(expectedResult.shape);
        expectArraysClose(await result.data(), await expectedResult.data());
    });
    it('input=2x3x3x2,f=3,s=1,d=2,p=same,chMul=2', async () => {
        const fSize = 3;
        const pad = 'same';
        const stride = 1;
        const inDepth = 2;
        const dilation = 2;
        const x = tf.tensor4d([[
                [
                    [0.52276146, 0.34775984], [0.4690094, 0.40816104],
                    [0.32390153, 0.23729074], [0.61366737, 0.7918105],
                    [0.9145211, 0.218611], [0.37787926, 0.23923647],
                    [0.23401344, 0.12519836]
                ],
                [
                    [0.6222534, 0.13273609], [0.7697753, 0.12160587],
                    [0.0448128, 0.94806635], [0.4199953, 0.7140714],
                    [0.01420832, 0.47453713], [0.02061439, 0.37226152],
                    [0.62741446, 0.23167181]
                ],
                [
                    [0.7257557, 0.14352751], [0.3011638, 0.3869065],
                    [0.09286129, 0.25151742], [0.7566397, 0.13099921],
                    [0.65324724, 0.38959372], [0.65826, 0.7505318],
                    [0.35919082, 0.85470796]
                ],
                [
                    [0.24827361, 0.2826661], [0.24717247, 0.27446854],
                    [0.27112448, 0.68068564], [0.11082292, 0.7948675],
                    [0.41535318, 0.659986], [0.22165525, 0.18149579],
                    [0.42273378, 0.9558281]
                ],
                [
                    [0.943074, 0.6799041], [0.78851473, 0.07249606],
                    [0.771909, 0.7925967], [0.9551083, 0.03087568],
                    [0.82589805, 0.94797385], [0.5895462, 0.5045923],
                    [0.9667754, 0.24292922]
                ],
                [
                    [0.67123663, 0.109761], [0.04002762, 0.51942277],
                    [0.37868536, 0.8467603], [0.77171385, 0.51604605],
                    [0.8192849, 0.38843668], [0.19607484, 0.5591624],
                    [0.45990825, 0.35768318]
                ],
                [
                    [0.67443585, 0.6256168], [0.9373623, 0.6498393],
                    [0.7623085, 0.13218105], [0.9349631, 0.7660191],
                    [0.50054944, 0.7738123], [0.30201948, 0.525643],
                    [0.30896342, 0.21111596]
                ]
            ]], [1, 7, 7, inDepth]);
        const w = tf.tensor4d([
            [
                [[0.65113723], [0.8699447]],
                [[0.267792], [0.9981787]],
                [[0.4913572], [0.33211958]]
            ],
            [
                [[0.5286497], [0.42418027]],
                [[0.01754463], [0.8365464]],
                [[0.17683995], [0.2874831]]
            ],
            [
                [[0.09339976], [0.57645476]],
                [[0.06616235], [0.8850273]],
                [[0.87009287], [0.20542204]]
            ]
        ], [fSize, fSize, inDepth, 1]);
        const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);
        expect(result.shape).toEqual([1, 7, 7, 2]);
        expectArraysClose(await result.data(), [
            0.19526604, 0.5378273, 0.795022, 0.9384107, 1.0860794, 0.7942326,
            0.9764694, 1.3974442, 0.5930813, 0.9848901, 0.44526684, 1.275759,
            0.572345, 1.1784878, 0.27117175, 0.773588, 0.20055711, 0.71320784,
            0.73477566, 1.8867722, 0.64123434, 1.6549369, 0.55551285, 2.0385633,
            0.24740812, 1.233143, 0.08528192, 1.6214795, 1.062326, 1.3828603,
            1.4494176, 1.1022222, 2.2350664, 2.283423, 1.5940895, 1.8871424,
            1.6627852, 2.4903212, 1.0405337, 2.0754304, 1.1508893, 1.9568737,
            0.6148571, 1.1505995, 1.1105528, 1.3823687, 1.4342139, 2.9909487,
            1.0210396, 2.6467443, 1.0563798, 3.3963797, 0.42652097, 2.274134,
            0.51121074, 2.264094, 1.1009313, 1.6042703, 1.510688, 1.2317145,
            2.025515, 2.3658662, 1.6722159, 2.0787857, 1.3785586, 2.895031,
            1.2915218, 2.2051222, 1.0423074, 2.4303207, 0.27844793, 0.84346974,
            0.25781655, 1.1208354, 0.9447272, 2.0111258, 0.3689065, 1.9052455,
            0.79137695, 2.355344, 0.5429248, 1.5593178, 0.8248403, 1.9922242,
            0.77847, 1.5032601, 0.8622418, 0.84645665, 1.6850245, 2.2958806,
            1.6242284, 1.329045, 1.6652328, 2.480535, 1.2793491, 1.2951884,
            1.0667037, 1.5720158
        ]);
    });
    it('Tensor3D is allowed', async () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const chMul = 3;
        const inDepth = 2;
        const x = tf.zeros([3, 3, inDepth]);
        const w = tf.zeros([fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([3, 3, inDepth * chMul]);
    });
    it('Pass null for dilations, which defaults to [1, 1]', () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const chMul = 3;
        const inDepth = 2;
        const dilations = null;
        const x = tf.zeros([3, 3, inDepth]);
        const w = tf.zeros([fSize, fSize, inDepth, chMul]);
        const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilations);
        expect(result.shape).toEqual([3, 3, inDepth * chMul]);
    });
    it('TensorLike', async () => {
        const pad = 'valid';
        const stride = 1;
        const x = [[
                [[0.230664], [0.987388], [0.0685208]],
                [[0.419224], [0.887861], [0.731641]],
                [[0.0741907], [0.409265], [0.351377]]
            ]];
        const w = [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        const expected = [1.07022, 1.03167, 0.67041, 0.778863];
        expectArraysClose(await result.data(), expected);
    });
    it('TensorLike Chained', async () => {
        const pad = 'valid';
        const stride = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];
        const result = x.depthwiseConv2d(w, stride, pad);
        expect(result.shape).toEqual([1, 2, 2, 1]);
        const expected = [1.07022, 1.03167, 0.67041, 0.778863];
        expectArraysClose(await result.data(), expected);
    });
    it('throws when passed x as a non-tensor', () => {
        const inputDepth = 1;
        const outputDepth = 1;
        const fSize = 1;
        const pad = 'same';
        const stride = 2;
        const dataFormat = 'NHWC';
        const dilation = 2;
        const w = tf.tensor4d([3], [fSize, fSize, inputDepth, outputDepth]);
        const e = /Argument 'x' passed to 'depthwiseConv2d' must be a Tensor/;
        expect(() => tf.depthwiseConv2d({}, w, stride, pad, dataFormat, dilation))
            .toThrowError(e);
    });
    it('throws when passed filter as a non-tensor', () => {
        const inputDepth = 1;
        const inputShape = [2, 2, inputDepth];
        const pad = 'same';
        const stride = 2;
        const dataFormat = 'NHWC';
        const dilation = 2;
        const x = tf.tensor3d([1, 2, 3, 4], inputShape);
        const e = /Argument 'filter' passed to 'depthwiseConv2d' must be a Tensor/;
        expect(() => tf.depthwiseConv2d(x, {}, stride, pad, dataFormat, dilation))
            .toThrowError(e);
    });
    it('throws when input is int32', async () => {
        const fSize = 2;
        const pad = 'valid';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, inDepth], 'int32');
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        const errRegex = /Argument 'x' passed to 'depthwiseConv2d' must be float32/;
        expect(() => tf.depthwiseConv2d(x, w, stride, pad)).toThrowError(errRegex);
    });
    it('throws when filter is int32', async () => {
        const fSize = 2;
        const pad = 'valid';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([1, 2, 3, 4], [fSize, fSize, inDepth, chMul], 'int32');
        const errRegex = /Argument 'filter' passed to 'depthwiseConv2d' must be float32/;
        expect(() => tf.depthwiseConv2d(x, w, stride, pad)).toThrowError(errRegex);
    });
    it('throws when dimRoundingMode is set and pad is same', () => {
        const fSize = 2;
        const pad = 'same';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        expect(() => tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is valid', () => {
        const fSize = 2;
        const pad = 'valid';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        expect(() => tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is a non-integer number', () => {
        const fSize = 2;
        const pad = 1.2;
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        expect(() => tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is explicit by non-integer ' +
        'number', () => {
        const fSize = 2;
        const pad = [[0, 0], [0, 2.1], [1, 1], [0, 0]];
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const w = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        expect(() => tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
            .toThrowError();
    });
    it('accepts a tensor-like object', async () => {
        const pad = 'valid';
        const stride = 1;
        // 1x3x3x1
        const x = [[
                [[0.230664], [0.987388], [0.0685208]],
                [[0.419224], [0.887861], [0.731641]],
                [[0.0741907], [0.409265], [0.351377]]
            ]];
        // 2x2x1x1
        const w = [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];
        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 2, 2, 1]);
        const expected = [1.07022, 1.03167, 0.67041, 0.778863];
        expectArraysClose(await result.data(), expected);
    });
});
describeWithFlags('depthwiseConv2d gradients', ALL_ENVS, () => {
    let images;
    let filter;
    let result;
    const stride = 1;
    const pad = 'same';
    beforeEach(() => {
        // two 2x2 RGB images => 2x2x2x3
        images = tf.tensor4d([
            [[[2, 3, 1], [3, 0, 2]], [[0, 4, 1], [3, 1, 3]]],
            [[[2, 1, 0], [0, 3, 3]], [[4, 0, 1], [1, 4, 1]]]
        ]);
        // 2x2 filters, chMul = 2 => 2x2x3x2
        filter = tf.tensor4d([
            [[[1, 1], [1, 1], [0, 0]], [[0, 1], [1, 1], [1, 1]]],
            [[[1, 0], [1, 1], [0, 0]], [[0, 1], [1, 0], [0, 0]]]
        ]);
        // result of convolution operatoin
        result = tf.tensor4d([
            [
                [[2, 8, 8, 7, 2, 2], [6, 3, 1, 1, 0, 0]],
                [[0, 3, 5, 5, 3, 3], [3, 3, 1, 1, 0, 0]]
            ],
            [
                [[6, 3, 8, 4, 3, 3], [1, 0, 7, 7, 0, 0]],
                [[4, 5, 4, 4, 1, 1], [1, 1, 4, 4, 0, 0]]
            ]
        ]);
    });
    it('wrt input', async () => {
        const { value, grad } = tf.valueAndGrad((x) => tf.depthwiseConv2d(x, filter, stride, pad))(images);
        expectArraysClose(await value.data(), await result.data());
        const expectedGrad = tf.tensor4d([
            [[[2., 2., 0.], [3., 4., 2.]], [[3., 4., 0.], [5., 7., 2.]]],
            [[[2., 2., 0.], [3., 4., 2.]], [[3., 4., 0.], [5., 7., 2.]]]
        ]);
        expectArraysClose(await grad.data(), await expectedGrad.data());
    });
    // The gradients of normal and depthwise 2D convolutions are actually the same
    // in the special case that dy = 1, so we also test the gradient of a function
    // of the output to disambiguate the two methods.
    it('wrt input, squared output', async () => {
        const grad = tf.grad((x) => tf.square(tf.depthwiseConv2d(x, filter, stride, pad)))(images);
        const expectedGrad = tf.tensor4d([
            [[[20., 30., 0.], [34., 34., 8.]], [[10., 50., 0.], [46., 44., 12.]]],
            [[[18., 24., 0.], [8., 52., 12.]], [[30., 40., 0.], [22., 76., 4.]]]
        ]);
        expectArraysClose(await grad.data(), await expectedGrad.data());
    });
    it('wrt filter', async () => {
        const { value, grad } = tf.valueAndGrad((f) => tf.depthwiseConv2d(images, f, stride, pad))(filter);
        expectArraysClose(await value.data(), await result.data());
        const expectedGrad = tf.tensor4d([
            [[[15., 15.], [16., 16.], [12., 12.]], [[7., 7.], [8., 8.], [9., 9.]]],
            [[[8., 8.], [9., 9.], [6., 6.]], [[4., 4.], [5., 5.], [4., 4.]]]
        ]);
        expectArraysClose(await grad.data(), await expectedGrad.data());
    });
    it('gradient with clones', async () => {
        const [dx, dFilter] = tf.grads((x, filter) => tf.depthwiseConv2d(x.clone(), filter.clone(), stride, pad).clone())([images, filter]);
        expect(dx.shape).toEqual(images.shape);
        expect(dFilter.shape).toEqual(filter.shape);
    });
    // Also disambiguate regular vs. depthwise filter gradients
    it('wrt filter, squared output', async () => {
        const grad = tf.grad((f) => tf.square(tf.depthwiseConv2d(images, f, stride, pad)))(filter);
        const expectedGrad = tf.tensor4d([
            [
                [[120., 122.], [180., 166.], [12., 12.]],
                [[20., 76.], [90., 66.], [46., 46.]]
            ],
            [
                [[86., 42.], [122., 114.], [10., 10.]],
                [[24., 54.], [80., 46.], [18., 18.]]
            ]
        ]);
        expectArraysClose(await grad.data(), await expectedGrad.data());
    });
    it('throws error on dilations > 1', () => {
        const grad = tf.grad((x) => tf.depthwiseConv2d(x, filter, stride, pad, 'NHWC', 2));
        expect(() => grad(images))
            .toThrowError(/dilation rates greater than 1 are not yet supported/);
    });
    it('wrt input, stride=2, pad=valid', async () => {
        const dx = tf.grad((x) => tf.depthwiseConv2d(x, filter, 2, 'valid'))(images);
        expectArraysClose(await dx.data(), [
            2., 2., 0., 1., 2., 2., 1., 2., 0., 1., 1., 0.,
            2., 2., 0., 1., 2., 2., 1., 2., 0., 1., 1., 0.
        ]);
        expect(dx.shape).toEqual([2, 2, 2, 3]);
    });
    it('wrt filter, stride=2, pad=valid', async () => {
        const df = tf.grad((f) => tf.depthwiseConv2d(images, f, 2, 'valid'))(filter);
        expectArraysClose(await df.data(), [
            4., 4., 4., 4., 1., 1., 3., 3., 3., 3., 5., 5.,
            4., 4., 4., 4., 2., 2., 4., 4., 5., 5., 4., 4.
        ]);
        expect(df.shape).toEqual([2, 2, 3, 2]);
    });
    it('gradient with clones', async () => {
        const fSize = 2;
        const pad = 'valid';
        const stride = 1;
        const chMul = 1;
        const inDepth = 1;
        const x = tf.tensor4d([
            0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
            0.0741907, 0.409265, 0.351377
        ], [1, 3, 3, inDepth]);
        const f = tf.tensor4d([0.303873, 0.229223, 0.144333, 0.803373], [fSize, fSize, inDepth, chMul]);
        const [dx, df] = tf.grads((x, f) => tf.depthwiseConv2d(x.clone(), f.clone(), stride, pad).clone())([x, f]);
        expectArraysClose(await dx.data(), [
            0.303873, 0.533096, 0.229223, 0.448206, 1.480802, 1.032596, 0.144333,
            0.947706, 0.803373
        ]);
        expect(dx.shape).toEqual([1, 3, 3, 1]);
        expectArraysClose(await df.data(), [2.525137, 2.6754108, 1.7905407, 2.380144]);
        expect(df.shape).toEqual([2, 2, 1, 1]);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGh3aXNlX2NvbnYyZF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZGVwdGh3aXNlX2NvbnYyZF90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxRQUFRLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFHL0MsaUJBQWlCLENBQUMsaUJBQWlCLEVBQUUsUUFBUSxFQUFFLEdBQUcsRUFBRTtJQUNsRCxFQUFFLENBQUMsMkNBQTJDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDekQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sR0FBRyxHQUFHLE9BQU8sQ0FBQztRQUNwQixNQUFNLE1BQU0sR0FBRyxDQUFDLENBQUM7UUFDakIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sT0FBTyxHQUFHLENBQUMsQ0FBQztRQUVsQixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQjtZQUNFLFFBQVEsRUFBRSxRQUFRLEVBQUUsU0FBUyxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUMzRCxTQUFTLEVBQUUsUUFBUSxFQUFFLFFBQVE7U0FDOUIsRUFDRCxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakIsQ0FBQyxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLENBQUMsRUFDeEMsQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FDakMsQ0FBQztRQUVGLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDckQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNDLE1BQU0sUUFBUSxHQUFHLENBQUMsT0FBTyxFQUFFLE9BQU8sRUFBRSxPQUFPLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFDdkQsaUJBQWlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDbkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsOENBQThDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDNUQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sR0FBRyxHQUNMLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQW9DLENBQUM7UUFDeEUsTUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDM0QsU0FBUyxFQUFFLFFBQVEsRUFBRSxRQUFRO1NBQzlCLEVBQ0QsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCLENBQUMsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxDQUFDLEVBQ3hDLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQ2pDLENBQUM7UUFFRixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMzQyxNQUFNLFFBQVEsR0FBRztZQUNmLFFBQVEsRUFBRSxRQUFRLEVBQUUsU0FBUyxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFNBQVM7WUFDdEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUM7U0FDMUQsQ0FBQztRQUNGLGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ25ELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDJDQUEyQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3pELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLEdBQUcsR0FBRyxPQUFPLENBQUM7UUFDcEIsTUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLE9BQU87WUFDbkUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1NBQ3ZDLEVBQ0QsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCO1lBQ0UsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUTtTQUNuQixFQUNELENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQ2pDLENBQUM7UUFFRixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMzQyxNQUFNLFFBQVEsR0FBRztZQUNmLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLE9BQU87WUFDbkUsUUFBUSxFQUFFLE9BQU87U0FDbEIsQ0FBQztRQUNGLGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ25ELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDJDQUEyQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3pELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLEdBQUcsR0FBRyxPQUFPLENBQUM7UUFDcEIsTUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sUUFBUSxHQUFHLENBQUMsQ0FBQztRQUNuQixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUM7UUFDaEIsTUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDO1FBRWxCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCO1lBQ0UsUUFBUSxFQUFFLFFBQVEsRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQzNELFNBQVMsRUFBRSxRQUFRLEVBQUUsUUFBUTtTQUM5QixFQUNELENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUN4QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQixDQUFDLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsQ0FBQyxFQUN4QyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUNqQyxDQUFDO1FBQ0YseURBQXlEO1FBQ3pELGdDQUFnQztRQUNoQyxNQUFNLFlBQVksR0FBRyxLQUFLLEdBQUcsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxRQUFRLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDMUQsTUFBTSxRQUFRLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDeEIsQ0FBQyxRQUFRLEVBQUUsQ0FBQyxFQUFFLFFBQVEsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxRQUFRLEVBQUUsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxFQUN2RCxDQUFDLFlBQVksRUFBRSxZQUFZLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUMvQyxDQUFDO1FBRUYsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLE1BQU0sRUFBRSxHQUFHLEVBQUUsTUFBTSxFQUFFLFFBQVEsQ0FBQyxDQUFDO1FBRXZFLE1BQU0sY0FBYyxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLFFBQVEsRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFFcEUsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsY0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ25ELGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLE1BQU0sY0FBYyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUM7SUFDdEUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsMkNBQTJDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDekQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sR0FBRyxHQUFHLE9BQU8sQ0FBQztRQUNwQixNQUFNLE1BQU0sR0FBRyxDQUFDLENBQUM7UUFDakIsTUFBTSxRQUFRLEdBQUcsQ0FBQyxDQUFDO1FBQ25CLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLE9BQU87WUFDbkUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1NBQ3ZDLEVBQ0QsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCO1lBQ0UsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUTtTQUNuQixFQUNELENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQ2pDLENBQUM7UUFDRix5REFBeUQ7UUFDekQsZ0NBQWdDO1FBQ2hDLE1BQU0sWUFBWSxHQUFHLEtBQUssR0FBRyxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLFFBQVEsR0FBRyxDQUFDLENBQUMsQ0FBQztRQUMxRCxNQUFNLFFBQVEsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUN4QjtZQUNFLFFBQVEsRUFBRSxDQUFDLEVBQUUsUUFBUSxFQUFFLENBQUMsRUFBRSxRQUFRLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDakQsUUFBUSxFQUFFLENBQUMsRUFBRSxRQUFRLEVBQUUsQ0FBQyxFQUFFLFFBQVEsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUNqRCxRQUFRLEVBQUUsQ0FBQyxFQUFFLFFBQVEsRUFBRSxDQUFDLEVBQUUsUUFBUTtTQUNuQyxFQUNELENBQUMsWUFBWSxFQUFFLFlBQVksRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQy9DLENBQUM7UUFFRixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsRUFBRSxNQUFNLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFFdkUsTUFBTSxjQUFjLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxDQUFDLEVBQUUsUUFBUSxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztRQUVwRSxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxjQUFjLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDbkQsaUJBQWlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsTUFBTSxjQUFjLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQztJQUN0RSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQywyQ0FBMkMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUN6RCxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUM7UUFDaEIsTUFBTSxHQUFHLEdBQUcsT0FBTyxDQUFDO1FBQ3BCLE1BQU0sTUFBTSxHQUFHLENBQUMsQ0FBQztRQUNqQixNQUFNLFFBQVEsR0FBRyxDQUFDLENBQUM7UUFDbkIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sT0FBTyxHQUFHLENBQUMsQ0FBQztRQUVsQixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQjtZQUNFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDcEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLE9BQU8sRUFBRyxRQUFRLEVBQUUsT0FBTztZQUNuRSxRQUFRLEVBQUUsUUFBUSxFQUFFLE9BQU8sRUFBRyxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7U0FDdkMsRUFDRCxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakIsQ0FBQyxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLENBQUMsRUFDeEMsQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FDakMsQ0FBQztRQUNGLHlEQUF5RDtRQUN6RCxnQ0FBZ0M7UUFDaEMsTUFBTSxZQUFZLEdBQUcsS0FBSyxHQUFHLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsUUFBUSxHQUFHLENBQUMsQ0FBQyxDQUFDO1FBQzFELE1BQU0sUUFBUSxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ3hCO1lBQ0UsUUFBUSxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLFFBQVEsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBUyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQVEsQ0FBQztZQUNoRSxDQUFDLEVBQVMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFTLENBQUMsRUFBRSxDQUFDLEVBQUUsUUFBUSxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLFFBQVE7U0FDL0QsRUFDRCxDQUFDLFlBQVksRUFBRSxZQUFZLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUMvQyxDQUFDO1FBRUYsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLE1BQU0sRUFBRSxHQUFHLEVBQUUsTUFBTSxFQUFFLFFBQVEsQ0FBQyxDQUFDO1FBRXZFLE1BQU0sY0FBYyxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLFFBQVEsRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFFcEUsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsY0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ25ELGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLE1BQU0sY0FBYyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUM7SUFDdEUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsMENBQTBDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDeEQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQztRQUNuQixNQUFNLE1BQU0sR0FBRyxDQUFDLENBQUM7UUFDakIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sT0FBTyxHQUFHLENBQUMsQ0FBQztRQUVsQixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQjtZQUNFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDcEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxTQUFTLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1NBQ3hDLEVBQ0QsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCO1lBQ0UsUUFBUSxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFNBQVMsRUFBRSxRQUFRO1lBQzVELFFBQVEsRUFBRSxRQUFRO1NBQ25CLEVBQ0QsQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDO1FBQ3BDLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDckQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTNDLE1BQU0sUUFBUSxHQUFHO1lBQ2YsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNuRSxPQUFPLEVBQUUsUUFBUSxFQUFFLE9BQU8sRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLE1BQU0sRUFBRSxRQUFRO1lBQ2hFLFNBQVMsRUFBRSxRQUFRLEVBQUUsT0FBTyxFQUFFLFNBQVM7U0FDeEMsQ0FBQztRQUNGLGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ25ELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDBDQUEwQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3hELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUM7UUFDbkIsTUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLE9BQU87WUFDbkUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1NBQ3ZDLEVBQ0QsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCO1lBQ0UsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUTtTQUNuQixFQUNELENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQ2pDLENBQUM7UUFFRixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMzQyxNQUFNLFFBQVEsR0FBRztZQUNmLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsT0FBTyxFQUFHLFFBQVE7WUFDcEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUSxFQUFFLE9BQU8sRUFBRyxRQUFRLEVBQUUsT0FBTyxFQUFHLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLFFBQVE7U0FDdkMsQ0FBQztRQUNGLGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ25ELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDBDQUEwQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3hELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUM7UUFDbkIsTUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sUUFBUSxHQUFHLENBQUMsQ0FBQztRQUNuQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDcEUsU0FBUyxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtTQUN4QyxFQUNELENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUV4QixNQUFNLENBQUMsR0FDSCxFQUFFLENBQUMsS0FBSyxDQUNGO1lBQ0UsRUFBRSxDQUFDLFFBQVEsQ0FDUCxDQUFDLFFBQVEsRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUFFLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO1lBQzlELEVBQUUsQ0FBQyxRQUFRLENBQ1AsQ0FBQyxTQUFTLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztTQUMvRCxFQUNELENBQUMsQ0FBQzthQUNILFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUV2Qix5REFBeUQ7UUFDekQsZ0NBQWdDO1FBQ2hDLE1BQU0sWUFBWSxHQUFHLEtBQUssR0FBRyxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLFFBQVEsR0FBRyxDQUFDLENBQUMsQ0FBQztRQUMxRCxNQUFNLFFBQVEsR0FDVixFQUFFLENBQUMsS0FBSyxDQUNGO1lBQ0UsRUFBRSxDQUFDLFFBQVEsQ0FDUCxDQUFDLFFBQVEsRUFBRSxDQUFDLEVBQUUsU0FBUyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLFFBQVEsRUFBRSxDQUFDLEVBQUUsUUFBUSxDQUFDLEVBQ3hELENBQUMsWUFBWSxFQUFFLFlBQVksQ0FBQyxDQUFDO1lBQ2pDLEVBQUUsQ0FBQyxRQUFRLENBQ1AsQ0FBQyxTQUFTLEVBQUUsQ0FBQyxFQUFFLFFBQVEsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxRQUFRLEVBQUUsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxFQUN4RCxDQUFDLFlBQVksRUFBRSxZQUFZLENBQUMsQ0FBQztTQUNsQyxFQUNELENBQUMsQ0FBQzthQUNILFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUV2QixNQUFNLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLFlBQVksRUFBRSxZQUFZLEVBQUUsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFekUsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLE1BQU0sRUFBRSxHQUFHLEVBQUUsTUFBTSxFQUFFLFFBQVEsQ0FBQyxDQUFDO1FBRXZFLE1BQU0sY0FBYyxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLFFBQVEsRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFFcEUsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsY0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ25ELGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLE1BQU0sY0FBYyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUM7SUFDdEUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsMENBQTBDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDeEQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sR0FBRyxHQUFHLE9BQU8sQ0FBQztRQUNwQixNQUFNLE1BQU0sR0FBRyxDQUFDLENBQUM7UUFDakIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sT0FBTyxHQUFHLENBQUMsQ0FBQztRQUVsQixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQjtZQUNFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDcEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLE9BQU8sRUFBRyxRQUFRLEVBQUUsT0FBTztZQUNuRSxRQUFRLEVBQUUsUUFBUSxFQUFFLE9BQU8sRUFBRyxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7U0FDdkMsRUFDRCxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRO1NBQ25CLEVBQ0QsQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FDakMsQ0FBQztRQUVGLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDckQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNDLE1BQU0sUUFBUSxHQUFHO1lBQ2YsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsT0FBTztZQUNuRSxRQUFRLEVBQUUsT0FBTztTQUNsQixDQUFDO1FBQ0YsaUJBQWlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDbkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsOENBQThDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDNUQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sR0FBRyxHQUNMLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQW9DLENBQUM7UUFDeEUsTUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLE9BQU87WUFDbkUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUcsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1NBQ3ZDLEVBQ0QsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCO1lBQ0UsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUNwRSxRQUFRLEVBQUUsUUFBUTtTQUNuQixFQUNELENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQ2pDLENBQUM7UUFFRixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMzQyxNQUFNLFFBQVEsR0FBRztZQUNmLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDbkUsT0FBTyxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsT0FBTyxFQUFFLFFBQVE7U0FDL0MsQ0FBQztRQUNGLGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ25ELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNDQUFzQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3BELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUM7UUFDbkIsTUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDcEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtTQUN2QyxFQUNELENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUN4QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQjtZQUNFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsT0FBTyxFQUFFLE9BQU87WUFDbEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsU0FBUztZQUNyRSxRQUFRLEVBQUUsUUFBUTtTQUNuQixFQUNELENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQUMsQ0FBQztRQUNwQyxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUUzQyxNQUFNLFFBQVEsR0FBRztZQUNmLE9BQU8sRUFBSSxRQUFRLEVBQUcsTUFBTSxFQUFJLE9BQU8sRUFBRyxRQUFRLEVBQUUsUUFBUTtZQUM1RCxPQUFPLEVBQUksT0FBTyxFQUFJLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDNUQsUUFBUSxFQUFHLFFBQVEsRUFBRyxPQUFPLEVBQUcsT0FBTyxFQUFHLE9BQU8sRUFBRyxRQUFRO1lBQzVELE9BQU8sRUFBSSxPQUFPLEVBQUksUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUTtZQUM1RCxTQUFTLEVBQUUsU0FBUyxFQUFFLE9BQU8sRUFBRyxPQUFPLEVBQUcsUUFBUSxFQUFFLFFBQVE7WUFDNUQsUUFBUSxFQUFHLFFBQVEsRUFBRyxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1NBQzdELENBQUM7UUFDRixpQkFBaUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxRQUFRLENBQUMsQ0FBQztJQUNuRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxzQ0FBc0MsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNwRCxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUM7UUFDaEIsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDO1FBQ25CLE1BQU0sTUFBTSxHQUFHLENBQUMsQ0FBQztRQUNqQixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUM7UUFDaEIsTUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDO1FBRWxCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ2pCO1lBQ0UsUUFBUSxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUcsUUFBUSxFQUFHLFFBQVEsRUFBRSxRQUFRO1lBQzdELFFBQVEsRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUFHLFNBQVMsRUFBRSxPQUFPLEVBQUcsUUFBUTtZQUM3RCxRQUFRLEVBQUUsUUFBUSxFQUFHLFFBQVEsRUFBRyxRQUFRLEVBQUcsT0FBTyxFQUFHLFFBQVE7WUFDN0QsUUFBUSxFQUFFLFFBQVEsRUFBRyxRQUFRLEVBQUcsUUFBUSxFQUFHLFFBQVEsRUFBRSxRQUFRO1lBQzdELFFBQVEsRUFBRSxRQUFRLEVBQUcsU0FBUyxFQUFFLFFBQVEsRUFBRyxPQUFPLEVBQUcsUUFBUTtZQUM3RCxRQUFRLEVBQUUsUUFBUSxFQUFHLFFBQVEsRUFBRyxRQUFRLEVBQUcsUUFBUSxFQUFFLFFBQVE7U0FDOUQsRUFDRCxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakI7WUFDRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRO1lBQ3BFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxNQUFNLEVBQUUsUUFBUSxFQUFFLFFBQVE7WUFDbEUsUUFBUSxFQUFFLE9BQU87U0FDbEIsRUFDRCxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUM7UUFDcEMsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztRQUNyRCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFM0MsTUFBTSxRQUFRLEdBQUc7WUFDZixRQUFRLEVBQUUsTUFBTSxFQUFJLFFBQVEsRUFBRSxRQUFRLEVBQUUsT0FBTyxFQUFJLE9BQU8sRUFBRyxRQUFRO1lBQ3JFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUcsUUFBUSxFQUFFLE9BQU87WUFDcEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsT0FBTyxFQUFHLFFBQVEsRUFBRyxNQUFNLEVBQUksUUFBUTtZQUNyRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsT0FBTyxFQUFJLFFBQVEsRUFBRSxRQUFRO1lBQ3JFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxTQUFTLEVBQUUsT0FBTyxFQUFHLFFBQVE7WUFDckUsUUFBUSxFQUFFLE9BQU8sRUFBRyxPQUFPLEVBQUcsT0FBTyxFQUFHLE9BQU8sRUFBSSxRQUFRLEVBQUUsT0FBTztZQUNwRSxRQUFRLEVBQUUsT0FBTyxFQUFHLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFHLFFBQVEsRUFBRSxRQUFRO1lBQ3JFLFFBQVEsRUFBRSxPQUFPLEVBQUcsT0FBTyxFQUFHLE9BQU8sRUFBRyxRQUFRLEVBQUcsT0FBTyxFQUFHLE9BQU87WUFDcEUsUUFBUSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRyxRQUFRLEVBQUUsUUFBUTtZQUNyRSxRQUFRLEVBQUUsUUFBUSxFQUFF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LE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDM0MsaUJBQWlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUU7WUFDckMsVUFBVSxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUksU0FBUyxFQUFHLFNBQVMsRUFBRyxTQUFTO1lBQ3BFLFNBQVMsRUFBRyxTQUFTLEVBQUUsU0FBUyxFQUFHLFNBQVMsRUFBRyxVQUFVLEVBQUUsUUFBUTtZQUNuRSxRQUFRLEVBQUksU0FBUyxFQUFFLFVBQVUsRUFBRSxRQUFRLEVBQUksVUFBVSxFQUFFLFVBQVU7WUFDckUsVUFBVSxFQUFFLFNBQVMsRUFBRSxVQUFVLEVBQUUsU0FBUyxFQUFHLFVBQVUsRUFBRSxTQUFTO1lBQ3BFLFVBQVUsRUFBRSxRQUFRLEVBQUcsVUFBVSxFQUFFLFNBQVMsRUFBRyxRQUFRLEVBQUksU0FBUztZQUNwRSxTQUFTLEVBQUcsU0FBUyxFQUFFLFNBQVMsRUFBRyxRQUFRLEVBQUksU0FBUyxFQUFHLFNBQVM7WUFDcEUsU0FBUyxFQUFHLFNBQVMsRUFBRSxTQUFTLEVBQUcsU0FBUyxFQUFHLFNBQVMsRUFBRyxTQUFTO1lBQ3BFLFNBQVMsRUFBRyxTQUFTLEVBQUUsU0FBUyxFQUFHLFNBQVMsRUFBRyxTQUFTLEVBQUcsU0FBUztZQUNwRSxTQUFTLEVBQUcsU0FBUyxFQUFFLFNBQVMsRUFBRyxTQUFTLEVBQUcsVUFBVSxFQUFFLFFBQVE7WUFDbkUsVUFBVSxFQUFFLFFBQVEsRUFBRyxTQUFTLEVBQUcsU0FBUyxFQUFHLFFBQVEsRUFBSSxTQUFTO1lBQ3BFLFFBQVEsRUFBSSxTQUFTLEVBQUUsU0FBUyxFQUFHLFNBQVMsRUFBRyxTQUFTLEVBQUcsUUFBUTtZQUNuRSxTQUFTLEVBQUcsU0FBUyxFQUFFLFNBQVMsRUFBRyxTQUFTLEVBQUcsVUFBVSxFQUFFLFVBQVU7WUFDckUsVUFBVSxFQUFFLFNBQVMsRUFBRSxTQUFTLEVBQUcsU0FBUyxFQUFHLFNBQVMsRUFB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USxFQUFFLFNBQVMsRUFBRSxTQUFTLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQztRQUNqRSxNQUFNLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi4vaW5kZXgnO1xuaW1wb3J0IHtBTExfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3N9IGZyb20gJy4uL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge2V4cGVjdEFycmF5c0Nsb3NlfSBmcm9tICcuLi90ZXN0X3V0aWwnO1xuaW1wb3J0IHtSYW5rfSBmcm9tICcuLi90eXBlcyc7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdkZXB0aHdpc2VDb252MkQnLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnaW5wdXQ9MXgzeDN4MSxmPTIscz0xLGQ9MSxwPXZhbGlkLGNoTXVsPTEnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9ICd2YWxpZCc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjIzMDY2NCwgMC45ODczODgsIDAuMDY4NTIwOCwgMC40MTkyMjQsIDAuODg3ODYxLCAwLjczMTY0MSxcbiAgICAgICAgICAwLjA3NDE5MDcsIDAuNDA5MjY1LCAwLjM1MTM3N1xuICAgICAgICBdLFxuICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuMzAzODczLCAwLjIyOTIyMywgMC4xNDQzMzMsIDAuODAzMzczXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICk7XG5cbiAgICBjb25zdCByZXN1bHQgPSB0Zi5kZXB0aHdpc2VDb252MmQoeCwgdywgc3RyaWRlLCBwYWQpO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzEsIDIsIDIsIDFdKTtcbiAgICBjb25zdCBleHBlY3RlZCA9IFsxLjA3MDIyLCAxLjAzMTY3LCAwLjY3MDQxLCAwLjc3ODg2M107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgZXhwZWN0ZWQpO1xuICB9KTtcblxuICBpdCgnaW5wdXQ9MXgzeDN4MSxmPTIscz0xLGQ9MSxwPWV4cGxpY2l0LGNoTXVsPTEnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9XG4gICAgICAgIFtbMCwgMF0sIFsxLCAyXSwgWzAsIDFdLCBbMCwgMF1dIGFzIHRmLmJhY2tlbmRfdXRpbC5FeHBsaWNpdFBhZGRpbmc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjIzMDY2NCwgMC45ODczODgsIDAuMDY4NTIwOCwgMC40MTkyMjQsIDAuODg3ODYxLCAwLjczMTY0MSxcbiAgICAgICAgICAwLjA3NDE5MDcsIDAuNDA5MjY1LCAwLjM1MTM3N1xuICAgICAgICBdLFxuICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuMzAzODczLCAwLjIyOTIyMywgMC4xNDQzMzMsIDAuODAzMzczXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICk7XG5cbiAgICBjb25zdCByZXN1bHQgPSB0Zi5kZXB0aHdpc2VDb252MmQoeCwgdywgc3RyaWRlLCBwYWQpO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzEsIDUsIDMsIDFdKTtcbiAgICBjb25zdCBleHBlY3RlZCA9IFtcbiAgICAgIDAuODI2NTMzLCAwLjE5NzU2MCwgMC4wMDk4ODk4LCAxLjA3MDIxNiwgMS4wMzE2NzUsIDAuMTI2NDIyLCAwLjY3MDQwOTYsXG4gICAgICAwLjc3ODg2MywgMC4yNzMwNDEsIDAuMTE2MzU3LCAwLjIwNDkwOCwgMC4xMDY3NzQsIDAsIDAsIDBcbiAgICBdO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTF4NXg1eDEsZj0zLHM9MSxkPTEscD12YWxpZCxjaE11bD0xJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGZTaXplID0gMztcbiAgICBjb25zdCBwYWQgPSAndmFsaWQnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgY2hNdWwgPSAxO1xuICAgIGNvbnN0IGluRGVwdGggPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4xNDkxOTQsIDAuMDg5MDA5LCAwLjY1NDg5MSwgMC4wODMzMjQsIDAuNTM3MDQzLCAwLjY0NDMzMSwgMC41NjMwMzcsXG4gICAgICAgICAgMC4yMTE4NTksIDAuNjMzNTAxLCAwLjE4NjQyNywgMC43NzcwMzQsIDAuNTAwMDEsICAwLjYwNzM0MSwgMC45NTMwMyxcbiAgICAgICAgICAwLjY5NjQ3OSwgMC4wNTAzODcsIDAuNjIwNDUsICAwLjcyODA0OSwgMC4wMjgwNDMsIDAuNDM3MDA5LCAwLjcxMjg4MSxcbiAgICAgICAgICAwLjc0MTkzNSwgMC45NzQ0NzQsIDAuNjIxMTAyLCAwLjE3MTQxMVxuICAgICAgICBdLFxuICAgICAgICBbMSwgNSwgNSwgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDAuMTI1Mzg2LCAwLjk3NTE5OSwgMC42NDA0MzcsIDAuMjgxODk1LCAwLjk5MDk2OCwgMC4zNDcyMDgsIDAuODg5NzAyLFxuICAgICAgICAgIDAuMTgwNjk1LCAwLjY5MTk5MlxuICAgICAgICBdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0sXG4gICAgKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgMywgMywgMV0pO1xuICAgIGNvbnN0IGV4cGVjdGVkID0gW1xuICAgICAgMi41NDAwMjIsIDIuNTA1ODg1LCAyLjQ1NDA2MiwgMi4zNTE3MDEsIDIuNDU5NjAxLCAzLjA3NjQyMSwgMy4yOTg0OCxcbiAgICAgIDMuNDM3NDIxLCAyLjkzNDE5XG4gICAgXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdpbnB1dD0xeDN4M3gxLGY9MixzPTEsZD0yLHA9dmFsaWQsY2hNdWw9MScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3QgcGFkID0gJ3ZhbGlkJztcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuICAgIGNvbnN0IGRpbGF0aW9uID0gMjtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjIzMDY2NCwgMC45ODczODgsIDAuMDY4NTIwOCwgMC40MTkyMjQsIDAuODg3ODYxLCAwLjczMTY0MSxcbiAgICAgICAgICAwLjA3NDE5MDcsIDAuNDA5MjY1LCAwLjM1MTM3N1xuICAgICAgICBdLFxuICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuMzAzODczLCAwLjIyOTIyMywgMC4xNDQzMzMsIDAuODAzMzczXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICk7XG4gICAgLy8gYWRkaW5nIGEgZGlsYXRpb24gcmF0ZSBpcyBlcXVpdmFsZW50IHRvIHVzaW5nIGEgZmlsdGVyXG4gICAgLy8gd2l0aCAwcyBmb3IgdGhlIGRpbGF0aW9uIHJhdGVcbiAgICBjb25zdCBmU2l6ZURpbGF0ZWQgPSBmU2l6ZSArIChmU2l6ZSAtIDEpICogKGRpbGF0aW9uIC0gMSk7XG4gICAgY29uc3Qgd0RpbGF0ZWQgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuMzAzODczLCAwLCAwLjIyOTIyMywgMCwgMCwgMCwgMC4xNDQzMzMsIDAsIDAuODAzMzczXSxcbiAgICAgICAgW2ZTaXplRGlsYXRlZCwgZlNpemVEaWxhdGVkLCBpbkRlcHRoLCBjaE11bF0sXG4gICAgKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCwgJ05IV0MnLCBkaWxhdGlvbik7XG5cbiAgICBjb25zdCBleHBlY3RlZFJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3RGlsYXRlZCwgc3RyaWRlLCBwYWQpO1xuXG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChleHBlY3RlZFJlc3VsdC5zaGFwZSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgYXdhaXQgZXhwZWN0ZWRSZXN1bHQuZGF0YSgpKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTF4NXg1eDEsZj0zLHM9MSxkPTIscD12YWxpZCxjaE11bD0xJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGZTaXplID0gMztcbiAgICBjb25zdCBwYWQgPSAndmFsaWQnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgZGlsYXRpb24gPSAyO1xuICAgIGNvbnN0IGNoTXVsID0gMTtcbiAgICBjb25zdCBpbkRlcHRoID0gMTtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDAuMTQ5MTk0LCAwLjA4OTAwOSwgMC42NTQ4OTEsIDAuMDgzMzI0LCAwLjUzNzA0MywgMC42NDQzMzEsIDAuNTYzMDM3LFxuICAgICAgICAgIDAuMjExODU5LCAwLjYzMzUwMSwgMC4xODY0MjcsIDAuNzc3MDM0LCAwLjUwMDAxLCAgMC42MDczNDEsIDAuOTUzMDMsXG4gICAgICAgICAgMC42OTY0NzksIDAuMDUwMzg3LCAwLjYyMDQ1LCAgMC43MjgwNDksIDAuMDI4MDQzLCAwLjQzNzAwOSwgMC43MTI4ODEsXG4gICAgICAgICAgMC43NDE5MzUsIDAuOTc0NDc0LCAwLjYyMTEwMiwgMC4xNzE0MTFcbiAgICAgICAgXSxcbiAgICAgICAgWzEsIDUsIDUsIGluRGVwdGhdKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjEyNTM4NiwgMC45NzUxOTksIDAuNjQwNDM3LCAwLjI4MTg5NSwgMC45OTA5NjgsIDAuMzQ3MjA4LCAwLjg4OTcwMixcbiAgICAgICAgICAwLjE4MDY5NSwgMC42OTE5OTJcbiAgICAgICAgXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICk7XG4gICAgLy8gYWRkaW5nIGEgZGlsYXRpb24gcmF0ZSBpcyBlcXVpdmFsZW50IHRvIHVzaW5nIGEgZmlsdGVyXG4gICAgLy8gd2l0aCAwcyBmb3IgdGhlIGRpbGF0aW9uIHJhdGVcbiAgICBjb25zdCBmU2l6ZURpbGF0ZWQgPSBmU2l6ZSArIChmU2l6ZSAtIDEpICogKGRpbGF0aW9uIC0gMSk7XG4gICAgY29uc3Qgd0RpbGF0ZWQgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDAuMTI1Mzg2LCAwLCAwLjk3NTE5OSwgMCwgMC42NDA0MzcsIDAsIDAsIDAsIDAsIDAsXG4gICAgICAgICAgMC4yODE4OTUsIDAsIDAuOTkwOTY4LCAwLCAwLjM0NzIwOCwgMCwgMCwgMCwgMCwgMCxcbiAgICAgICAgICAwLjg4OTcwMiwgMCwgMC4xODA2OTUsIDAsIDAuNjkxOTkyXG4gICAgICAgIF0sXG4gICAgICAgIFtmU2l6ZURpbGF0ZWQsIGZTaXplRGlsYXRlZCwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICk7XG5cbiAgICBjb25zdCByZXN1bHQgPSB0Zi5kZXB0aHdpc2VDb252MmQoeCwgdywgc3RyaWRlLCBwYWQsICdOSFdDJywgZGlsYXRpb24pO1xuXG4gICAgY29uc3QgZXhwZWN0ZWRSZXN1bHQgPSB0Zi5kZXB0aHdpc2VDb252MmQoeCwgd0RpbGF0ZWQsIHN0cmlkZSwgcGFkKTtcblxuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoZXhwZWN0ZWRSZXN1bHQuc2hhcGUpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGF3YWl0IGV4cGVjdGVkUmVzdWx0LmRhdGEoKSk7XG4gIH0pO1xuXG4gIGl0KCdpbnB1dD0xeDV4NXgxLGY9MixzPTEsZD00LHA9dmFsaWQsY2hNdWw9MScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3QgcGFkID0gJ3ZhbGlkJztcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuICAgIGNvbnN0IGRpbGF0aW9uID0gNDtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjE0OTE5NCwgMC4wODkwMDksIDAuNjU0ODkxLCAwLjA4MzMyNCwgMC41MzcwNDMsIDAuNjQ0MzMxLCAwLjU2MzAzNyxcbiAgICAgICAgICAwLjIxMTg1OSwgMC42MzM1MDEsIDAuMTg2NDI3LCAwLjc3NzAzNCwgMC41MDAwMSwgIDAuNjA3MzQxLCAwLjk1MzAzLFxuICAgICAgICAgIDAuNjk2NDc5LCAwLjA1MDM4NywgMC42MjA0NSwgIDAuNzI4MDQ5LCAwLjAyODA0MywgMC40MzcwMDksIDAuNzEyODgxLFxuICAgICAgICAgIDAuNzQxOTM1LCAwLjk3NDQ3NCwgMC42MjExMDIsIDAuMTcxNDExXG4gICAgICAgIF0sXG4gICAgICAgIFsxLCA1LCA1LCBpbkRlcHRoXSk7XG4gICAgY29uc3QgdyA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMC4xMjUzODYsIDAuOTc1MTk5LCAwLjY0MDQzNywgMC4yODE4OTVdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0sXG4gICAgKTtcbiAgICAvLyBhZGRpbmcgYSBkaWxhdGlvbiByYXRlIGlzIGVxdWl2YWxlbnQgdG8gdXNpbmcgYSBmaWx0ZXJcbiAgICAvLyB3aXRoIDBzIGZvciB0aGUgZGlsYXRpb24gcmF0ZVxuICAgIGNvbnN0IGZTaXplRGlsYXRlZCA9IGZTaXplICsgKGZTaXplIC0gMSkgKiAoZGlsYXRpb24gLSAxKTtcbiAgICBjb25zdCB3RGlsYXRlZCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4xMjUzODYsIDAsIDAsIDAsIDAuOTc1MTk5LCAwLCAwLCAwLCAgICAgICAgMCwgMCwgMCwgMCwgICAgICAgMCxcbiAgICAgICAgICAwLCAgICAgICAgMCwgMCwgMCwgMCwgICAgICAgIDAsIDAsIDAuNjQwNDM3LCAwLCAwLCAwLCAwLjI4MTg5NVxuICAgICAgICBdLFxuICAgICAgICBbZlNpemVEaWxhdGVkLCBmU2l6ZURpbGF0ZWQsIGluRGVwdGgsIGNoTXVsXSxcbiAgICApO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZGVwdGh3aXNlQ29udjJkKHgsIHcsIHN0cmlkZSwgcGFkLCAnTkhXQycsIGRpbGF0aW9uKTtcblxuICAgIGNvbnN0IGV4cGVjdGVkUmVzdWx0ID0gdGYuZGVwdGh3aXNlQ29udjJkKHgsIHdEaWxhdGVkLCBzdHJpZGUsIHBhZCk7XG5cbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKGV4cGVjdGVkUmVzdWx0LnNoYXBlKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBhd2FpdCBleHBlY3RlZFJlc3VsdC5kYXRhKCkpO1xuICB9KTtcblxuICBpdCgnaW5wdXQ9MXgzeDN4MixmPTIscz0xLGQ9MSxwPXNhbWUsY2hNdWw9MScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgY2hNdWwgPSAxO1xuICAgIGNvbnN0IGluRGVwdGggPSAyO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4xMTEwNTcsIDAuNjYxODE4LCAwLjcwMTk3OSwgMC40MjQzNjIsIDAuOTkyODU0LCAwLjQxNzU5OSwgMC40MjMwMzYsXG4gICAgICAgICAgMC41MDA0OTksIDAuMzY4NDg0LCAwLjcxNDEzNSwgMC40NTY2OTMsIDAuNTMxMDU4LCAwLjYzNjYzNiwgMC4zNDUwMjQsXG4gICAgICAgICAgMC4wNTA2MzAzLCAwLjc4OTY4MiwgMC4xNzc0NzMsIDAuNzkzNTY5XG4gICAgICAgIF0sXG4gICAgICAgIFsxLCAzLCAzLCBpbkRlcHRoXSk7XG4gICAgY29uc3QgdyA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC42MTQyOTMsIDAuMDY0ODAxMSwgMC4xMDExMTMsIDAuNDUyODg3LCAwLjA1ODI3NDYsIDAuNDI2NDgxLFxuICAgICAgICAgIDAuODcyNzQzLCAwLjc2NTc2N1xuICAgICAgICBdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0pO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgMywgMywgMl0pO1xuXG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbXG4gICAgICAwLjQ4NTQ0NSwgMC45OTUzODksIDAuOTUxNjYsIDAuOTI3ODU2LCAwLjYzNjUxNiwgMC4yNTM1NDcsIDAuMzc4NDE0LFxuICAgICAgMS4xMDc3MSwgMC40MzAzNzMsIDEuMjMxMjYsIDAuMjkwODg1LCAwLjM3Mjg1NSwgMC4zOTYyLCAwLjM3OTk5NSxcbiAgICAgIDAuMDQ5MDQ2NiwgMC40MTA1NjksIDAuMTA5MDIsIDAuMDUxNDI0MlxuICAgIF07XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgZXhwZWN0ZWQpO1xuICB9KTtcblxuICBpdCgnaW5wdXQ9MXg1eDV4MSxmPTMscz0xLGQ9MSxwPXNhbWUsY2hNdWw9MScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmU2l6ZSA9IDM7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgY2hNdWwgPSAxO1xuICAgIGNvbnN0IGluRGVwdGggPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4xNDkxOTQsIDAuMDg5MDA5LCAwLjY1NDg5MSwgMC4wODMzMjQsIDAuNTM3MDQzLCAwLjY0NDMzMSwgMC41NjMwMzcsXG4gICAgICAgICAgMC4yMTE4NTksIDAuNjMzNTAxLCAwLjE4NjQyNywgMC43NzcwMzQsIDAuNTAwMDEsICAwLjYwNzM0MSwgMC45NTMwMyxcbiAgICAgICAgICAwLjY5NjQ3OSwgMC4wNTAzODcsIDAuNjIwNDUsICAwLjcyODA0OSwgMC4wMjgwNDMsIDAuNDM3MDA5LCAwLjcxMjg4MSxcbiAgICAgICAgICAwLjc0MTkzNSwgMC45NzQ0NzQsIDAuNjIxMTAyLCAwLjE3MTQxMVxuICAgICAgICBdLFxuICAgICAgICBbMSwgNSwgNSwgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDAuMTI1Mzg2LCAwLjk3NTE5OSwgMC42NDA0MzcsIDAuMjgxODk1LCAwLjk5MDk2OCwgMC4zNDcyMDgsIDAuODg5NzAyLFxuICAgICAgICAgIDAuMTgwNjk1LCAwLjY5MTk5MlxuICAgICAgICBdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0sXG4gICAgKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgNSwgNSwgMV0pO1xuICAgIGNvbnN0IGV4cGVjdGVkID0gW1xuICAgICAgMC42ODQ3OTYsIDEuMTc5MjUxLCAxLjY4MDU5MywgMC44ODU2MTUsIDEuMTUyOTk1LCAxLjUyMjkxLCAgMi41NDAwMjIsXG4gICAgICAyLjUwNTg4NSwgMi40NTQwNjIsIDEuODcxMjU4LCAyLjM3MTAxNSwgMi4zNTE3MDEsIDIuNDU5NjAxLCAzLjA3NjQyMSxcbiAgICAgIDEuMzIzOTk0LCAxLjk4NTU3MiwgMy4yOTg0OCwgIDMuNDM3NDIxLCAyLjkzNDE5LCAgMS44MjMyMzgsIDEuNDEwNTQ1LFxuICAgICAgMi4zNTIxODYsIDIuMTk2MjIsICAxLjM0ODIxOCwgMC43NzQ2MzVcbiAgICBdO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTF4M3gzeDIsZj0yLHM9MSxkPTIscD1zYW1lLGNoTXVsPTEnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuICAgIGNvbnN0IGRpbGF0aW9uID0gMjtcbiAgICBjb25zdCBpbkRlcHRoID0gMjtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDAuMTExMDU3LCAwLjY2MTgxOCwgMC43MDE5NzksIDAuNDI0MzYyLCAwLjk5Mjg1NCwgMC40MTc1OTksIDAuNDIzMDM2LFxuICAgICAgICAgIDAuNTAwNDk5LCAwLjM2ODQ4NCwgMC43MTQxMzUsIDAuNDU2NjkzLCAwLjUzMTA1OCwgMC42MzY2MzYsIDAuMzQ1MDI0LFxuICAgICAgICAgIDAuMDUwNjMwMywgMC43ODk2ODIsIDAuMTc3NDczLCAwLjc5MzU2OVxuICAgICAgICBdLFxuICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuXG4gICAgY29uc3QgdzogdGYuVGVuc29yNEQgPVxuICAgICAgICB0Zi5zdGFjayhcbiAgICAgICAgICAgICAgW1xuICAgICAgICAgICAgICAgIHRmLnRlbnNvcjJkKFxuICAgICAgICAgICAgICAgICAgICBbMC42MTQyOTMsIDAuMDY0ODAxMSwgMC4xMDExMTMsIDAuNDUyODg3XSwgW2ZTaXplLCBmU2l6ZV0pLFxuICAgICAgICAgICAgICAgIHRmLnRlbnNvcjJkKFxuICAgICAgICAgICAgICAgICAgICBbMC4wNTgyNzQ2LCAwLjQyNjQ4MSwgMC44NzI3NDMsIDAuNzY1NzY3XSwgW2ZTaXplLCBmU2l6ZV0pXG4gICAgICAgICAgICAgIF0sXG4gICAgICAgICAgICAgIDIpXG4gICAgICAgICAgICAuZXhwYW5kRGltcygzKTtcblxuICAgIC8vIGFkZGluZyBhIGRpbGF0aW9uIHJhdGUgaXMgZXF1aXZhbGVudCB0byB1c2luZyBhIGZpbHRlclxuICAgIC8vIHdpdGggMHMgZm9yIHRoZSBkaWxhdGlvbiByYXRlXG4gICAgY29uc3QgZlNpemVEaWxhdGVkID0gZlNpemUgKyAoZlNpemUgLSAxKSAqIChkaWxhdGlvbiAtIDEpO1xuICAgIGNvbnN0IHdEaWxhdGVkOiB0Zi5UZW5zb3I0RCA9XG4gICAgICAgIHRmLnN0YWNrKFxuICAgICAgICAgICAgICBbXG4gICAgICAgICAgICAgICAgdGYudGVuc29yMmQoXG4gICAgICAgICAgICAgICAgICAgIFswLjYxNDI5MywgMCwgMC4wNjQ4MDExLCAwLCAwLCAwLCAwLjEwMTExMywgMCwgMC40NTI4ODddLFxuICAgICAgICAgICAgICAgICAgICBbZlNpemVEaWxhdGVkLCBmU2l6ZURpbGF0ZWRdKSxcbiAgICAgICAgICAgICAgICB0Zi50ZW5zb3IyZChcbiAgICAgICAgICAgICAgICAgICAgWzAuMDU4Mjc0NiwgMCwgMC40MjY0ODEsIDAsIDAsIDAsIDAuODcyNzQzLCAwLCAwLjc2NTc2N10sXG4gICAgICAgICAgICAgICAgICAgIFtmU2l6ZURpbGF0ZWQsIGZTaXplRGlsYXRlZF0pXG4gICAgICAgICAgICAgIF0sXG4gICAgICAgICAgICAgIDIpXG4gICAgICAgICAgICAuZXhwYW5kRGltcygzKTtcblxuICAgIGV4cGVjdCh3RGlsYXRlZC5zaGFwZSkudG9FcXVhbChbZlNpemVEaWxhdGVkLCBmU2l6ZURpbGF0ZWQsIGluRGVwdGgsIDFdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCwgJ05IV0MnLCBkaWxhdGlvbik7XG5cbiAgICBjb25zdCBleHBlY3RlZFJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3RGlsYXRlZCwgc3RyaWRlLCBwYWQpO1xuXG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChleHBlY3RlZFJlc3VsdC5zaGFwZSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgYXdhaXQgZXhwZWN0ZWRSZXN1bHQuZGF0YSgpKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTF4NXg1eDEsZj0zLHM9MSxkPTIscD1zYW1lLGNoTXVsPTEnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAzO1xuICAgIGNvbnN0IHBhZCA9ICd2YWxpZCc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjE0OTE5NCwgMC4wODkwMDksIDAuNjU0ODkxLCAwLjA4MzMyNCwgMC41MzcwNDMsIDAuNjQ0MzMxLCAwLjU2MzAzNyxcbiAgICAgICAgICAwLjIxMTg1OSwgMC42MzM1MDEsIDAuMTg2NDI3LCAwLjc3NzAzNCwgMC41MDAwMSwgIDAuNjA3MzQxLCAwLjk1MzAzLFxuICAgICAgICAgIDAuNjk2NDc5LCAwLjA1MDM4NywgMC42MjA0NSwgIDAuNzI4MDQ5LCAwLjAyODA0MywgMC40MzcwMDksIDAuNzEyODgxLFxuICAgICAgICAgIDAuNzQxOTM1LCAwLjk3NDQ3NCwgMC42MjExMDIsIDAuMTcxNDExXG4gICAgICAgIF0sXG4gICAgICAgIFsxLCA1LCA1LCBpbkRlcHRoXSk7XG4gICAgY29uc3QgdyA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4xMjUzODYsIDAuOTc1MTk5LCAwLjY0MDQzNywgMC4yODE4OTUsIDAuOTkwOTY4LCAwLjM0NzIwOCwgMC44ODk3MDIsXG4gICAgICAgICAgMC4xODA2OTUsIDAuNjkxOTkyXG4gICAgICAgIF0sXG4gICAgICAgIFtmU2l6ZSwgZlNpemUsIGluRGVwdGgsIGNoTXVsXSxcbiAgICApO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZGVwdGh3aXNlQ29udjJkKHgsIHcsIHN0cmlkZSwgcGFkKTtcbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKFsxLCAzLCAzLCAxXSk7XG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbXG4gICAgICAyLjU0MDAyMiwgMi41MDU4ODUsIDIuNDU0MDYyLCAyLjM1MTcwMSwgMi40NTk2MDEsIDMuMDc2NDIxLCAzLjI5ODQ4LFxuICAgICAgMy40Mzc0MjEsIDIuOTM0MTlcbiAgICBdO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTF4NXg1eDEsZj0zLHM9MSxkPTIscD1leHBsaWNpdCxjaE11bD0xJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGZTaXplID0gMztcbiAgICBjb25zdCBwYWQgPVxuICAgICAgICBbWzAsIDBdLCBbMCwgMF0sIFswLCAxXSwgWzAsIDFdXSBhcyB0Zi5iYWNrZW5kX3V0aWwuRXhwbGljaXRQYWRkaW5nO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgY2hNdWwgPSAxO1xuICAgIGNvbnN0IGluRGVwdGggPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4xNDkxOTQsIDAuMDg5MDA5LCAwLjY1NDg5MSwgMC4wODMzMjQsIDAuNTM3MDQzLCAwLjY0NDMzMSwgMC41NjMwMzcsXG4gICAgICAgICAgMC4yMTE4NTksIDAuNjMzNTAxLCAwLjE4NjQyNywgMC43NzcwMzQsIDAuNTAwMDEsICAwLjYwNzM0MSwgMC45NTMwMyxcbiAgICAgICAgICAwLjY5NjQ3OSwgMC4wNTAzODcsIDAuNjIwNDUsICAwLjcyODA0OSwgMC4wMjgwNDMsIDAuNDM3MDA5LCAwLjcxMjg4MSxcbiAgICAgICAgICAwLjc0MTkzNSwgMC45NzQ0NzQsIDAuNjIxMTAyLCAwLjE3MTQxMVxuICAgICAgICBdLFxuICAgICAgICBbMSwgNSwgNSwgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDAuMTI1Mzg2LCAwLjk3NTE5OSwgMC42NDA0MzcsIDAuMjgxODk1LCAwLjk5MDk2OCwgMC4zNDcyMDgsIDAuODg5NzAyLFxuICAgICAgICAgIDAuMTgwNjk1LCAwLjY5MTk5MlxuICAgICAgICBdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0sXG4gICAgKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgMywgNCwgMV0pO1xuICAgIGNvbnN0IGV4cGVjdGVkID0gW1xuICAgICAgMi41NDAwMjIsIDIuNTA1ODg1LCAyLjQ1NDA2MiwgMS44NzEyNTgsIDIuMzUxNzAsIDIuNDU5NjAxLCAzLjA3NjQyMSxcbiAgICAgIDEuMzIzOTksIDMuMjk4NDgwLCAzLjQzNzQyMSwgMi45MzQxOSwgMS44MjMyMzhcbiAgICBdO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTF4M3gzeDIsZj0yLHM9MSxwPXNhbWUsY2hNdWw9MicsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgY2hNdWwgPSAyO1xuICAgIGNvbnN0IGluRGVwdGggPSAyO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC42NzU3MDcsIDAuNzU4NTY3LCAwLjQxMzUyOSwgMC45NjM5NjcsIDAuMjE3MjkxLCAwLjEwMTMzNSwgMC44MDQyMzEsXG4gICAgICAgICAgMC4zMjk2NzMsIDAuOTI0NTAzLCAwLjcyODc0MiwgMC4xODAyMTcsIDAuMjEwNDU5LCAwLjEzMzg2OSwgMC42NTA4MjcsXG4gICAgICAgICAgMC4wNDc2MTMsIDAuNTU0Nzk1LCAwLjY1MzM2NSwgMC40NDIxOTZcbiAgICAgICAgXSxcbiAgICAgICAgWzEsIDMsIDMsIGluRGVwdGhdKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjM0NzE1NCwgMC4zODY2OTIsIDAuMzI3MTkxLCAwLjQ4Mzc4NCwgMC41OTE4MDcsIDAuMjQyNjMsIDAuOTUxODIsXG4gICAgICAgICAgMC4xNzQzNTMsIDAuNTkyMTM2LCAwLjYyMzQ2OSwgMC45ODgyNDQsIDAuNjYwNzMxLCAwLjk0NjUzNCwgMC4wODAxMzY1LFxuICAgICAgICAgIDAuODY0ODg5LCAwLjg3NDYwMlxuICAgICAgICBdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0pO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgMywgMywgNF0pO1xuXG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbXG4gICAgICAxLjgzMDU5LCAgIDAuOTM3MTI1LCAgMi4xMjE4LCAgIDEuMzkwMjQsICAwLjk5MDE2NywgMC44MDM0NzIsXG4gICAgICAxLjMxNDA1LCAgIDEuMTQ5NTksICAgMC4xODIxNDcsIDAuMTk2Mzg1LCAwLjI0MTE0MSwgMC4xODgwODEsXG4gICAgICAwLjk1MDY1NiwgIDAuNjIyNTgxLCAgMS45MjQ1MSwgIDEuMjAxNzksICAxLjA3NDIyLCAgMC40ODMyNjgsXG4gICAgICAxLjM2OTQ4LCAgIDEuMTQyNTYsICAgMC40NDk0NDQsIDAuNDc3MDQyLCAwLjUwNTg1NywgMC4zOTM5ODksXG4gICAgICAwLjA3NDY1MDksIDAuMDYzMzE4NCwgMC43NDEwMSwgIDAuNDExNTksICAwLjQwMzE5NSwgMC4xNzY5MzgsXG4gICAgICAwLjYwMjQxNSwgIDAuMzQ1NDk5LCAgMC4yMjY4MTksIDAuMjUyNjUxLCAwLjE0NDY4MiwgMC4yMTM5MjdcbiAgICBdO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTJ4M3gzeDIsZj0yLHM9MSxwPXNhbWUsY2hNdWw9MicsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgY2hNdWwgPSAyO1xuICAgIGNvbnN0IGluRGVwdGggPSAyO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4yNjE5NDUsIDAuMDUyODExMywgMC42NTY2OTgsICAwLjEyNzM0NSwgIDAuNjEwMDM5LCAwLjE2OTEzMSxcbiAgICAgICAgICAwLjQ1ODY0NywgMC4wOTg4Mjg4LCAwLjk2NjEwOSwgIDAuMDQyMTc0NywgMC44MjAzNSwgIDAuMjc0NzExLFxuICAgICAgICAgIDAuMzU5Mzc3LCAwLjUxMjExMywgIDAuNjg5NjgyLCAgMC45NDE1NzEsICAwLjMxOTYxLCAgMC43NDM4MjYsXG4gICAgICAgICAgMC44NTgxNDcsIDAuOTg0NzY2LCAgMC45MjY5NzMsICAwLjU3OTU5NywgIDAuNDQ0MTA0LCAwLjUwNTk2OSxcbiAgICAgICAgICAwLjI0MTQzNywgMC45Mzc5OTksICAwLjA5NTcwNzQsIDAuNzczNjExLCAgMC40NjAyMywgIDAuNDY5Mzc5LFxuICAgICAgICAgIDAuMzYzNzg5LCAwLjI2OTc0NSwgIDAuNDg2MTM2LCAgMC44OTQyMTUsICAwLjc5NDI5OSwgMC43MjQ2MTVcbiAgICAgICAgXSxcbiAgICAgICAgWzIsIDMsIDMsIGluRGVwdGhdKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjI0MDM0NywgMC45MDYzNTIsIDAuNDc4NjU3LCAwLjgyNTkxOCwgMC4zODA3NjksIDAuMTg0NzA1LCAwLjIzODI0MSxcbiAgICAgICAgICAwLjIwMTkwNywgMC4yOTQwODcsIDAuMTgxMTY1LCAwLjE5MTMwMywgMC43MjI1LCAwLjQzMDA2NCwgMC45MDA2MjIsXG4gICAgICAgICAgMC42NzAzMzgsIDAuMzM0NzhcbiAgICAgICAgXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdKTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi5kZXB0aHdpc2VDb252MmQoeCwgdywgc3RyaWRlLCBwYWQpO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzIsIDMsIDMsIDRdKTtcblxuICAgIGNvbnN0IGV4cGVjdGVkID0gW1xuICAgICAgMC44NjMzNzksIDEuMzExOSwgICAwLjEwMjc5NSwgMC4xNTQ4NTMsIDEuMDI3MDQsICAgMS42MjE3MywgIDAuMjkzNDY2LFxuICAgICAgMC4yNjE3NjQsIDAuMzg3ODc2LCAwLjcwMTUyOSwgMC4xMzM1MDgsIDAuMzM4MTY3LCAgMC44ODAzOTUsIDEuMjgwMzksXG4gICAgICAwLjc4NjQ5MiwgMC43NzUzNjEsIDAuODg0ODQ1LCAxLjQzOTk1LCAgMC43NjQzNzQsICAxLjAxOTYsICAgMC4yOTExNjIsXG4gICAgICAwLjgwMTQyOCwgMC4yNzM3ODgsIDAuNzY0MzAzLCAwLjM0ODk4NSwgMC40NTMxMSwgICAwLjQ2OTQ0NywgMC42MTMwNzMsXG4gICAgICAwLjI4NzQ2MSwgMC42ODQxMjgsIDAuNjI3ODk5LCAwLjkyNzg0NCwgMC4wNzY4MTc0LCAwLjI4OTY4LCAgMC4zNTYwMzcsXG4gICAgICAwLjYxNDMzOSwgMC42NzEzOCwgIDEuMDc4OTQsICAxLjMwNzQ3LCAgMS44NjcwNSwgICAwLjYxNzk3MSwgMS4zNTQwMixcbiAgICAgIDAuODYwNjA3LCAxLjI5NjkzLCAgMC4yNDIwODcsIDAuNDg1ODkyLCAwLjMzMTk3OSwgIDAuNzU3MDE1LCAwLjQxMDUyNyxcbiAgICAgIDAuNzQwMjM1LCAxLjI4NDMxLCAgMS40MjUxNiwgIDAuNjgyODEsICAwLjk3NTE4NSwgIDEuMTM4OTIsICAxLjYyMjM3LFxuICAgICAgMC4zNDQyMDgsIDAuNTYxMDI5LCAwLjM2MzI5MiwgMC45MTEyMDMsIDAuMjcyNTQxLCAgMC40MTk1MTMsIDAuMzQyMTU0LFxuICAgICAgMC40MDMzMzUsIDAuNDE5Mjg2LCAwLjU4NzMyMSwgMC42MDA2NTUsIDAuODg0ODUzLCAgMC4xOTA5MDcsIDAuNzE5OTE0LFxuICAgICAgMC4zNDY4NDIsIDAuNTk4NDcyXG4gICAgXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdpbnB1dD0yeDN4M3gyLGY9MixzPTEsZD0yLHA9c2FtZSxjaE11bD0yJyxcbiAgICAgYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IGZTaXplID0gMjtcbiAgICAgICBjb25zdCBwYWQgPSAnc2FtZSc7XG4gICAgICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICAgICBjb25zdCBpbkRlcHRoID0gMjtcbiAgICAgICBjb25zdCBkaWxhdGlvbiA9IDI7XG4gICAgICAgY29uc3Qgbm9EaWxhdGlvbiA9IDE7XG5cbiAgICAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgICAgIFtcbiAgICAgICAgICAgICAwLjI2MTk0NSwgMC4wNTI4MTEzLCAwLjY1NjY5OCwgIDAuMTI3MzQ1LCAgMC42MTAwMzksIDAuMTY5MTMxLFxuICAgICAgICAgICAgIDAuNDU4NjQ3LCAwLjA5ODgyODgsIDAuOTY2MTA5LCAgMC4wNDIxNzQ3LCAwLjgyMDM1LCAgMC4yNzQ3MTEsXG4gICAgICAgICAgICAgMC4zNTkzNzcsIDAuNTEyMTEzLCAgMC42ODk2ODIsICAwLjk0MTU3MSwgIDAuMzE5NjEsICAwLjc0MzgyNixcbiAgICAgICAgICAgICAwLjg1ODE0NywgMC45ODQ3NjYsICAwLjkyNjk3MywgIDAuNTc5NTk3LCAgMC40NDQxMDQsIDAuNTA1OTY5LFxuICAgICAgICAgICAgIDAuMjQxNDM3LCAwLjkzNzk5OSwgIDAuMDk1NzA3NCwgMC43NzM2MTEsICAwLjQ2MDIzLCAgMC40NjkzNzksXG4gICAgICAgICAgICAgMC4zNjM3ODksIDAuMjY5NzQ1LCAgMC40ODYxMzYsICAwLjg5NDIxNSwgIDAuNzk0Mjk5LCAwLjcyNDYxNVxuICAgICAgICAgICBdLFxuICAgICAgICAgICBbMiwgMywgMywgaW5EZXB0aF0pO1xuXG4gICAgICAgY29uc3QgdyA9IHRmLnN0YWNrKFxuICAgICAgICAgICAgICAgICAgICAgW1xuICAgICAgICAgICAgICAgICAgICAgICB0Zi5zdGFjayhcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGYudGVuc29yMmQoXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbMC4yNDAzNDcsIDAuOTA2MzUyLCAwLjQ3ODY1NywgMC44MjU5MThdLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgW2ZTaXplLCBmU2l6ZV0pLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0Zi50ZW5zb3IyZChcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFswLjM4MDc2OSwgMC4xODQ3MDUsIDAuMjM4MjQxLCAwLjIwMTkwN10sXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbZlNpemUsIGZTaXplXSlcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIF0sXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyKSxcbiAgICAgICAgICAgICAgICAgICAgICAgdGYuc3RhY2soXG4gICAgICAgICAgICAgICAgICAgICAgICAgICBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjJkKFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgWzAuMjk0MDg3LCAwLjE4MTE2NSwgMC4xOTEzMDMsIDAuNzIyNV0sXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbZlNpemUsIGZTaXplXSksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjJkKFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgWzAuNDMwMDY0LCAwLjkwMDYyMiwgMC42NzAzMzgsIDAuMzM0NzhdLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgW2ZTaXplLCBmU2l6ZV0pXG4gICAgICAgICAgICAgICAgICAgICAgICAgICBdLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMilcbiAgICAgICAgICAgICAgICAgICAgIF0sXG4gICAgICAgICAgICAgICAgICAgICAzKSBhcyB0Zi5UZW5zb3I0RDtcblxuICAgICAgIGNvbnN0IGZTaXplRGlsYXRlZCA9IGZTaXplICsgKGZTaXplIC0gMSkgKiAoZGlsYXRpb24gLSAxKTtcbiAgICAgICBjb25zdCB3RGlsYXRlZCA9IHRmLnN0YWNrKFtcbiAgICAgIHRmLnN0YWNrKFxuICAgICAgICAgIFtcbiAgICAgICAgICAgIHRmLnRlbnNvcjJkKFxuICAgICAgICAgICAgICBbMC4yNDAzNDcsIDAsIDAuOTA2MzUyLCAwLCAwLCAwLCAwLjQ3ODY1NywgMCwgMC44MjU5MThdLFxuICAgICAgICAgICAgICBbZlNpemVEaWxhdGVkLCBmU2l6ZURpbGF0ZWRdKSxcbiAgICAgICAgICAgIHRmLnRlbnNvcjJkKFxuICAgICAgICAgICAgICBbMC4zODA3NjksIDAsIDAuMTg0NzA1LCAwLCAwLCAwLCAwLjIzODI0MSwgMCwgMC4yMDE5MDddLFxuICAgICAgICAgICAgICBbZlNpemVEaWxhdGVkLCBmU2l6ZURpbGF0ZWRdKVxuICAgICAgICAgIF0sXG4gICAgICAgICAgMiksXG4gICAgICB0Zi5zdGFjayhcbiAgICAgICAgICBbXG4gICAgICAgICAgICB0Zi50ZW5zb3IyZChbMC4yOTQwODcsIDAsIDAuMTgxMTY1LCAwLCAwLCAwLCAwLjE5MTMwMywgMCwgMC43MjI1XSxcbiAgICAgICAgICAgICAgW2ZTaXplRGlsYXRlZCwgZlNpemVEaWxhdGVkXSksXG4gICAgICAgICAgICB0Zi50ZW5zb3IyZChcbiAgICAgICAgICAgICAgWzAuNDMwMDY0LCAwLCAwLjkwMDYyMiwgMCwgMCwgMCwgMC42NzAzMzgsIDAsIDAuMzM0NzhdLFxuICAgICAgICAgICAgICBbZlNpemVEaWxhdGVkLCBmU2l6ZURpbGF0ZWRdKVxuICAgICAgICAgIF0sXG4gICAgICAgICAgMilcbiAgICBdLCAzKSBhcyB0Zi5UZW5zb3I0RDtcblxuICAgICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCwgJ05IV0MnLCBkaWxhdGlvbik7XG5cbiAgICAgICBjb25zdCBleHBlY3RlZFJlc3VsdCA9XG4gICAgICAgICAgIHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3RGlsYXRlZCwgc3RyaWRlLCBwYWQsICdOSFdDJywgbm9EaWxhdGlvbik7XG5cbiAgICAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKGV4cGVjdGVkUmVzdWx0LnNoYXBlKTtcbiAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBhd2FpdCBleHBlY3RlZFJlc3VsdC5kYXRhKCkpO1xuICAgICB9KTtcblxuICBpdCgnaW5wdXQ9MngzeDN4MixmPTMscz0xLGQ9MixwPXNhbWUsY2hNdWw9MicsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmU2l6ZSA9IDM7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDI7XG4gICAgY29uc3QgZGlsYXRpb24gPSAyO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbW1xuICAgICAgICAgIFtcbiAgICAgICAgICAgIFswLjUyMjc2MTQ2LCAwLjM0Nzc1OTg0XSwgWzAuNDY5MDA5NCwgMC40MDgxNjEwNF0sXG4gICAgICAgICAgICBbMC4zMjM5MDE1MywgMC4yMzcyOTA3NF0sIFswLjYxMzY2NzM3LCAwLjc5MTgxMDVdLFxuICAgICAgICAgICAgWzAuOTE0NTIxMSwgMC4yMTg2MTFdLCBbMC4zNzc4NzkyNiwgMC4yMzkyMzY0N10sXG4gICAgICAgICAgICBbMC4yMzQwMTM0NCwgMC4xMjUxOTgzNl1cbiAgICAgICAgICBdLFxuXG4gICAgICAgICAgW1xuICAgICAgICAgICAgWzAuNjIyMjUzNCwgMC4xMzI3MzYwOV0sIFswLjc2OTc3NTMsIDAuMTIxNjA1ODddLFxuICAgICAgICAgICAgWzAuMDQ0ODEyOCwgMC45NDgwNjYzNV0sIFswLjQxOTk5NTMsIDAuNzE0MDcxNF0sXG4gICAgICAgICAgICBbMC4wMTQyMDgzMiwgMC40NzQ1MzcxM10sIFswLjAyMDYxNDM5LCAwLjM3MjI2MTUyXSxcbiAgICAgICAgICAgIFswLjYyNzQxNDQ2LCAwLjIzMTY3MTgxXVxuICAgICAgICAgIF0sXG5cbiAgICAgICAgICBbXG4gICAgICAgICAgICBbMC43MjU3NTU3LCAwLjE0MzUyNzUxXSwgWzAuMzAxMTYzOCwgMC4zODY5MDY1XSxcbiAgICAgICAgICAgIFswLjA5Mjg2MTI5LCAwLjI1MTUxNzQyXSwgWzAuNzU2NjM5NywgMC4xMzA5OTkyMV0sXG4gICAgICAgICAgICBbMC42NTMyNDcyNCwgMC4zODk1OTM3Ml0sIFswLjY1ODI2LCAwLjc1MDUzMThdLFxuICAgICAgICAgICAgWzAuMzU5MTkwODIsIDAuODU0NzA3OTZdXG4gICAgICAgICAgXSxcblxuICAgICAgICAgIFtcbiAgICAgICAgICAgIFswLjI0ODI3MzYxLCAwLjI4MjY2NjFdLCBbMC4yNDcxNzI0NywgMC4yNzQ0Njg1NF0sXG4gICAgICAgICAgICBbMC4yNzExMjQ0OCwgMC42ODA2ODU2NF0sIFswLjExMDgyMjkyLCAwLjc5NDg2NzVdLFxuICAgICAgICAgICAgWzAuNDE1MzUzMTgsIDAuNjU5OTg2XSwgWzAuMjIxNjU1MjUsIDAuMTgxNDk1NzldLFxuICAgICAgICAgICAgWzAuNDIyNzMzNzgsIDAuOTU1ODI4MV1cbiAgICAgICAgICBdLFxuXG4gICAgICAgICAgW1xuICAgICAgICAgICAgWzAuOTQzMDc0LCAwLjY3OTkwNDFdLCBbMC43ODg1MTQ3MywgMC4wNzI0OTYwNl0sXG4gICAgICAgICAgICBbMC43NzE5MDksIDAuNzkyNTk2N10sIFswLjk1NTEwODMsIDAuMDMwODc1NjhdLFxuICAgICAgICAgICAgWzAuODI1ODk4MDUsIDAuOTQ3OTczODVdLCBbMC41ODk1NDYyLCAwLjUwNDU5MjNdLFxuICAgICAgICAgICAgWzAuOTY2Nzc1NCwgMC4yNDI5MjkyMl1cbiAgICAgICAgICBdLFxuXG4gICAgICAgICAgW1xuICAgICAgICAgICAgWzAuNjcxMjM2NjMsIDAuMTA5NzYxXSwgWzAuMDQwMDI3NjIsIDAuNTE5NDIyNzddLFxuICAgICAgICAgICAgWzAuMzc4Njg1MzYsIDAuODQ2NzYwM10sIFswLjc3MTcxMzg1LCAwLjUxNjA0NjA1XSxcbiAgICAgICAgICAgIFswLjgxOTI4NDksIDAuMzg4NDM2NjhdLCBbMC4xOTYwNzQ4NCwgMC41NTkxNjI0XSxcbiAgICAgICAgICAgIFswLjQ1OTkwODI1LCAwLjM1NzY4MzE4XVxuICAgICAgICAgIF0sXG5cbiAgICAgICAgICBbXG4gICAgICAgICAgICBbMC42NzQ0MzU4NSwgMC42MjU2MTY4XSwgWzAuOTM3MzYyMywgMC42NDk4MzkzXSxcbiAgICAgICAgICAgIFswLjc2MjMwODUsIDAuMTMyMTgxMDVdLCBbMC45MzQ5NjMxLCAwLjc2NjAxOTFdLFxuICAgICAgICAgICAgWzAuNTAwNTQ5NDQsIDAuNzczODEyM10sIFswLjMwMjAxOTQ4LCAwLjUyNTY0M10sXG4gICAgICAgICAgICBbMC4zMDg5NjM0MiwgMC4yMTExMTU5Nl1cbiAgICAgICAgICBdXG4gICAgICAgIF1dLFxuICAgICAgICBbMSwgNywgNywgaW5EZXB0aF0pO1xuXG4gICAgY29uc3QgdyA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgW1xuICAgICAgICAgICAgW1swLjY1MTEzNzIzXSwgWzAuODY5OTQ0N11dLFxuXG4gICAgICAgICAgICBbWzAuMjY3NzkyXSwgWzAuOTk4MTc4N11dLFxuXG4gICAgICAgICAgICBbWzAuNDkxMzU3Ml0sIFswLjMzMjExOTU4XV1cbiAgICAgICAgICBdLFxuICAgICAgICAgIFtcbiAgICAgICAgICAgIFtbMC41Mjg2NDk3XSwgWzAuNDI0MTgwMjddXSxcblxuICAgICAgICAgICAgW1swLjAxNzU0NDYzXSwgWzAuODM2NTQ2NF1dLFxuXG4gICAgICAgICAgICBbWzAuMTc2ODM5OTVdLCBbMC4yODc0ODMxXV1cbiAgICAgICAgICBdLFxuICAgICAgICAgIFtcbiAgICAgICAgICAgIFtbMC4wOTMzOTk3Nl0sIFswLjU3NjQ1NDc2XV0sXG5cbiAgICAgICAgICAgIFtbMC4wNjYxNjIzNV0sIFswLjg4NTAyNzNdXSxcblxuICAgICAgICAgICAgW1swLjg3MDA5Mjg3XSwgWzAuMjA1NDIyMDRdXVxuICAgICAgICAgIF1cbiAgICAgICAgXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgMV0sXG4gICAgKTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi5kZXB0aHdpc2VDb252MmQoeCwgdywgc3RyaWRlLCBwYWQsICdOSFdDJywgZGlsYXRpb24pO1xuXG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgNywgNywgMl0pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFtcbiAgICAgIDAuMTk1MjY2MDQsIDAuNTM3ODI3MywgMC43OTUwMjIsICAgMC45Mzg0MTA3LCAgMS4wODYwNzk0LCAgMC43OTQyMzI2LFxuICAgICAgMC45NzY0Njk0LCAgMS4zOTc0NDQyLCAwLjU5MzA4MTMsICAwLjk4NDg5MDEsICAwLjQ0NTI2Njg0LCAxLjI3NTc1OSxcbiAgICAgIDAuNTcyMzQ1LCAgIDEuMTc4NDg3OCwgMC4yNzExNzE3NSwgMC43NzM1ODgsICAgMC4yMDA1NTcxMSwgMC43MTMyMDc4NCxcbiAgICAgIDAuNzM0Nzc1NjYsIDEuODg2NzcyMiwgMC42NDEyMzQzNCwgMS42NTQ5MzY5LCAgMC41NTU1MTI4NSwgMi4wMzg1NjMzLFxuICAgICAgMC4yNDc0MDgxMiwgMS4yMzMxNDMsICAwLjA4NTI4MTkyLCAxLjYyMTQ3OTUsICAxLjA2MjMyNiwgICAxLjM4Mjg2MDMsXG4gICAgICAxLjQ0OTQxNzYsICAxLjEwMjIyMjIsIDIuMjM1MDY2NCwgIDIuMjgzNDIzLCAgIDEuNTk0MDg5NSwgIDEuODg3MTQyNCxcbiAgICAgIDEuNjYyNzg1MiwgIDIuNDkwMzIxMiwgMS4wNDA1MzM3LCAgMi4wNzU0MzA0LCAgMS4xNTA4ODkzLCAgMS45NTY4NzM3LFxuICAgICAgMC42MTQ4NTcxLCAgMS4xNTA1OTk1LCAxLjExMDU1MjgsICAxLjM4MjM2ODcsICAxLjQzNDIxMzksICAyLjk5MDk0ODcsXG4gICAgICAxLjAyMTAzOTYsICAyLjY0Njc0NDMsIDEuMDU2Mzc5OCwgIDMuMzk2Mzc5NywgIDAuNDI2NTIwOTcsIDIuMjc0MTM0LFxuICAgICAgMC41MTEyMTA3NCwgMi4yNjQwOTQsICAxLjEwMDkzMTMsICAxLjYwNDI3MDMsICAxLjUxMDY4OCwgICAxLjIzMTcxNDUsXG4gICAgICAyLjAyNTUxNSwgICAyLjM2NTg2NjIsIDEuNjcyMjE1OSwgIDIuMDc4Nzg1NywgIDEuMzc4NTU4NiwgIDIuODk1MDMxLFxuICAgICAgMS4yOTE1MjE4LCAgMi4yMDUxMjIyLCAxLjA0MjMwNzQsICAyLjQzMDMyMDcsICAwLjI3ODQ0NzkzLCAwLjg0MzQ2OTc0LFxuICAgICAgMC4yNTc4MTY1NSwgMS4xMjA4MzU0LCAwLjk0NDcyNzIsICAyLjAxMTEyNTgsICAwLjM2ODkwNjUsICAxLjkwNTI0NTUsXG4gICAgICAwLjc5MTM3Njk1LCAyLjM1NTM0NCwgIDAuNTQyOTI0OCwgIDEuNTU5MzE3OCwgIDAuODI0ODQwMywgIDEuOTkyMjI0MixcbiAgICAgIDAuNzc4NDcsICAgIDEuNTAzMjYwMSwgMC44NjIyNDE4LCAgMC44NDY0NTY2NSwgMS42ODUwMjQ1LCAgMi4yOTU4ODA2LFxuICAgICAgMS42MjQyMjg0LCAgMS4zMjkwNDUsICAxLjY2NTIzMjgsICAyLjQ4MDUzNSwgICAxLjI3OTM0OTEsICAxLjI5NTE4ODQsXG4gICAgICAxLjA2NjcwMzcsICAxLjU3MjAxNThcbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJ1RlbnNvcjNEIGlzIGFsbG93ZWQnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuICAgIGNvbnN0IGNoTXVsID0gMztcbiAgICBjb25zdCBpbkRlcHRoID0gMjtcblxuICAgIGNvbnN0IHggPSB0Zi56ZXJvczxSYW5rLlIzPihbMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi56ZXJvczxSYW5rLlI0PihbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0pO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMywgMywgaW5EZXB0aCAqIGNoTXVsXSk7XG4gIH0pO1xuXG4gIGl0KCdQYXNzIG51bGwgZm9yIGRpbGF0aW9ucywgd2hpY2ggZGVmYXVsdHMgdG8gWzEsIDFdJywgKCkgPT4ge1xuICAgIGNvbnN0IGZTaXplID0gMjtcbiAgICBjb25zdCBwYWQgPSAnc2FtZSc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBjaE11bCA9IDM7XG4gICAgY29uc3QgaW5EZXB0aCA9IDI7XG4gICAgY29uc3QgZGlsYXRpb25zOiBbbnVtYmVyLCBudW1iZXJdID0gbnVsbDtcblxuICAgIGNvbnN0IHggPSB0Zi56ZXJvczxSYW5rLlIzPihbMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi56ZXJvczxSYW5rLlI0PihbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0pO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCwgJ05IV0MnLCBkaWxhdGlvbnMpO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzMsIDMsIGluRGVwdGggKiBjaE11bF0pO1xuICB9KTtcblxuICBpdCgnVGVuc29yTGlrZScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBwYWQgPSAndmFsaWQnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gW1tcbiAgICAgIFtbMC4yMzA2NjRdLCBbMC45ODczODhdLCBbMC4wNjg1MjA4XV0sXG4gICAgICBbWzAuNDE5MjI0XSwgWzAuODg3ODYxXSwgWzAuNzMxNjQxXV0sXG4gICAgICBbWzAuMDc0MTkwN10sIFswLjQwOTI2NV0sIFswLjM1MTM3N11dXG4gICAgXV07XG4gICAgY29uc3QgdyA9IFtbW1swLjMwMzg3M11dLCBbWzAuMjI5MjIzXV1dLCBbW1swLjE0NDMzM11dLCBbWzAuODAzMzczXV1dXTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG5cbiAgICBjb25zdCBleHBlY3RlZCA9IFsxLjA3MDIyLCAxLjAzMTY3LCAwLjY3MDQxLCAwLjc3ODg2M107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgZXhwZWN0ZWQpO1xuICB9KTtcbiAgaXQoJ1RlbnNvckxpa2UgQ2hhaW5lZCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBwYWQgPSAndmFsaWQnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjIzMDY2NCwgMC45ODczODgsIDAuMDY4NTIwOCwgMC40MTkyMjQsIDAuODg3ODYxLCAwLjczMTY0MSxcbiAgICAgICAgICAwLjA3NDE5MDcsIDAuNDA5MjY1LCAwLjM1MTM3N1xuICAgICAgICBdLFxuICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSBbW1tbMC4zMDM4NzNdXSwgW1swLjIyOTIyM11dXSwgW1tbMC4xNDQzMzNdXSwgW1swLjgwMzM3M11dXV07XG5cbiAgICBjb25zdCByZXN1bHQgPSB4LmRlcHRod2lzZUNvbnYyZCh3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgMiwgMiwgMV0pO1xuXG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbMS4wNzAyMiwgMS4wMzE2NywgMC42NzA0MSwgMC43Nzg4NjNdO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyB3aGVuIHBhc3NlZCB4IGFzIGEgbm9uLXRlbnNvcicsICgpID0+IHtcbiAgICBjb25zdCBpbnB1dERlcHRoID0gMTtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gICAgY29uc3QgZlNpemUgPSAxO1xuICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICBjb25zdCBzdHJpZGUgPSAyO1xuICAgIGNvbnN0IGRhdGFGb3JtYXQgPSAnTkhXQyc7XG4gICAgY29uc3QgZGlsYXRpb24gPSAyO1xuXG4gICAgY29uc3QgdyA9IHRmLnRlbnNvcjRkKFszXSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IGUgPSAvQXJndW1lbnQgJ3gnIHBhc3NlZCB0byAnZGVwdGh3aXNlQ29udjJkJyBtdXN0IGJlIGEgVGVuc29yLztcbiAgICBleHBlY3QoXG4gICAgICAgICgpID0+IHRmLmRlcHRod2lzZUNvbnYyZChcbiAgICAgICAgICAgIHt9IGFzIHRmLlRlbnNvcjNELCB3LCBzdHJpZGUsIHBhZCwgZGF0YUZvcm1hdCwgZGlsYXRpb24pKVxuICAgICAgICAudG9UaHJvd0Vycm9yKGUpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gcGFzc2VkIGZpbHRlciBhcyBhIG5vbi10ZW5zb3InLCAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzIsIDIsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICBjb25zdCBzdHJpZGUgPSAyO1xuICAgIGNvbnN0IGRhdGFGb3JtYXQgPSAnTkhXQyc7XG4gICAgY29uc3QgZGlsYXRpb24gPSAyO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjNkKFsxLCAyLCAzLCA0XSwgaW5wdXRTaGFwZSk7XG5cbiAgICBjb25zdCBlID0gL0FyZ3VtZW50ICdmaWx0ZXInIHBhc3NlZCB0byAnZGVwdGh3aXNlQ29udjJkJyBtdXN0IGJlIGEgVGVuc29yLztcbiAgICBleHBlY3QoXG4gICAgICAgICgpID0+IHRmLmRlcHRod2lzZUNvbnYyZChcbiAgICAgICAgICAgIHgsIHt9IGFzIHRmLlRlbnNvcjRELCBzdHJpZGUsIHBhZCwgZGF0YUZvcm1hdCwgZGlsYXRpb24pKVxuICAgICAgICAudG9UaHJvd0Vycm9yKGUpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gaW5wdXQgaXMgaW50MzInLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9ICd2YWxpZCc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID1cbiAgICAgICAgdGYudGVuc29yNGQoWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDldLCBbMSwgMywgMywgaW5EZXB0aF0sICdpbnQzMicpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuMzAzODczLCAwLjIyOTIyMywgMC4xNDQzMzMsIDAuODAzMzczXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICk7XG5cbiAgICBjb25zdCBlcnJSZWdleCA9IC9Bcmd1bWVudCAneCcgcGFzc2VkIHRvICdkZXB0aHdpc2VDb252MmQnIG11c3QgYmUgZmxvYXQzMi87XG4gICAgZXhwZWN0KCgpID0+IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCkpLnRvVGhyb3dFcnJvcihlcnJSZWdleCk7XG4gIH0pO1xuXG4gIGl0KCd0aHJvd3Mgd2hlbiBmaWx0ZXIgaXMgaW50MzInLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9ICd2YWxpZCc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDldLCBbMSwgMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzEsIDIsIDMsIDRdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0sXG4gICAgICAgICdpbnQzMicsXG4gICAgKTtcblxuICAgIGNvbnN0IGVyclJlZ2V4ID1cbiAgICAgICAgL0FyZ3VtZW50ICdmaWx0ZXInIHBhc3NlZCB0byAnZGVwdGh3aXNlQ29udjJkJyBtdXN0IGJlIGZsb2F0MzIvO1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5kZXB0aHdpc2VDb252MmQoeCwgdywgc3RyaWRlLCBwYWQpKS50b1Rocm93RXJyb3IoZXJyUmVnZXgpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gZGltUm91bmRpbmdNb2RlIGlzIHNldCBhbmQgcGFkIGlzIHNhbWUnLCAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuICAgIGNvbnN0IGNoTXVsID0gMTtcbiAgICBjb25zdCBpbkRlcHRoID0gMTtcbiAgICBjb25zdCBkaW1Sb3VuZGluZ01vZGUgPSAncm91bmQnO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMC4yMzA2NjQsIDAuOTg3Mzg4LCAwLjA2ODUyMDgsIDAuNDE5MjI0LCAwLjg4Nzg2MSwgMC43MzE2NDEsXG4gICAgICAgICAgMC4wNzQxOTA3LCAwLjQwOTI2NSwgMC4zNTEzNzdcbiAgICAgICAgXSxcbiAgICAgICAgWzEsIDMsIDMsIGluRGVwdGhdKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFswLjMwMzg3MywgMC4yMjkyMjMsIDAuMTQ0MzMzLCAwLjgwMzM3M10sXG4gICAgICAgIFtmU2l6ZSwgZlNpemUsIGluRGVwdGgsIGNoTXVsXSxcbiAgICApO1xuICAgIGV4cGVjdChcbiAgICAgICAgKCkgPT4gdGYuZGVwdGh3aXNlQ29udjJkKFxuICAgICAgICAgICAgeCwgdywgc3RyaWRlLCBwYWQsICdOSFdDJywgMSwgZGltUm91bmRpbmdNb2RlKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gZGltUm91bmRpbmdNb2RlIGlzIHNldCBhbmQgcGFkIGlzIHZhbGlkJywgKCkgPT4ge1xuICAgIGNvbnN0IGZTaXplID0gMjtcbiAgICBjb25zdCBwYWQgPSAndmFsaWQnO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgY29uc3QgY2hNdWwgPSAxO1xuICAgIGNvbnN0IGluRGVwdGggPSAxO1xuICAgIGNvbnN0IGRpbVJvdW5kaW5nTW9kZSA9ICdyb3VuZCc7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjIzMDY2NCwgMC45ODczODgsIDAuMDY4NTIwOCwgMC40MTkyMjQsIDAuODg3ODYxLCAwLjczMTY0MSxcbiAgICAgICAgICAwLjA3NDE5MDcsIDAuNDA5MjY1LCAwLjM1MTM3N1xuICAgICAgICBdLFxuICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuMzAzODczLCAwLjIyOTIyMywgMC4xNDQzMzMsIDAuODAzMzczXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICk7XG4gICAgZXhwZWN0KFxuICAgICAgICAoKSA9PiB0Zi5kZXB0aHdpc2VDb252MmQoXG4gICAgICAgICAgICB4LCB3LCBzdHJpZGUsIHBhZCwgJ05IV0MnLCAxLCBkaW1Sb3VuZGluZ01vZGUpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xuXG4gIGl0KCd0aHJvd3Mgd2hlbiBkaW1Sb3VuZGluZ01vZGUgaXMgc2V0IGFuZCBwYWQgaXMgYSBub24taW50ZWdlciBudW1iZXInLFxuICAgICAoKSA9PiB7XG4gICAgICAgY29uc3QgZlNpemUgPSAyO1xuICAgICAgIGNvbnN0IHBhZCA9IDEuMjtcbiAgICAgICBjb25zdCBzdHJpZGUgPSAxO1xuICAgICAgIGNvbnN0IGNoTXVsID0gMTtcbiAgICAgICBjb25zdCBpbkRlcHRoID0gMTtcbiAgICAgICBjb25zdCBkaW1Sb3VuZGluZ01vZGUgPSAncm91bmQnO1xuXG4gICAgICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICAgICBbXG4gICAgICAgICAgICAgMC4yMzA2NjQsIDAuOTg3Mzg4LCAwLjA2ODUyMDgsIDAuNDE5MjI0LCAwLjg4Nzg2MSwgMC43MzE2NDEsXG4gICAgICAgICAgICAgMC4wNzQxOTA3LCAwLjQwOTI2NSwgMC4zNTEzNzdcbiAgICAgICAgICAgXSxcbiAgICAgICAgICAgWzEsIDMsIDMsIGluRGVwdGhdKTtcbiAgICAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgICAgIFswLjMwMzg3MywgMC4yMjkyMjMsIDAuMTQ0MzMzLCAwLjgwMzM3M10sXG4gICAgICAgICAgIFtmU2l6ZSwgZlNpemUsIGluRGVwdGgsIGNoTXVsXSxcbiAgICAgICApO1xuICAgICAgIGV4cGVjdChcbiAgICAgICAgICAgKCkgPT4gdGYuZGVwdGh3aXNlQ29udjJkKFxuICAgICAgICAgICAgICAgeCwgdywgc3RyaWRlLCBwYWQsICdOSFdDJywgMSwgZGltUm91bmRpbmdNb2RlKSlcbiAgICAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICAgICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gZGltUm91bmRpbmdNb2RlIGlzIHNldCBhbmQgcGFkIGlzIGV4cGxpY2l0IGJ5IG5vbi1pbnRlZ2VyICcgK1xuICAgICAgICAgJ251bWJlcicsXG4gICAgICgpID0+IHtcbiAgICAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgICAgY29uc3QgcGFkID0gW1swLCAwXSwgWzAsIDIuMV0sIFsxLCAxXSwgWzAsIDBdXSBhc1xuICAgICAgICAgICB0Zi5iYWNrZW5kX3V0aWwuRXhwbGljaXRQYWRkaW5nO1xuICAgICAgIGNvbnN0IHN0cmlkZSA9IDE7XG4gICAgICAgY29uc3QgY2hNdWwgPSAxO1xuICAgICAgIGNvbnN0IGluRGVwdGggPSAxO1xuICAgICAgIGNvbnN0IGRpbVJvdW5kaW5nTW9kZSA9ICdyb3VuZCc7XG5cbiAgICAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgICAgIFtcbiAgICAgICAgICAgICAwLjIzMDY2NCwgMC45ODczODgsIDAuMDY4NTIwOCwgMC40MTkyMjQsIDAuODg3ODYxLCAwLjczMTY0MSxcbiAgICAgICAgICAgICAwLjA3NDE5MDcsIDAuNDA5MjY1LCAwLjM1MTM3N1xuICAgICAgICAgICBdLFxuICAgICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuICAgICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgICAgWzAuMzAzODczLCAwLjIyOTIyMywgMC4xNDQzMzMsIDAuODAzMzczXSxcbiAgICAgICAgICAgW2ZTaXplLCBmU2l6ZSwgaW5EZXB0aCwgY2hNdWxdLFxuICAgICAgICk7XG4gICAgICAgZXhwZWN0KFxuICAgICAgICAgICAoKSA9PiB0Zi5kZXB0aHdpc2VDb252MmQoXG4gICAgICAgICAgICAgICB4LCB3LCBzdHJpZGUsIHBhZCwgJ05IV0MnLCAxLCBkaW1Sb3VuZGluZ01vZGUpKVxuICAgICAgICAgICAudG9UaHJvd0Vycm9yKCk7XG4gICAgIH0pO1xuXG4gIGl0KCdhY2NlcHRzIGEgdGVuc29yLWxpa2Ugb2JqZWN0JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHBhZCA9ICd2YWxpZCc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICAvLyAxeDN4M3gxXG4gICAgY29uc3QgeCA9IFtbXG4gICAgICBbWzAuMjMwNjY0XSwgWzAuOTg3Mzg4XSwgWzAuMDY4NTIwOF1dLFxuICAgICAgW1swLjQxOTIyNF0sIFswLjg4Nzg2MV0sIFswLjczMTY0MV1dLFxuICAgICAgW1swLjA3NDE5MDddLCBbMC40MDkyNjVdLCBbMC4zNTEzNzddXVxuICAgIF1dO1xuICAgIC8vIDJ4MngxeDFcbiAgICBjb25zdCB3ID0gW1tbWzAuMzAzODczXV0sIFtbMC4yMjkyMjNdXV0sIFtbWzAuMTQ0MzMzXV0sIFtbMC44MDMzNzNdXV1dO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmRlcHRod2lzZUNvbnYyZCh4LCB3LCBzdHJpZGUsIHBhZCk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgMiwgMiwgMV0pO1xuXG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbMS4wNzAyMiwgMS4wMzE2NywgMC42NzA0MSwgMC43Nzg4NjNdO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ2RlcHRod2lzZUNvbnYyZCBncmFkaWVudHMnLCBBTExfRU5WUywgKCkgPT4ge1xuICBsZXQgaW1hZ2VzOiB0Zi5UZW5zb3I0RDtcbiAgbGV0IGZpbHRlcjogdGYuVGVuc29yNEQ7XG4gIGxldCByZXN1bHQ6IHRmLlRlbnNvcjREO1xuICBjb25zdCBzdHJpZGUgPSAxO1xuICBjb25zdCBwYWQgPSAnc2FtZSc7XG5cbiAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgLy8gdHdvIDJ4MiBSR0IgaW1hZ2VzID0+IDJ4MngyeDNcbiAgICBpbWFnZXMgPSB0Zi50ZW5zb3I0ZChbXG4gICAgICBbW1syLCAzLCAxXSwgWzMsIDAsIDJdXSwgW1swLCA0LCAxXSwgWzMsIDEsIDNdXV0sXG4gICAgICBbW1syLCAxLCAwXSwgWzAsIDMsIDNdXSwgW1s0LCAwLCAxXSwgWzEsIDQsIDFdXV1cbiAgICBdKTtcbiAgICAvLyAyeDIgZmlsdGVycywgY2hNdWwgPSAyID0+IDJ4MngzeDJcbiAgICBmaWx0ZXIgPSB0Zi50ZW5zb3I0ZChbXG4gICAgICBbW1sxLCAxXSwgWzEsIDFdLCBbMCwgMF1dLCBbWzAsIDFdLCBbMSwgMV0sIFsxLCAxXV1dLFxuICAgICAgW1tbMSwgMF0sIFsxLCAxXSwgWzAsIDBdXSwgW1swLCAxXSwgWzEsIDBdLCBbMCwgMF1dXVxuICAgIF0pO1xuICAgIC8vIHJlc3VsdCBvZiBjb252b2x1dGlvbiBvcGVyYXRvaW5cbiAgICByZXN1bHQgPSB0Zi50ZW5zb3I0ZChbXG4gICAgICBbXG4gICAgICAgIFtbMiwgOCwgOCwgNywgMiwgMl0sIFs2LCAzLCAxLCAxLCAwLCAwXV0sXG4gICAgICAgIFtbMCwgMywgNSwgNSwgMywgM10sIFszLCAzLCAxLCAxLCAwLCAwXV1cbiAgICAgIF0sXG4gICAgICBbXG4gICAgICAgIFtbNiwgMywgOCwgNCwgMywgM10sIFsxLCAwLCA3LCA3LCAwLCAwXV0sXG4gICAgICAgIFtbNCwgNSwgNCwgNCwgMSwgMV0sIFsxLCAxLCA0LCA0LCAwLCAwXV1cbiAgICAgIF1cbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJ3dydCBpbnB1dCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB7dmFsdWUsIGdyYWR9ID0gdGYudmFsdWVBbmRHcmFkKFxuICAgICAgICAoeDogdGYuVGVuc29yNEQpID0+IHRmLmRlcHRod2lzZUNvbnYyZCh4LCBmaWx0ZXIsIHN0cmlkZSwgcGFkKSkoaW1hZ2VzKTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHZhbHVlLmRhdGEoKSwgYXdhaXQgcmVzdWx0LmRhdGEoKSk7XG5cbiAgICBjb25zdCBleHBlY3RlZEdyYWQgPSB0Zi50ZW5zb3I0ZChbXG4gICAgICBbW1syLiwgMi4sIDAuXSwgWzMuLCA0LiwgMi5dXSwgW1szLiwgNC4sIDAuXSwgWzUuLCA3LiwgMi5dXV0sXG4gICAgICBbW1syLiwgMi4sIDAuXSwgWzMuLCA0LiwgMi5dXSwgW1szLiwgNC4sIDAuXSwgWzUuLCA3LiwgMi5dXV1cbiAgICBdKTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGdyYWQuZGF0YSgpLCBhd2FpdCBleHBlY3RlZEdyYWQuZGF0YSgpKTtcbiAgfSk7XG5cbiAgLy8gVGhlIGdyYWRpZW50cyBvZiBub3JtYWwgYW5kIGRlcHRod2lzZSAyRCBjb252b2x1dGlvbnMgYXJlIGFjdHVhbGx5IHRoZSBzYW1lXG4gIC8vIGluIHRoZSBzcGVjaWFsIGNhc2UgdGhhdCBkeSA9IDEsIHNvIHdlIGFsc28gdGVzdCB0aGUgZ3JhZGllbnQgb2YgYSBmdW5jdGlvblxuICAvLyBvZiB0aGUgb3V0cHV0IHRvIGRpc2FtYmlndWF0ZSB0aGUgdHdvIG1ldGhvZHMuXG4gIGl0KCd3cnQgaW5wdXQsIHNxdWFyZWQgb3V0cHV0JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGdyYWQgPSB0Zi5ncmFkKFxuICAgICAgICAoeDogdGYuVGVuc29yNEQpID0+XG4gICAgICAgICAgICB0Zi5zcXVhcmUodGYuZGVwdGh3aXNlQ29udjJkKHgsIGZpbHRlciwgc3RyaWRlLCBwYWQpKSkoaW1hZ2VzKTtcblxuICAgIGNvbnN0IGV4cGVjdGVkR3JhZCA9IHRmLnRlbnNvcjRkKFtcbiAgICAgIFtbWzIwLiwgMzAuLCAwLl0sIFszNC4sIDM0LiwgOC5dXSwgW1sxMC4sIDUwLiwgMC5dLCBbNDYuLCA0NC4sIDEyLl1dXSxcbiAgICAgIFtbWzE4LiwgMjQuLCAwLl0sIFs4LiwgNTIuLCAxMi5dXSwgW1szMC4sIDQwLiwgMC5dLCBbMjIuLCA3Ni4sIDQuXV1dXG4gICAgXSk7XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBncmFkLmRhdGEoKSwgYXdhaXQgZXhwZWN0ZWRHcmFkLmRhdGEoKSk7XG4gIH0pO1xuXG4gIGl0KCd3cnQgZmlsdGVyJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHt2YWx1ZSwgZ3JhZH0gPSB0Zi52YWx1ZUFuZEdyYWQoXG4gICAgICAgIChmOiB0Zi5UZW5zb3I0RCkgPT4gdGYuZGVwdGh3aXNlQ29udjJkKGltYWdlcywgZiwgc3RyaWRlLCBwYWQpKShmaWx0ZXIpO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgdmFsdWUuZGF0YSgpLCBhd2FpdCByZXN1bHQuZGF0YSgpKTtcblxuICAgIGNvbnN0IGV4cGVjdGVkR3JhZCA9IHRmLnRlbnNvcjRkKFtcbiAgICAgIFtbWzE1LiwgMTUuXSwgWzE2LiwgMTYuXSwgWzEyLiwgMTIuXV0sIFtbNy4sIDcuXSwgWzguLCA4Ll0sIFs5LiwgOS5dXV0sXG4gICAgICBbW1s4LiwgOC5dLCBbOS4sIDkuXSwgWzYuLCA2Ll1dLCBbWzQuLCA0Ll0sIFs1LiwgNS5dLCBbNC4sIDQuXV1dXG4gICAgXSk7XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBncmFkLmRhdGEoKSwgYXdhaXQgZXhwZWN0ZWRHcmFkLmRhdGEoKSk7XG4gIH0pO1xuXG4gIGl0KCdncmFkaWVudCB3aXRoIGNsb25lcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBbZHgsIGRGaWx0ZXJdID0gdGYuZ3JhZHMoXG4gICAgICAgICh4OiB0Zi5UZW5zb3I0RCwgZmlsdGVyOiB0Zi5UZW5zb3I0RCkgPT5cbiAgICAgICAgICAgIHRmLmRlcHRod2lzZUNvbnYyZCh4LmNsb25lKCksIGZpbHRlci5jbG9uZSgpLCBzdHJpZGUsIHBhZCkuY2xvbmUoKSkoXG4gICAgICAgIFtpbWFnZXMsIGZpbHRlcl0pO1xuICAgIGV4cGVjdChkeC5zaGFwZSkudG9FcXVhbChpbWFnZXMuc2hhcGUpO1xuICAgIGV4cGVjdChkRmlsdGVyLnNoYXBlKS50b0VxdWFsKGZpbHRlci5zaGFwZSk7XG4gIH0pO1xuXG4gIC8vIEFsc28gZGlzYW1iaWd1YXRlIHJlZ3VsYXIgdnMuIGRlcHRod2lzZSBmaWx0ZXIgZ3JhZGllbnRzXG4gIGl0KCd3cnQgZmlsdGVyLCBzcXVhcmVkIG91dHB1dCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBncmFkID0gdGYuZ3JhZChcbiAgICAgICAgKGY6IHRmLlRlbnNvcjREKSA9PlxuICAgICAgICAgICAgdGYuc3F1YXJlKHRmLmRlcHRod2lzZUNvbnYyZChpbWFnZXMsIGYsIHN0cmlkZSwgcGFkKSkpKGZpbHRlcik7XG5cbiAgICBjb25zdCBleHBlY3RlZEdyYWQgPSB0Zi50ZW5zb3I0ZChbXG4gICAgICBbXG4gICAgICAgIFtbMTIwLiwgMTIyLl0sIFsxODAuLCAxNjYuXSwgWzEyLiwgMTIuXV0sXG4gICAgICAgIFtbMjAuLCA3Ni5dLCBbOTAuLCA2Ni5dLCBbNDYuLCA0Ni5dXVxuICAgICAgXSxcbiAgICAgIFtcbiAgICAgICAgW1s4Ni4sIDQyLl0sIFsxMjIuLCAxMTQuXSwgWzEwLiwgMTAuXV0sXG4gICAgICAgIFtbMjQuLCA1NC5dLCBbODAuLCA0Ni5dLCBbMTguLCAxOC5dXVxuICAgICAgXVxuICAgIF0pO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgZ3JhZC5kYXRhKCksIGF3YWl0IGV4cGVjdGVkR3JhZC5kYXRhKCkpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIGVycm9yIG9uIGRpbGF0aW9ucyA+IDEnLCAoKSA9PiB7XG4gICAgY29uc3QgZ3JhZCA9IHRmLmdyYWQoXG4gICAgICAgICh4OiB0Zi5UZW5zb3I0RCkgPT5cbiAgICAgICAgICAgIHRmLmRlcHRod2lzZUNvbnYyZCh4LCBmaWx0ZXIsIHN0cmlkZSwgcGFkLCAnTkhXQycsIDIpKTtcblxuICAgIGV4cGVjdCgoKSA9PiBncmFkKGltYWdlcykpXG4gICAgICAgIC50b1Rocm93RXJyb3IoL2RpbGF0aW9uIHJhdGVzIGdyZWF0ZXIgdGhhbiAxIGFyZSBub3QgeWV0IHN1cHBvcnRlZC8pO1xuICB9KTtcblxuICBpdCgnd3J0IGlucHV0LCBzdHJpZGU9MiwgcGFkPXZhbGlkJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGR4ID0gdGYuZ3JhZChcbiAgICAgICAgKHg6IHRmLlRlbnNvcjREKSA9PiB0Zi5kZXB0aHdpc2VDb252MmQoeCwgZmlsdGVyLCAyLCAndmFsaWQnKSkoaW1hZ2VzKTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGR4LmRhdGEoKSwgW1xuICAgICAgMi4sIDIuLCAwLiwgMS4sIDIuLCAyLiwgMS4sIDIuLCAwLiwgMS4sIDEuLCAwLixcbiAgICAgIDIuLCAyLiwgMC4sIDEuLCAyLiwgMi4sIDEuLCAyLiwgMC4sIDEuLCAxLiwgMC5cbiAgICBdKTtcbiAgICBleHBlY3QoZHguc2hhcGUpLnRvRXF1YWwoWzIsIDIsIDIsIDNdKTtcbiAgfSk7XG5cbiAgaXQoJ3dydCBmaWx0ZXIsIHN0cmlkZT0yLCBwYWQ9dmFsaWQnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZGYgPSB0Zi5ncmFkKFxuICAgICAgICAoZjogdGYuVGVuc29yNEQpID0+IHRmLmRlcHRod2lzZUNvbnYyZChpbWFnZXMsIGYsIDIsICd2YWxpZCcpKShmaWx0ZXIpO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgZGYuZGF0YSgpLCBbXG4gICAgICA0LiwgNC4sIDQuLCA0LiwgMS4sIDEuLCAzLiwgMy4sIDMuLCAzLiwgNS4sIDUuLFxuICAgICAgNC4sIDQuLCA0LiwgNC4sIDIuLCAyLiwgNC4sIDQuLCA1LiwgNS4sIDQuLCA0LlxuICAgIF0pO1xuICAgIGV4cGVjdChkZi5zaGFwZSkudG9FcXVhbChbMiwgMiwgMywgMl0pO1xuICB9KTtcblxuICBpdCgnZ3JhZGllbnQgd2l0aCBjbG9uZXMnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IHBhZCA9ICd2YWxpZCc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBjaE11bCA9IDE7XG4gICAgY29uc3QgaW5EZXB0aCA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICAwLjIzMDY2NCwgMC45ODczODgsIDAuMDY4NTIwOCwgMC40MTkyMjQsIDAuODg3ODYxLCAwLjczMTY0MSxcbiAgICAgICAgICAwLjA3NDE5MDcsIDAuNDA5MjY1LCAwLjM1MTM3N1xuICAgICAgICBdLFxuICAgICAgICBbMSwgMywgMywgaW5EZXB0aF0pO1xuXG4gICAgY29uc3QgZiA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMC4zMDM4NzMsIDAuMjI5MjIzLCAwLjE0NDMzMywgMC44MDMzNzNdLFxuICAgICAgICBbZlNpemUsIGZTaXplLCBpbkRlcHRoLCBjaE11bF0sXG4gICAgKTtcblxuICAgIGNvbnN0IFtkeCwgZGZdID0gdGYuZ3JhZHMoXG4gICAgICAgICh4OiB0Zi5UZW5zb3I0RCwgZjogdGYuVGVuc29yNEQpID0+XG4gICAgICAgICAgICB0Zi5kZXB0aHdpc2VDb252MmQoeC5jbG9uZSgpLCBmLmNsb25lKCksIHN0cmlkZSwgcGFkKS5jbG9uZSgpKShcbiAgICAgICAgW3gsIGZdKTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGR4LmRhdGEoKSwgW1xuICAgICAgMC4zMDM4NzMsIDAuNTMzMDk2LCAwLjIyOTIyMywgMC40NDgyMDYsIDEuNDgwODAyLCAxLjAzMjU5NiwgMC4xNDQzMzMsXG4gICAgICAwLjk0NzcwNiwgMC44MDMzNzNcbiAgICBdKTtcbiAgICBleHBlY3QoZHguc2hhcGUpLnRvRXF1YWwoWzEsIDMsIDMsIDFdKTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgICAgICBhd2FpdCBkZi5kYXRhKCksIFsyLjUyNTEzNywgMi42NzU0MTA4LCAxLjc5MDU0MDcsIDIuMzgwMTQ0XSk7XG4gICAgZXhwZWN0KGRmLnNoYXBlKS50b0VxdWFsKFsyLCAyLCAxLCAxXSk7XG4gIH0pO1xufSk7XG4iXX0=
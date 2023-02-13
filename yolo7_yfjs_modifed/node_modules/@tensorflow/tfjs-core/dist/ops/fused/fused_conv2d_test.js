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
import * as tf from '../../index';
import { ALL_ENVS, describeWithFlags } from '../../jasmine_util';
import { expectArraysClose } from '../../test_util';
function generateCaseInputs(totalSizeTensor, totalSizeFilter) {
    const inp = new Array(totalSizeTensor);
    const filt = new Array(totalSizeFilter);
    for (let i = 0; i < totalSizeTensor; i++) {
        inp[i] = i * 0.001 - totalSizeTensor * 0.001 / 2;
    }
    for (let i = 0; i < totalSizeFilter; i++) {
        const sign = i % 2 === 0 ? -1 : 1;
        filt[i] = i * 0.001 * sign;
    }
    return { input: inp, filter: filt };
}
describeWithFlags('fused conv2d', ALL_ENVS, () => {
    it('basic', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({ x, filter: w, strides: stride, pad });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [-5, 2, -11, 5, -17, 8, -23, 11, -29, 14, -35, 17, -41, 20, -47, 23];
        expectArraysClose(await result.data(), expected);
    });
    it('basic with relu', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'relu'
        });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [0, 2, 0, 5, 0, 8, 0, 11, 0, 14, 0, 17, 0, 20, 0, 23];
        expectArraysClose(await result.data(), expected);
    });
    it('relu with stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=1 p=same', async () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 1;
        const fSize = 3;
        const pad = 'same';
        const stride = [2, 2];
        // TODO(annxingyuan): Make this test work with large inputs
        // https://github.com/tensorflow/tfjs/issues/3143
        const inputData = [];
        for (let i = 0; i < xSize * xSize * inputDepth; i++) {
            inputData.push(i % 5);
        }
        const wData = [];
        for (let i = 0; i < fSize * fSize * inputDepth * outputDepth; i++) {
            wData.push(i % 5);
        }
        const x = tf.tensor4d(inputData, inputShape);
        const w = tf.tensor4d(wData, [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'relu'
        });
        expect(result.shape).toEqual([1, 4, 4, 1]);
        expectArraysClose(await result.data(), new Float32Array([
            854, 431, 568, 382, 580, 427, 854, 288, 431, 568,
            580, 289, 285, 570, 285, 258
        ]));
    });
    it('relu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same', async () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = 'same';
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'relu',
            bias
        });
        expect(result.shape).toEqual([1, 4, 4, 8]);
        expectArraysClose(await result.data(), new Float32Array([
            25.75398063659668,
            0,
            26.857805252075195,
            0,
            33.961631774902344,
            0,
            30.065458297729492,
            0,
            23.118206024169922,
            0,
            24.212820053100586,
            0,
            31.307422637939453,
            0,
            27.402034759521484,
            0,
            20.482431411743164,
            0,
            21.567821502685547,
            0,
            28.653217315673828,
            0,
            24.73861312866211,
            0,
            11.078080177307129,
            0,
            12.130399703979492,
            0,
            19.182720184326172,
            0,
            15.235037803649902,
            0,
            4.6677775382995605,
            0.31717729568481445,
            5.697869777679443,
            0,
            12.727968215942383,
            2.2569849491119385,
            8.758066177368164,
            4.226885795593262,
            2.0319995880126953,
            2.9575586318969727,
            3.052880048751831,
            1.9366796016693115,
            10.073760032653809,
            4.915799617767334,
            6.094639778137207,
            6.89492130279541,
            0,
            5.5979437828063965,
            0.4078875780105591,
            4.586280822753906,
            7.419551849365234,
            7.5746169090271,
            3.43121600151062,
            9.562952041625977,
            0,
            6.404943943023682,
            0,
            5.401776313781738,
            6.5998077392578125,
            8.398608207702637,
            2.602976083755493,
            10.395440101623535,
            0,
            21.440250396728516,
            0,
            20.483882904052734,
            0,
            23.527509689331055,
            0,
            25.571144104003906,
            0,
            24.080629348754883,
            0,
            23.133480072021484,
            0,
            26.186328887939453,
            0,
            28.239177703857422,
            0,
            26.721012115478516,
            0,
            25.783079147338867,
            0,
            28.84514808654785,
            0,
            30.907209396362305,
            0,
            18.914127349853516,
            0,
            17.960111618041992,
            0,
            21.006093978881836,
            0,
            23.052082061767578,
            0,
            17.89089584350586,
            0,
            16.95684814453125,
            0,
            20.022798538208008,
            0,
            22.088754653930664,
            0,
            19.06132698059082,
            0,
            18.133424758911133,
            0,
            21.205520629882812,
            0,
            23.27761459350586,
            0,
            20.23175811767578,
            0,
            19.309999465942383,
            0,
            22.388240814208984,
            0,
            24.46647834777832,
            0,
            13.584352493286133,
            0,
            12.6395845413208,
            0,
            15.694815635681152,
            0,
            17.750045776367188
        ]));
    });
    it('prelu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same', async () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = 'same';
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const preluActivationWeights = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'prelu',
            preluActivationWeights,
            bias
        });
        expect(result.shape).toEqual([1, 4, 4, 8]);
        expectArraysClose(await result.data(), new Float32Array([
            25.75398063659668, -41.61178970336914, 26.857805252075195,
            -87.63885498046875, 33.961631774902344, -114.0812759399414,
            30.065458297729492, -136.93893432617188, 23.118206024169922,
            -36.33102035522461, 24.212820053100586, -77.04048156738281,
            31.307422637939453, -98.12835693359375, 27.402034759521484,
            -115.5947265625, 20.482431411743164, -31.050262451171875,
            21.567821502685547, -66.44209289550781, 28.653217315673828,
            -82.17544555664062, 24.73861312866211, -94.25041198730469,
            11.078080177307129, -12.208478927612305, 12.130399703979492,
            -28.626232147216797, 19.182720184326172, -25.253299713134766,
            15.235037803649902, -18.08960723876953, 4.6677775382995605,
            0.31717729568481445, 5.697869777679443, -2.8516759872436523,
            12.727968215942383, 2.2569849491119385, 8.758066177368164,
            4.226885795593262, 2.0319995880126953, 2.9575586318969727,
            3.052880048751831, 1.9366796016693115, 10.073760032653809,
            4.915799617767334, 6.094639778137207, 6.89492130279541,
            -0.6037763357162476, 5.5979437828063965, 0.4078875780105591,
            4.586280822753906, 7.419551849365234, 7.5746169090271,
            3.43121600151062, 9.562952041625977, -1.4065279960632324,
            6.404943943023682, -1.2100803852081299, 5.401776313781738,
            6.5998077392578125, 8.398608207702637, 2.602976083755493,
            10.395440101623535, -16.418434143066406, 21.440250396728516,
            -46.38618850708008, 20.483882904052734, -42.52848815917969,
            23.527509689331055, -87.84530639648438, 25.571144104003906,
            -19.054208755493164, 24.080629348754883, -54.32115936279297,
            23.133480072021484, -55.79951477050781, 26.186328887939453,
            -106.48924255371094, 28.239177703857422, -21.689987182617188,
            26.721012115478516, -62.25614929199219, 25.783079147338867,
            -69.070556640625, 28.84514808654785, -125.13325500488281,
            30.907209396362305, -13.891133308410645, 18.914127349853516,
            -38.81135940551758, 17.960111618041992, -29.915504455566406,
            21.006093978881836, -70.20361328125, 23.052082061767578,
            -12.857919692993164, 17.89089584350586, -35.771610260009766,
            16.95684814453125, -24.949115753173828, 20.022798538208008,
            -63.39042282104492, 22.088754653930664, -14.02528190612793,
            19.06132698059082, -39.2921257019043, 18.133424758911133,
            -30.847349166870117, 21.205520629882812, -71.69097137451172,
            23.27761459350586, -15.192638397216797, 20.23175811767578,
            -42.8126335144043, 19.309999465942383, -36.74560546875,
            22.388240814208984, -79.99152374267578, 24.46647834777832,
            -8.556736946105957, 13.584352493286133, -22.835901260375977,
            12.6395845413208, -3.336000442504883, 15.694815635681152,
            -33.0570182800293, 17.750045776367188
        ]));
    });
    it('relu6 bias stride 2 x=[1,8,8,16] f=[3,3,16,8] s=[2,2] d=8 p=same', async () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = 'same';
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'relu6',
            bias
        });
        expect(result.shape).toEqual([1, 4, 4, 8]);
        const resultData = await result.data();
        expectArraysClose(resultData, new Float32Array([
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            4.6677775382995605,
            0.31717729568481445,
            5.697869777679443,
            0,
            6,
            2.2569849491119385,
            6,
            4.226885795593262,
            2.0319995880126953,
            2.9575586318969727,
            3.052880048751831,
            1.9366796016693115,
            6,
            4.915799617767334,
            6,
            6,
            0,
            5.5979437828063965,
            0.4078875780105591,
            4.586280822753906,
            6,
            6,
            3.43121600151062,
            6,
            0,
            6,
            0,
            5.401776313781738,
            6,
            6,
            2.602976083755493,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6,
            0,
            6
        ]));
    });
    it('leakyrelu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same', async () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = 'same';
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const leakyreluAlpha = 0.3;
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha,
            bias
        });
        expect(result.shape).toEqual([1, 4, 4, 8]);
        expectArraysClose(await result.data(), new Float32Array([
            25.75398063659668, -6.241768836975098, 26.857805252075195,
            -6.5729146003723145, 33.961631774902344, -5.704063892364502,
            30.065458297729492, -5.135210037231445, 23.118206024169922,
            -5.449653148651123, 24.212820053100586, -5.778036117553711,
            31.307422637939453, -4.906418323516846, 27.402034759521484,
            -4.334802627563477, 20.482431411743164, -4.657539367675781,
            21.567821502685547, -4.983157157897949, 28.653217315673828,
            -4.108772277832031, 24.73861312866211, -3.534390687942505,
            11.078080177307129, -1.8312718868255615, 12.130399703979492,
            -2.1469674110412598, 19.182720184326172, -1.262665033340454,
            15.235037803649902, -0.6783602833747864, 4.6677775382995605,
            0.31717729568481445, 5.697869777679443, -0.21387571096420288,
            12.727968215942383, 2.2569849491119385, 8.758066177368164,
            4.226885795593262, 2.0319995880126953, 2.9575586318969727,
            3.052880048751831, 1.9366796016693115, 10.073760032653809,
            4.915799617767334, 6.094639778137207, 6.89492130279541,
            -0.18113291263580322, 5.5979437828063965, 0.4078875780105591,
            4.586280822753906, 7.419551849365234, 7.5746169090271,
            3.43121600151062, 9.562952041625977, -0.42195841670036316,
            6.404943943023682, -0.12100804597139359, 5.401776313781738,
            6.5998077392578125, 8.398608207702637, 2.602976083755493,
            10.395440101623535, -4.925530433654785, 21.440250396728516,
            -4.6386189460754395, 20.483882904052734, -2.5517091751098633,
            23.527509689331055, -3.764799118041992, 25.571144104003906,
            -5.7162628173828125, 24.080629348754883, -5.432116508483887,
            23.133480072021484, -3.347970962524414, 26.186328887939453,
            -4.5638251304626465, 28.239177703857422, -6.5069966316223145,
            26.721012115478516, -6.225615501403809, 25.783079147338867,
            -4.144233703613281, 28.84514808654785, -5.36285400390625,
            30.907209396362305, -4.167340278625488, 18.914127349853516,
            -3.881135940551758, 17.960111618041992, -1.794930338859558,
            21.006093978881836, -3.0087265968322754, 23.052082061767578,
            -3.8573760986328125, 17.89089584350586, -3.5771610736846924,
            16.95684814453125, -1.4969470500946045, 20.022798538208008,
            -2.7167325019836426, 22.088754653930664, -4.207584857940674,
            19.06132698059082, -3.9292125701904297, 18.133424758911133,
            -1.8508410453796387, 21.205520629882812, -3.0724704265594482,
            23.27761459350586, -4.557791709899902, 20.23175811767578,
            -4.28126335144043, 19.309999465942383, -2.2047364711761475,
            22.388240814208984, -3.428208351135254, 24.46647834777832,
            -2.567021131515503, 13.584352493286133, -2.283590316772461,
            12.6395845413208, -0.20016004145145416, 15.694815635681152,
            -1.41672945022583, 17.750045776367188
        ]));
    });
    it('throws when dimRoundingMode is set and pad is same', () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = 'same';
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const leakyreluAlpha = 0.3;
        expect(() => tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha,
            bias,
            dimRoundingMode: 'round'
        }))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is valid', () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = 'valid';
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const leakyreluAlpha = 0.3;
        expect(() => tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha,
            bias,
            dimRoundingMode: 'round'
        }))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is a non-integer number', () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = 1.2;
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const leakyreluAlpha = 0.3;
        expect(() => tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha,
            bias,
            dimRoundingMode: 'round'
        }))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is explicit by non-integer ' +
        'number', () => {
        const inputDepth = 16;
        const xSize = 8;
        const inputShape = [1, xSize, xSize, inputDepth];
        const outputDepth = 8;
        const fSize = 3;
        const pad = [[0, 0], [0, 2.1], [1, 1], [0, 0]];
        const stride = [2, 2];
        const inputs = generateCaseInputs(1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth * outputDepth);
        const x = tf.tensor4d(inputs.input, inputShape);
        const w = tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
        const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
        const leakyreluAlpha = 0.3;
        expect(() => tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha,
            bias,
            dimRoundingMode: 'round'
        }))
            .toThrowError();
    });
    it('basic with bias', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            bias: tf.tensor1d([5, 6])
        });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [0, 8, -6, 11, -12, 14, -18, 17, -24, 20, -30, 23, -36, 26, -42, 29];
        expectArraysClose(await result.data(), expected);
    });
    it('basic with explicit padding', async () => {
        const inputDepth = 1;
        const outputDepth = 1;
        const pad = [[0, 0], [1, 2], [0, 1], [0, 0]];
        const stride = 1;
        const dataFormat = 'NHWC';
        const dilation = 1;
        const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2, inputDepth]);
        const w = tf.tensor4d([3, 1, 5, 0, 2, 7, 8, 9], [4, 2, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({ x, filter: w, strides: stride, pad, dataFormat, dilations: dilation });
        const resultData = await result.data();
        expect(result.shape).toEqual([4, 2, 1]);
        expectArraysClose(resultData, [133, 66, 200, 102, 108, 58, 56, 58]);
    });
    it('basic with elu', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'elu'
        });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [-0.99326, 2, -1, 5, -1, 8, -1, 11, -1, 14, -1, 17, -1, 20, -1, 23];
        expectArraysClose(await result.data(), expected);
    });
    it('basic with prelu', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const alpha = tf.tensor3d([0.25, 0.75], [1, 1, 2]);
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'prelu',
            preluActivationWeights: alpha
        });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [
            -1.25, 2, -2.75, 5, -4.25, 8, -5.75, 11, -7.25, 14, -8.75, 17, -10.25, 20,
            -11.75, 23
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('basic with leakyrelu', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const alpha = 0.3;
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha: alpha
        });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [
            -1.5, 2, -3.3000001907348633, 5, -5.100000381469727, 8,
            -6.900000095367432, 11, -8.700000762939453, 14, -10.5, 17,
            -12.300000190734863, 20, -14.100000381469727, 23
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('basic with sigmoid', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const alpha = 0.3;
        const w = tf.tensor4d([-0.1, 0.1, -0.2, 0.05], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'sigmoid',
            leakyreluAlpha: alpha
        });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [
            0.3775407, 0.549834, 0.24973989, 0.6224593, 0.15446526, 0.6899744,
            0.09112296, 0.7502601, 0.0521535, 0.80218387, 0.02931219, 0.84553474,
            0.0163025, 0.8807971, 0.0090133, 0.908877
        ];
        expectArraysClose(await result.data(), expected);
    });
    it('basic with broadcasted bias and relu', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides: stride,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            bias: tf.scalar(5),
            activation: 'relu'
        });
        expect(result.shape).toEqual([2, 2, 2, 2]);
        const expected = [0, 7, 0, 10, 0, 13, 0, 16, 0, 19, 0, 22, 0, 25, 0, 28];
        expectArraysClose(await result.data(), expected);
    });
    it('im2row', async () => {
        const inputDepth = 1;
        const inputShape = [4, 4, inputDepth];
        const outputDepth = 3;
        const fSize = 1;
        const pad = 'same';
        const strides = [2, 2];
        const x = tf.tensor3d([
            10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ], inputShape);
        const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({ x, filter: w, strides, pad });
        expectArraysClose(await result.data(), [10, 5, 10, 50, 25, 50, -10, -5, -10, -50, -25, -50]);
    });
    it('im2row with relu', async () => {
        const inputDepth = 1;
        const inputShape = [4, 4, inputDepth];
        const outputDepth = 3;
        const fSize = 1;
        const pad = 'same';
        const strides = [2, 2];
        const x = tf.tensor3d([
            10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ], inputShape);
        const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'relu'
        });
        expectArraysClose(await result.data(), [10, 5, 10, 50, 25, 50, 0, 0, 0, 0, 0, 0]);
    });
    it('im2row with prelu', async () => {
        const inputDepth = 1;
        const inputShape = [4, 4, inputDepth];
        const outputDepth = 3;
        const fSize = 1;
        const pad = 'same';
        const strides = [2, 2];
        const x = tf.tensor3d([
            10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ], inputShape);
        const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
        const alpha = tf.tensor3d([0.5], [1, 1, inputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'prelu',
            preluActivationWeights: alpha
        });
        expectArraysClose(await result.data(), [10, 5, 10, 50, 25, 50, -5, -2.5, -5, -25, -12.5, -25]);
    });
    it('im2row with leakyrelu', async () => {
        const inputDepth = 1;
        const inputShape = [4, 4, inputDepth];
        const outputDepth = 3;
        const fSize = 1;
        const pad = 'same';
        const strides = [2, 2];
        const x = tf.tensor3d([
            10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ], inputShape);
        const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
        const alpha = 0.3;
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha: alpha
        });
        expectArraysClose(await result.data(), [
            10, 5, 10, 50, 25, 50, -3, -1.5, -3, -15.000000953674316,
            -7.500000476837158, -15.000000953674316
        ]);
    });
    it('pointwise with prelu', async () => {
        const inputDepth = 1;
        const inputShape = [4, 4, inputDepth];
        const outputDepth = 3;
        const fSize = 1;
        const pad = 'same';
        const strides = [1, 1];
        const x = tf.tensor3d([
            10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ], inputShape);
        const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
        const alpha = tf.tensor3d([0.5], [1, 1, inputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'prelu',
            preluActivationWeights: alpha
        });
        expectArraysClose(await result.data(), [
            10, 5, 10, 30, 15, 30, 50, 25, 50, 70, 35, 70,
            20, 10, 20, 40, 20, 40, 60, 30, 60, 80, 40, 80,
            -5, -2.5, -5, -15, -7.5, -15, -25, -12.5, -25, -35, -17.5, -35,
            -10, -5, -10, -20, -10, -20, -30, -15, -30, -40, -20, -40
        ]);
    });
    it('pointwise with leakyrelu', async () => {
        const inputDepth = 1;
        const inputShape = [4, 4, inputDepth];
        const outputDepth = 3;
        const fSize = 1;
        const pad = 'same';
        const strides = [1, 1];
        const x = tf.tensor3d([
            10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ], inputShape);
        const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
        const alpha = 0.3;
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            activation: 'leakyrelu',
            leakyreluAlpha: alpha
        });
        expectArraysClose(await result.data(), [
            10,
            5,
            10,
            30,
            15,
            30,
            50,
            25,
            50,
            70,
            35,
            70,
            20,
            10,
            20,
            40,
            20,
            40,
            60,
            30,
            60,
            80,
            40,
            80,
            -3,
            -1.5,
            -3,
            -9,
            -4.5,
            -9,
            -15.000000953674316,
            -7.500000476837158,
            -15.000000953674316,
            -21,
            -10.5,
            -21,
            -6,
            -3,
            -6,
            -12,
            -6,
            -12,
            -18,
            -9,
            -18,
            -24,
            -12,
            -24
        ]);
    });
    it('im2row with broadcasted bias and relu', async () => {
        const inputDepth = 1;
        const inputShape = [4, 4, inputDepth];
        const outputDepth = 3;
        const fSize = 1;
        const pad = 'same';
        const strides = [2, 2];
        const x = tf.tensor3d([
            10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ], inputShape);
        const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
        const result = tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            bias: tf.scalar(5),
            activation: 'relu'
        });
        expectArraysClose(await result.data(), [15, 10, 15, 55, 30, 55, 0, 0, 0, 0, 0, 0]);
    });
    it('backProp input x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
        const inputDepth = 1;
        const outputDepth = 1;
        const inputShape = [2, 3, 3, inputDepth];
        const filterSize = 2;
        const strides = 1;
        const pad = 0;
        const filterShape = [filterSize, filterSize, inputDepth, outputDepth];
        const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
        const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);
        const grads = tf.grads((x) => tf.fused.conv2d({ x, filter, strides, pad }));
        const [dx] = grads([x], dy);
        expect(dx.shape).toEqual(x.shape);
        expectArraysClose(await dx.data(), [-3, 2, 1, -8, 1.5, 0.5, -4, 1, 0, -3, 2, 1, -8, 1.5, 0.5, -4, 1, 0]);
    });
    it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
        const inputDepth = 1;
        const outputDepth = 1;
        const inputShape = [2, 3, 3, inputDepth];
        const filterSize = 2;
        const strides = 1;
        const pad = 0;
        const filterShape = [filterSize, filterSize, inputDepth, outputDepth];
        const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
        const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);
        const grads = tf.grads((x, filter) => tf.fused.conv2d({ x, filter, strides, pad }));
        const [dx, dfilter] = grads([x, filter], dy);
        expect(dx.shape).toEqual(x.shape);
        expectArraysClose(await dx.data(), [-3, 2, 1, -8, 1.5, 0.5, -4, 1, 0, -3, 2, 1, -8, 1.5, 0.5, -4, 1, 0]);
        expect(dfilter.shape).toEqual(filterShape);
        expectArraysClose(await dfilter.data(), [26, 38, 62, 74]);
    });
    it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias', async () => {
        const inputDepth = 1;
        const outputDepth = 1;
        const inputShape = [2, 3, 3, inputDepth];
        const filterSize = 2;
        const strides = 1;
        const pad = 0;
        const filterShape = [filterSize, filterSize, inputDepth, outputDepth];
        const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
        const bias = tf.ones([2, 2, 2, 1]);
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
        const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);
        const fusedGrads = tf.grads((x, w, b) => tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            bias: b
        }));
        const [dxFused, dfilterFused, dbiasFused] = fusedGrads([x, filter, bias], dy);
        const grads = tf.grads((x, filter, bias) => {
            const conv = tf.conv2d(x, filter, strides, pad);
            const sum = tf.add(conv, bias);
            return sum;
        });
        const [dx, dfilter, dbias] = grads([x, filter, bias], dy);
        expectArraysClose(await dxFused.array(), await dx.array());
        expectArraysClose(await dfilterFused.array(), await dfilter.array());
        expectArraysClose(await dbiasFused.array(), await dbias.array());
    });
    it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and relu', async () => {
        const inputDepth = 1;
        const outputDepth = 1;
        const inputShape = [2, 3, 3, inputDepth];
        const filterSize = 2;
        const strides = 1;
        const pad = 0;
        const filterShape = [filterSize, filterSize, inputDepth, outputDepth];
        const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
        const bias = tf.ones([2, 2, 2, 1]);
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
        const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);
        const fusedGrads = tf.grads((x, w, b) => tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            bias: b,
            activation: 'relu'
        }));
        const [dxFused, dfilterFused, dbiasFused] = fusedGrads([x, filter, bias], dy);
        const grads = tf.grads((x, filter, bias) => {
            const conv = tf.conv2d(x, filter, strides, pad);
            const sum = tf.add(conv, bias);
            return tf.relu(sum);
        });
        const [dx, dfilter, dbias] = grads([x, filter, bias], dy);
        expectArraysClose(await dxFused.array(), await dx.array());
        expectArraysClose(await dfilterFused.array(), await dfilter.array());
        expectArraysClose(await dbiasFused.array(), await dbias.array());
    });
    it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and elu', async () => {
        const inputDepth = 1;
        const outputDepth = 1;
        const inputShape = [2, 3, 3, inputDepth];
        const filterSize = 2;
        const strides = 1;
        const pad = 0;
        const filterShape = [filterSize, filterSize, inputDepth, outputDepth];
        const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
        const bias = tf.ones([2, 2, 2, 1]);
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
        const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);
        const fusedGrads = tf.grads((x, w, b) => tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            bias: b,
            activation: 'elu'
        }));
        const [dxFused, dfilterFused, dbiasFused] = fusedGrads([x, filter, bias], dy);
        const grads = tf.grads((x, filter, bias) => {
            const conv = tf.conv2d(x, filter, strides, pad);
            const sum = tf.add(conv, bias);
            return tf.elu(sum);
        });
        const [dx, dfilter, dbias] = grads([x, filter, bias], dy);
        expectArraysClose(await dxFused.array(), await dx.array());
        expectArraysClose(await dfilterFused.array(), await dfilter.array());
        expectArraysClose(await dbiasFused.array(), await dbias.array());
    });
    it('throws when input is int32', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape, 'int32');
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);
        expect(() => tf.fused.conv2d({ x, filter: w, strides: stride, pad }))
            .toThrowError(/Argument 'x' passed to 'conv2d' must be float32/);
    });
    it('throws when filter is int32', async () => {
        const inputDepth = 2;
        const inShape = [2, 2, 2, inputDepth];
        const outputDepth = 2;
        const fSize = 1;
        const pad = 0;
        const stride = 1;
        const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
        const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth], 'int32');
        expect(() => tf.fused.conv2d({ x, filter: w, strides: stride, pad }))
            .toThrowError(/Argument 'filter' passed to 'conv2d' must be float32/);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZnVzZWRfY29udjJkX3Rlc3QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9mdXNlZC9mdXNlZF9jb252MmRfdGVzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEtBQUssRUFBRSxNQUFNLGFBQWEsQ0FBQztBQUNsQyxPQUFPLEVBQUMsUUFBUSxFQUFFLGlCQUFpQixFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDL0QsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFbEQsU0FBUyxrQkFBa0IsQ0FBQyxlQUF1QixFQUFFLGVBQXVCO0lBQzFFLE1BQU0sR0FBRyxHQUFHLElBQUksS0FBSyxDQUFDLGVBQWUsQ0FBQyxDQUFDO0lBQ3ZDLE1BQU0sSUFBSSxHQUFHLElBQUksS0FBSyxDQUFDLGVBQWUsQ0FBQyxDQUFDO0lBRXhDLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxlQUFlLEVBQUUsQ0FBQyxFQUFFLEVBQUU7UUFDeEMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxLQUFLLEdBQUcsZUFBZSxHQUFHLEtBQUssR0FBRyxDQUFDLENBQUM7S0FDbEQ7SUFDRCxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsZUFBZSxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQ3hDLE1BQU0sSUFBSSxHQUFHLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2xDLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsS0FBSyxHQUFHLElBQUksQ0FBQztLQUM1QjtJQUVELE9BQU8sRUFBQyxLQUFLLEVBQUUsR0FBRyxFQUFFLE1BQU0sRUFBRSxJ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Q2pCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixDQUFDO1lBQ0Qsa0JBQWtCO1lBQ2xCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixtQkFBbUI7WUFDbkIsaUJBQWlCO1lBQ2pCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsa0JBQWtCO1lBQ2xCLGlCQUFpQjtZQUNqQixpQkFBaUI7WUFDakIsa0JBQWtCO1lBQ2xCLGtCQUFrQjtZQUNsQixpQkFBaUI7WUFDakIsa0JBQWtCO1lBQ2xCLGtCQUFrQjtZQUNsQixpQkFBaUI7WUFDakIsaUJBQWlCO1lBQ2pCLGdCQUFnQjtZQUNoQixDQUFDO1lBQ0Qsa0JBQWtCO1lBQ2xCLGtCQUFrQjtZQUNsQixpQkFBaUI7WUFDakIsaUJBQWlCO1lBQ2pCLGVBQWU7WUFDZixnQkFBZ0I7WUFDaEIsaUJBQWlCO1lBQ2pCLENBQUM7WUFDRCxpQkFBaUI7WUFDakIsQ0FBQztZQUNELGlCQUFpQjtZQUNqQixrQkFBa0I7WUFDbEIsaUJBQWlCO1lBQ2pCLGlCQUFpQjtZQUNqQixrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixDQUFDO1lBQ0Qsa0JBQWtCO1lBQ2xCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixDQUFDO1lBQ0Qsa0JBQWtCO1lBQ2xCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixDQUFDO1lBQ0Qsa0JBQWtCO1lBQ2xCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixDQUFDO1lBQ0QsaUJBQWlCO1lBQ2pCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixDQUFDO1lBQ0Qsa0JBQWtCO1lBQ2xCLENBQUM7WUFDRCxrQkFBa0I7WUFDbEIsQ0FBQztZQUNELGtCQUFrQjtZQUNsQixDQUFDO1lBQ0QsaUJBQWlCO1lBQ2pCLENB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QUFDLENBQUM7SUFFSCxFQUFFLENBQUMsMENBQTBDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDeEQsTUFBTSxVQUFVLEdBQUcsQ0FBQyxDQUFDO1FBQ3JCLE1BQU0sV0FBVyxHQUFHLENBQUMsQ0FBQztRQUN0QixNQUFNLFVBQVUsR0FBcUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQztRQUMzRSxNQUFNLFVBQVUsR0FBRyxDQUFDLENBQUM7UUFDckIsTUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDO1FBQ2xCLE1BQU0sR0FBRyxHQUFHLENBQUMsQ0FBQztRQUVkLE1BQU0sV0FBVyxHQUNiLENBQUMsVUFBVSxFQUFFLFVBQVUsRUFBRSxVQUFVLEVBQUUsV0FBVyxDQUFDLENBQUM7UUFDdEQsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsRUFBRSxXQUFXLENBQUMsQ0FBQztRQUUxRCxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsVUFBVSxDQUFDLENBQUM7UUFDeEUsTUFBTSxFQUFFLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFL0QsTUFBTSxLQUFLLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FDbEIsQ0FBQyxDQUFjLEVBQUUsTUFBbUIsRUFBRSxFQUFFLENBQ3BDLEVBQUUsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLEVBQUMsQ0FBQyxFQUFFLE1BQU0sRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3BELE1BQU0sQ0FBQyxFQUFFLEVBQUUsT0FBTyxDQUFDLEdBQUcsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1FBRTdDLE1BQU0sQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNsQyxpQkFBaUIsQ0FDYixNQUFNLEVBQUUsQ0FBQyxJQUFJLEVBQUUsRUFDZixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUUxRSxNQUFNLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzQyxpQkFBaUIsQ0FBQyxNQUFNLE9BQU8sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsb0RBQW9ELEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDbEUsTUFBTSxVQUFVLEdBQUcsQ0FBQyxDQUFDO1FBQ3JCLE1BQU0sV0FBVyxHQUFHLENBQUMsQ0FBQztRQUN0QixNQUFNLFVBQVUsR0FBcUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQztRQUMzRSxNQUFNLFVBQVUsR0FBRyxDQUFDLENBQUM7UUFDckIsTUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDO1FBQ2xCLE1BQU0sR0FBRyxHQUFHLENBQUMsQ0FBQztRQUVkLE1BQU0sV0FBVyxHQUNiLENBQUMsVUFBVSxFQUFFLFVBQVUsRUFBRSxVQUFVLEVBQUUsV0FBVyxDQUFDLENBQUM7UUFDdEQsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsRUFBRSxXQUFXLENBQUMsQ0FBQztRQUMxRCxNQUFNLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUVuQyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsVUFBVSxDQUFDLENBQUM7UUFDeEUsTUFBTSxFQUFFLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFL0QsTUFBTSxVQUFV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UUFFakIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDakIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztRQUN0RSxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLFdBQVcsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBRXhFLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxFQUFDLENBQUMsRUFBRSxNQUFNLEVBQUUsQ0FBQyxFQUFFLE9BQU8sRUFBRSxNQUFNLEVBQUUsR0FBRyxFQUFDLENBQUMsQ0FBQzthQUM5RCxZQUFZLENBQUMsc0RBQXNELENBQUMsQ0FBQztJQUM1RSxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZiBmcm9tICcuLi8uLi9pbmRleCc7XG5pbXBvcnQge0FMTF9FTlZTLCBkZXNjcmliZVdpdGhGbGFnc30gZnJvbSAnLi4vLi4vamFzbWluZV91dGlsJztcbmltcG9ydCB7ZXhwZWN0QXJyYXlzQ2xvc2V9IGZyb20gJy4uLy4uL3Rlc3RfdXRpbCc7XG5cbmZ1bmN0aW9uIGdlbmVyYXRlQ2FzZUlucHV0cyh0b3RhbFNpemVUZW5zb3I6IG51bWJlciwgdG90YWxTaXplRmlsdGVyOiBudW1iZXIpIHtcbiAgY29uc3QgaW5wID0gbmV3IEFycmF5KHRvdGFsU2l6ZVRlbnNvcik7XG4gIGNvbnN0IGZpbHQgPSBuZXcgQXJyYXkodG90YWxTaXplRmlsdGVyKTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IHRvdGFsU2l6ZVRlbnNvcjsgaSsrKSB7XG4gICAgaW5wW2ldID0gaSAqIDAuMDAxIC0gdG90YWxTaXplVGVuc29yICogMC4wMDEgLyAyO1xuICB9XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgdG90YWxTaXplRmlsdGVyOyBpKyspIHtcbiAgICBjb25zdCBzaWduID0gaSAlIDIgPT09IDAgPyAtMSA6IDE7XG4gICAgZmlsdFtpXSA9IGkgKiAwLjAwMSAqIHNpZ247XG4gIH1cblxuICByZXR1cm4ge2lucHV0OiBpbnAsIGZpbHRlcjogZmlsdH07XG59XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdmdXNlZCBjb252MmQnLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnYmFzaWMnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDI7XG4gICAgY29uc3QgaW5TaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbMiwgMiwgMiwgaW5wdXREZXB0aF07XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGZTaXplID0gMTtcbiAgICBjb25zdCBwYWQgPSAwO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxMCwgMTEsIDEyLCAxMywgMTQsIDE1LCAxNl0sIGluU2hhcGUpO1xuICAgIGNvbnN0IHcgPVxuICAgICAgICB0Zi50ZW5zb3I0ZChbLTEsIDEsIC0yLCAwLjVdLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZnVzZWQuY29udjJkKHt4LCBmaWx0ZXI6IHcsIHN0cmlkZXM6IHN0cmlkZSwgcGFkfSk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMiwgMiwgMiwgMl0pO1xuICAgIGNvbnN0IGV4cGVjdGVkID1cbiAgICAgICAgWy01LCAyLCAtMTEsIDUsIC0xNywgOCwgLTIzLCAxMSwgLTI5LCAxNCwgLTM1LCAxNywgLTQxLCAyMCwgLTQ3LCAyM107XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdiYXNpYyB3aXRoIHJlbHUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDI7XG4gICAgY29uc3QgaW5TaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbMiwgMiwgMiwgaW5wdXREZXB0aF07XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGZTaXplID0gMTtcbiAgICBjb25zdCBwYWQgPSAwO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxMCwgMTEsIDEyLCAxMywgMTQsIDE1LCAxNl0sIGluU2hhcGUpO1xuICAgIGNvbnN0IHcgPVxuICAgICAgICB0Zi50ZW5zb3I0ZChbLTEsIDEsIC0yLCAwLjVdLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZnVzZWQuY29udjJkKHtcbiAgICAgIHgsXG4gICAgICBmaWx0ZXI6IHcsXG4gICAgICBzdHJpZGVzOiBzdHJpZGUsXG4gICAgICBwYWQsXG4gICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICBkaWxhdGlvbnM6IFsxLCAxXSxcbiAgICAgIGFjdGl2YXRpb246ICdyZWx1J1xuICAgIH0pO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzIsIDIsIDIsIDJdKTtcbiAgICBjb25zdCBleHBlY3RlZCA9IFswLCAyLCAwLCA1LCAwLCA4LCAwLCAxMSwgMCwgMTQsIDAsIDE3LCAwLCAyMCwgMCwgMjNdO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgZXhwZWN0ZWQpO1xuICB9KTtcblxuICBpdCgncmVsdSB3aXRoIHN0cmlkZSAyIHg9WzEsOCw4LDE2XSBmPVszLDMsMTYsMV0gcz1bMiwyXSBkPTEgcD1zYW1lJyxcbiAgICAgYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IGlucHV0RGVwdGggPSAxNjtcbiAgICAgICBjb25zdCB4U2l6ZSA9IDg7XG4gICAgICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICAgICBbMSwgeFNpemUsIHhTaXplLCBpbnB1dERlcHRoXTtcbiAgICAgICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gICAgICAgY29uc3QgZlNpemUgPSAzO1xuICAgICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICAgICBjb25zdCBzdHJpZGU6IFtudW1iZXIsIG51bWJlcl0gPSBbMiwgMl07XG5cbiAgICAgICAvLyBUT0RPKGFubnhpbmd5dWFuKTogTWFrZSB0aGlzIHRlc3Qgd29yayB3aXRoIGxhcmdlIGlucHV0c1xuICAgICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzMxNDNcbiAgICAgICBjb25zdCBpbnB1dERhdGEgPSBbXTtcbiAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHhTaXplICogeFNpemUgKiBpbnB1dERlcHRoOyBpKyspIHtcbiAgICAgICAgIGlucHV0RGF0YS5wdXNoKGkgJSA1KTtcbiAgICAgICB9XG5cbiAgICAgICBjb25zdCB3RGF0YSA9IFtdO1xuICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgZlNpemUgKiBmU2l6ZSAqIGlucHV0RGVwdGggKiBvdXRwdXREZXB0aDsgaSsrKSB7XG4gICAgICAgICB3RGF0YS5wdXNoKGkgJSA1KTtcbiAgICAgICB9XG5cbiAgICAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoaW5wdXREYXRhLCBpbnB1dFNoYXBlKTtcbiAgICAgICBjb25zdCB3ID0gdGYudGVuc29yNGQod0RhdGEsIFtmU2l6ZSwgZlNpemUsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXSk7XG5cbiAgICAgICBjb25zdCByZXN1bHQgPSB0Zi5mdXNlZC5jb252MmQoe1xuICAgICAgICAgeCxcbiAgICAgICAgIGZpbHRlcjogdyxcbiAgICAgICAgIHN0cmlkZXM6IHN0cmlkZSxcbiAgICAgICAgIHBhZCxcbiAgICAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgICAgIGRpbGF0aW9uczogWzEsIDFdLFxuICAgICAgICAgYWN0aXZhdGlvbjogJ3JlbHUnXG4gICAgICAgfSk7XG4gICAgICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgNCwgNCwgMV0pO1xuICAgICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIG5ldyBGbG9hdDMyQXJyYXkoW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgODU0LCA0MzEsIDU2OCwgMzgyLCA1ODAsIDQyNywgODU0LCAyODgsIDQzMSwgNTY4LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNTgwLCAyODksIDI4NSwgNTcwLCAyODUsIDI1OFxuICAgICAgICAgICAgICAgICAgICAgICAgIF0pKTtcbiAgICAgfSk7XG5cbiAgaXQoJ3JlbHUgYmlhcyBzdHJpZGUgMiB4PVsxLDgsOCwxNl0gZj1bMywzLDE2LDFdIHM9WzIsMl0gZD04IHA9c2FtZScsXG4gICAgIGFzeW5jICgpID0+IHtcbiAgICAgICBjb25zdCBpbnB1dERlcHRoID0gMTY7XG4gICAgICAgY29uc3QgeFNpemUgPSA4O1xuICAgICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICAgWzEsIHhTaXplLCB4U2l6ZSwgaW5wdXREZXB0aF07XG4gICAgICAgY29uc3Qgb3V0cHV0RGVwdGggPSA4O1xuICAgICAgIGNvbnN0IGZTaXplID0gMztcbiAgICAgICBjb25zdCBwYWQgPSAnc2FtZSc7XG4gICAgICAgY29uc3Qgc3RyaWRlOiBbbnVtYmVyLCBudW1iZXJdID0gWzIsIDJdO1xuXG4gICAgICAgY29uc3QgaW5wdXRzID0gZ2VuZXJhdGVDYXNlSW5wdXRzKFxuICAgICAgICAgICAxICogeFNpemUgKiB4U2l6ZSAqIGlucHV0RGVwdGgsXG4gICAgICAgICAgIGZTaXplICogZlNpemUgKiBpbnB1dERlcHRoICogb3V0cHV0RGVwdGgpO1xuICAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChpbnB1dHMuaW5wdXQsIGlucHV0U2hhcGUpO1xuICAgICAgIGNvbnN0IHcgPVxuICAgICAgICAgICB0Zi50ZW5zb3I0ZChpbnB1dHMuZmlsdGVyLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuICAgICAgIGNvbnN0IGJpYXMgPSB0Zi50ZW5zb3IxZChbMSwgNCwgMiwgMywgOSwgNiwgNSwgOF0pO1xuICAgICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICAgICB4LFxuICAgICAgICAgZmlsdGVyOiB3LFxuICAgICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgICAgcGFkLFxuICAgICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICAgICBhY3RpdmF0aW9uOiAncmVsdScsXG4gICAgICAgICBiaWFzXG4gICAgICAgfSk7XG4gICAgICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMSwgNCwgNCwgOF0pO1xuICAgICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIG5ldyBGbG9hdDMyQXJyYXkoW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjUuNzUzOTgwNjM2NTk2NjgsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjYuODU3ODA1MjUyMDc1MTk1LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDMzLjk2MTYzMTc3NDkwMjM0NCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAzMC4wNjU0NTgyOTc3Mjk0OTIsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjMuMTE4MjA2MDI0MTY5OTIyLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDI0LjIxMjgyMDA1MzEwMDU4NixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAzMS4zMDc0MjI2Mzc5Mzk0NTMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjcuNDAyMDM0NzU5NTIxNDg0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIwLjQ4MjQzMTQxMTc0MzE2NCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyMS41Njc4MjE1MDI2ODU1NDcsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjguNjUzMjE3MzE1NjczODI4LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDI0LjczODYxMzEyODY2MjExLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDExLjA3ODA4MDE3NzMwNzEyOSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAxMi4xMzAzOTk3MDM5Nzk0OTIsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTkuMTgyNzIwMTg0MzI2MTcyLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDE1LjIzNTAzNzgwMzY0OTkwMixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA0LjY2Nzc3NzUzODI5OTU2MDUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLjMxNzE3NzI5NTY4NDgxNDQ1LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNS42OTc4Njk3Nzc2Nzk0NDMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTIuNzI3OTY4MjE1OTQyMzgzLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMi4yNTY5ODQ5NDkxMTE5Mzg1LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgOC43NTgwNjYxNzczNjgxNjQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA0LjIyNjg4NTc5NTU5MzI2MixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIuMDMxOTk5NTg4MDEyNjk1MyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIuOTU3NTU4NjMxODk2OTcyNyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDMuMDUyODgwMDQ4NzUxODMxLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMS45MzY2Nzk2MDE2NjkzMTE1LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTAuMDczNzYwMDMyNjUzODA5LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNC45MTU3OTk2MTc3NjczMzQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LjA5NDYzOTc3ODEzNzIwNyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYuODk0OTIxMzAyNzk1NDEsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNS41OTc5NDM3ODI4MDYzOTY1LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMC40MDc4ODc1NzgwMTA1NTkxLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNC41ODYyODA4MjI3NTM5MDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA3LjQxOTU1MTg0OTM2NTIzNCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDcuNTc0NjE2OTA5MDI3MSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDMuNDMxMjE2MDAxNTEwNjIsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA5LjU2Mjk1MjA0MTYyNTk3NyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LjQwNDk0Mzk0MzAyMzY4MixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA1LjQwMTc3NjMxMzc4MTczOCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYuNTk5ODA3NzM5MjU3ODEyNSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDguMzk4NjA4MjA3NzAyNjM3LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMi42MDI5NzYwODM3NTU0OTMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAxMC4zOTU0NDAxMDE2MjM1MzUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjEuNDQwMjUwMzk2NzI4NTE2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIwLjQ4Mzg4MjkwNDA1MjczNCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyMy41Mjc1MDk2ODkzMzEwNTUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjUuNTcxMTQ0MTA0MDAzOTA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDI0LjA4MDYyOTM0ODc1NDg4MyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyMy4xMzM0ODAwNzIwMjE0ODQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjYuMTg2MzI4ODg3OTM5NDUzLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDI4LjIzOTE3NzcwMzg1NzQyMixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyNi43MjEwMTIxMTU0Nzg1MTYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjUuNzgzMDc5MTQ3MzM4ODY3LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDI4Ljg0NTE0ODA4NjU0Nzg1LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDMwLjkwNzIwOTM5NjM2MjMwNSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAxOC45MTQxMjczNDk4NTM1MTYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTcuOTYwMTExNjE4MDQxOTkyLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIxLjAwNjA5Mzk3ODg4MTgzNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyMy4wNTIwODIwNjE3Njc1NzgsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTcuODkwODk1ODQzNTA1ODYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTYuOTU2ODQ4MTQ0NTMxMjUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjAuMDIyNzk4NTM4MjA4MDA4LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIyLjA4ODc1NDY1MzkzMDY2NCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAxOS4wNjEzMjY5ODA1OTA4MixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAxOC4xMzM0MjQ3NTg5MTExMzMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjEuMjA1NTIwNjI5ODgyODEyLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIzLjI3NzYxNDU5MzUwNTg2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDIwLjIzMTc1ODExNzY3NTc4LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDE5LjMwOTk5OTQ2NTk0MjM4MyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyMi4zODgyNDA4MTQyMDg5ODQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMjQuNDY2NDc4MzQ3Nzc4MzIsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTMuNTg0MzUyNDkzMjg2MTMzLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDEyLjYzOTU4NDU0MTMyMDgsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMTUuNjk0ODE1NjM1NjgxMTUyLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDE3Ljc1MDA0NTc3NjM2NzE4OFxuICAgICAgICAgICAgICAgICAgICAgICAgIF0pKTtcbiAgICAgfSk7XG5cbiAgaXQoJ3ByZWx1IGJpYXMgc3RyaWRlIDIgeD1bMSw4LDgsMTZdIGY9WzMsMywxNiwxXSBzPVsyLDJdIGQ9OCBwPXNhbWUnLFxuICAgICBhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaW5wdXREZXB0aCA9IDE2O1xuICAgICAgIGNvbnN0IHhTaXplID0gODtcbiAgICAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgICAgIFsxLCB4U2l6ZSwgeFNpemUsIGlucHV0RGVwdGhdO1xuICAgICAgIGNvbnN0IG91dHB1dERlcHRoID0gODtcbiAgICAgICBjb25zdCBmU2l6ZSA9IDM7XG4gICAgICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgICAgIGNvbnN0IHN0cmlkZTogW251bWJlciwgbnVtYmVyXSA9IFsyLCAyXTtcblxuICAgICAgIGNvbnN0IGlucHV0cyA9IGdlbmVyYXRlQ2FzZUlucHV0cyhcbiAgICAgICAgICAgMSAqIHhTaXplICogeFNpemUgKiBpbnB1dERlcHRoLFxuICAgICAgICAgICBmU2l6ZSAqIGZTaXplICogaW5wdXREZXB0aCAqIG91dHB1dERlcHRoKTtcbiAgICAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoaW5wdXRzLmlucHV0LCBpbnB1dFNoYXBlKTtcbiAgICAgICBjb25zdCB3ID1cbiAgICAgICAgICAgdGYudGVuc29yNGQoaW5wdXRzLmZpbHRlciwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICAgICBjb25zdCBiaWFzID0gdGYudGVuc29yMWQoWzEsIDQsIDIsIDMsIDksIDYsIDUsIDhdKTtcbiAgICAgICBjb25zdCBwcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gdGYudGVuc29yMWQoWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDhdKTtcblxuICAgICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICAgICB4LFxuICAgICAgICAgZmlsdGVyOiB3LFxuICAgICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgICAgcGFkLFxuICAgICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICAgICBhY3RpdmF0aW9uOiAncHJlbHUnLFxuICAgICAgICAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyxcbiAgICAgICAgIGJpYXNcbiAgICAgICB9KTtcbiAgICAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKFsxLCA0LCA0LCA4XSk7XG4gICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoXG4gICAgICAgICAgIGF3YWl0IHJlc3VsdC5kYXRhKCksIG5ldyBGbG9hdDMyQXJyYXkoW1xuICAgICAgICAgICAgIDI1Ljc1Mzk4MDYzNjU5NjY4LCAgIC00MS42MTE3ODk3MDMzNjkxNCwgIDI2Ljg1NzgwNTI1MjA3NTE5NSxcbiAgICAgICAgICAgICAtODcuNjM4ODU0OTgwNDY4NzUsICAzMy45NjE2MzE3NzQ5MDIzNDQsICAtMTE0LjA4MTI3NTkzOTk0MTQsXG4gICAgICAgICAgICAgMzAuMDY1NDU4Mjk3NzI5NDkyLCAgLTEzNi45Mzg5MzQzMjYxNzE4OCwgMjMuMTE4MjA2MDI0MTY5OTIyLFxuICAgICAgICAgICAgIC0zNi4zMzEwMjAzNTUyMjQ2MSwgIDI0LjIxMjgyMDA1MzEwMDU4NiwgIC03Ny4wNDA0ODE1NjczODI4MSxcbiAgICAgICAgICAgICAzMS4zMDc0MjI2Mzc5Mzk0NTMsICAtOTguMTI4MzU2OTMzNTkzNzUsICAyNy40MDIwMzQ3NTk1MjE0ODQsXG4gICAgICAgICAgICAgLTExNS41OTQ3MjY1NjI1LCAgICAgMjAuNDgyNDMxNDExNzQzMTY0LCAgLTMxLjA1MDI2MjQ1MTE3MTg3NSxcbiAgICAgICAgICAgICAyMS41Njc4MjE1MDI2ODU1NDcsICAtNjYuNDQyMDkyODk1NTA3ODEsICAyOC42NTMyMTczMTU2NzM4MjgsXG4gICAgICAgICAgICAgLTgyLjE3NTQ0NTU1NjY0MDYyLCAgMjQuNzM4NjEzMTI4NjYyMTEsICAgLTk0LjI1MDQxMTk4NzMwNDY5LFxuICAgICAgICAgICAgIDExLjA3ODA4MDE3NzMwNzEyOSwgIC0xMi4yMDg0Nzg5Mjc2MTIzMDUsIDEyLjEzMDM5OTcwMzk3OTQ5MixcbiAgICAgICAgICAgICAtMjguNjI2MjMyMTQ3MjE2Nzk3LCAxOS4xODI3MjAxODQzMjYxNzIsICAtMjUuMjUzMjk5NzEzMTM0NzY2LFxuICAgICAgICAgICAgIDE1LjIzNTAzNzgwMzY0OTkwMiwgIC0xOC4wODk2MDcyMzg3Njk1MywgIDQuNjY3Nzc3NTM4Mjk5NTYwNSxcbiAgICAgICAgICAgICAwLjMxNzE3NzI5NTY4NDgxNDQ1LCA1LjY5Nzg2OTc3NzY3OTQ0MywgICAtMi44NTE2NzU5ODcyNDM2NTIzLFxuICAgICAgICAgICAgIDEyLjcyNzk2ODIxNTk0MjM4MywgIDIuMjU2OTg0OTQ5MTExOTM4NSwgIDguNzU4MDY2MTc3MzY4MTY0LFxuICAgICAgICAgICAgIDQuMjI2ODg1Nzk1NTkzMjYyLCAgIDIuMDMxOTk5NTg4MDEyNjk1MywgIDIuOTU3NTU4NjMxODk2OTcyNyxcbiAgICAgICAgICAgICAzLjA1Mjg4MDA0ODc1MTgzMSwgICAxLjkzNjY3OTYwMTY2OTMxMTUsICAxMC4wNzM3NjAwMzI2NTM4MDksXG4gICAgICAgICAgICAgNC45MTU3OTk2MTc3NjczMzQsICAgNi4wOTQ2Mzk3NzgxMzcyMDcsICAgNi44OTQ5MjEzMDI3OTU0MSxcbiAgICAgICAgICAgICAtMC42MDM3NzYzMzU3MTYyNDc2LCA1LjU5Nzk0Mzc4MjgwNjM5NjUsICAwLjQwNzg4NzU3ODAxMDU1OTEsXG4gICAgICAgICAgICAgNC41ODYyODA4MjI3NTM5MDYsICAgNy40MTk1NTE4NDkzNjUyMzQsICAgNy41NzQ2MTY5MDkwMjcxLFxuICAgICAgICAgICAgIDMuNDMxMjE2MDAxNTEwNjIsICAgIDkuNTYyOTUyMDQxNjI1OTc3LCAgIC0xLjQwNjUyNzk5NjA2MzIzMjQsXG4gICAgICAgICAgICAgNi40MDQ5NDM5NDMwMjM2ODIsICAgLTEuMjEwMDgwMzg1MjA4MTI5OSwgNS40MDE3NzYzMTM3ODE3MzgsXG4gICAgICAgICAgICAgNi41OTk4MDc3MzkyNTc4MTI1LCAgOC4zOTg2MDgyMDc3MDI2MzcsICAgMi42MDI5NzYwODM3NTU0OTMsXG4gICAgICAgICAgICAgMTAuMzk1NDQwMTAxNjIzNTM1LCAgLTE2LjQxODQzNDE0MzA2NjQwNiwgMjEuNDQwMjUwMzk2NzI4NTE2LFxuICAgICAgICAgICAgIC00Ni4zODYxODg1MDcwODAwOCwgIDIwLjQ4Mzg4MjkwNDA1MjczNCwgIC00Mi41Mjg0ODgxNTkxNzk2OSxcbiAgICAgICAgICAgICAyMy41Mjc1MDk2ODkzMzEwNTUsICAtODcuODQ1MzA2Mzk2NDg0MzgsICAyNS41NzExNDQxMDQwMDM5MDYsXG4gICAgICAgICAgICAgLTE5LjA1NDIwODc1NTQ5MzE2NCwgMjQuMDgwNjI5MzQ4NzU0ODgzLCAgLTU0LjMyMTE1OTM2Mjc5Mjk3LFxuICAgICAgICAgICAgIDIzLjEzMzQ4MDA3MjAyMTQ4NCwgIC01NS43OTk1MTQ3NzA1MDc4MSwgIDI2LjE4NjMyODg4NzkzOTQ1MyxcbiAgICAgICAgICAgICAtMTA2LjQ4OTI0MjU1MzcxMDk0LCAyOC4yMzkxNzc3MDM4NTc0MjIsICAtMjEuNjg5OTg3MTgyNjE3MTg4LFxuICAgICAgICAgICAgIDI2LjcyMTAxMjExNTQ3ODUxNiwgIC02Mi4yNTYxNDkyOTE5OTIxOSwgIDI1Ljc4MzA3OTE0NzMzODg2NyxcbiAgICAgICAgICAgICAtNjkuMDcwNTU2NjQwNjI1LCAgICAyOC44NDUxNDgwODY1NDc4NSwgICAtMTI1LjEzMzI1NTAwNDg4MjgxLFxuICAgICAgICAgICAgIDMwLjkwNzIwOTM5NjM2MjMwNSwgIC0xMy44OTExMzMzMDg0MTA2NDUsIDE4LjkxNDEyNzM0OTg1MzUxNixcbiAgICAgICAgICAgICAtMzguODExMzU5NDA1NTE3NTgsICAxNy45NjAxMTE2MTgwNDE5OTIsICAtMjkuOTE1NTA0NDU1NTY2NDA2LFxuICAgICAgICAgICAgIDIxLjAwNjA5Mzk3ODg4MTgzNiwgIC03MC4yMDM2MTMyODEyNSwgICAgIDIzLjA1MjA4MjA2MTc2NzU3OCxcbiAgICAgICAgICAgICAtMTIuODU3OTE5NjkyOTkzMTY0LCAxNy44OTA4OTU4NDM1MDU4NiwgICAtMzUuNzcxNjEwMjYwMDA5NzY2LFxuICAgICAgICAgICAgIDE2Ljk1Njg0ODE0NDUzMTI1LCAgIC0yNC45NDkxMTU3NTMxNzM4MjgsIDIwLjAyMjc5ODUzODIwODAwOCxcbiAgICAgICAgICAgICAtNjMuMzkwNDIyODIxMDQ0OTIsICAyMi4wODg3NTQ2NTM5MzA2NjQsICAtMTQuMDI1MjgxOTA2MTI3OTMsXG4gICAgICAgICAgICAgMTkuMDYxMzI2OTgwNTkwODIsICAgLTM5LjI5MjEyNTcwMTkwNDMsICAgMTguMTMzNDI0NzU4OTExMTMzLFxuICAgICAgICAgICAgIC0zMC44NDczNDkxNjY4NzAxMTcsIDIxLjIwNTUyMDYyOTg4MjgxMiwgIC03MS42OTA5NzEzNzQ1MTE3MixcbiAgICAgICAgICAgICAyMy4yNzc2MTQ1OTM1MDU4NiwgICAtMTUuMTkyNjM4Mzk3MjE2Nzk3LCAyMC4yMzE3NTgxMTc2NzU3OCxcbiAgICAgICAgICAgICAtNDIuODEyNjMzNTE0NDA0MywgICAxOS4zMDk5OTk0NjU5NDIzODMsICAtMzYuNzQ1NjA1NDY4NzUsXG4gICAgICAgICAgICAgMjIuMzg4MjQwODE0MjA4OTg0LCAgLTc5Ljk5MTUyMzc0MjY3NTc4LCAgMjQuNDY2NDc4MzQ3Nzc4MzIsXG4gICAgICAgICAgICAgLTguNTU2NzM2OTQ2MTA1OTU3LCAgMTMuNTg0MzUyNDkzMjg2MTMzLCAgLTIyLjgzNTkwMTI2MDM3NTk3NyxcbiAgICAgICAgICAgICAxMi42Mzk1ODQ1NDEzMjA4LCAgICAtMy4zMzYwMDA0NDI1MDQ4ODMsICAxNS42OTQ4MTU2MzU2ODExNTIsXG4gICAgICAgICAgICAgLTMzLjA1NzAxODI4MDAyOTMsICAgMTcuNzUwMDQ1Nzc2MzY3MTg4XG4gICAgICAgICAgIF0pKTtcbiAgICAgfSk7XG5cbiAgaXQoJ3JlbHU2IGJpYXMgc3RyaWRlIDIgeD1bMSw4LDgsMTZdIGY9WzMsMywxNiw4XSBzPVsyLDJdIGQ9OCBwPXNhbWUnLFxuICAgICBhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaW5wdXREZXB0aCA9IDE2O1xuICAgICAgIGNvbnN0IHhTaXplID0gODtcbiAgICAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgICAgIFsxLCB4U2l6ZSwgeFNpemUsIGlucHV0RGVwdGhdO1xuICAgICAgIGNvbnN0IG91dHB1dERlcHRoID0gODtcbiAgICAgICBjb25zdCBmU2l6ZSA9IDM7XG4gICAgICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgICAgIGNvbnN0IHN0cmlkZTogW251bWJlciwgbnVtYmVyXSA9IFsyLCAyXTtcblxuICAgICAgIGNvbnN0IGlucHV0cyA9IGdlbmVyYXRlQ2FzZUlucHV0cyhcbiAgICAgICAgICAgMSAqIHhTaXplICogeFNpemUgKiBpbnB1dERlcHRoLFxuICAgICAgICAgICBmU2l6ZSAqIGZTaXplICogaW5wdXREZXB0aCAqIG91dHB1dERlcHRoKTtcbiAgICAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoaW5wdXRzLmlucHV0LCBpbnB1dFNoYXBlKTtcbiAgICAgICBjb25zdCB3ID1cbiAgICAgICAgICAgdGYudGVuc29yNGQoaW5wdXRzLmZpbHRlciwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICAgICBjb25zdCBiaWFzID0gdGYudGVuc29yMWQoWzEsIDQsIDIsIDMsIDksIDYsIDUsIDhdKTtcblxuICAgICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICAgICB4LFxuICAgICAgICAgZmlsdGVyOiB3LFxuICAgICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgICAgcGFkLFxuICAgICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICAgICBhY3RpdmF0aW9uOiAncmVsdTYnLFxuICAgICAgICAgYmlhc1xuICAgICAgIH0pO1xuICAgICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzEsIDQsIDQsIDhdKTtcbiAgICAgICBjb25zdCByZXN1bHREYXRhID0gYXdhaXQgcmVzdWx0LmRhdGEoKTtcbiAgICAgICBleHBlY3RBcnJheXNDbG9zZShyZXN1bHREYXRhLCBuZXcgRmxvYXQzMkFycmF5KFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNC42Njc3Nzc1MzgyOTk1NjA1LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMC4zMTcxNzcyOTU2ODQ4MTQ0NSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDUuNjk3ODY5Nzc3Njc5NDQzLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyLjI1Njk4NDk0OTExMTkzODUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNC4yMjY4ODU3OTU1OTMyNjIsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyLjAzMTk5OTU4ODAxMjY5NTMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyLjk1NzU1ODYzMTg5Njk3MjcsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAzLjA1Mjg4MDA0ODc1MTgzMSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDEuOTM2Njc5NjAxNjY5MzExNSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA0LjkxNTc5OTYxNzc2NzMzNCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDUuNTk3OTQzNzgyODA2Mzk2NSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAuNDA3ODg3NTc4MDEwNTU5MSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDQuNTg2MjgwODIyNzUzOTA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAzLjQzMTIxNjAwMTUxMDYyLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDUuNDAxNzc2MzEzNzgxNzM4LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAyLjYwMjk3NjA4Mzc1NTQ5MyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDYsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgNixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA2XG4gICAgICAgICAgICAgICAgICAgICAgICAgXSkpO1xuICAgICB9KTtcblxuICBpdCgnbGVha3lyZWx1IGJpYXMgc3RyaWRlIDIgeD1bMSw4LDgsMTZdIGY9WzMsMywxNiwxXSBzPVsyLDJdIGQ9OCBwPXNhbWUnLFxuICAgICBhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaW5wdXREZXB0aCA9IDE2O1xuICAgICAgIGNvbnN0IHhTaXplID0gODtcbiAgICAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgICAgIFsxLCB4U2l6ZSwgeFNpemUsIGlucHV0RGVwdGhdO1xuICAgICAgIGNvbnN0IG91dHB1dERlcHRoID0gODtcbiAgICAgICBjb25zdCBmU2l6ZSA9IDM7XG4gICAgICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgICAgIGNvbnN0IHN0cmlkZTogW251bWJlciwgbnVtYmVyXSA9IFsyLCAyXTtcblxuICAgICAgIGNvbnN0IGlucHV0cyA9IGdlbmVyYXRlQ2FzZUlucHV0cyhcbiAgICAgICAgICAgMSAqIHhTaXplICogeFNpemUgKiBpbnB1dERlcHRoLFxuICAgICAgICAgICBmU2l6ZSAqIGZTaXplICogaW5wdXREZXB0aCAqIG91dHB1dERlcHRoKTtcbiAgICAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoaW5wdXRzLmlucHV0LCBpbnB1dFNoYXBlKTtcbiAgICAgICBjb25zdCB3ID1cbiAgICAgICAgICAgdGYudGVuc29yNGQoaW5wdXRzLmZpbHRlciwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICAgICBjb25zdCBiaWFzID0gdGYudGVuc29yMWQoWzEsIDQsIDIsIDMsIDksIDYsIDUsIDhdKTtcbiAgICAgICBjb25zdCBsZWFreXJlbHVBbHBoYSA9IDAuMztcblxuICAgICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICAgICB4LFxuICAgICAgICAgZmlsdGVyOiB3LFxuICAgICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgICAgcGFkLFxuICAgICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICAgICBhY3RpdmF0aW9uOiAnbGVha3lyZWx1JyxcbiAgICAgICAgIGxlYWt5cmVsdUFscGhhLFxuICAgICAgICAgYmlhc1xuICAgICAgIH0pO1xuICAgICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzEsIDQsIDQsIDhdKTtcbiAgICAgICBleHBlY3RBcnJheXNDbG9zZShcbiAgICAgICAgICAgYXdhaXQgcmVzdWx0LmRhdGEoKSwgbmV3IEZsb2F0MzJBcnJheShbXG4gICAgICAgICAgICAgMjUuNzUzOTgwNjM2NTk2NjgsICAgIC02LjI0MTc2ODgzNjk3NTA5OCwgICAyNi44NTc4MDUyNTIwNzUxOTUsXG4gICAgICAgICAgICAgLTYuNTcyOTE0NjAwMzcyMzE0NSwgIDMzLjk2MTYzMTc3NDkwMjM0NCwgICAtNS43MDQwNjM4OTIzNjQ1MDIsXG4gICAgICAgICAgICAgMzAuMDY1NDU4Mjk3NzI5NDkyLCAgIC01LjEzNTIxMDAzNzIzMTQ0NSwgICAyMy4xMTgyMDYwMjQxNjk5MjIsXG4gICAgICAgICAgICAgLTUuNDQ5NjUzMTQ4NjUxMTIzLCAgIDI0LjIxMjgyMDA1MzEwMDU4NiwgICAtNS43NzgwMzYxMTc1NTM3MTEsXG4gICAgICAgICAgICAgMzEuMzA3NDIyNjM3OTM5NDUzLCAgIC00LjkwNjQxODMyMzUxNjg0NiwgICAyNy40MDIwMzQ3NTk1MjE0ODQsXG4gICAgICAgICAgICAgLTQuMzM0ODAyNjI3NTYzNDc3LCAgIDIwLjQ4MjQzMTQxMTc0MzE2NCwgICAtNC42NTc1MzkzNjc2NzU3ODEsXG4gICAgICAgICAgICAgMjEuNTY3ODIxNTAyNjg1NTQ3LCAgIC00Ljk4MzE1NzE1Nzg5Nzk0OSwgICAyOC42NTMyMTczMTU2NzM4MjgsXG4gICAgICAgICAgICAgLTQuMTA4NzcyMjc3ODMyMDMxLCAgIDI0LjczODYxMzEyODY2MjExLCAgICAtMy41MzQzOTA2ODc5NDI1MDUsXG4gICAgICAgICAgICAgMTEuMDc4MDgwMTc3MzA3MTI5LCAgIC0xLjgzMTI3MTg4NjgyNTU2MTUsICAxMi4xMzAzOTk3MDM5Nzk0OTIsXG4gICAgICAgICAgICAgLTIuMTQ2OTY3NDExMDQxMjU5OCwgIDE5LjE4MjcyMDE4NDMyNjE3MiwgICAtMS4yNjI2NjUwMzMzNDA0NTQsXG4gICAgICAgICAgICAgMTUuMjM1MDM3ODAzNjQ5OTAyLCAgIC0wLjY3ODM2MDI4MzM3NDc4NjQsICA0LjY2Nzc3NzUzODI5OTU2MDUsXG4gICAgICAgICAgICAgMC4zMTcxNzcyOTU2ODQ4MTQ0NSwgIDUuNjk3ODY5Nzc3Njc5NDQzLCAgICAtMC4yMTM4NzU3MTA5NjQyMDI4OCxcbiAgICAgICAgICAgICAxMi43Mjc5NjgyMTU5NDIzODMsICAgMi4yNTY5ODQ5NDkxMTE5Mzg1LCAgIDguNzU4MDY2MTc3MzY4MTY0LFxuICAgICAgICAgICAgIDQuMjI2ODg1Nzk1NTkzMjYyLCAgICAyLjAzMTk5OTU4ODAxMjY5NTMsICAgMi45NTc1NTg2MzE4OTY5NzI3LFxuICAgICAgICAgICAgIDMuMDUyODgwMDQ4NzUxODMxLCAgICAxLjkzNjY3OTYwMTY2OTMxMTUsICAgMTAuMDczNzYwMDMyNjUzODA5LFxuICAgICAgICAgICAgIDQuOTE1Nzk5NjE3NzY3MzM0LCAgICA2LjA5NDYzOTc3ODEzNzIwNywgICAgNi44OTQ5MjEzMDI3OTU0MSxcbiAgICAgICAgICAgICAtMC4xODExMzI5MTI2MzU4MDMyMiwgNS41OTc5NDM3ODI4MDYzOTY1LCAgIDAuNDA3ODg3NTc4MDEwNTU5MSxcbiAgICAgICAgICAgICA0LjU4NjI4MDgyMjc1MzkwNiwgICAgNy40MTk1NTE4NDkzNjUyMzQsICAgIDcuNTc0NjE2OTA5MDI3MSxcbiAgICAgICAgICAgICAzLjQzMTIxNjAwMTUxMDYyLCAgICAgOS41NjI5NTIwNDE2MjU5NzcsICAgIC0wLjQyMTk1ODQxNjcwMDM2MzE2LFxuICAgICAgICAgICAgIDYuNDA0OTQzOTQzMDIzNjgyLCAgICAtMC4xMjEwMDgwNDU5NzEzOTM1OSwgNS40MDE3NzYzMTM3ODE3MzgsXG4gICAgICAgICAgICAgNi41OTk4MDc3MzkyNTc4MTI1LCAgIDguMzk4NjA4MjA3NzAyNjM3LCAgICAyLjYwMjk3NjA4Mzc1NTQ5MyxcbiAgICAgICAgICAgICAxMC4zOTU0NDAxMDE2MjM1MzUsICAgLTQuOTI1NTMwNDMzNjU0Nzg1LCAgIDIxLjQ0MDI1MDM5NjcyODUxNixcbiAgICAgICAgICAgICAtNC42Mzg2MTg5NDYwNzU0Mzk1LCAgMjAuNDgzODgyOTA0MDUyNzM0LCAgIC0yLjU1MTcwOTE3NTEwOTg2MzMsXG4gICAgICAgICAgICAgMjMuNTI3NTA5Njg5MzMxMDU1LCAgIC0zLjc2NDc5OTExODA0MTk5MiwgICAyNS41NzExNDQxMDQwMDM5MDYsXG4gICAgICAgICAgICAgLTUuNzE2MjYyODE3MzgyODEyNSwgIDI0LjA4MDYyOTM0ODc1NDg4MywgICAtNS40MzIxMTY1MDg0ODM4ODcsXG4gICAgICAgICAgICAgMjMuMTMzNDgwMDcyMDIxNDg0LCAgIC0zLjM0Nzk3MDk2MjUyNDQxNCwgICAyNi4xODYzMjg4ODc5Mzk0NTMsXG4gICAgICAgICAgICAgLTQuNTYzODI1MTMwNDYyNjQ2NSwgIDI4LjIzOTE3NzcwMzg1NzQyMiwgICAtNi41MDY5OTY2MzE2MjIzMTQ1LFxuICAgICAgICAgICAgIDI2LjcyMTAxMjExNTQ3ODUxNiwgICAtNi4yMjU2MTU1MDE0MDM4MDksICAgMjUuNzgzMDc5MTQ3MzM4ODY3LFxuICAgICAgICAgICAgIC00LjE0NDIzMzcwMzYxMzI4MSwgICAyOC44NDUxNDgwODY1NDc4NSwgICAgLTUuMzYyODU0MDAzOTA2MjUsXG4gICAgICAgICAgICAgMzAuOTA3MjA5Mzk2MzYyMzA1LCAgIC00LjE2NzM0MDI3ODYyNTQ4OCwgICAxOC45MTQxMjczNDk4NTM1MTYsXG4gICAgICAgICAgICAgLTMuODgxMTM1OTQwNTUxNzU4LCAgIDE3Ljk2MDExMTYxODA0MTk5MiwgICAtMS43OTQ5MzAzMzg4NTk1NTgsXG4gICAgICAgICAgICAgMjEuMDA2MDkzOTc4ODgxODM2LCAgIC0zLjAwODcyNjU5NjgzMjI3NTQsICAyMy4wNTIwODIwNjE3Njc1NzgsXG4gICAgICAgICAgICAgLTMuODU3Mzc2MDk4NjMyODEyNSwgIDE3Ljg5MDg5NTg0MzUwNTg2LCAgICAtMy41NzcxNjEwNzM2ODQ2OTI0LFxuICAgICAgICAgICAgIDE2Ljk1Njg0ODE0NDUzMTI1LCAgICAtMS40OTY5NDcwNTAwOTQ2MDQ1LCAgMjAuMDIyNzk4NTM4MjA4MDA4LFxuICAgICAgICAgICAgIC0yLjcxNjczMjUwMTk4MzY0MjYsICAyMi4wODg3NTQ2NTM5MzA2NjQsICAgLTQuMjA3NTg0ODU3OTQwNjc0LFxuICAgICAgICAgICAgIDE5LjA2MTMyNjk4MDU5MDgyLCAgICAtMy45MjkyMTI1NzAxOTA0Mjk3LCAgMTguMTMzNDI0NzU4OTExMTMzLFxuICAgICAgICAgICAgIC0xLjg1MDg0MTA0NTM3OTYzODcsICAyMS4yMDU1MjA2Mjk4ODI4MTIsICAgLTMuMDcyNDcwNDI2NTU5NDQ4MixcbiAgICAgICAgICAgICAyMy4yNzc2MTQ1OTM1MDU4NiwgICAgLTQuNTU3NzkxNzA5ODk5OTAyLCAgIDIwLjIzMTc1ODExNzY3NTc4LFxuICAgICAgICAgICAgIC00LjI4MTI2MzM1MTQ0MDQzLCAgICAxOS4zMDk5OTk0NjU5NDIzODMsICAgLTIuMjA0NzM2NDcxMTc2MTQ3NSxcbiAgICAgICAgICAgICAyMi4zODgyNDA4MTQyMDg5ODQsICAgLTMuNDI4MjA4MzUxMTM1MjU0LCAgIDI0LjQ2NjQ3ODM0Nzc3ODMyLFxuICAgICAgICAgICAgIC0yLjU2NzAyMTEzMTUxNTUwMywgICAxMy41ODQzNTI0OTMyODYxMzMsICAgLTIuMjgzNTkwMzE2NzcyNDYxLFxuICAgICAgICAgICAgIDEyLjYzOTU4NDU0MTMyMDgsICAgICAtMC4yMDAxNjAwNDE0NTE0NTQxNiwgMTUuNjk0ODE1NjM1NjgxMTUyLFxuICAgICAgICAgICAgIC0xLjQxNjcyOTQ1MDIyNTgzLCAgICAxNy43NTAwNDU3NzYzNjcxODhcbiAgICAgICAgICAgXSkpO1xuICAgICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gZGltUm91bmRpbmdNb2RlIGlzIHNldCBhbmQgcGFkIGlzIHNhbWUnLCAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE2O1xuICAgIGNvbnN0IHhTaXplID0gODtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgIFsxLCB4U2l6ZSwgeFNpemUsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gODtcbiAgICBjb25zdCBmU2l6ZSA9IDM7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZTogW251bWJlciwgbnVtYmVyXSA9IFsyLCAyXTtcblxuICAgIGNvbnN0IGlucHV0cyA9IGdlbmVyYXRlQ2FzZUlucHV0cyhcbiAgICAgICAgMSAqIHhTaXplICogeFNpemUgKiBpbnB1dERlcHRoLFxuICAgICAgICBmU2l6ZSAqIGZTaXplICogaW5wdXREZXB0aCAqIG91dHB1dERlcHRoKTtcbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoaW5wdXRzLmlucHV0LCBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID1cbiAgICAgICAgdGYudGVuc29yNGQoaW5wdXRzLmZpbHRlciwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICBjb25zdCBiaWFzID0gdGYudGVuc29yMWQoWzEsIDQsIDIsIDMsIDksIDYsIDUsIDhdKTtcbiAgICBjb25zdCBsZWFreXJlbHVBbHBoYSA9IDAuMztcblxuICAgIGV4cGVjdChcbiAgICAgICAgKCkgPT4gdGYuZnVzZWQuY29udjJkKFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICB4LFxuICAgICAgICAgICAgICBmaWx0ZXI6IHcsXG4gICAgICAgICAgICAgIHN0cmlkZXM6IHN0cmlkZSxcbiAgICAgICAgICAgICAgcGFkLFxuICAgICAgICAgICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICAgICAgICAgIGRpbGF0aW9uczogWzEsIDFdLFxuICAgICAgICAgICAgICBhY3RpdmF0aW9uOiAnbGVha3lyZWx1JyxcbiAgICAgICAgICAgICAgbGVha3lyZWx1QWxwaGEsXG4gICAgICAgICAgICAgIGJpYXMsXG4gICAgICAgICAgICAgIGRpbVJvdW5kaW5nTW9kZTogJ3JvdW5kJ1xuICAgICAgICAgICAgfSkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyB3aGVuIGRpbVJvdW5kaW5nTW9kZSBpcyBzZXQgYW5kIHBhZCBpcyB2YWxpZCcsICgpID0+IHtcbiAgICBjb25zdCBpbnB1dERlcHRoID0gMTY7XG4gICAgY29uc3QgeFNpemUgPSA4O1xuICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgWzEsIHhTaXplLCB4U2l6ZSwgaW5wdXREZXB0aF07XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSA4O1xuICAgIGNvbnN0IGZTaXplID0gMztcbiAgICBjb25zdCBwYWQgPSAndmFsaWQnO1xuICAgIGNvbnN0IHN0cmlkZTogW251bWJlciwgbnVtYmVyXSA9IFsyLCAyXTtcblxuICAgIGNvbnN0IGlucHV0cyA9IGdlbmVyYXRlQ2FzZUlucHV0cyhcbiAgICAgICAgMSAqIHhTaXplICogeFNpemUgKiBpbnB1dERlcHRoLFxuICAgICAgICBmU2l6ZSAqIGZTaXplICogaW5wdXREZXB0aCAqIG91dHB1dERlcHRoKTtcbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoaW5wdXRzLmlucHV0LCBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID1cbiAgICAgICAgdGYudGVuc29yNGQoaW5wdXRzLmZpbHRlciwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICBjb25zdCBiaWFzID0gdGYudGVuc29yMWQoWzEsIDQsIDIsIDMsIDksIDYsIDUsIDhdKTtcbiAgICBjb25zdCBsZWFreXJlbHVBbHBoYSA9IDAuMztcblxuICAgIGV4cGVjdChcbiAgICAgICAgKCkgPT4gdGYuZnVzZWQuY29udjJkKFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICB4LFxuICAgICAgICAgICAgICBmaWx0ZXI6IHcsXG4gICAgICAgICAgICAgIHN0cmlkZXM6IHN0cmlkZSxcbiAgICAgICAgICAgICAgcGFkLFxuICAgICAgICAgICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICAgICAgICAgIGRpbGF0aW9uczogWzEsIDFdLFxuICAgICAgICAgICAgICBhY3RpdmF0aW9uOiAnbGVha3lyZWx1JyxcbiAgICAgICAgICAgICAgbGVha3lyZWx1QWxwaGEsXG4gICAgICAgICAgICAgIGJpYXMsXG4gICAgICAgICAgICAgIGRpbVJvdW5kaW5nTW9kZTogJ3JvdW5kJ1xuICAgICAgICAgICAgfSkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyB3aGVuIGRpbVJvdW5kaW5nTW9kZSBpcyBzZXQgYW5kIHBhZCBpcyBhIG5vbi1pbnRlZ2VyIG51bWJlcicsXG4gICAgICgpID0+IHtcbiAgICAgICBjb25zdCBpbnB1dERlcHRoID0gMTY7XG4gICAgICAgY29uc3QgeFNpemUgPSA4O1xuICAgICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICAgWzEsIHhTaXplLCB4U2l6ZSwgaW5wdXREZXB0aF07XG4gICAgICAgY29uc3Qgb3V0cHV0RGVwdGggPSA4O1xuICAgICAgIGNvbnN0IGZTaXplID0gMztcbiAgICAgICBjb25zdCBwYWQgPSAxLjI7XG4gICAgICAgY29uc3Qgc3RyaWRlOiBbbnVtYmVyLCBudW1iZXJdID0gWzIsIDJdO1xuXG4gICAgICAgY29uc3QgaW5wdXRzID0gZ2VuZXJhdGVDYXNlSW5wdXRzKFxuICAgICAgICAgICAxICogeFNpemUgKiB4U2l6ZSAqIGlucHV0RGVwdGgsXG4gICAgICAgICAgIGZTaXplICogZlNpemUgKiBpbnB1dERlcHRoICogb3V0cHV0RGVwdGgpO1xuICAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChpbnB1dHMuaW5wdXQsIGlucHV0U2hhcGUpO1xuICAgICAgIGNvbnN0IHcgPVxuICAgICAgICAgICB0Zi50ZW5zb3I0ZChpbnB1dHMuZmlsdGVyLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuICAgICAgIGNvbnN0IGJpYXMgPSB0Zi50ZW5zb3IxZChbMSwgNCwgMiwgMywgOSwgNiwgNSwgOF0pO1xuICAgICAgIGNvbnN0IGxlYWt5cmVsdUFscGhhID0gMC4zO1xuXG4gICAgICAgZXhwZWN0KFxuICAgICAgICAgICAoKSA9PiB0Zi5mdXNlZC5jb252MmQoXG4gICAgICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgIHgsXG4gICAgICAgICAgICAgICAgIGZpbHRlcjogdyxcbiAgICAgICAgICAgICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgICAgICAgICAgICBwYWQsXG4gICAgICAgICAgICAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgICAgICAgICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICAgICAgICAgICAgIGFjdGl2YXRpb246ICdsZWFreXJlbHUnLFxuICAgICAgICAgICAgICAgICBsZWFreXJlbHVBbHBoYSxcbiAgICAgICAgICAgICAgICAgYmlhcyxcbiAgICAgICAgICAgICAgICAgZGltUm91bmRpbmdNb2RlOiAncm91bmQnXG4gICAgICAgICAgICAgICB9KSlcbiAgICAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICAgICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gZGltUm91bmRpbmdNb2RlIGlzIHNldCBhbmQgcGFkIGlzIGV4cGxpY2l0IGJ5IG5vbi1pbnRlZ2VyICcgK1xuICAgICAgICAgJ251bWJlcicsXG4gICAgICgpID0+IHtcbiAgICAgICBjb25zdCBpbnB1dERlcHRoID0gMTY7XG4gICAgICAgY29uc3QgeFNpemUgPSA4O1xuICAgICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICAgWzEsIHhTaXplLCB4U2l6ZSwgaW5wdXREZXB0aF07XG4gICAgICAgY29uc3Qgb3V0cHV0RGVwdGggPSA4O1xuICAgICAgIGNvbnN0IGZTaXplID0gMztcbiAgICAgICBjb25zdCBwYWQgPSBbWzAsIDBdLCBbMCwgMi4xXSwgWzEsIDFdLCBbMCwgMF1dIGFzXG4gICAgICAgICAgIHRmLmJhY2tlbmRfdXRpbC5FeHBsaWNpdFBhZGRpbmc7XG4gICAgICAgY29uc3Qgc3RyaWRlOiBbbnVtYmVyLCBudW1iZXJdID0gWzIsIDJdO1xuXG4gICAgICAgY29uc3QgaW5wdXRzID0gZ2VuZXJhdGVDYXNlSW5wdXRzKFxuICAgICAgICAgICAxICogeFNpemUgKiB4U2l6ZSAqIGlucHV0RGVwdGgsXG4gICAgICAgICAgIGZTaXplICogZlNpemUgKiBpbnB1dERlcHRoICogb3V0cHV0RGVwdGgpO1xuICAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChpbnB1dHMuaW5wdXQsIGlucHV0U2hhcGUpO1xuICAgICAgIGNvbnN0IHcgPVxuICAgICAgICAgICB0Zi50ZW5zb3I0ZChpbnB1dHMuZmlsdGVyLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuICAgICAgIGNvbnN0IGJpYXMgPSB0Zi50ZW5zb3IxZChbMSwgNCwgMiwgMywgOSwgNiwgNSwgOF0pO1xuICAgICAgIGNvbnN0IGxlYWt5cmVsdUFscGhhID0gMC4zO1xuXG4gICAgICAgZXhwZWN0KFxuICAgICAgICAgICAoKSA9PiB0Zi5mdXNlZC5jb252MmQoXG4gICAgICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgIHgsXG4gICAgICAgICAgICAgICAgIGZpbHRlcjogdyxcbiAgICAgICAgICAgICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgICAgICAgICAgICBwYWQsXG4gICAgICAgICAgICAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgICAgICAgICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICAgICAgICAgICAgIGFjdGl2YXRpb246ICdsZWFreXJlbHUnLFxuICAgICAgICAgICAgICAgICBsZWFreXJlbHVBbHBoYSxcbiAgICAgICAgICAgICAgICAgYmlhcyxcbiAgICAgICAgICAgICAgICAgZGltUm91bmRpbmdNb2RlOiAncm91bmQnXG4gICAgICAgICAgICAgICB9KSlcbiAgICAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICAgICB9KTtcblxuICBpdCgnYmFzaWMgd2l0aCBiaWFzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzIsIDIsIDIsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gMDtcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMTAsIDExLCAxMiwgMTMsIDE0LCAxNSwgMTZdLCBpblNoYXBlKTtcbiAgICBjb25zdCB3ID1cbiAgICAgICAgdGYudGVuc29yNGQoWy0xLCAxLCAtMiwgMC41XSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICB4LFxuICAgICAgZmlsdGVyOiB3LFxuICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgcGFkLFxuICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICBiaWFzOiB0Zi50ZW5zb3IxZChbNSwgNl0pXG4gICAgfSk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMiwgMiwgMiwgMl0pO1xuICAgIGNvbnN0IGV4cGVjdGVkID1cbiAgICAgICAgWzAsIDgsIC02LCAxMSwgLTEyLCAxNCwgLTE4LCAxNywgLTI0LCAyMCwgLTMwLCAyMywgLTM2LCAyNiwgLTQyLCAyOV07XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdiYXNpYyB3aXRoIGV4cGxpY2l0IHBhZGRpbmcnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IHBhZCA9XG4gICAgICAgIFtbMCwgMF0sIFsxLCAyXSwgWzAsIDFdLCBbMCwgMF1dIGFzIHRmLmJhY2tlbmRfdXRpbC5FeHBsaWNpdFBhZGRpbmc7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICBjb25zdCBkYXRhRm9ybWF0ID0gJ05IV0MnO1xuICAgIGNvbnN0IGRpbGF0aW9uID0gMTtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3IzZChbMSwgMiwgMywgNCwgNSwgNiwgNywgOF0sIFs0LCAyLCBpbnB1dERlcHRoXSk7XG4gICAgY29uc3QgdyA9XG4gICAgICAgIHRmLnRlbnNvcjRkKFszLCAxLCA1LCAwLCAyLCA3LCA4LCA5XSwgWzQsIDIsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXSk7XG5cbiAgICBjb25zdCByZXN1bHQgPSB0Zi5mdXNlZC5jb252MmQoXG4gICAgICAgIHt4LCBmaWx0ZXI6IHcsIHN0cmlkZXM6IHN0cmlkZSwgcGFkLCBkYXRhRm9ybWF0LCBkaWxhdGlvbnM6IGRpbGF0aW9ufSk7XG5cbiAgICBjb25zdCByZXN1bHREYXRhID0gYXdhaXQgcmVzdWx0LmRhdGEoKTtcbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKFs0LCAyLCAxXSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UocmVzdWx0RGF0YSwgWzEzMywgNjYsIDIwMCwgMTAyLCAxMDgsIDU4LCA1NiwgNThdKTtcbiAgfSk7XG5cbiAgaXQoJ2Jhc2ljIHdpdGggZWx1JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzIsIDIsIDIsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gMDtcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMTAsIDExLCAxMiwgMTMsIDE0LCAxNSwgMTZdLCBpblNoYXBlKTtcbiAgICBjb25zdCB3ID1cbiAgICAgICAgdGYudGVuc29yNGQoWy0xLCAxLCAtMiwgMC41XSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICB4LFxuICAgICAgZmlsdGVyOiB3LFxuICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgcGFkLFxuICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICBhY3RpdmF0aW9uOiAnZWx1J1xuICAgIH0pO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzIsIDIsIDIsIDJdKTtcbiAgICBjb25zdCBleHBlY3RlZCA9XG4gICAgICAgIFstMC45OTMyNiwgMiwgLTEsIDUsIC0xLCA4LCAtMSwgMTEsIC0xLCAxNCwgLTEsIDE3LCAtMSwgMjAsIC0xLCAyM107XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdiYXNpYyB3aXRoIHByZWx1JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzIsIDIsIDIsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gMDtcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMTAsIDExLCAxMiwgMTMsIDE0LCAxNSwgMTZdLCBpblNoYXBlKTtcbiAgICBjb25zdCBhbHBoYSA9IHRmLnRlbnNvcjNkKFswLjI1LCAwLjc1XSwgWzEsIDEsIDJdKTtcbiAgICBjb25zdCB3ID1cbiAgICAgICAgdGYudGVuc29yNGQoWy0xLCAxLCAtMiwgMC41XSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICB4LFxuICAgICAgZmlsdGVyOiB3LFxuICAgICAgc3RyaWRlczogc3RyaWRlLFxuICAgICAgcGFkLFxuICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICBhY3RpdmF0aW9uOiAncHJlbHUnLFxuICAgICAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0czogYWxwaGFcbiAgICB9KTtcbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKFsyLCAyLCAyLCAyXSk7XG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbXG4gICAgICAtMS4yNSwgMiwgLTIuNzUsIDUsIC00LjI1LCA4LCAtNS43NSwgMTEsIC03LjI1LCAxNCwgLTguNzUsIDE3LCAtMTAuMjUsIDIwLFxuICAgICAgLTExLjc1LCAyM1xuICAgIF07XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdiYXNpYyB3aXRoIGxlYWt5cmVsdScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBpbnB1dERlcHRoID0gMjtcbiAgICBjb25zdCBpblNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsyLCAyLCAyLCBpbnB1dERlcHRoXTtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IDI7XG4gICAgY29uc3QgZlNpemUgPSAxO1xuICAgIGNvbnN0IHBhZCA9IDA7XG4gICAgY29uc3Qgc3RyaWRlID0gMTtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxMSwgMTIsIDEzLCAxNCwgMTUsIDE2XSwgaW5TaGFwZSk7XG4gICAgY29uc3QgYWxwaGEgPSAwLjM7XG4gICAgY29uc3QgdyA9XG4gICAgICAgIHRmLnRlbnNvcjRkKFstMSwgMSwgLTIsIDAuNV0sIFtmU2l6ZSwgZlNpemUsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXSk7XG5cbiAgICBjb25zdCByZXN1bHQgPSB0Zi5mdXNlZC5jb252MmQoe1xuICAgICAgeCxcbiAgICAgIGZpbHRlcjogdyxcbiAgICAgIHN0cmlkZXM6IHN0cmlkZSxcbiAgICAgIHBhZCxcbiAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgIGRpbGF0aW9uczogWzEsIDFdLFxuICAgICAgYWN0aXZhdGlvbjogJ2xlYWt5cmVsdScsXG4gICAgICBsZWFreXJlbHVBbHBoYTogYWxwaGFcbiAgICB9KTtcbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKFsyLCAyLCAyLCAyXSk7XG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbXG4gICAgICAtMS41LCAyLCAtMy4zMDAwMDAxOTA3MzQ4NjMzLCA1LCAtNS4xMDAwMDAzODE0Njk3MjcsIDgsXG4gICAgICAtNi45MDAwMDAwOTUzNjc0MzIsIDExLCAtOC43MDAwMDA3NjI5Mzk0NTMsIDE0LCAtMTAuNSwgMTcsXG4gICAgICAtMTIuMzAwMDAwMTkwNzM0ODYzLCAyMCwgLTE0LjEwMDAwMDM4MTQ2OTcyNywgMjNcbiAgICBdO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgZXhwZWN0ZWQpO1xuICB9KTtcblxuICBpdCgnYmFzaWMgd2l0aCBzaWdtb2lkJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzIsIDIsIDIsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gMDtcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMTAsIDExLCAxMiwgMTMsIDE0LCAxNSwgMTZdLCBpblNoYXBlKTtcbiAgICBjb25zdCBhbHBoYSA9IDAuMztcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFstMC4xLCAwLjEsIC0wLjIsIDAuMDVdLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZnVzZWQuY29udjJkKHtcbiAgICAgIHgsXG4gICAgICBmaWx0ZXI6IHcsXG4gICAgICBzdHJpZGVzOiBzdHJpZGUsXG4gICAgICBwYWQsXG4gICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICBkaWxhdGlvbnM6IFsxLCAxXSxcbiAgICAgIGFjdGl2YXRpb246ICdzaWdtb2lkJyxcbiAgICAgIGxlYWt5cmVsdUFscGhhOiBhbHBoYVxuICAgIH0pO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzIsIDIsIDIsIDJdKTtcbiAgICBjb25zdCBleHBlY3RlZCA9IFtcbiAgICAgIDAuMzc3NTQwNywgMC41NDk4MzQsIDAuMjQ5NzM5ODksIDAuNjIyNDU5MywgMC4xNTQ0NjUyNiwgMC42ODk5NzQ0LFxuICAgICAgMC4wOTExMjI5NiwgMC43NTAyNjAxLCAwLjA1MjE1MzUsIDAuODAyMTgzODcsIDAuMDI5MzEyMTksIDAuODQ1NTM0NzQsXG4gICAgICAwLjAxNjMwMjUsIDAuODgwNzk3MSwgMC4wMDkwMTMzLCAwLjkwODg3N1xuICAgIF07XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdiYXNpYyB3aXRoIGJyb2FkY2FzdGVkIGJpYXMgYW5kIHJlbHUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDI7XG4gICAgY29uc3QgaW5TaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbMiwgMiwgMiwgaW5wdXREZXB0aF07XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGZTaXplID0gMTtcbiAgICBjb25zdCBwYWQgPSAwO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxMCwgMTEsIDEyLCAxMywgMTQsIDE1LCAxNl0sIGluU2hhcGUpO1xuICAgIGNvbnN0IHcgPVxuICAgICAgICB0Zi50ZW5zb3I0ZChbLTEsIDEsIC0yLCAwLjVdLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZnVzZWQuY29udjJkKHtcbiAgICAgIHgsXG4gICAgICBmaWx0ZXI6IHcsXG4gICAgICBzdHJpZGVzOiBzdHJpZGUsXG4gICAgICBwYWQsXG4gICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICBkaWxhdGlvbnM6IFsxLCAxXSxcbiAgICAgIGJpYXM6IHRmLnNjYWxhcig1KSxcbiAgICAgIGFjdGl2YXRpb246ICdyZWx1J1xuICAgIH0pO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzIsIDIsIDIsIDJdKTtcbiAgICBjb25zdCBleHBlY3RlZCA9IFswLCA3LCAwLCAxMCwgMCwgMTMsIDAsIDE2LCAwLCAxOSwgMCwgMjIsIDAsIDI1LCAwLCAyOF07XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdpbTJyb3cnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzQsIDQsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMztcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0gPSBbMiwgMl07XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yM2QoXG4gICAgICAgIFtcbiAgICAgICAgICAxMCwgMzAsIDUwLCA3MCwgMjAsIDQwLCA2MCwgODAsIC0xMCwgLTMwLCAtNTAsIC03MCwgLTIwLCAtNDAsIC02MCwgLTgwXG4gICAgICAgIF0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChbMSwgMC41LCAxXSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7eCwgZmlsdGVyOiB3LCBzdHJpZGVzLCBwYWR9KTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgICAgICBhd2FpdCByZXN1bHQuZGF0YSgpLFxuICAgICAgICBbMTAsIDUsIDEwLCA1MCwgMjUsIDUwLCAtMTAsIC01LCAtMTAsIC01MCwgLTI1LCAtNTBdKTtcbiAgfSk7XG5cbiAgaXQoJ2ltMnJvdyB3aXRoIHJlbHUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzQsIDQsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMztcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0gPSBbMiwgMl07XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yM2QoXG4gICAgICAgIFtcbiAgICAgICAgICAxMCwgMzAsIDUwLCA3MCwgMjAsIDQwLCA2MCwgODAsIC0xMCwgLTMwLCAtNTAsIC03MCwgLTIwLCAtNDAsIC02MCwgLTgwXG4gICAgICAgIF0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChbMSwgMC41LCAxXSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICB4LFxuICAgICAgZmlsdGVyOiB3LFxuICAgICAgc3RyaWRlcyxcbiAgICAgIHBhZCxcbiAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgIGRpbGF0aW9uczogWzEsIDFdLFxuICAgICAgYWN0aXZhdGlvbjogJ3JlbHUnXG4gICAgfSk7XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShcbiAgICAgICAgYXdhaXQgcmVzdWx0LmRhdGEoKSwgWzEwLCA1LCAxMCwgNTAsIDI1LCA1MCwgMCwgMCwgMCwgMCwgMCwgMF0pO1xuICB9KTtcblxuICBpdCgnaW0ycm93IHdpdGggcHJlbHUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzQsIDQsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMztcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0gPSBbMiwgMl07XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yM2QoXG4gICAgICAgIFtcbiAgICAgICAgICAxMCwgMzAsIDUwLCA3MCwgMjAsIDQwLCA2MCwgODAsIC0xMCwgLTMwLCAtNTAsIC03MCwgLTIwLCAtNDAsIC02MCwgLTgwXG4gICAgICAgIF0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChbMSwgMC41LCAxXSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICBjb25zdCBhbHBoYSA9IHRmLnRlbnNvcjNkKFswLjVdLCBbMSwgMSwgaW5wdXREZXB0aF0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZnVzZWQuY29udjJkKHtcbiAgICAgIHgsXG4gICAgICBmaWx0ZXI6IHcsXG4gICAgICBzdHJpZGVzLFxuICAgICAgcGFkLFxuICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICBhY3RpdmF0aW9uOiAncHJlbHUnLFxuICAgICAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0czogYWxwaGFcbiAgICB9KTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgICAgICBhd2FpdCByZXN1bHQuZGF0YSgpLFxuICAgICAgICBbMTAsIDUsIDEwLCA1MCwgMjUsIDUwLCAtNSwgLTIuNSwgLTUsIC0yNSwgLTEyLjUsIC0yNV0pO1xuICB9KTtcblxuICBpdCgnaW0ycm93IHdpdGggbGVha3lyZWx1JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs0LCA0LCBpbnB1dERlcHRoXTtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IDM7XG4gICAgY29uc3QgZlNpemUgPSAxO1xuICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICBjb25zdCBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdID0gWzIsIDJdO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjNkKFxuICAgICAgICBbXG4gICAgICAgICAgMTAsIDMwLCA1MCwgNzAsIDIwLCA0MCwgNjAsIDgwLCAtMTAsIC0zMCwgLTUwLCAtNzAsIC0yMCwgLTQwLCAtNjAsIC04MFxuICAgICAgICBdLFxuICAgICAgICBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoWzEsIDAuNSwgMV0sIFtmU2l6ZSwgZlNpemUsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXSk7XG4gICAgY29uc3QgYWxwaGEgPSAwLjM7XG5cbiAgICBjb25zdCByZXN1bHQgPSB0Zi5mdXNlZC5jb252MmQoe1xuICAgICAgeCxcbiAgICAgIGZpbHRlcjogdyxcbiAgICAgIHN0cmlkZXMsXG4gICAgICBwYWQsXG4gICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICBkaWxhdGlvbnM6IFsxLCAxXSxcbiAgICAgIGFjdGl2YXRpb246ICdsZWFreXJlbHUnLFxuICAgICAgbGVha3lyZWx1QWxwaGE6IGFscGhhXG4gICAgfSk7XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBbXG4gICAgICAxMCwgNSwgMTAsIDUwLCAyNSwgNTAsIC0zLCAtMS41LCAtMywgLTE1LjAwMDAwMDk1MzY3NDMxNixcbiAgICAgIC03LjUwMDAwMDQ3NjgzNzE1OCwgLTE1LjAwMDAwMDk1MzY3NDMxNlxuICAgIF0pO1xuICB9KTtcblxuICBpdCgncG9pbnR3aXNlIHdpdGggcHJlbHUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzQsIDQsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMztcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0gPSBbMSwgMV07XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yM2QoXG4gICAgICAgIFtcbiAgICAgICAgICAxMCwgMzAsIDUwLCA3MCwgMjAsIDQwLCA2MCwgODAsIC0xMCwgLTMwLCAtNTAsIC03MCwgLTIwLCAtNDAsIC02MCwgLTgwXG4gICAgICAgIF0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChbMSwgMC41LCAxXSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICBjb25zdCBhbHBoYSA9IHRmLnRlbnNvcjNkKFswLjVdLCBbMSwgMSwgaW5wdXREZXB0aF0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuZnVzZWQuY29udjJkKHtcbiAgICAgIHgsXG4gICAgICBmaWx0ZXI6IHcsXG4gICAgICBzdHJpZGVzLFxuICAgICAgcGFkLFxuICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICBhY3RpdmF0aW9uOiAncHJlbHUnLFxuICAgICAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0czogYWxwaGFcbiAgICB9KTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFtcbiAgICAgIDEwLCAgNSwgICAgMTAsICAzMCwgIDE1LCAgIDMwLCAgNTAsICAyNSwgICAgNTAsICA3MCwgIDM1LCAgICA3MCxcbiAgICAgIDIwLCAgMTAsICAgMjAsICA0MCwgIDIwLCAgIDQwLCAgNjAsICAzMCwgICAgNjAsICA4MCwgIDQwLCAgICA4MCxcbiAgICAgIC01LCAgLTIuNSwgLTUsICAtMTUsIC03LjUsIC0xNSwgLTI1LCAtMTIuNSwgLTI1LCAtMzUsIC0xNy41LCAtMzUsXG4gICAgICAtMTAsIC01LCAgIC0xMCwgLTIwLCAtMTAsICAtMjAsIC0zMCwgLTE1LCAgIC0zMCwgLTQwLCAtMjAsICAgLTQwXG4gICAgXSk7XG4gIH0pO1xuXG4gIGl0KCdwb2ludHdpc2Ugd2l0aCBsZWFreXJlbHUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzQsIDQsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMztcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl0gPSBbMSwgMV07XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yM2QoXG4gICAgICAgIFtcbiAgICAgICAgICAxMCwgMzAsIDUwLCA3MCwgMjAsIDQwLCA2MCwgODAsIC0xMCwgLTMwLCAtNTAsIC03MCwgLTIwLCAtNDAsIC02MCwgLTgwXG4gICAgICAgIF0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChbMSwgMC41LCAxXSwgW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdKTtcbiAgICBjb25zdCBhbHBoYSA9IDAuMztcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICB4LFxuICAgICAgZmlsdGVyOiB3LFxuICAgICAgc3RyaWRlcyxcbiAgICAgIHBhZCxcbiAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgIGRpbGF0aW9uczogWzEsIDFdLFxuICAgICAgYWN0aXZhdGlvbjogJ2xlYWt5cmVsdScsXG4gICAgICBsZWFreXJlbHVBbHBoYTogYWxwaGFcbiAgICB9KTtcblxuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFtcbiAgICAgIDEwLFxuICAgICAgNSxcbiAgICAgIDEwLFxuICAgICAgMzAsXG4gICAgICAxNSxcbiAgICAgIDMwLFxuICAgICAgNTAsXG4gICAgICAyNSxcbiAgICAgIDUwLFxuICAgICAgNzAsXG4gICAgICAzNSxcbiAgICAgIDcwLFxuICAgICAgMjAsXG4gICAgICAxMCxcbiAgICAgIDIwLFxuICAgICAgNDAsXG4gICAgICAyMCxcbiAgICAgIDQwLFxuICAgICAgNjAsXG4gICAgICAzMCxcbiAgICAgIDYwLFxuICAgICAgODAsXG4gICAgICA0MCxcbiAgICAgIDgwLFxuICAgICAgLTMsXG4gICAgICAtMS41LFxuICAgICAgLTMsXG4gICAgICAtOSxcbiAgICAgIC00LjUsXG4gICAgICAtOSxcbiAgICAgIC0xNS4wMDAwMDA5NTM2NzQzMTYsXG4gICAgICAtNy41MDAwMDA0NzY4MzcxNTgsXG4gICAgICAtMTUuMDAwMDAwOTUzNjc0MzE2LFxuICAgICAgLTIxLFxuICAgICAgLTEwLjUsXG4gICAgICAtMjEsXG4gICAgICAtNixcbiAgICAgIC0zLFxuICAgICAgLTYsXG4gICAgICAtMTIsXG4gICAgICAtNixcbiAgICAgIC0xMixcbiAgICAgIC0xOCxcbiAgICAgIC05LFxuICAgICAgLTE4LFxuICAgICAgLTI0LFxuICAgICAgLTEyLFxuICAgICAgLTI0XG4gICAgXSk7XG4gIH0pO1xuXG4gIGl0KCdpbTJyb3cgd2l0aCBicm9hZGNhc3RlZCBiaWFzIGFuZCByZWx1JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs0LCA0LCBpbnB1dERlcHRoXTtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IDM7XG4gICAgY29uc3QgZlNpemUgPSAxO1xuICAgIGNvbnN0IHBhZCA9ICdzYW1lJztcbiAgICBjb25zdCBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdID0gWzIsIDJdO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjNkKFxuICAgICAgICBbXG4gICAgICAgICAgMTAsIDMwLCA1MCwgNzAsIDIwLCA0MCwgNjAsIDgwLCAtMTAsIC0zMCwgLTUwLCAtNzAsIC0yMCwgLTQwLCAtNjAsIC04MFxuICAgICAgICBdLFxuICAgICAgICBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoWzEsIDAuNSwgMV0sIFtmU2l6ZSwgZlNpemUsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXSk7XG5cbiAgICBjb25zdCByZXN1bHQgPSB0Zi5mdXNlZC5jb252MmQoe1xuICAgICAgeCxcbiAgICAgIGZpbHRlcjogdyxcbiAgICAgIHN0cmlkZXMsXG4gICAgICBwYWQsXG4gICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICBkaWxhdGlvbnM6IFsxLCAxXSxcbiAgICAgIGJpYXM6IHRmLnNjYWxhcig1KSxcbiAgICAgIGFjdGl2YXRpb246ICdyZWx1J1xuICAgIH0pO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoXG4gICAgICAgIGF3YWl0IHJlc3VsdC5kYXRhKCksIFsxNSwgMTAsIDE1LCA1NSwgMzAsIDU1LCAwLCAwLCAwLCAwLCAwLCAwXSk7XG4gIH0pO1xuXG4gIGl0KCdiYWNrUHJvcCBpbnB1dCB4PVsyLDMsMywxXSBmPVsyLDIsMSwxXSBzPTEgcD0wJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMTtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsyLCAzLCAzLCBpbnB1dERlcHRoXTtcbiAgICBjb25zdCBmaWx0ZXJTaXplID0gMjtcbiAgICBjb25zdCBzdHJpZGVzID0gMTtcbiAgICBjb25zdCBwYWQgPSAwO1xuXG4gICAgY29uc3QgZmlsdGVyU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgW2ZpbHRlclNpemUsIGZpbHRlclNpemUsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXTtcbiAgICBjb25zdCBmaWx0ZXIgPSB0Zi50ZW5zb3I0ZChbLTEsIDEsIC0yLCAwLjVdLCBmaWx0ZXJTaGFwZSk7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5XSwgaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgZHkgPSB0Zi50ZW5zb3I0ZChbMywgMSwgMiwgMCwgMywgMSwgMiwgMF0sIFsyLCAyLCAyLCAxXSk7XG5cbiAgICBjb25zdCBncmFkcyA9IHRmLmdyYWRzKFxuICAgICAgICAoeDogdGYuVGVuc29yNEQpID0+IHRmLmZ1c2VkLmNvbnYyZCh7eCwgZmlsdGVyLCBzdHJpZGVzLCBwYWR9KSk7XG4gICAgY29uc3QgW2R4XSA9IGdyYWRzKFt4XSwgZHkpO1xuXG4gICAgZXhwZWN0KGR4LnNoYXBlKS50b0VxdWFsKHguc2hhcGUpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgICAgICBhd2FpdCBkeC5kYXRhKCksXG4gICAgICAgIFstMywgMiwgMSwgLTgsIDEuNSwgMC41LCAtNCwgMSwgMCwgLTMsIDIsIDEsIC04LCAxLjUsIDAuNSwgLTQsIDEsIDBdKTtcbiAgfSk7XG5cbiAgaXQoJ2dyYWRpZW50IHg9WzIsMywzLDFdIGY9WzIsMiwxLDFdIHM9MSBwPTAnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzIsIDMsIDMsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IGZpbHRlclNpemUgPSAyO1xuICAgIGNvbnN0IHN0cmlkZXMgPSAxO1xuICAgIGNvbnN0IHBhZCA9IDA7XG5cbiAgICBjb25zdCBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICBbZmlsdGVyU2l6ZSwgZmlsdGVyU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdO1xuICAgIGNvbnN0IGZpbHRlciA9IHRmLnRlbnNvcjRkKFstMSwgMSwgLTIsIDAuNV0sIGZpbHRlclNoYXBlKTtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDldLCBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBkeSA9IHRmLnRlbnNvcjRkKFszLCAxLCAyLCAwLCAzLCAxLCAyLCAwXSwgWzIsIDIsIDIsIDFdKTtcblxuICAgIGNvbnN0IGdyYWRzID0gdGYuZ3JhZHMoXG4gICAgICAgICh4OiB0Zi5UZW5zb3I0RCwgZmlsdGVyOiB0Zi5UZW5zb3I0RCkgPT5cbiAgICAgICAgICAgIHRmLmZ1c2VkLmNvbnYyZCh7eCwgZmlsdGVyLCBzdHJpZGVzLCBwYWR9KSk7XG4gICAgY29uc3QgW2R4LCBkZmlsdGVyXSA9IGdyYWRzKFt4LCBmaWx0ZXJdLCBkeSk7XG5cbiAgICBleHBlY3QoZHguc2hhcGUpLnRvRXF1YWwoeC5zaGFwZSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoXG4gICAgICAgIGF3YWl0IGR4LmRhdGEoKSxcbiAgICAgICAgWy0zLCAyLCAxLCAtOCwgMS41LCAwLjUsIC00LCAxLCAwLCAtMywgMiwgMSwgLTgsIDEuNSwgMC41LCAtNCwgMSwgMF0pO1xuXG4gICAgZXhwZWN0KGRmaWx0ZXIuc2hhcGUpLnRvRXF1YWwoZmlsdGVyU2hhcGUpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGRmaWx0ZXIuZGF0YSgpLCBbMjYsIDM4LCA2MiwgNzRdKTtcbiAgfSk7XG5cbiAgaXQoJ2dyYWRpZW50IHg9WzIsMywzLDFdIGY9WzIsMiwxLDFdIHM9MSBwPTAgd2l0aCBiaWFzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMTtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsyLCAzLCAzLCBpbnB1dERlcHRoXTtcbiAgICBjb25zdCBmaWx0ZXJTaXplID0gMjtcbiAgICBjb25zdCBzdHJpZGVzID0gMTtcbiAgICBjb25zdCBwYWQgPSAwO1xuXG4gICAgY29uc3QgZmlsdGVyU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgW2ZpbHRlclNpemUsIGZpbHRlclNpemUsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXTtcbiAgICBjb25zdCBmaWx0ZXIgPSB0Zi50ZW5zb3I0ZChbLTEsIDEsIC0yLCAwLjVdLCBmaWx0ZXJTaGFwZSk7XG4gICAgY29uc3QgYmlhcyA9IHRmLm9uZXMoWzIsIDIsIDIsIDFdKTtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDldLCBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBkeSA9IHRmLnRlbnNvcjRkKFszLCAxLCAyLCAwLCAzLCAxLCAyLCAwXSwgWzIsIDIsIDIsIDFdKTtcblxuICAgIGNvbnN0IGZ1c2VkR3JhZHMgPVxuICAgICAgICB0Zi5ncmFkcygoeDogdGYuVGVuc29yNEQsIHc6IHRmLlRlbnNvcjRELCBiKSA9PiB0Zi5mdXNlZC5jb252MmQoe1xuICAgICAgICAgIHgsXG4gICAgICAgICAgZmlsdGVyOiB3LFxuICAgICAgICAgIHN0cmlkZXMsXG4gICAgICAgICAgcGFkLFxuICAgICAgICAgIGRhdGFGb3JtYXQ6ICdOSFdDJyxcbiAgICAgICAgICBkaWxhdGlvbnM6IFsxLCAxXSxcbiAgICAgICAgICBiaWFzOiBiXG4gICAgICAgIH0pKTtcbiAgICBjb25zdCBbZHhGdXNlZCwgZGZpbHRlckZ1c2VkLCBkYmlhc0Z1c2VkXSA9XG4gICAgICAgIGZ1c2VkR3JhZHMoW3gsIGZpbHRlciwgYmlhc10sIGR5KTtcblxuICAgIGNvbnN0IGdyYWRzID0gdGYuZ3JhZHMoKHg6IHRmLlRlbnNvcjRELCBmaWx0ZXI6IHRmLlRlbnNvcjRELCBiaWFzKSA9PiB7XG4gICAgICBjb25zdCBjb252ID0gdGYuY29udjJkKHgsIGZpbHRlciwgc3RyaWRlcywgcGFkKTtcbiAgICAgIGNvbnN0IHN1bSA9IHRmLmFkZChjb252LCBiaWFzKTtcbiAgICAgIHJldHVybiBzdW07XG4gICAgfSk7XG4gICAgY29uc3QgW2R4LCBkZmlsdGVyLCBkYmlhc10gPSBncmFkcyhbeCwgZmlsdGVyLCBiaWFzXSwgZHkpO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgZHhGdXNlZC5hcnJheSgpLCBhd2FpdCBkeC5hcnJheSgpKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBkZmlsdGVyRnVzZWQuYXJyYXkoKSwgYXdhaXQgZGZpbHRlci5hcnJheSgpKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBkYmlhc0Z1c2VkLmFycmF5KCksIGF3YWl0IGRiaWFzLmFycmF5KCkpO1xuICB9KTtcblxuICBpdCgnZ3JhZGllbnQgeD1bMiwzLDMsMV0gZj1bMiwyLDEsMV0gcz0xIHA9MCB3aXRoIGJpYXMgYW5kIHJlbHUnLFxuICAgICBhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgICAgY29uc3Qgb3V0cHV0RGVwdGggPSAxO1xuICAgICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICAgWzIsIDMsIDMsIGlucHV0RGVwdGhdO1xuICAgICAgIGNvbnN0IGZpbHRlclNpemUgPSAyO1xuICAgICAgIGNvbnN0IHN0cmlkZXMgPSAxO1xuICAgICAgIGNvbnN0IHBhZCA9IDA7XG5cbiAgICAgICBjb25zdCBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICAgICBbZmlsdGVyU2l6ZSwgZmlsdGVyU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdO1xuICAgICAgIGNvbnN0IGZpbHRlciA9IHRmLnRlbnNvcjRkKFstMSwgMSwgLTIsIDAuNV0sIGZpbHRlclNoYXBlKTtcbiAgICAgICBjb25zdCBiaWFzID0gdGYub25lcyhbMiwgMiwgMiwgMV0pO1xuXG4gICAgICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICAgICBbMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOV0sIGlucHV0U2hhcGUpO1xuICAgICAgIGNvbnN0IGR5ID0gdGYudGVuc29yNGQoWzMsIDEsIDIsIDAsIDMsIDEsIDIsIDBdLCBbMiwgMiwgMiwgMV0pO1xuXG4gICAgICAgY29uc3QgZnVzZWRHcmFkcyA9XG4gICAgICAgICAgIHRmLmdyYWRzKCh4OiB0Zi5UZW5zb3I0RCwgdzogdGYuVGVuc29yNEQsIGIpID0+IHRmLmZ1c2VkLmNvbnYyZCh7XG4gICAgICAgICAgICAgeCxcbiAgICAgICAgICAgICBmaWx0ZXI6IHcsXG4gICAgICAgICAgICAgc3RyaWRlcyxcbiAgICAgICAgICAgICBwYWQsXG4gICAgICAgICAgICAgZGF0YUZvcm1hdDogJ05IV0MnLFxuICAgICAgICAgICAgIGRpbGF0aW9uczogWzEsIDFdLFxuICAgICAgICAgICAgIGJpYXM6IGIsXG4gICAgICAgICAgICAgYWN0aXZhdGlvbjogJ3JlbHUnXG4gICAgICAgICAgIH0pKTtcbiAgICAgICBjb25zdCBbZHhGdXNlZCwgZGZpbHRlckZ1c2VkLCBkYmlhc0Z1c2VkXSA9XG4gICAgICAgICAgIGZ1c2VkR3JhZHMoW3gsIGZpbHRlciwgYmlhc10sIGR5KTtcblxuICAgICAgIGNvbnN0IGdyYWRzID0gdGYuZ3JhZHMoKHg6IHRmLlRlbnNvcjRELCBmaWx0ZXI6IHRmLlRlbnNvcjRELCBiaWFzKSA9PiB7XG4gICAgICAgICBjb25zdCBjb252ID0gdGYuY29udjJkKHgsIGZpbHRlciwgc3RyaWRlcywgcGFkKTtcbiAgICAgICAgIGNvbnN0IHN1bSA9IHRmLmFkZChjb252LCBiaWFzKTtcbiAgICAgICAgIHJldHVybiB0Zi5yZWx1KHN1bSk7XG4gICAgICAgfSk7XG4gICAgICAgY29uc3QgW2R4LCBkZmlsdGVyLCBkYmlhc10gPSBncmFkcyhbeCwgZmlsdGVyLCBiaWFzXSwgZHkpO1xuXG4gICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgZHhGdXNlZC5hcnJheSgpLCBhd2FpdCBkeC5hcnJheSgpKTtcbiAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBkZmlsdGVyRnVzZWQuYXJyYXkoKSwgYXdhaXQgZGZpbHRlci5hcnJheSgpKTtcbiAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBkYmlhc0Z1c2VkLmFycmF5KCksIGF3YWl0IGRiaWFzLmFycmF5KCkpO1xuICAgICB9KTtcblxuICBpdCgnZ3JhZGllbnQgeD1bMiwzLDMsMV0gZj1bMiwyLDEsMV0gcz0xIHA9MCB3aXRoIGJpYXMgYW5kIGVsdScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBpbnB1dERlcHRoID0gMTtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbMiwgMywgMywgaW5wdXREZXB0aF07XG4gICAgY29uc3QgZmlsdGVyU2l6ZSA9IDI7XG4gICAgY29uc3Qgc3RyaWRlcyA9IDE7XG4gICAgY29uc3QgcGFkID0gMDtcblxuICAgIGNvbnN0IGZpbHRlclNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgIFtmaWx0ZXJTaXplLCBmaWx0ZXJTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF07XG4gICAgY29uc3QgZmlsdGVyID0gdGYudGVuc29yNGQoWy0xLCAxLCAtMiwgMC41XSwgZmlsdGVyU2hhcGUpO1xuICAgIGNvbnN0IGJpYXMgPSB0Zi5vbmVzKFsyLCAyLCAyLCAxXSk7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5XSwgaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgZHkgPSB0Zi50ZW5zb3I0ZChbMywgMSwgMiwgMCwgMywgMSwgMiwgMF0sIFsyLCAyLCAyLCAxXSk7XG5cbiAgICBjb25zdCBmdXNlZEdyYWRzID1cbiAgICAgICAgdGYuZ3JhZHMoKHg6IHRmLlRlbnNvcjRELCB3OiB0Zi5UZW5zb3I0RCwgYikgPT4gdGYuZnVzZWQuY29udjJkKHtcbiAgICAgICAgICB4LFxuICAgICAgICAgIGZpbHRlcjogdyxcbiAgICAgICAgICBzdHJpZGVzLFxuICAgICAgICAgIHBhZCxcbiAgICAgICAgICBkYXRhRm9ybWF0OiAnTkhXQycsXG4gICAgICAgICAgZGlsYXRpb25zOiBbMSwgMV0sXG4gICAgICAgICAgYmlhczogYixcbiAgICAgICAgICBhY3RpdmF0aW9uOiAnZWx1J1xuICAgICAgICB9KSk7XG4gICAgY29uc3QgW2R4RnVzZWQsIGRmaWx0ZXJGdXNlZCwgZGJpYXNGdXNlZF0gPVxuICAgICAgICBmdXNlZEdyYWRzKFt4LCBmaWx0ZXIsIGJpYXNdLCBkeSk7XG5cbiAgICBjb25zdCBncmFkcyA9IHRmLmdyYWRzKCh4OiB0Zi5UZW5zb3I0RCwgZmlsdGVyOiB0Zi5UZW5zb3I0RCwgYmlhcykgPT4ge1xuICAgICAgY29uc3QgY29udiA9IHRmLmNvbnYyZCh4LCBmaWx0ZXIsIHN0cmlkZXMsIHBhZCk7XG4gICAgICBjb25zdCBzdW0gPSB0Zi5hZGQoY29udiwgYmlhcyk7XG4gICAgICByZXR1cm4gdGYuZWx1KHN1bSk7XG4gICAgfSk7XG4gICAgY29uc3QgW2R4LCBkZmlsdGVyLCBkYmlhc10gPSBncmFkcyhbeCwgZmlsdGVyLCBiaWFzXSwgZHkpO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgZHhGdXNlZC5hcnJheSgpLCBhd2FpdCBkeC5hcnJheSgpKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBkZmlsdGVyRnVzZWQuYXJyYXkoKSwgYXdhaXQgZGZpbHRlci5hcnJheSgpKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBkYmlhc0Z1c2VkLmFycmF5KCksIGF3YWl0IGRiaWFzLmFycmF5KCkpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gaW5wdXQgaXMgaW50MzInLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDI7XG4gICAgY29uc3QgaW5TaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbMiwgMiwgMiwgaW5wdXREZXB0aF07XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGZTaXplID0gMTtcbiAgICBjb25zdCBwYWQgPSAwO1xuICAgIGNvbnN0IHN0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxMCwgMTEsIDEyLCAxMywgMTQsIDE1LCAxNl0sIGluU2hhcGUsXG4gICAgICAgICdpbnQzMicpO1xuICAgIGNvbnN0IHcgPVxuICAgICAgICB0Zi50ZW5zb3I0ZChbLTEsIDEsIC0yLCAwLjVdLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF0pO1xuXG4gICAgZXhwZWN0KCgpID0+IHRmLmZ1c2VkLmNvbnYyZCh7eCwgZmlsdGVyOiB3LCBzdHJpZGVzOiBzdHJpZGUsIHBhZH0pKVxuICAgICAgICAudG9UaHJvd0Vycm9yKC9Bcmd1bWVudCAneCcgcGFzc2VkIHRvICdjb252MmQnIG11c3QgYmUgZmxvYXQzMi8pO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gZmlsdGVyIGlzIGludDMyJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzIsIDIsIDIsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBmU2l6ZSA9IDE7XG4gICAgY29uc3QgcGFkID0gMDtcbiAgICBjb25zdCBzdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMTAsIDExLCAxMiwgMTMsIDE0LCAxNSwgMTZdLCBpblNoYXBlKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFstMSwgMSwgLTIsIDAuNV0sIFtmU2l6ZSwgZlNpemUsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXSwgJ2ludDMyJyk7XG5cbiAgICBleHBlY3QoKCkgPT4gdGYuZnVzZWQuY29udjJkKHt4LCBmaWx0ZXI6IHcsIHN0cmlkZXM6IHN0cmlkZSwgcGFkfSkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoL0FyZ3VtZW50ICdmaWx0ZXInIHBhc3NlZCB0byAnY29udjJkJyBtdXN0IGJlIGZsb2F0MzIvKTtcbiAgfSk7XG59KTtcbiJdfQ==
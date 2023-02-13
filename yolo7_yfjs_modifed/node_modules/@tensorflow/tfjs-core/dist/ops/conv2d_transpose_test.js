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
describeWithFlags('conv2dTranspose', ALL_ENVS, () => {
    it('input=2x2x1,d2=1,f=2,s=1,p=0', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 1;
        const inputShape = [1, 1, origOutputDepth];
        const fSize = 2;
        const origPad = 0;
        const origStride = 1;
        const x = tf.tensor3d([2], inputShape);
        const w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        const result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
        const expected = [6, 2, 10, 0];
        expect(result.shape).toEqual([2, 2, 1]);
        expectArraysClose(await result.data(), expected);
    });
    it('input=3x3x1,d2=1,f=2,s=2,p=same', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 4;
        const inputShape = [1, 2, 2, origOutputDepth];
        const fSize = 2;
        const origPad = 'same';
        const origStride = 2;
        const x = tf.tensor4d([
            1.24, 1.66, 0.9, 1.39, 0.16, 0.27, 0.42, 0.61, 0.04, 0.17, 0.34, 0.28,
            0., 0.06, 0.14, 0.24
        ], inputShape);
        const w = tf.tensor4d([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.], [fSize, fSize, origInputDepth, origOutputDepth]);
        const result = tf.conv2dTranspose(x, w, [1, 3, 3, 1], origStride, origPad);
        const expected = [7.63, 28.39, 2.94, 49.15, 69.91, 14.62, 1.69, 5.01, 1.06];
        expect(result.shape).toEqual([1, 3, 3, 1]);
        expectArraysClose(await result.data(), expected);
    });
    it('input=3x3x1,d2=1,f=2,s=2,p=explicit', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 4;
        const inputShape = [1, 2, 2, origOutputDepth];
        const fSize = 2;
        const origPad = [[0, 0], [0, 1], [0, 1], [0, 0]];
        const origStride = 2;
        const x = tf.tensor4d([
            1.24, 1.66, 0.9, 1.39, 0.16, 0.27, 0.42, 0.61, 0.04, 0.17, 0.34, 0.28,
            0., 0.06, 0.14, 0.24
        ], inputShape);
        const w = tf.tensor4d([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.], [fSize, fSize, origInputDepth, origOutputDepth]);
        const result = tf.conv2dTranspose(x, w, [1, 3, 3, 1], origStride, origPad);
        const expected = [7.63, 28.39, 2.94, 49.15, 69.91, 14.62, 1.69, 5.01, 1.06];
        expect(result.shape).toEqual([1, 3, 3, 1]);
        expectArraysClose(await result.data(), expected);
    });
    it('input=2x2x1,d2=1,f=2,s=1,p=0, batch=2', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 1;
        const inputShape = [2, 1, 1, origOutputDepth];
        const fSize = 2;
        const origPad = 0;
        const origStride = 1;
        const x = tf.tensor4d([2, 3], inputShape);
        const w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        const result = tf.conv2dTranspose(x, w, [2, 2, 2, 1], origStride, origPad);
        const expected = [6, 2, 10, 0, 9, 3, 15, 0];
        expect(result.shape).toEqual([2, 2, 2, 1]);
        expectArraysClose(await result.data(), expected);
    });
    it('input=2x2x2,output=3x3x2,f=2,s=2,inDepth=2,p=same', async () => {
        const origInputDepth = 2;
        const origOutputDepth = 2;
        const inputShape = [1, 2, 2, origOutputDepth];
        const fSize = 2;
        const origPad = 'same';
        const origStride = 2;
        const x = tf.tensor4d([0., 1., 2., 3., 4., 5., 6., 7.], inputShape);
        const w = tf.tensor4d([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.], [fSize, fSize, origInputDepth, origOutputDepth]);
        const result = tf.conv2dTranspose(x, w, [1, 3, 3, origInputDepth], origStride, origPad);
        const expected = [1, 3, 5, 7, 3, 13, 9, 11, 13, 15, 43, 53, 5, 23, 41, 59, 7, 33.];
        expect(result.shape).toEqual([1, 3, 3, origInputDepth]);
        expectArraysClose(await result.data(), expected);
    });
    it('throws when dimRoundingMode is set and pad is same', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 4;
        const inputShape = [1, 2, 2, origOutputDepth];
        const fSize = 2;
        const origPad = 'same';
        const origStride = 2;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            1.24, 1.66, 0.9, 1.39, 0.16, 0.27, 0.42, 0.61, 0.04, 0.17, 0.34, 0.28,
            0., 0.06, 0.14, 0.24
        ], inputShape);
        const w = tf.tensor4d([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(() => tf.conv2dTranspose(x, w, [1, 3, 3, 1], origStride, origPad, dimRoundingMode))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is valid', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 4;
        const inputShape = [1, 2, 2, origOutputDepth];
        const fSize = 2;
        const origPad = 'valid';
        const origStride = 2;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            1.24, 1.66, 0.9, 1.39, 0.16, 0.27, 0.42, 0.61, 0.04, 0.17, 0.34, 0.28,
            0., 0.06, 0.14, 0.24
        ], inputShape);
        const w = tf.tensor4d([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(() => tf.conv2dTranspose(x, w, [1, 3, 3, 1], origStride, origPad, dimRoundingMode))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is a non-integer number', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 4;
        const inputShape = [1, 2, 2, origOutputDepth];
        const fSize = 2;
        const origPad = 1.2;
        const origStride = 2;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            1.24, 1.66, 0.9, 1.39, 0.16, 0.27, 0.42, 0.61, 0.04, 0.17, 0.34,
            0.28, 0., 0.06, 0.14, 0.24
        ], inputShape);
        const w = tf.tensor4d([
            0., 1., 2., 3., 4., 5., 6., 7., 8.,
            9., 10., 11., 12., 13., 14., 15.
        ], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(() => tf.conv2dTranspose(x, w, [1, 3, 3, 1], origStride, origPad, dimRoundingMode))
            .toThrowError();
    });
    it('throws when dimRoundingMode is set and pad is explicit by non-integer ' +
        'number', async () => {
        const origInputDepth = 1;
        const origOutputDepth = 4;
        const inputShape = [1, 2, 2, origOutputDepth];
        const fSize = 2;
        const origPad = [[0, 0], [0, 1.1], [0, 1], [0, 0]];
        const origStride = 2;
        const dimRoundingMode = 'round';
        const x = tf.tensor4d([
            1.24, 1.66, 0.9, 1.39, 0.16, 0.27, 0.42, 0.61, 0.04, 0.17, 0.34,
            0.28, 0., 0.06, 0.14, 0.24
        ], inputShape);
        const w = tf.tensor4d([
            0., 1., 2., 3., 4., 5., 6., 7., 8.,
            9., 10., 11., 12., 13., 14., 15.
        ], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(() => tf.conv2dTranspose(x, w, [1, 3, 3, 1], origStride, origPad, dimRoundingMode))
            .toThrowError();
    });
    // Reference (Python) TensorFlow code:
    //
    // ```py
    // import numpy as np
    // import tensorflow as tf
    //
    // tf.enable_eager_execution()
    //
    // x = tf.constant(np.array([[
    //     [[-0.14656299], [0.32942239], [-1.90302866]],
    //     [[-0.06487813], [-2.02637842], [-1.83669377]],
    //     [[0.82650784], [-0.89249092], [0.01207666]]
    // ]]).astype(np.float32))
    // filt = tf.constant(np.array([
    //     [[[-0.48280062], [1.26770487]], [[-0.83083738], [0.54341856]]],
    //     [[[-0.274904], [0.73111374]], [[2.01885189], [-2.68975237]]]
    // ]).astype(np.float32))
    //
    // with tf.GradientTape() as g:
    //   g.watch(x)
    //   g.watch(filt)
    //   y = tf.keras.backend.conv2d_transpose(x, filt, [1, 4, 4, 2])
    //   print(y)
    // (x_grad, filt_grad) = g.gradient(y, [x, filt])
    //
    // print("x_grad = %s" % x_grad)
    // print("filt_grad = %s" % filt_grad)
    // ```
    it('gradient with clones input=[1,3,3,1] f=[2,2,2,1] s=1 padding=valid', async () => {
        const inputDepth = 1;
        const outputDepth = 2;
        const inputShape = [1, 3, 3, inputDepth];
        const filterSize = 2;
        const stride = 1;
        const pad = 'valid';
        const filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        const x = tf.tensor4d([[
                [[-0.14656299], [0.32942239], [-1.90302866]],
                [[-0.06487813], [-2.02637842], [-1.83669377]],
                [[0.82650784], [-0.89249092], [0.01207666]]
            ]], inputShape);
        const filt = tf.tensor4d([
            [[[-0.48280062], [1.26770487]], [[-0.83083738], [0.54341856]]],
            [[[-0.274904], [0.73111374]], [[2.01885189], [-2.68975237]]]
        ], filterShape);
        const grads = tf.grads((x, filter) => tf.conv2dTranspose(x.clone(), filter.clone(), [1, 4, 4, outputDepth], stride, pad)
            .clone());
        const dy = tf.ones([1, 4, 4, outputDepth]);
        const [xGrad, filtGrad] = grads([x, filt], dy);
        const expectedXGrad = tf.ones([1, 3, 3, 1]).mul(tf.scalar(0.2827947));
        expectArraysClose(await xGrad.data(), await expectedXGrad.data());
        const expectedFiltGrad = tf.ones([2, 2, 2, 1]).mul(tf.scalar(-5.70202599));
        expectArraysClose(await filtGrad.data(), await expectedFiltGrad.data());
    });
    // Reference (Python) TensorFlow code:
    //
    // ```py
    // import numpy as np
    // import tensorflow as tf
    //
    // tf.enable_eager_execution()
    //
    // x = tf.constant(np.array([
    //     [[[-0.36541713], [-0.53973116]], [[0.01731674], [0.90227772]]]
    // ]).astype(np.float32))
    // filt = tf.constant(np.array([
    //     [[[-0.01423461], [-1.00267384]], [[1.61163029], [0.66302646]]],
    //     [[[-0.46900087], [-0.78649444]], [[0.87780536], [-0.84551637]]]
    // ]).astype(np.float32))
    //
    // with tf.GradientTape() as g:
    //   g.watch(x)
    //   g.watch(filt)
    //   y = tf.keras.backend.conv2d_transpose(x, filt, [1, 4, 4, 2], strides=(2,
    //   2)) print(y)
    // (x_grad, filt_grad) = g.gradient(y, [x, filt])
    //
    // print("x_grad = %s" % -x_grad)
    // print("filt_grad = %s" % -filt_grad)
    // ```
    it('gradient input=[1,2,2,1] f=[2,2,2,1] s=[2,2] padding=valid', async () => {
        const inputDepth = 1;
        const outputDepth = 2;
        const inputShape = [1, 2, 2, inputDepth];
        const filterSize = 2;
        const stride = [2, 2];
        const pad = 'valid';
        const filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        const x = tf.tensor4d([[[[-0.36541713], [-0.53973116]], [[0.01731674], [0.90227772]]]], inputShape);
        const filt = tf.tensor4d([
            [[[-0.01423461], [-1.00267384]], [[1.61163029], [0.66302646]]],
            [[[-0.46900087], [-0.78649444]], [[0.87780536], [-0.84551637]]]
        ], filterShape);
        const grads = tf.grads((x, filter) => tf.conv2dTranspose(x, filter, [1, 4, 4, outputDepth], stride, pad));
        const dy = tf.ones([1, 4, 4, outputDepth]).mul(tf.scalar(-1));
        const [xGrad, filtGrad] = grads([x, filt], dy);
        const expectedXGrad = tf.ones([1, 2, 2, 1]).mul(tf.scalar(-0.03454196));
        expectArraysClose(await xGrad.data(), await expectedXGrad.data());
        expect(xGrad.shape).toEqual([1, 2, 2, 1]);
        const expectedFiltGrad = tf.ones([2, 2, 2, 1]).mul(tf.scalar(-0.01444618));
        expectArraysClose(await filtGrad.data(), await expectedFiltGrad.data());
        expect(filtGrad.shape).toEqual([2, 2, 2, 1]);
    });
    // Reference (Python) TensorFlow code:
    //
    // ```py
    // import numpy as np
    // import tensorflow as tf
    //
    // tf.enable_eager_execution()
    //
    // x = tf.constant(np.array([[
    //     [[1.52433065], [-0.77053435], [-0.64562341]],
    //     [[0.77962889], [1.58413887], [-0.25581856]],
    //     [[-0.58966221], [0.05411662], [0.70749138]]
    // ]]).astype(np.float32))
    // filt = tf.constant(np.array([
    //     [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
    //     [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
    // ]).astype(np.float32))
    //
    // with tf.GradientTape() as g:
    //   g.watch(x)
    //   g.watch(filt)
    //   y = tf.keras.backend.conv2d_transpose(
    //       x, filt, [1, 3, 3, 2], strides=(1, 1), padding='same')
    // (x_grad, filt_grad) = g.gradient(y, [x, filt])
    //
    // print("x_grad = %s" % x_grad)
    // print("filt_grad = %s" % filt_grad)
    // ```
    it('gradient input=[1,3,3,1] f=[2,2,2,1] s=[1,1] padding=same', async () => {
        const inputDepth = 1;
        const outputDepth = 2;
        const inputShape = [1, 3, 3, inputDepth];
        const filterSize = 2;
        const stride = [1, 1];
        const pad = 'same';
        const filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        const x = tf.tensor4d([[
                [[1.52433065], [-0.77053435], [-0.64562341]],
                [[0.77962889], [1.58413887], [-0.25581856]],
                [[-0.58966221], [0.05411662], [0.70749138]]
            ]], inputShape);
        const filt = tf.tensor4d([
            [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
            [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
        ], filterShape);
        const grads = tf.grads((x, filter) => tf.conv2dTranspose(x, filter, [1, 3, 3, outputDepth], stride, pad));
        const dy = tf.ones([1, 3, 3, outputDepth]);
        const [xGrad, filtGrad] = grads([x, filt], dy);
        expectArraysClose(await xGrad.array(), [[
                [[1.30709858], [1.30709858], [-0.92814366]],
                [[1.30709858], [1.30709858], [-0.92814366]],
                [[1.19666437], [1.19666437], [-0.85476589]]
            ]]);
        expectArraysClose(await filtGrad.array(), [
            [[[2.38806788], [2.38806788]], [[2.58201847], [2.58201847]]],
            [[[2.2161221], [2.2161221]], [[3.11756406], [3.11756406]]]
        ]);
    });
    it('gradient input=[1,3,3,1] f=[2,2,2,1] s=[1,1] p=explicit', async () => {
        const inputDepth = 1;
        const outputDepth = 2;
        const inputShape = [1, 3, 3, inputDepth];
        const filterSize = 2;
        const stride = [1, 1];
        const pad = [[0, 0], [0, 1], [0, 1], [0, 0]];
        const filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        const x = tf.tensor4d([[
                [[1.52433065], [-0.77053435], [-0.64562341]],
                [[0.77962889], [1.58413887], [-0.25581856]],
                [[-0.58966221], [0.05411662], [0.70749138]]
            ]], inputShape);
        const filt = tf.tensor4d([
            [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
            [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
        ], filterShape);
        const grads = tf.grads((x, filter) => tf.conv2dTranspose(x, filter, [1, 3, 3, outputDepth], stride, pad));
        const dy = tf.ones([1, 3, 3, outputDepth]);
        const [xGrad, filtGrad] = grads([x, filt], dy);
        expectArraysClose(await xGrad.array(), [[
                [[1.30709858], [1.30709858], [-0.92814366]],
                [[1.30709858], [1.30709858], [-0.92814366]],
                [[1.19666437], [1.19666437], [-0.85476589]]
            ]]);
        expectArraysClose(await filtGrad.array(), [
            [[[2.38806788], [2.38806788]], [[2.58201847], [2.58201847]]],
            [[[2.2161221], [2.2161221]], [[3.11756406], [3.11756406]]]
        ]);
    });
    // Reference (Python) TensorFlow code:
    //
    // ```py
    // import numpy as np
    // import tensorflow as tf
    //
    // tf.enable_eager_execution()
    //
    // x = tf.constant(np.array([[
    //     [[1.52433065], [-0.77053435]], [[0.77962889], [1.58413887]],
    // ]]).astype(np.float32))
    // filt = tf.constant(np.array([
    //     [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
    //     [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
    // ]).astype(np.float32))
    //
    // with tf.GradientTape() as g:
    //   g.watch(x)
    //   g.watch(filt)
    //   y = tf.keras.backend.conv2d_transpose(
    //       x, filt, [1, 3, 3, 2], strides=(2, 2), padding='same')
    //   print(y.shape)
    // (x_grad, filt_grad) = g.gradient(y, [x, filt])
    //
    // print("x_grad = %s" % x_grad)
    // print("filt_grad = %s" % filt_grad)
    // ```
    it('gradient input=[1,2,2,2] f=[2,2,2,1] s=[2,2] padding=same', async () => {
        const inputDepth = 2;
        const outputDepth = 2;
        const inputShape = [1, 2, 2, inputDepth];
        const filterSize = 2;
        const stride = [2, 2];
        const pad = 'same';
        const filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        const x = tf.tensor4d([[
                [[-1.81506593, 1.00900095], [-0.05199118, 0.26311377]],
                [[-1.18469792, -0.34780521], [2.04971242, -0.65154692]]
            ]], inputShape);
        const filt = tf.tensor4d([
            [
                [[0.19529686, -0.79594708], [0.70314057, -0.06081263]],
                [[0.28724744, 0.88522715], [-0.51824096, -0.97120989]]
            ],
            [
                [[0.51872197, -1.17569193], [1.28316791, -0.81225092]],
                [[-0.44221532, 0.70058174], [-0.4849217, 0.03806348]]
            ]
        ], filterShape);
        const grads = tf.grads((x, filter) => tf.conv2dTranspose(x, filter, [1, 3, 3, outputDepth], stride, pad));
        const dy = tf.ones([1, 3, 3, outputDepth]);
        const [xGrad, filtGrad] = grads([x, filt], dy);
        expectArraysClose(await xGrad.data(), [
            1.54219678, -2.19204008, 2.70032732, -2.84470257, 0.66744391, -0.94274245,
            0.89843743, -0.85675972
        ]);
        expect(xGrad.shape).toEqual([1, 2, 2, 2]);
        expectArraysClose(await filtGrad.data(), [
            -1.00204261, 0.27276259, -1.00204261, 0.27276259, -2.99976385, 0.66119574,
            -2.99976385, 0.66119574, -1.86705711, 1.27211472, -1.86705711, 1.27211472,
            -1.81506593, 1.00900095, -1.81506593, 1.00900095
        ]);
        expect(filtGrad.shape).toEqual([2, 2, 2, 2]);
    });
    it('throws when x is not rank 3', () => {
        const origInputDepth = 1;
        const origOutputDepth = 1;
        const fSize = 2;
        const origPad = 0;
        const origStride = 1;
        // tslint:disable-next-line:no-any
        const x = tf.tensor2d([2, 2], [2, 1]);
        const w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(() => tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
            .toThrowError();
    });
    it('throws when weights is not rank 4', () => {
        const origInputDepth = 1;
        const origOutputDepth = 1;
        const inputShape = [1, 1, origOutputDepth];
        const fSize = 2;
        const origPad = 0;
        const origStride = 1;
        const x = tf.tensor3d([2], inputShape);
        // tslint:disable-next-line:no-any
        const w = tf.tensor3d([3, 1, 5, 0], [fSize, fSize, origInputDepth]);
        expect(() => tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
            .toThrowError();
    });
    it('throws when x depth does not match weights original output depth', () => {
        const origInputDepth = 1;
        const origOutputDepth = 2;
        const wrongOrigOutputDepth = 3;
        const inputShape = [1, 1, origOutputDepth];
        const fSize = 2;
        const origPad = 0;
        const origStride = 1;
        const x = tf.tensor3d([2, 2], inputShape);
        const w = tf.randomNormal([fSize, fSize, origInputDepth, wrongOrigOutputDepth]);
        expect(() => tf.conv2dTranspose(x, w, [2, 2, 2], origStride, origPad))
            .toThrowError();
    });
    it('throws when passed x as a non-tensor', () => {
        const origInputDepth = 1;
        const origOutputDepth = 1;
        const fSize = 2;
        const origPad = 0;
        const origStride = 1;
        const w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(() => tf.conv2dTranspose({}, w, [2, 2, 1], origStride, origPad))
            .toThrowError(/Argument 'x' passed to 'conv2dTranspose' must be a Tensor/);
    });
    it('throws when passed filter as a non-tensor', () => {
        const origOutputDepth = 1;
        const inputShape = [1, 1, origOutputDepth];
        const origPad = 0;
        const origStride = 1;
        const x = tf.tensor3d([2], inputShape);
        expect(() => tf.conv2dTranspose(x, {}, [2, 2, 1], origStride, origPad))
            .toThrowError(/Argument 'filter' passed to 'conv2dTranspose' must be a Tensor/);
    });
    it('accepts a tensor-like object', async () => {
        const origPad = 0;
        const origStride = 1;
        const x = [[[2]]]; // 1x1x1
        const w = [[[[3]], [[1]]], [[[5]], [[0]]]]; // 2x2x1x1
        const result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
        const expected = [6, 2, 10, 0];
        expect(result.shape).toEqual([2, 2, 1]);
        expectArraysClose(await result.data(), expected);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjJkX3RyYW5zcG9zZV90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29udjJkX3RyYW5zcG9zZV90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxRQUFRLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFHL0MsaUJBQWlCLENBQUMsaUJBQWlCLEVBQUUsUUFBUSxFQUFFLEdBQUcsRUFBRTtJQUNsRCxFQUFFLENBQUMsOEJBQThCLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDNUMsTUFBTSxjQUFjLEdBQUcsQ0FBQyxDQUFDO1FBQ3pCLE1BQU0sZUFBZSxHQUFHLENBQUMsQ0FBQztRQUMxQixNQUFNLFVBQVUsR0FBNkIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLGVBQWUsQ0FBQyxDQUFDO1FBQ3JFLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNoQixNQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFDbEIsTUFBTSxVQUFVLEdBQUcsQ0FBQyxDQUFDO1FBRXJCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQztRQUN2QyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUNqQixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxjQUFjLEVBQUUsZUFBZSxDQUFDLENBQUMsQ0FBQztRQUVuRSxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFB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Q0FBQyxDQUFDO2FBQ3pELFlBQVksQ0FDVCxnRUFBZ0UsQ0FBQyxDQUFDO0lBQzVFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDhCQUE4QixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzVDLE1BQU0sT0FBTyxHQUFHLENBQUMsQ0FBQztRQUNsQixNQUFNLFVBQVUsR0FBRyxDQUFDLENBQUM7UUFFckIsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQTJCLFFBQVE7UUFDckQsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBRSxVQUFVO1FBRXZELE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsVUFBVSxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3hFLE1BQU0sUUFBUSxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFFL0IsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDeEMsaUJBQWlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDbkQsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi4vaW5kZXgnO1xuaW1wb3J0IHtBTExfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3N9IGZyb20gJy4uL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge2V4cGVjdEFycmF5c0Nsb3NlfSBmcm9tICcuLi90ZXN0X3V0aWwnO1xuaW1wb3J0IHtSYW5rfSBmcm9tICcuLi90eXBlcyc7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdjb252MmRUcmFuc3Bvc2UnLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnaW5wdXQ9MngyeDEsZDI9MSxmPTIscz0xLHA9MCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gMTtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbMSwgMSwgb3JpZ091dHB1dERlcHRoXTtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3Qgb3JpZ1BhZCA9IDA7XG4gICAgY29uc3Qgb3JpZ1N0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yM2QoWzJdLCBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFszLCAxLCA1LCAwXSwgW2ZTaXplLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIG9yaWdPdXRwdXREZXB0aF0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gdGYuY29udjJkVHJhbnNwb3NlKHgsIHcsIFsyLCAyLCAxXSwgb3JpZ1N0cmlkZSwgb3JpZ1BhZCk7XG4gICAgY29uc3QgZXhwZWN0ZWQgPSBbNiwgMiwgMTAsIDBdO1xuXG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbMiwgMiwgMV0pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIGV4cGVjdGVkKTtcbiAgfSk7XG5cbiAgaXQoJ2lucHV0PTN4M3gxLGQyPTEsZj0yLHM9MixwPXNhbWUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgb3JpZ0lucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG9yaWdPdXRwdXREZXB0aCA9IDQ7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICBbMSwgMiwgMiwgb3JpZ091dHB1dERlcHRoXTtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3Qgb3JpZ1BhZCA9ICdzYW1lJztcbiAgICBjb25zdCBvcmlnU3RyaWRlID0gMjtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDEuMjQsIDEuNjYsIDAuOSwgMS4zOSwgMC4xNiwgMC4yNywgMC40MiwgMC42MSwgMC4wNCwgMC4xNywgMC4zNCwgMC4yOCxcbiAgICAgICAgICAwLiwgMC4wNiwgMC4xNCwgMC4yNFxuICAgICAgICBdLFxuICAgICAgICBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFswLiwgMS4sIDIuLCAzLiwgNC4sIDUuLCA2LiwgNy4sIDguLCA5LiwgMTAuLCAxMS4sIDEyLiwgMTMuLCAxNC4sIDE1Ll0sXG4gICAgICAgIFtmU2l6ZSwgZlNpemUsIG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmNvbnYyZFRyYW5zcG9zZSh4LCB3LCBbMSwgMywgMywgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQpO1xuICAgIGNvbnN0IGV4cGVjdGVkID0gWzcuNjMsIDI4LjM5LCAyLjk0LCA0OS4xNSwgNjkuOTEsIDE0LjYyLCAxLjY5LCA1LjAxLCAxLjA2XTtcblxuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzEsIDMsIDMsIDFdKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdpbnB1dD0zeDN4MSxkMj0xLGY9MixzPTIscD1leHBsaWNpdCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gNDtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgIFsxLCAyLCAyLCBvcmlnT3V0cHV0RGVwdGhdO1xuICAgIGNvbnN0IGZTaXplID0gMjtcbiAgICBjb25zdCBvcmlnUGFkID1cbiAgICAgICAgW1swLCAwXSwgWzAsIDFdLCBbMCwgMV0sIFswLCAwXV0gYXMgdGYuYmFja2VuZF91dGlsLkV4cGxpY2l0UGFkZGluZztcbiAgICBjb25zdCBvcmlnU3RyaWRlID0gMjtcblxuICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIDEuMjQsIDEuNjYsIDAuOSwgMS4zOSwgMC4xNiwgMC4yNywgMC40MiwgMC42MSwgMC4wNCwgMC4xNywgMC4zNCwgMC4yOCxcbiAgICAgICAgICAwLiwgMC4wNiwgMC4xNCwgMC4yNFxuICAgICAgICBdLFxuICAgICAgICBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFswLiwgMS4sIDIuLCAzLiwgNC4sIDUuLCA2LiwgNy4sIDguLCA5LiwgMTAuLCAxMS4sIDEyLiwgMTMuLCAxNC4sIDE1Ll0sXG4gICAgICAgIFtmU2l6ZSwgZlNpemUsIG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmNvbnYyZFRyYW5zcG9zZSh4LCB3LCBbMSwgMywgMywgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQpO1xuICAgIGNvbnN0IGV4cGVjdGVkID0gWzcuNjMsIDI4LjM5LCAyLjk0LCA0OS4xNSwgNjkuOTEsIDE0LjYyLCAxLjY5LCA1LjAxLCAxLjA2XTtcblxuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzEsIDMsIDMsIDFdKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xuXG4gIGl0KCdpbnB1dD0yeDJ4MSxkMj0xLGY9MixzPTEscD0wLCBiYXRjaD0yJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gMTtcbiAgICBjb25zdCBvcmlnT3V0cHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgWzIsIDEsIDEsIG9yaWdPdXRwdXREZXB0aF07XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IG9yaWdQYWQgPSAwO1xuICAgIGNvbnN0IG9yaWdTdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFsyLCAzXSwgaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgdyA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbMywgMSwgNSwgMF0sIFtmU2l6ZSwgZlNpemUsIG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmNvbnYyZFRyYW5zcG9zZSh4LCB3LCBbMiwgMiwgMiwgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQpO1xuICAgIGNvbnN0IGV4cGVjdGVkID0gWzYsIDIsIDEwLCAwLCA5LCAzLCAxNSwgMF07XG5cbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKFsyLCAyLCAyLCAxXSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgZXhwZWN0ZWQpO1xuICB9KTtcblxuICBpdCgnaW5wdXQ9MngyeDIsb3V0cHV0PTN4M3gyLGY9MixzPTIsaW5EZXB0aD0yLHA9c2FtZScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDI7XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgIFsxLCAyLCAyLCBvcmlnT3V0cHV0RGVwdGhdO1xuICAgIGNvbnN0IGZTaXplID0gMjtcbiAgICBjb25zdCBvcmlnUGFkID0gJ3NhbWUnO1xuICAgIGNvbnN0IG9yaWdTdHJpZGUgPSAyO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFswLiwgMS4sIDIuLCAzLiwgNC4sIDUuLCA2LiwgNy5dLCBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFswLiwgMS4sIDIuLCAzLiwgNC4sIDUuLCA2LiwgNy4sIDguLCA5LiwgMTAuLCAxMS4sIDEyLiwgMTMuLCAxNC4sIDE1Ll0sXG4gICAgICAgIFtmU2l6ZSwgZlNpemUsIG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGhdKTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmNvbnYyZFRyYW5zcG9zZShcbiAgICAgICAgeCwgdywgWzEsIDMsIDMsIG9yaWdJbnB1dERlcHRoXSwgb3JpZ1N0cmlkZSwgb3JpZ1BhZCk7XG4gICAgY29uc3QgZXhwZWN0ZWQgPVxuICAgICAgICBbMSwgMywgNSwgNywgMywgMTMsIDksIDExLCAxMywgMTUsIDQzLCA1MywgNSwgMjMsIDQxLCA1OSwgNywgMzMuXTtcblxuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzEsIDMsIDMsIG9yaWdJbnB1dERlcHRoXSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgZXhwZWN0ZWQpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gZGltUm91bmRpbmdNb2RlIGlzIHNldCBhbmQgcGFkIGlzIHNhbWUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgb3JpZ0lucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG9yaWdPdXRwdXREZXB0aCA9IDQ7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICBbMSwgMiwgMiwgb3JpZ091dHB1dERlcHRoXTtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3Qgb3JpZ1BhZCA9ICdzYW1lJztcbiAgICBjb25zdCBvcmlnU3RyaWRlID0gMjtcbiAgICBjb25zdCBkaW1Sb3VuZGluZ01vZGUgPSAncm91bmQnO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMS4yNCwgMS42NiwgMC45LCAxLjM5LCAwLjE2LCAwLjI3LCAwLjQyLCAwLjYxLCAwLjA0LCAwLjE3LCAwLjM0LCAwLjI4LFxuICAgICAgICAgIDAuLCAwLjA2LCAwLjE0LCAwLjI0XG4gICAgICAgIF0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuLCAxLiwgMi4sIDMuLCA0LiwgNS4sIDYuLCA3LiwgOC4sIDkuLCAxMC4sIDExLiwgMTIuLCAxMy4sIDE0LiwgMTUuXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIG9yaWdPdXRwdXREZXB0aF0pO1xuXG4gICAgZXhwZWN0KFxuICAgICAgICAoKSA9PiB0Zi5jb252MmRUcmFuc3Bvc2UoXG4gICAgICAgICAgICB4LCB3LCBbMSwgMywgMywgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQsIGRpbVJvdW5kaW5nTW9kZSkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyB3aGVuIGRpbVJvdW5kaW5nTW9kZSBpcyBzZXQgYW5kIHBhZCBpcyB2YWxpZCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gNDtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgIFsxLCAyLCAyLCBvcmlnT3V0cHV0RGVwdGhdO1xuICAgIGNvbnN0IGZTaXplID0gMjtcbiAgICBjb25zdCBvcmlnUGFkID0gJ3ZhbGlkJztcbiAgICBjb25zdCBvcmlnU3RyaWRlID0gMjtcbiAgICBjb25zdCBkaW1Sb3VuZGluZ01vZGUgPSAncm91bmQnO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgMS4yNCwgMS42NiwgMC45LCAxLjM5LCAwLjE2LCAwLjI3LCAwLjQyLCAwLjYxLCAwLjA0LCAwLjE3LCAwLjM0LCAwLjI4LFxuICAgICAgICAgIDAuLCAwLjA2LCAwLjE0LCAwLjI0XG4gICAgICAgIF0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgWzAuLCAxLiwgMi4sIDMuLCA0LiwgNS4sIDYuLCA3LiwgOC4sIDkuLCAxMC4sIDExLiwgMTIuLCAxMy4sIDE0LiwgMTUuXSxcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIG9yaWdPdXRwdXREZXB0aF0pO1xuXG4gICAgZXhwZWN0KFxuICAgICAgICAoKSA9PiB0Zi5jb252MmRUcmFuc3Bvc2UoXG4gICAgICAgICAgICB4LCB3LCBbMSwgMywgMywgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQsIGRpbVJvdW5kaW5nTW9kZSkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyB3aGVuIGRpbVJvdW5kaW5nTW9kZSBpcyBzZXQgYW5kIHBhZCBpcyBhIG5vbi1pbnRlZ2VyIG51bWJlcicsXG4gICAgIGFzeW5jICgpID0+IHtcbiAgICAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gICAgICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gNDtcbiAgICAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgICAgIFsxLCAyLCAyLCBvcmlnT3V0cHV0RGVwdGhdO1xuICAgICAgIGNvbnN0IGZTaXplID0gMjtcbiAgICAgICBjb25zdCBvcmlnUGFkID0gMS4yO1xuICAgICAgIGNvbnN0IG9yaWdTdHJpZGUgPSAyO1xuICAgICAgIGNvbnN0IGRpbVJvdW5kaW5nTW9kZSA9ICdyb3VuZCc7XG5cbiAgICAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgICAgIFtcbiAgICAgICAgICAgICAxLjI0LCAxLjY2LCAwLjksIDEuMzksIDAuMTYsIDAuMjcsIDAuNDIsIDAuNjEsIDAuMDQsIDAuMTcsIDAuMzQsXG4gICAgICAgICAgICAgMC4yOCwgMC4sIDAuMDYsIDAuMTQsIDAuMjRcbiAgICAgICAgICAgXSxcbiAgICAgICAgICAgaW5wdXRTaGFwZSk7XG4gICAgICAgY29uc3QgdyA9IHRmLnRlbnNvcjRkKFxuICAgICAgICAgICBbXG4gICAgICAgICAgICAgMC4sIDEuLCAyLiwgMy4sIDQuLCA1LiwgNi4sIDcuLCA4LixcbiAgICAgICAgICAgICA5LiwgMTAuLCAxMS4sIDEyLiwgMTMuLCAxNC4sIDE1LlxuICAgICAgICAgICBdLFxuICAgICAgICAgICBbZlNpemUsIGZTaXplLCBvcmlnSW5wdXREZXB0aCwgb3JpZ091dHB1dERlcHRoXSk7XG5cbiAgICAgICBleHBlY3QoXG4gICAgICAgICAgICgpID0+IHRmLmNvbnYyZFRyYW5zcG9zZShcbiAgICAgICAgICAgICAgeCwgdywgWzEsIDMsIDMsIDFdLCBvcmlnU3RyaWRlLCBvcmlnUGFkLCBkaW1Sb3VuZGluZ01vZGUpKVxuICAgICAgICAgICAudG9UaHJvd0Vycm9yKCk7XG4gICAgIH0pO1xuXG4gIGl0KCd0aHJvd3Mgd2hlbiBkaW1Sb3VuZGluZ01vZGUgaXMgc2V0IGFuZCBwYWQgaXMgZXhwbGljaXQgYnkgbm9uLWludGVnZXIgJyArXG4gICAgICAgICAnbnVtYmVyJyxcbiAgICAgYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gMTtcbiAgICAgICBjb25zdCBvcmlnT3V0cHV0RGVwdGggPSA0O1xuICAgICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICAgWzEsIDIsIDIsIG9yaWdPdXRwdXREZXB0aF07XG4gICAgICAgY29uc3QgZlNpemUgPSAyO1xuICAgICAgIGNvbnN0IG9yaWdQYWQgPSBbWzAsIDBdLCBbMCwgMS4xXSwgWzAsIDFdLCBbMCwgMF1dIGFzXG4gICAgICAgICAgIHRmLmJhY2tlbmRfdXRpbC5FeHBsaWNpdFBhZGRpbmc7XG4gICAgICAgY29uc3Qgb3JpZ1N0cmlkZSA9IDI7XG4gICAgICAgY29uc3QgZGltUm91bmRpbmdNb2RlID0gJ3JvdW5kJztcblxuICAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgICAgW1xuICAgICAgICAgICAgIDEuMjQsIDEuNjYsIDAuOSwgMS4zOSwgMC4xNiwgMC4yNywgMC40MiwgMC42MSwgMC4wNCwgMC4xNywgMC4zNCxcbiAgICAgICAgICAgICAwLjI4LCAwLiwgMC4wNiwgMC4xNCwgMC4yNFxuICAgICAgICAgICBdLFxuICAgICAgICAgICBpbnB1dFNoYXBlKTtcbiAgICAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgICAgIFtcbiAgICAgICAgICAgICAwLiwgMS4sIDIuLCAzLiwgNC4sIDUuLCA2LiwgNy4sIDguLFxuICAgICAgICAgICAgIDkuLCAxMC4sIDExLiwgMTIuLCAxMy4sIDE0LiwgMTUuXG4gICAgICAgICAgIF0sXG4gICAgICAgICAgIFtmU2l6ZSwgZlNpemUsIG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGhdKTtcblxuICAgICAgIGV4cGVjdChcbiAgICAgICAgICAgKCkgPT4gdGYuY29udjJkVHJhbnNwb3NlKFxuICAgICAgICAgICAgICB4LCB3LCBbMSwgMywgMywgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQsIGRpbVJvdW5kaW5nTW9kZSkpXG4gICAgICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgICAgfSk7XG5cbiAgLy8gUmVmZXJlbmNlIChQeXRob24pIFRlbnNvckZsb3cgY29kZTpcbiAgLy9cbiAgLy8gYGBgcHlcbiAgLy8gaW1wb3J0IG51bXB5IGFzIG5wXG4gIC8vIGltcG9ydCB0ZW5zb3JmbG93IGFzIHRmXG4gIC8vXG4gIC8vIHRmLmVuYWJsZV9lYWdlcl9leGVjdXRpb24oKVxuICAvL1xuICAvLyB4ID0gdGYuY29uc3RhbnQobnAuYXJyYXkoW1tcbiAgLy8gICAgIFtbLTAuMTQ2NTYyOTldLCBbMC4zMjk0MjIzOV0sIFstMS45MDMwMjg2Nl1dLFxuICAvLyAgICAgW1stMC4wNjQ4NzgxM10sIFstMi4wMjYzNzg0Ml0sIFstMS44MzY2OTM3N11dLFxuICAvLyAgICAgW1swLjgyNjUwNzg0XSwgWy0wLjg5MjQ5MDkyXSwgWzAuMDEyMDc2NjZdXVxuICAvLyBdXSkuYXN0eXBlKG5wLmZsb2F0MzIpKVxuICAvLyBmaWx0ID0gdGYuY29uc3RhbnQobnAuYXJyYXkoW1xuICAvLyAgICAgW1tbLTAuNDgyODAwNjJdLCBbMS4yNjc3MDQ4N11dLCBbWy0wLjgzMDgzNzM4XSwgWzAuNTQzNDE4NTZdXV0sXG4gIC8vICAgICBbW1stMC4yNzQ5MDRdLCBbMC43MzExMTM3NF1dLCBbWzIuMDE4ODUxODldLCBbLTIuNjg5NzUyMzddXV1cbiAgLy8gXSkuYXN0eXBlKG5wLmZsb2F0MzIpKVxuICAvL1xuICAvLyB3aXRoIHRmLkdyYWRpZW50VGFwZSgpIGFzIGc6XG4gIC8vICAgZy53YXRjaCh4KVxuICAvLyAgIGcud2F0Y2goZmlsdClcbiAgLy8gICB5ID0gdGYua2VyYXMuYmFja2VuZC5jb252MmRfdHJhbnNwb3NlKHgsIGZpbHQsIFsxLCA0LCA0LCAyXSlcbiAgLy8gICBwcmludCh5KVxuICAvLyAoeF9ncmFkLCBmaWx0X2dyYWQpID0gZy5ncmFkaWVudCh5LCBbeCwgZmlsdF0pXG4gIC8vXG4gIC8vIHByaW50KFwieF9ncmFkID0gJXNcIiAlIHhfZ3JhZClcbiAgLy8gcHJpbnQoXCJmaWx0X2dyYWQgPSAlc1wiICUgZmlsdF9ncmFkKVxuICAvLyBgYGBcbiAgaXQoJ2dyYWRpZW50IHdpdGggY2xvbmVzIGlucHV0PVsxLDMsMywxXSBmPVsyLDIsMiwxXSBzPTEgcGFkZGluZz12YWxpZCcsXG4gICAgIGFzeW5jICgpID0+IHtcbiAgICAgICBjb25zdCBpbnB1dERlcHRoID0gMTtcbiAgICAgICBjb25zdCBvdXRwdXREZXB0aCA9IDI7XG4gICAgICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICAgICBbMSwgMywgMywgaW5wdXREZXB0aF07XG4gICAgICAgY29uc3QgZmlsdGVyU2l6ZSA9IDI7XG4gICAgICAgY29uc3Qgc3RyaWRlID0gMTtcbiAgICAgICBjb25zdCBwYWQgPSAndmFsaWQnO1xuXG4gICAgICAgY29uc3QgZmlsdGVyU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICAgW2ZpbHRlclNpemUsIGZpbHRlclNpemUsIG91dHB1dERlcHRoLCBpbnB1dERlcHRoXTtcblxuICAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgICAgW1tcbiAgICAgICAgICAgICBbWy0wLjE0NjU2Mjk5XSwgWzAuMzI5NDIyMzldLCBbLTEuOTAzMDI4NjZdXSxcbiAgICAgICAgICAgICBbWy0wLjA2NDg3ODEzXSwgWy0yLjAyNjM3ODQyXSwgWy0xLjgzNjY5Mzc3XV0sXG4gICAgICAgICAgICAgW1swLjgyNjUwNzg0XSwgWy0wLjg5MjQ5MDkyXSwgWzAuMDEyMDc2NjZdXVxuICAgICAgICAgICBdXSxcbiAgICAgICAgICAgaW5wdXRTaGFwZSk7XG4gICAgICAgY29uc3QgZmlsdCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICAgICBbXG4gICAgICAgICAgICAgW1tbLTAuNDgyODAwNjJdLCBbMS4yNjc3MDQ4N11dLCBbWy0wLjgzMDgzNzM4XSwgWzAuNTQzNDE4NTZdXV0sXG4gICAgICAgICAgICAgW1tbLTAuMjc0OTA0XSwgWzAuNzMxMTEzNzRdXSwgW1syLjAxODg1MTg5XSwgWy0yLjY4OTc1MjM3XV1dXG4gICAgICAgICAgIF0sXG4gICAgICAgICAgIGZpbHRlclNoYXBlKTtcblxuICAgICAgIGNvbnN0IGdyYWRzID0gdGYuZ3JhZHMoXG4gICAgICAgICAgICh4OiB0Zi5UZW5zb3I0RCwgZmlsdGVyOiB0Zi5UZW5zb3I0RCkgPT5cbiAgICAgICAgICAgICAgIHRmLmNvbnYyZFRyYW5zcG9zZShcbiAgICAgICAgICAgICAgICAgICAgIHguY2xvbmUoKSwgZmlsdGVyLmNsb25lKCksIFsxLCA0LCA0LCBvdXRwdXREZXB0aF0sIHN0cmlkZSxcbiAgICAgICAgICAgICAgICAgICAgIHBhZClcbiAgICAgICAgICAgICAgICAgICAuY2xvbmUoKSk7XG4gICAgICAgY29uc3QgZHkgPSB0Zi5vbmVzKFsxLCA0LCA0LCBvdXRwdXREZXB0aF0pO1xuICAgICAgIGNvbnN0IFt4R3JhZCwgZmlsdEdyYWRdID0gZ3JhZHMoW3gsIGZpbHRdLCBkeSk7XG5cbiAgICAgICBjb25zdCBleHBlY3RlZFhHcmFkID0gdGYub25lcyhbMSwgMywgMywgMV0pLm11bCh0Zi5zY2FsYXIoMC4yODI3OTQ3KSk7XG4gICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgeEdyYWQuZGF0YSgpLCBhd2FpdCBleHBlY3RlZFhHcmFkLmRhdGEoKSk7XG4gICAgICAgY29uc3QgZXhwZWN0ZWRGaWx0R3JhZCA9XG4gICAgICAgICAgIHRmLm9uZXMoWzIsIDIsIDIsIDFdKS5tdWwodGYuc2NhbGFyKC01LjcwMjAyNTk5KSk7XG4gICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgZmlsdEdyYWQuZGF0YSgpLCBhd2FpdCBleHBlY3RlZEZpbHRHcmFkLmRhdGEoKSk7XG4gICAgIH0pO1xuXG4gIC8vIFJlZmVyZW5jZSAoUHl0aG9uKSBUZW5zb3JGbG93IGNvZGU6XG4gIC8vXG4gIC8vIGBgYHB5XG4gIC8vIGltcG9ydCBudW1weSBhcyBucFxuICAvLyBpbXBvcnQgdGVuc29yZmxvdyBhcyB0ZlxuICAvL1xuICAvLyB0Zi5lbmFibGVfZWFnZXJfZXhlY3V0aW9uKClcbiAgLy9cbiAgLy8geCA9IHRmLmNvbnN0YW50KG5wLmFycmF5KFtcbiAgLy8gICAgIFtbWy0wLjM2NTQxNzEzXSwgWy0wLjUzOTczMTE2XV0sIFtbMC4wMTczMTY3NF0sIFswLjkwMjI3NzcyXV1dXG4gIC8vIF0pLmFzdHlwZShucC5mbG9hdDMyKSlcbiAgLy8gZmlsdCA9IHRmLmNvbnN0YW50KG5wLmFycmF5KFtcbiAgLy8gICAgIFtbWy0wLjAxNDIzNDYxXSwgWy0xLjAwMjY3Mzg0XV0sIFtbMS42MTE2MzAyOV0sIFswLjY2MzAyNjQ2XV1dLFxuICAvLyAgICAgW1tbLTAuNDY5MDAwODddLCBbLTAuNzg2NDk0NDRdXSwgW1swLjg3NzgwNTM2XSwgWy0wLjg0NTUxNjM3XV1dXG4gIC8vIF0pLmFzdHlwZShucC5mbG9hdDMyKSlcbiAgLy9cbiAgLy8gd2l0aCB0Zi5HcmFkaWVudFRhcGUoKSBhcyBnOlxuICAvLyAgIGcud2F0Y2goeClcbiAgLy8gICBnLndhdGNoKGZpbHQpXG4gIC8vICAgeSA9IHRmLmtlcmFzLmJhY2tlbmQuY29udjJkX3RyYW5zcG9zZSh4LCBmaWx0LCBbMSwgNCwgNCwgMl0sIHN0cmlkZXM9KDIsXG4gIC8vICAgMikpIHByaW50KHkpXG4gIC8vICh4X2dyYWQsIGZpbHRfZ3JhZCkgPSBnLmdyYWRpZW50KHksIFt4LCBmaWx0XSlcbiAgLy9cbiAgLy8gcHJpbnQoXCJ4X2dyYWQgPSAlc1wiICUgLXhfZ3JhZClcbiAgLy8gcHJpbnQoXCJmaWx0X2dyYWQgPSAlc1wiICUgLWZpbHRfZ3JhZClcbiAgLy8gYGBgXG4gIGl0KCdncmFkaWVudCBpbnB1dD1bMSwyLDIsMV0gZj1bMiwyLDIsMV0gcz1bMiwyXSBwYWRkaW5nPXZhbGlkJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxLCAyLCAyLCBpbnB1dERlcHRoXTtcbiAgICBjb25zdCBmaWx0ZXJTaXplID0gMjtcbiAgICBjb25zdCBzdHJpZGU6IFtudW1iZXIsIG51bWJlcl0gPSBbMiwgMl07XG4gICAgY29uc3QgcGFkID0gJ3ZhbGlkJztcblxuICAgIGNvbnN0IGZpbHRlclNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgIFtmaWx0ZXJTaXplLCBmaWx0ZXJTaXplLCBvdXRwdXREZXB0aCwgaW5wdXREZXB0aF07XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtbW1stMC4zNjU0MTcxM10sIFstMC41Mzk3MzExNl1dLCBbWzAuMDE3MzE2NzRdLCBbMC45MDIyNzc3Ml1dXV0sXG4gICAgICAgIGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IGZpbHQgPSB0Zi50ZW5zb3I0ZChcbiAgICAgICAgW1xuICAgICAgICAgIFtbWy0wLjAxNDIzNDYxXSwgWy0xLjAwMjY3Mzg0XV0sIFtbMS42MTE2MzAyOV0sIFswLjY2MzAyNjQ2XV1dLFxuICAgICAgICAgIFtbWy0wLjQ2OTAwMDg3XSwgWy0wLjc4NjQ5NDQ0XV0sIFtbMC44Nzc4MDUzNl0sIFstMC44NDU1MTYzN11dXVxuICAgICAgICBdLFxuICAgICAgICBmaWx0ZXJTaGFwZSk7XG5cbiAgICBjb25zdCBncmFkcyA9IHRmLmdyYWRzKFxuICAgICAgICAoeDogdGYuVGVuc29yNEQsIGZpbHRlcjogdGYuVGVuc29yNEQpID0+XG4gICAgICAgICAgICB0Zi5jb252MmRUcmFuc3Bvc2UoeCwgZmlsdGVyLCBbMSwgNCwgNCwgb3V0cHV0RGVwdGhdLCBzdHJpZGUsIHBhZCkpO1xuICAgIGNvbnN0IGR5ID0gdGYub25lcyhbMSwgNCwgNCwgb3V0cHV0RGVwdGhdKS5tdWwodGYuc2NhbGFyKC0xKSk7XG4gICAgY29uc3QgW3hHcmFkLCBmaWx0R3JhZF0gPSBncmFkcyhbeCwgZmlsdF0sIGR5KTtcblxuICAgIGNvbnN0IGV4cGVjdGVkWEdyYWQgPSB0Zi5vbmVzKFsxLCAyLCAyLCAxXSkubXVsKHRmLnNjYWxhcigtMC4wMzQ1NDE5NikpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHhHcmFkLmRhdGEoKSwgYXdhaXQgZXhwZWN0ZWRYR3JhZC5kYXRhKCkpO1xuICAgIGV4cGVjdCh4R3JhZC5zaGFwZSkudG9FcXVhbChbMSwgMiwgMiwgMV0pO1xuXG4gICAgY29uc3QgZXhwZWN0ZWRGaWx0R3JhZCA9IHRmLm9uZXMoWzIsIDIsIDIsIDFdKS5tdWwodGYuc2NhbGFyKC0wLjAxNDQ0NjE4KSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgZmlsdEdyYWQuZGF0YSgpLCBhd2FpdCBleHBlY3RlZEZpbHRHcmFkLmRhdGEoKSk7XG4gICAgZXhwZWN0KGZpbHRHcmFkLnNoYXBlKS50b0VxdWFsKFsyLCAyLCAyLCAxXSk7XG4gIH0pO1xuXG4gIC8vIFJlZmVyZW5jZSAoUHl0aG9uKSBUZW5zb3JGbG93IGNvZGU6XG4gIC8vXG4gIC8vIGBgYHB5XG4gIC8vIGltcG9ydCBudW1weSBhcyBucFxuICAvLyBpbXBvcnQgdGVuc29yZmxvdyBhcyB0ZlxuICAvL1xuICAvLyB0Zi5lbmFibGVfZWFnZXJfZXhlY3V0aW9uKClcbiAgLy9cbiAgLy8geCA9IHRmLmNvbnN0YW50KG5wLmFycmF5KFtbXG4gIC8vICAgICBbWzEuNTI0MzMwNjVdLCBbLTAuNzcwNTM0MzVdLCBbLTAuNjQ1NjIzNDFdXSxcbiAgLy8gICAgIFtbMC43Nzk2Mjg4OV0sIFsxLjU4NDEzODg3XSwgWy0wLjI1NTgxODU2XV0sXG4gIC8vICAgICBbWy0wLjU4OTY2MjIxXSwgWzAuMDU0MTE2NjJdLCBbMC43MDc0OTEzOF1dXG4gIC8vIF1dKS5hc3R5cGUobnAuZmxvYXQzMikpXG4gIC8vIGZpbHQgPSB0Zi5jb25zdGFudChucC5hcnJheShbXG4gIC8vICAgICBbW1swLjExMTc4Mzg4XSwgWy0wLjk2NjU0OTc3XV0sIFtbMS4yMTAyMTI5Nl0sIFswLjg0MTIxNzI5XV1dLFxuICAvLyAgICAgW1tbMC4zNDk2ODMzOF0sIFstMC40MjMwNjExNF1dLCBbWzEuMjczOTU3MzNdLCBbLTEuMDkwMTQ1MzVdXV1cbiAgLy8gXSkuYXN0eXBlKG5wLmZsb2F0MzIpKVxuICAvL1xuICAvLyB3aXRoIHRmLkdyYWRpZW50VGFwZSgpIGFzIGc6XG4gIC8vICAgZy53YXRjaCh4KVxuICAvLyAgIGcud2F0Y2goZmlsdClcbiAgLy8gICB5ID0gdGYua2VyYXMuYmFja2VuZC5jb252MmRfdHJhbnNwb3NlKFxuICAvLyAgICAgICB4LCBmaWx0LCBbMSwgMywgMywgMl0sIHN0cmlkZXM9KDEsIDEpLCBwYWRkaW5nPSdzYW1lJylcbiAgLy8gKHhfZ3JhZCwgZmlsdF9ncmFkKSA9IGcuZ3JhZGllbnQoeSwgW3gsIGZpbHRdKVxuICAvL1xuICAvLyBwcmludChcInhfZ3JhZCA9ICVzXCIgJSB4X2dyYWQpXG4gIC8vIHByaW50KFwiZmlsdF9ncmFkID0gJXNcIiAlIGZpbHRfZ3JhZClcbiAgLy8gYGBgXG4gIGl0KCdncmFkaWVudCBpbnB1dD1bMSwzLDMsMV0gZj1bMiwyLDIsMV0gcz1bMSwxXSBwYWRkaW5nPXNhbWUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEsIDMsIDMsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IGZpbHRlclNpemUgPSAyO1xuICAgIGNvbnN0IHN0cmlkZTogW251bWJlciwgbnVtYmVyXSA9IFsxLCAxXTtcbiAgICBjb25zdCBwYWQgPSAnc2FtZSc7XG5cbiAgICBjb25zdCBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICBbZmlsdGVyU2l6ZSwgZmlsdGVyU2l6ZSwgb3V0cHV0RGVwdGgsIGlucHV0RGVwdGhdO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbW1xuICAgICAgICAgIFtbMS41MjQzMzA2NV0sIFstMC43NzA1MzQzNV0sIFstMC42NDU2MjM0MV1dLFxuICAgICAgICAgIFtbMC43Nzk2Mjg4OV0sIFsxLjU4NDEzODg3XSwgWy0wLjI1NTgxODU2XV0sXG4gICAgICAgICAgW1stMC41ODk2NjIyMV0sIFswLjA1NDExNjYyXSwgWzAuNzA3NDkxMzhdXVxuICAgICAgICBdXSxcbiAgICAgICAgaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgZmlsdCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbXG4gICAgICAgICAgW1tbMC4xMTE3ODM4OF0sIFstMC45NjY1NDk3N11dLCBbWzEuMjEwMjEyOTZdLCBbMC44NDEyMTcyOV1dXSxcbiAgICAgICAgICBbW1swLjM0OTY4MzM4XSwgWy0wLjQyMzA2MTE0XV0sIFtbMS4yNzM5NTczM10sIFstMS4wOTAxNDUzNV1dXVxuICAgICAgICBdLFxuICAgICAgICBmaWx0ZXJTaGFwZSk7XG5cbiAgICBjb25zdCBncmFkcyA9IHRmLmdyYWRzKFxuICAgICAgICAoeDogdGYuVGVuc29yNEQsIGZpbHRlcjogdGYuVGVuc29yNEQpID0+XG4gICAgICAgICAgICB0Zi5jb252MmRUcmFuc3Bvc2UoeCwgZmlsdGVyLCBbMSwgMywgMywgb3V0cHV0RGVwdGhdLCBzdHJpZGUsIHBhZCkpO1xuICAgIGNvbnN0IGR5ID0gdGYub25lcyhbMSwgMywgMywgb3V0cHV0RGVwdGhdKTtcbiAgICBjb25zdCBbeEdyYWQsIGZpbHRHcmFkXSA9IGdyYWRzKFt4LCBmaWx0XSwgZHkpO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgeEdyYWQuYXJyYXkoKSwgW1tcbiAgICAgICAgICAgICAgICAgICAgICAgIFtbMS4zMDcwOTg1OF0sIFsxLjMwNzA5ODU4XSwgWy0wLjkyODE0MzY2XV0sXG4gICAgICAgICAgICAgICAgICAgICAgICBbWzEuMzA3MDk4NThdLCBbMS4zMDcwOTg1OF0sIFstMC45MjgxNDM2Nl1dLFxuICAgICAgICAgICAgICAgICAgICAgICAgW1sxLjE5NjY2NDM3XSwgWzEuMTk2NjY0MzddLCBbLTAuODU0NzY1ODldXVxuICAgICAgICAgICAgICAgICAgICAgIF1dKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBmaWx0R3JhZC5hcnJheSgpLCBbXG4gICAgICBbW1syLjM4ODA2Nzg4XSwgWzIuMzg4MDY3ODhdXSwgW1syLjU4MjAxODQ3XSwgWzIuNTgyMDE4NDddXV0sXG4gICAgICBbW1syLjIxNjEyMjFdLCBbMi4yMTYxMjIxXV0sIFtbMy4xMTc1NjQwNl0sIFszLjExNzU2NDA2XV1dXG4gICAgXSk7XG4gIH0pO1xuXG4gIGl0KCdncmFkaWVudCBpbnB1dD1bMSwzLDMsMV0gZj1bMiwyLDIsMV0gcz1bMSwxXSBwPWV4cGxpY2l0JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxLCAzLCAzLCBpbnB1dERlcHRoXTtcbiAgICBjb25zdCBmaWx0ZXJTaXplID0gMjtcbiAgICBjb25zdCBzdHJpZGU6IFtudW1iZXIsIG51bWJlcl0gPSBbMSwgMV07XG4gICAgY29uc3QgcGFkID1cbiAgICAgICAgW1swLCAwXSwgWzAsIDFdLCBbMCwgMV0sIFswLCAwXV0gYXMgdGYuYmFja2VuZF91dGlsLkV4cGxpY2l0UGFkZGluZztcblxuICAgIGNvbnN0IGZpbHRlclNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICAgIFtmaWx0ZXJTaXplLCBmaWx0ZXJTaXplLCBvdXRwdXREZXB0aCwgaW5wdXREZXB0aF07XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtbXG4gICAgICAgICAgW1sxLjUyNDMzMDY1XSwgWy0wLjc3MDUzNDM1XSwgWy0wLjY0NTYyMzQxXV0sXG4gICAgICAgICAgW1swLjc3OTYyODg5XSwgWzEuNTg0MTM4ODddLCBbLTAuMjU1ODE4NTZdXSxcbiAgICAgICAgICBbWy0wLjU4OTY2MjIxXSwgWzAuMDU0MTE2NjJdLCBbMC43MDc0OTEzOF1dXG4gICAgICAgIF1dLFxuICAgICAgICBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBmaWx0ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICBbW1swLjExMTc4Mzg4XSwgWy0wLjk2NjU0OTc3XV0sIFtbMS4yMTAyMTI5Nl0sIFswLjg0MTIxNzI5XV1dLFxuICAgICAgICAgIFtbWzAuMzQ5NjgzMzhdLCBbLTAuNDIzMDYxMTRdXSwgW1sxLjI3Mzk1NzMzXSwgWy0xLjA5MDE0NTM1XV1dXG4gICAgICAgIF0sXG4gICAgICAgIGZpbHRlclNoYXBlKTtcblxuICAgIGNvbnN0IGdyYWRzID0gdGYuZ3JhZHMoXG4gICAgICAgICh4OiB0Zi5UZW5zb3I0RCwgZmlsdGVyOiB0Zi5UZW5zb3I0RCkgPT5cbiAgICAgICAgICAgIHRmLmNvbnYyZFRyYW5zcG9zZSh4LCBmaWx0ZXIsIFsxLCAzLCAzLCBvdXRwdXREZXB0aF0sIHN0cmlkZSwgcGFkKSk7XG4gICAgY29uc3QgZHkgPSB0Zi5vbmVzKFsxLCAzLCAzLCBvdXRwdXREZXB0aF0pO1xuICAgIGNvbnN0IFt4R3JhZCwgZmlsdEdyYWRdID0gZ3JhZHMoW3gsIGZpbHRdLCBkeSk7XG5cbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB4R3JhZC5hcnJheSgpLCBbW1xuICAgICAgICAgICAgICAgICAgICAgICAgW1sxLjMwNzA5ODU4XSwgWzEuMzA3MDk4NThdLCBbLTAuOTI4MTQzNjZdXSxcbiAgICAgICAgICAgICAgICAgICAgICAgIFtbMS4zMDcwOTg1OF0sIFsxLjMwNzA5ODU4XSwgWy0wLjkyODE0MzY2XV0sXG4gICAgICAgICAgICAgICAgICAgICAgICBbWzEuMTk2NjY0MzddLCBbMS4xOTY2NjQzN10sIFstMC44NTQ3NjU4OV1dXG4gICAgICAgICAgICAgICAgICAgICAgXV0pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGZpbHRHcmFkLmFycmF5KCksIFtcbiAgICAgIFtbWzIuMzg4MDY3ODhdLCBbMi4zODgwNjc4OF1dLCBbWzIuNTgyMDE4NDddLCBbMi41ODIwMTg0N11dXSxcbiAgICAgIFtbWzIuMjE2MTIyMV0sIFsyLjIxNjEyMjFdXSwgW1szLjExNzU2NDA2XSwgWzMuMTE3NTY0MDZdXV1cbiAgICBdKTtcbiAgfSk7XG5cbiAgLy8gUmVmZXJlbmNlIChQeXRob24pIFRlbnNvckZsb3cgY29kZTpcbiAgLy9cbiAgLy8gYGBgcHlcbiAgLy8gaW1wb3J0IG51bXB5IGFzIG5wXG4gIC8vIGltcG9ydCB0ZW5zb3JmbG93IGFzIHRmXG4gIC8vXG4gIC8vIHRmLmVuYWJsZV9lYWdlcl9leGVjdXRpb24oKVxuICAvL1xuICAvLyB4ID0gdGYuY29uc3RhbnQobnAuYXJyYXkoW1tcbiAgLy8gICAgIFtbMS41MjQzMzA2NV0sIFstMC43NzA1MzQzNV1dLCBbWzAuNzc5NjI4ODldLCBbMS41ODQxMzg4N11dLFxuICAvLyBdXSkuYXN0eXBlKG5wLmZsb2F0MzIpKVxuICAvLyBmaWx0ID0gdGYuY29uc3RhbnQobnAuYXJyYXkoW1xuICAvLyAgICAgW1tbMC4xMTE3ODM4OF0sIFstMC45NjY1NDk3N11dLCBbWzEuMjEwMjEyOTZdLCBbMC44NDEyMTcyOV1dXSxcbiAgLy8gICAgIFtbWzAuMzQ5NjgzMzhdLCBbLTAuNDIzMDYxMTRdXSwgW1sxLjI3Mzk1NzMzXSwgWy0xLjA5MDE0NTM1XV1dXG4gIC8vIF0pLmFzdHlwZShucC5mbG9hdDMyKSlcbiAgLy9cbiAgLy8gd2l0aCB0Zi5HcmFkaWVudFRhcGUoKSBhcyBnOlxuICAvLyAgIGcud2F0Y2goeClcbiAgLy8gICBnLndhdGNoKGZpbHQpXG4gIC8vICAgeSA9IHRmLmtlcmFzLmJhY2tlbmQuY29udjJkX3RyYW5zcG9zZShcbiAgLy8gICAgICAgeCwgZmlsdCwgWzEsIDMsIDMsIDJdLCBzdHJpZGVzPSgyLCAyKSwgcGFkZGluZz0nc2FtZScpXG4gIC8vICAgcHJpbnQoeS5zaGFwZSlcbiAgLy8gKHhfZ3JhZCwgZmlsdF9ncmFkKSA9IGcuZ3JhZGllbnQoeSwgW3gsIGZpbHRdKVxuICAvL1xuICAvLyBwcmludChcInhfZ3JhZCA9ICVzXCIgJSB4X2dyYWQpXG4gIC8vIHByaW50KFwiZmlsdF9ncmFkID0gJXNcIiAlIGZpbHRfZ3JhZClcbiAgLy8gYGBgXG4gIGl0KCdncmFkaWVudCBpbnB1dD1bMSwyLDIsMl0gZj1bMiwyLDIsMV0gcz1bMiwyXSBwYWRkaW5nPXNhbWUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IDI7XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSAyO1xuICAgIGNvbnN0IGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEsIDIsIDIsIGlucHV0RGVwdGhdO1xuICAgIGNvbnN0IGZpbHRlclNpemUgPSAyO1xuICAgIGNvbnN0IHN0cmlkZTogW251bWJlciwgbnVtYmVyXSA9IFsyLCAyXTtcbiAgICBjb25zdCBwYWQgPSAnc2FtZSc7XG5cbiAgICBjb25zdCBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICBbZmlsdGVyU2l6ZSwgZmlsdGVyU2l6ZSwgb3V0cHV0RGVwdGgsIGlucHV0RGVwdGhdO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjRkKFxuICAgICAgICBbW1xuICAgICAgICAgIFtbLTEuODE1MDY1OTMsIDEuMDA5MDAwOTVdLCBbLTAuMDUxOTkxMTgsIDAuMjYzMTEzNzddXSxcbiAgICAgICAgICBbWy0xLjE4NDY5NzkyLCAtMC4zNDc4MDUyMV0sIFsyLjA0OTcxMjQyLCAtMC42NTE1NDY5Ml1dXG4gICAgICAgIF1dLFxuICAgICAgICBpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBmaWx0ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFtcbiAgICAgICAgICBbXG4gICAgICAgICAgICBbWzAuMTk1Mjk2ODYsIC0wLjc5NTk0NzA4XSwgWzAuNzAzMTQwNTcsIC0wLjA2MDgxMjYzXV0sXG4gICAgICAgICAgICBbWzAuMjg3MjQ3NDQsIDAuODg1MjI3MTVdLCBbLTAuNTE4MjQwOTYsIC0wLjk3MTIwOTg5XV1cbiAgICAgICAgICBdLFxuICAgICAgICAgIFtcbiAgICAgICAgICAgIFtbMC41MTg3MjE5NywgLTEuMTc1NjkxOTNdLCBbMS4yODMxNjc5MSwgLTAuODEyMjUwOTJdXSxcbiAgICAgICAgICAgIFtbLTAuNDQyMjE1MzIsIDAuNzAwNTgxNzRdLCBbLTAuNDg0OTIxNywgMC4wMzgwNjM0OF1dXG4gICAgICAgICAgXVxuICAgICAgICBdLFxuICAgICAgICBmaWx0ZXJTaGFwZSk7XG5cbiAgICBjb25zdCBncmFkcyA9IHRmLmdyYWRzKFxuICAgICAgICAoeDogdGYuVGVuc29yNEQsIGZpbHRlcjogdGYuVGVuc29yNEQpID0+XG4gICAgICAgICAgICB0Zi5jb252MmRUcmFuc3Bvc2UoeCwgZmlsdGVyLCBbMSwgMywgMywgb3V0cHV0RGVwdGhdLCBzdHJpZGUsIHBhZCkpO1xuICAgIGNvbnN0IGR5ID0gdGYub25lcyhbMSwgMywgMywgb3V0cHV0RGVwdGhdKTtcbiAgICBjb25zdCBbeEdyYWQsIGZpbHRHcmFkXSA9IGdyYWRzKFt4LCBmaWx0XSwgZHkpO1xuXG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgeEdyYWQuZGF0YSgpLCBbXG4gICAgICAxLjU0MjE5Njc4LCAtMi4xOTIwNDAwOCwgMi43MDAzMjczMiwgLTIuODQ0NzAyNTcsIDAuNjY3NDQzOTEsIC0wLjk0Mjc0MjQ1LFxuICAgICAgMC44OTg0Mzc0MywgLTAuODU2NzU5NzJcbiAgICBdKTtcbiAgICBleHBlY3QoeEdyYWQuc2hhcGUpLnRvRXF1YWwoWzEsIDIsIDIsIDJdKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBmaWx0R3JhZC5kYXRhKCksIFtcbiAgICAgIC0xLjAwMjA0MjYxLCAwLjI3Mjc2MjU5LCAtMS4wMDIwNDI2MSwgMC4yNzI3NjI1OSwgLTIuOTk5NzYzODUsIDAuNjYxMTk1NzQsXG4gICAgICAtMi45OTk3NjM4NSwgMC42NjExOTU3NCwgLTEuODY3MDU3MTEsIDEuMjcyMTE0NzIsIC0xLjg2NzA1NzExLCAxLjI3MjExNDcyLFxuICAgICAgLTEuODE1MDY1OTMsIDEuMDA5MDAwOTUsIC0xLjgxNTA2NTkzLCAxLjAwOTAwMDk1XG4gICAgXSk7XG4gICAgZXhwZWN0KGZpbHRHcmFkLnNoYXBlKS50b0VxdWFsKFsyLCAyLCAyLCAyXSk7XG4gIH0pO1xuXG4gIGl0KCd0aHJvd3Mgd2hlbiB4IGlzIG5vdCByYW5rIDMnLCAoKSA9PiB7XG4gICAgY29uc3Qgb3JpZ0lucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG9yaWdPdXRwdXREZXB0aCA9IDE7XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IG9yaWdQYWQgPSAwO1xuICAgIGNvbnN0IG9yaWdTdHJpZGUgPSAxO1xuXG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIGNvbnN0IHg6IGFueSA9IHRmLnRlbnNvcjJkKFsyLCAyXSwgWzIsIDFdKTtcbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFszLCAxLCA1LCAwXSwgW2ZTaXplLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIG9yaWdPdXRwdXREZXB0aF0pO1xuXG4gICAgZXhwZWN0KCgpID0+IHRmLmNvbnYyZFRyYW5zcG9zZSh4LCB3LCBbMiwgMiwgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xuXG4gIGl0KCd0aHJvd3Mgd2hlbiB3ZWlnaHRzIGlzIG5vdCByYW5rIDQnLCAoKSA9PiB7XG4gICAgY29uc3Qgb3JpZ0lucHV0RGVwdGggPSAxO1xuICAgIGNvbnN0IG9yaWdPdXRwdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEsIDEsIG9yaWdPdXRwdXREZXB0aF07XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IG9yaWdQYWQgPSAwO1xuICAgIGNvbnN0IG9yaWdTdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjNkKFsyXSwgaW5wdXRTaGFwZSk7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIGNvbnN0IHc6IGFueSA9IHRmLnRlbnNvcjNkKFszLCAxLCA1LCAwXSwgW2ZTaXplLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGhdKTtcblxuICAgIGV4cGVjdCgoKSA9PiB0Zi5jb252MmRUcmFuc3Bvc2UoeCwgdywgWzIsIDIsIDFdLCBvcmlnU3RyaWRlLCBvcmlnUGFkKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4geCBkZXB0aCBkb2VzIG5vdCBtYXRjaCB3ZWlnaHRzIG9yaWdpbmFsIG91dHB1dCBkZXB0aCcsICgpID0+IHtcbiAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gMjtcbiAgICBjb25zdCB3cm9uZ09yaWdPdXRwdXREZXB0aCA9IDM7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEsIDEsIG9yaWdPdXRwdXREZXB0aF07XG4gICAgY29uc3QgZlNpemUgPSAyO1xuICAgIGNvbnN0IG9yaWdQYWQgPSAwO1xuICAgIGNvbnN0IG9yaWdTdHJpZGUgPSAxO1xuXG4gICAgY29uc3QgeCA9IHRmLnRlbnNvcjNkKFsyLCAyXSwgaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgdyA9IHRmLnJhbmRvbU5vcm1hbDxSYW5rLlI0PihcbiAgICAgICAgW2ZTaXplLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIHdyb25nT3JpZ091dHB1dERlcHRoXSk7XG5cbiAgICBleHBlY3QoKCkgPT4gdGYuY29udjJkVHJhbnNwb3NlKHgsIHcsIFsyLCAyLCAyXSwgb3JpZ1N0cmlkZSwgb3JpZ1BhZCkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyB3aGVuIHBhc3NlZCB4IGFzIGEgbm9uLXRlbnNvcicsICgpID0+IHtcbiAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gMTtcbiAgICBjb25zdCBmU2l6ZSA9IDI7XG4gICAgY29uc3Qgb3JpZ1BhZCA9IDA7XG4gICAgY29uc3Qgb3JpZ1N0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB3ID0gdGYudGVuc29yNGQoXG4gICAgICAgIFszLCAxLCA1LCAwXSwgW2ZTaXplLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIG9yaWdPdXRwdXREZXB0aF0pO1xuXG4gICAgZXhwZWN0KFxuICAgICAgICAoKSA9PiB0Zi5jb252MmRUcmFuc3Bvc2UoXG4gICAgICAgICAgICB7fSBhcyB0Zi5UZW5zb3IzRCwgdywgWzIsIDIsIDFdLCBvcmlnU3RyaWRlLCBvcmlnUGFkKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcihcbiAgICAgICAgICAgIC9Bcmd1bWVudCAneCcgcGFzc2VkIHRvICdjb252MmRUcmFuc3Bvc2UnIG11c3QgYmUgYSBUZW5zb3IvKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyB3aGVuIHBhc3NlZCBmaWx0ZXIgYXMgYSBub24tdGVuc29yJywgKCkgPT4ge1xuICAgIGNvbnN0IG9yaWdPdXRwdXREZXB0aCA9IDE7XG4gICAgY29uc3QgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEsIDEsIG9yaWdPdXRwdXREZXB0aF07XG4gICAgY29uc3Qgb3JpZ1BhZCA9IDA7XG4gICAgY29uc3Qgb3JpZ1N0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gdGYudGVuc29yM2QoWzJdLCBpbnB1dFNoYXBlKTtcblxuICAgIGV4cGVjdChcbiAgICAgICAgKCkgPT4gdGYuY29udjJkVHJhbnNwb3NlKFxuICAgICAgICAgICAgeCwge30gYXMgdGYuVGVuc29yNEQsIFsyLCAyLCAxXSwgb3JpZ1N0cmlkZSwgb3JpZ1BhZCkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoXG4gICAgICAgICAgICAvQXJndW1lbnQgJ2ZpbHRlcicgcGFzc2VkIHRvICdjb252MmRUcmFuc3Bvc2UnIG11c3QgYmUgYSBUZW5zb3IvKTtcbiAgfSk7XG5cbiAgaXQoJ2FjY2VwdHMgYSB0ZW5zb3ItbGlrZSBvYmplY3QnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgb3JpZ1BhZCA9IDA7XG4gICAgY29uc3Qgb3JpZ1N0cmlkZSA9IDE7XG5cbiAgICBjb25zdCB4ID0gW1tbMl1dXTsgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyAxeDF4MVxuICAgIGNvbnN0IHcgPSBbW1tbM11dLCBbWzFdXV0sIFtbWzVdXSwgW1swXV1dXTsgIC8vIDJ4MngxeDFcblxuICAgIGNvbnN0IHJlc3VsdCA9IHRmLmNvbnYyZFRyYW5zcG9zZSh4LCB3LCBbMiwgMiwgMV0sIG9yaWdTdHJpZGUsIG9yaWdQYWQpO1xuICAgIGNvbnN0IGV4cGVjdGVkID0gWzYsIDIsIDEwLCAwXTtcblxuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzIsIDIsIDFdKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBleHBlY3RlZCk7XG4gIH0pO1xufSk7XG4iXX0=
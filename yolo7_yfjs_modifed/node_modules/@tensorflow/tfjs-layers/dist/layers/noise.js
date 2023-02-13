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
 * TensorFlow.js Layers: Noise Layers.
 */
import { add, greaterEqual, mul, randomUniform, serialization, tidy } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { Layer } from '../engine/topology';
import { getExactlyOneTensor } from '../utils/types_utils';
export class GaussianNoise extends Layer {
    constructor(args) {
        super(args);
        this.supportsMasking = true;
        this.stddev = args.stddev;
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = { stddev: this.stddev };
        Object.assign(config, baseConfig);
        return config;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            const input = getExactlyOneTensor(inputs);
            const noised = () => add(K.randomNormal(input.shape, 0, this.stddev), input);
            const output = K.inTrainPhase(noised, () => input, kwargs['training'] || false);
            return output;
        });
    }
}
/** @nocollapse */
GaussianNoise.className = 'GaussianNoise';
serialization.registerClass(GaussianNoise);
export class GaussianDropout extends Layer {
    constructor(args) {
        super(args);
        this.supportsMasking = true;
        this.rate = args.rate;
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = { rate: this.rate };
        Object.assign(config, baseConfig);
        return config;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            const input = getExactlyOneTensor(inputs);
            if (this.rate > 0 && this.rate < 1) {
                const noised = () => {
                    const stddev = Math.sqrt(this.rate / (1 - this.rate));
                    return mul(input, K.randomNormal(input.shape, 1, stddev));
                };
                return K.inTrainPhase(noised, () => input, kwargs['training'] || false);
            }
            return input;
        });
    }
}
/** @nocollapse */
GaussianDropout.className = 'GaussianDropout';
serialization.registerClass(GaussianDropout);
/**
 * Applies Alpha Dropout to the input.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
 * to their original values, in order to ensure the self-normalizing property
 * even after this dropout.
 * Alpha Dropout fits well to Scaled Exponential Linear Units
 * by randomly setting activations to the negative saturation value.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *   - `noise_shape`: A 1-D `Tensor` of type `int32`, representing the
 *     shape for randomly generated keep/drop flags.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 */
export class AlphaDropout extends Layer {
    constructor(args) {
        super(args);
        this.supportsMasking = true;
        this.rate = args.rate;
        this.noiseShape = args.noiseShape;
    }
    _getNoiseShape(inputs) {
        return this.noiseShape || getExactlyOneTensor(inputs).shape;
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = { rate: this.rate };
        Object.assign(config, baseConfig);
        return config;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            if (this.rate < 1 && this.rate > 0) {
                const noiseShape = this._getNoiseShape(inputs);
                const droppedInputs = () => {
                    const input = getExactlyOneTensor(inputs);
                    const alpha = 1.6732632423543772848170429916717;
                    const scale = 1.0507009873554804934193349852946;
                    const alphaP = -alpha * scale;
                    let keptIdx = greaterEqual(randomUniform(noiseShape), this.rate);
                    keptIdx = K.cast(keptIdx, 'float32'); // get default dtype.
                    // Get affine transformation params.
                    const a = ((1 - this.rate) * (1 + this.rate * alphaP ** 2)) ** -0.5;
                    const b = -a * alphaP * this.rate;
                    // Apply mask.
                    const x = add(mul(input, keptIdx), mul(add(keptIdx, -1), alphaP));
                    return add(mul(x, a), b);
                };
                return K.inTrainPhase(droppedInputs, () => getExactlyOneTensor(inputs), kwargs['training'] || false);
            }
            return inputs;
        });
    }
}
/** @nocollapse */
AlphaDropout.className = 'AlphaDropout';
serialization.registerClass(AlphaDropout);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibm9pc2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvbGF5ZXJzL25vaXNlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUg7O0dBRUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUFFLFlBQVksRUFBRSxHQUFHLEVBQUUsYUFBYSxFQUFFLGFBQWEsRUFBVSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV6RyxPQUFPLEtBQUssQ0FBQyxNQUFNLHlCQUF5QixDQUFDO0FBQzdDLE9BQU8sRUFBQyxLQUFLLEVBQVksTUFBTSxvQkFBb0IsQ0FBQztBQUdwRCxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQU96RCxNQUFNLE9BQU8sYUFBYyxTQUFRLEtBQUs7SUFLdEMsWUFBWSxJQUF1QjtRQUNqQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDWixJQUFJLENBQUMsZUFBZSxHQUFHLElBQUksQ0FBQztRQUM1QixJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUM7SUFDNUIsQ0FBQztJQUVELGtCQUFrQixDQUFDLFVBQXlCO1FBQzFDLE9BQU8sVUFBVSxDQUFDO0lBQ3BCLENBQUM7SUFFRCxTQUFTO1FBQ1AsTUFBTSxVQUFVLEdBQUcsS0FBSyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ3JDLE1BQU0sTUFBTSxHQUFHLEVBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxNQUFNLEVBQUMsQ0FBQztRQUNyQyxNQUFNLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNsQyxPQUFPLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBRUQsSUFBSSxDQUFDLE1BQXVCLEVBQUUsTUFBYztRQUMxQyxPQUFPLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDZixJQUFJLENBQUMsY0FBYyxDQUFDLE1BQU0sRUFB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LEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxPQUFPLEVBQUUsU0FBUyxDQUFDLENBQUMsQ0FBRSxxQkFBcUI7b0JBRTVELG9DQUFvQztvQkFDcEMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLElBQUksR0FBRyxNQUFNLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQztvQkFDcEUsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUM7b0JBRWxDLGNBQWM7b0JBQ2QsTUFBTSxDQUFDLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsT0FBTyxDQUFDLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxPQUFPLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO29CQUVsRSxPQUFPLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO2dCQUMzQixDQUFDLENBQUM7Z0JBQ0YsT0FBTyxDQUFDLENBQUMsWUFBWSxDQUNqQixhQUFhLEVBQUUsR0FBRyxFQUFFLENBQUMsbUJBQW1CLENBQUMsTUFBTSxDQUFDLEVBQ2hELE1BQU0sQ0FBQyxVQUFVLENBQUMsSUFBSSxLQUFLLENBQUMsQ0FBQzthQUNsQztZQUNELE9BQU8sTUFBTSxDQUFDO1FBQ2hCLENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQzs7QUEzREQsa0JBQWtCO0FBQ1gsc0JBQVMsR0FBRyxjQUFjLENBQUM7QUE0RHBDLGFBQWEsQ0FBQyxhQUFhLENBQUMsWUFBWSxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcclxuICogQGxpY2Vuc2VcclxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xyXG4gKlxyXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcclxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XHJcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxyXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxyXG4gKi9cclxuXHJcbi8qKlxyXG4gKiBUZW5zb3JGbG93LmpzIExheWVyczogTm9pc2UgTGF5ZXJzLlxyXG4gKi9cclxuXHJcbmltcG9ydCB7YWRkLCBncmVhdGVyRXF1YWwsIG11bCwgcmFuZG9tVW5pZm9ybSwgc2VyaWFsaXphdGlvbiwgVGVuc29yLCB0aWR5fSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xyXG5cclxuaW1wb3J0ICogYXMgSyBmcm9tICcuLi9iYWNrZW5kL3RmanNfYmFja2VuZCc7XHJcbmltcG9ydCB7TGF5ZXIsIExheWVyQXJnc30gZnJvbSAnLi4vZW5naW5lL3RvcG9sb2d5JztcclxuaW1wb3J0IHtTaGFwZX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XHJcbmltcG9ydCB7S3dhcmdzfSBmcm9tICcuLi90eXBlcyc7XHJcbmltcG9ydCB7Z2V0RXhhY3RseU9uZVRlbnNvcn0gZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xyXG5cclxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIEdhdXNzaWFuTm9pc2VBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcclxuICAvKiogU3RhbmRhcmQgRGV2aWF0aW9uLiAgKi9cclxuICBzdGRkZXY6IG51bWJlcjtcclxufVxyXG5cclxuZXhwb3J0IGNsYXNzIEdhdXNzaWFuTm9pc2UgZXh0ZW5kcyBMYXllciB7XHJcbiAgLyoqIEBub2NvbGxhcHNlICovXHJcbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdHYXVzc2lhbk5vaXNlJztcclxuICByZWFkb25seSBzdGRkZXY6IG51bWJlcjtcclxuXHJcbiAgY29uc3RydWN0b3IoYXJnczogR2F1c3NpYW5Ob2lzZUFyZ3MpIHtcclxuICAgIHN1cGVyKGFyZ3MpO1xyXG4gICAgdGhpcy5zdXBwb3J0c01hc2tpbmcgPSB0cnVlO1xyXG4gICAgdGhpcy5zdGRkZXYgPSBhcmdzLnN0ZGRldjtcclxuICB9XHJcblxyXG4gIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XHJcbiAgICByZXR1cm4gaW5wdXRTaGFwZTtcclxuICB9XHJcblxyXG4gIGdldENvbmZpZygpIHtcclxuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcclxuICAgIGNvbnN0IGNvbmZpZyA9IHtzdGRkZXY6IHRoaXMuc3RkZGV2fTtcclxuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcclxuICAgIHJldHVybiBjb25maWc7XHJcbiAgfVxyXG5cclxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XHJcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XHJcbiAgICAgIHRoaXMuaW52b2tlQ2FsbEhvb2soaW5wdXRzLCBrd2FyZ3MpO1xyXG4gICAgICBjb25zdCBpbnB1dCA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKTtcclxuICAgICAgY29uc3Qgbm9pc2VkID0gKCkgPT5cclxuICAgICAgICAgIGFkZChLLnJhbmRvbU5vcm1hbChpbnB1dC5zaGFwZSwgMCwgdGhpcy5zdGRkZXYpLCBpbnB1dCk7XHJcbiAgICAgIGNvbnN0IG91dHB1dCA9XHJcbiAgICAgICAgICBLLmluVHJhaW5QaGFzZShub2lzZWQsICgpID0+IGlucHV0LCBrd2FyZ3NbJ3RyYWluaW5nJ10gfHwgZmFsc2UpO1xyXG4gICAgICByZXR1cm4gb3V0cHV0O1xyXG4gICAgfSk7XHJcbiAgfVxyXG59XHJcbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhHYXVzc2lhbk5vaXNlKTtcclxuXHJcbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBHYXVzc2lhbkRyb3BvdXRBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcclxuICAvKiogZHJvcCBwcm9iYWJpbGl0eS4gICovXHJcbiAgcmF0ZTogbnVtYmVyO1xyXG59XHJcblxyXG5leHBvcnQgY2xhc3MgR2F1c3NpYW5Ecm9wb3V0IGV4dGVuZHMgTGF5ZXIge1xyXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xyXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnR2F1c3NpYW5Ecm9wb3V0JztcclxuICByZWFkb25seSByYXRlOiBudW1iZXI7XHJcblxyXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IEdhdXNzaWFuRHJvcG91dEFyZ3MpIHtcclxuICAgIHN1cGVyKGFyZ3MpO1xyXG4gICAgdGhpcy5zdXBwb3J0c01hc2tpbmcgPSB0cnVlO1xyXG4gICAgdGhpcy5yYXRlID0gYXJncy5yYXRlO1xyXG4gIH1cclxuXHJcbiAgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcclxuICAgIHJldHVybiBpbnB1dFNoYXBlO1xyXG4gIH1cclxuXHJcbiAgZ2V0Q29uZmlnKCkge1xyXG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xyXG4gICAgY29uc3QgY29uZmlnID0ge3JhdGU6IHRoaXMucmF0ZX07XHJcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XHJcbiAgICByZXR1cm4gY29uZmlnO1xyXG4gIH1cclxuXHJcbiAgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xyXG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xyXG4gICAgICB0aGlzLmludm9rZUNhbGxIb29rKGlucHV0cywga3dhcmdzKTtcclxuICAgICAgY29uc3QgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XHJcbiAgICAgIGlmICh0aGlzLnJhdGUgPiAwICYmIHRoaXMucmF0ZSA8IDEpIHtcclxuICAgICAgICBjb25zdCBub2lzZWQgPSAoKSA9PiB7XHJcbiAgICAgICAgICBjb25zdCBzdGRkZXYgPSBNYXRoLnNxcnQodGhpcy5yYXRlIC8gKDEgLSB0aGlzLnJhdGUpKTtcclxuICAgICAgICAgIHJldHVybiBtdWwoaW5wdXQsIEsucmFuZG9tTm9ybWFsKGlucHV0LnNoYXBlLCAxLCBzdGRkZXYpKTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHJldHVybiBLLmluVHJhaW5QaGFzZShub2lzZWQsICgpID0+IGlucHV0LCBrd2FyZ3NbJ3RyYWluaW5nJ10gfHwgZmFsc2UpO1xyXG4gICAgICB9XHJcbiAgICAgIHJldHVybiBpbnB1dDtcclxuICAgIH0pO1xyXG4gIH1cclxufVxyXG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoR2F1c3NpYW5Ecm9wb3V0KTtcclxuXHJcbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBBbHBoYURyb3BvdXRBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcclxuICAvKiogZHJvcCBwcm9iYWJpbGl0eS4gICovXHJcbiAgcmF0ZTogbnVtYmVyO1xyXG4gIC8qKlxyXG4gICAqIEEgMS1EIGBUZW5zb3JgIG9mIHR5cGUgYGludDMyYCwgcmVwcmVzZW50aW5nIHRoZVxyXG4gICAqIHNoYXBlIGZvciByYW5kb21seSBnZW5lcmF0ZWQga2VlcC9kcm9wIGZsYWdzLlxyXG4gICAqL1xyXG4gIG5vaXNlU2hhcGU/OiBTaGFwZTtcclxufVxyXG5cclxuLyoqXHJcbiAqIEFwcGxpZXMgQWxwaGEgRHJvcG91dCB0byB0aGUgaW5wdXQuXHJcbiAqXHJcbiAqIEFzIGl0IGlzIGEgcmVndWxhcml6YXRpb24gbGF5ZXIsIGl0IGlzIG9ubHkgYWN0aXZlIGF0IHRyYWluaW5nIHRpbWUuXHJcbiAqXHJcbiAqIEFscGhhIERyb3BvdXQgaXMgYSBgRHJvcG91dGAgdGhhdCBrZWVwcyBtZWFuIGFuZCB2YXJpYW5jZSBvZiBpbnB1dHNcclxuICogdG8gdGhlaXIgb3JpZ2luYWwgdmFsdWVzLCBpbiBvcmRlciB0byBlbnN1cmUgdGhlIHNlbGYtbm9ybWFsaXppbmcgcHJvcGVydHlcclxuICogZXZlbiBhZnRlciB0aGlzIGRyb3BvdXQuXHJcbiAqIEFscGhhIERyb3BvdXQgZml0cyB3ZWxsIHRvIFNjYWxlZCBFeHBvbmVudGlhbCBMaW5lYXIgVW5pdHNcclxuICogYnkgcmFuZG9tbHkgc2V0dGluZyBhY3RpdmF0aW9ucyB0byB0aGUgbmVnYXRpdmUgc2F0dXJhdGlvbiB2YWx1ZS5cclxuICpcclxuICogQXJndW1lbnRzOlxyXG4gKiAgIC0gYHJhdGVgOiBmbG9hdCwgZHJvcCBwcm9iYWJpbGl0eSAoYXMgd2l0aCBgRHJvcG91dGApLlxyXG4gKiAgICAgVGhlIG11bHRpcGxpY2F0aXZlIG5vaXNlIHdpbGwgaGF2ZVxyXG4gKiAgICAgc3RhbmRhcmQgZGV2aWF0aW9uIGBzcXJ0KHJhdGUgLyAoMSAtIHJhdGUpKWAuXHJcbiAqICAgLSBgbm9pc2Vfc2hhcGVgOiBBIDEtRCBgVGVuc29yYCBvZiB0eXBlIGBpbnQzMmAsIHJlcHJlc2VudGluZyB0aGVcclxuICogICAgIHNoYXBlIGZvciByYW5kb21seSBnZW5lcmF0ZWQga2VlcC9kcm9wIGZsYWdzLlxyXG4gKlxyXG4gKiBJbnB1dCBzaGFwZTpcclxuICogICBBcmJpdHJhcnkuIFVzZSB0aGUga2V5d29yZCBhcmd1bWVudCBgaW5wdXRTaGFwZWBcclxuICogICAodHVwbGUgb2YgaW50ZWdlcnMsIGRvZXMgbm90IGluY2x1ZGUgdGhlIHNhbXBsZXMgYXhpcylcclxuICogICB3aGVuIHVzaW5nIHRoaXMgbGF5ZXIgYXMgdGhlIGZpcnN0IGxheWVyIGluIGEgbW9kZWwuXHJcbiAqXHJcbiAqIE91dHB1dCBzaGFwZTpcclxuICogICBTYW1lIHNoYXBlIGFzIGlucHV0LlxyXG4gKlxyXG4gKiBSZWZlcmVuY2VzOlxyXG4gKiAgIC0gW1NlbGYtTm9ybWFsaXppbmcgTmV1cmFsIE5ldHdvcmtzXShodHRwczovL2FyeGl2Lm9yZy9hYnMvMTcwNi4wMjUxNSlcclxuICovXHJcbmV4cG9ydCBjbGFzcyBBbHBoYURyb3BvdXQgZXh0ZW5kcyBMYXllciB7XHJcbiAgLyoqIEBub2NvbGxhcHNlICovXHJcbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdBbHBoYURyb3BvdXQnO1xyXG4gIHJlYWRvbmx5IHJhdGU6IG51bWJlcjtcclxuICByZWFkb25seSBub2lzZVNoYXBlOiBTaGFwZTtcclxuXHJcbiAgY29uc3RydWN0b3IoYXJnczogQWxwaGFEcm9wb3V0QXJncykge1xyXG4gICAgc3VwZXIoYXJncyk7XHJcbiAgICB0aGlzLnN1cHBvcnRzTWFza2luZyA9IHRydWU7XHJcbiAgICB0aGlzLnJhdGUgPSBhcmdzLnJhdGU7XHJcbiAgICB0aGlzLm5vaXNlU2hhcGUgPSBhcmdzLm5vaXNlU2hhcGU7XHJcbiAgfVxyXG5cclxuICBfZ2V0Tm9pc2VTaGFwZShpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSkge1xyXG4gICAgcmV0dXJuIHRoaXMubm9pc2VTaGFwZSB8fCBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cykuc2hhcGU7XHJcbiAgfVxyXG5cclxuICBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xyXG4gICAgcmV0dXJuIGlucHV0U2hhcGU7XHJcbiAgfVxyXG5cclxuICBnZXRDb25maWcoKSB7XHJcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XHJcbiAgICBjb25zdCBjb25maWcgPSB7cmF0ZTogdGhpcy5yYXRlfTtcclxuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcclxuICAgIHJldHVybiBjb25maWc7XHJcbiAgfVxyXG5cclxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XHJcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XHJcbiAgICAgIGlmICh0aGlzLnJhdGUgPCAxICYmIHRoaXMucmF0ZSA+IDApIHtcclxuICAgICAgICBjb25zdCBub2lzZVNoYXBlID0gdGhpcy5fZ2V0Tm9pc2VTaGFwZShpbnB1dHMpO1xyXG5cclxuICAgICAgICBjb25zdCBkcm9wcGVkSW5wdXRzID0gKCkgPT4ge1xyXG4gICAgICAgICAgY29uc3QgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XHJcblxyXG4gICAgICAgICAgY29uc3QgYWxwaGEgPSAxLjY3MzI2MzI0MjM1NDM3NzI4NDgxNzA0Mjk5MTY3MTc7XHJcbiAgICAgICAgICBjb25zdCBzY2FsZSA9IDEuMDUwNzAwOTg3MzU1NDgwNDkzNDE5MzM0OTg1Mjk0NjtcclxuXHJcbiAgICAgICAgICBjb25zdCBhbHBoYVAgPSAtYWxwaGEgKiBzY2FsZTtcclxuXHJcbiAgICAgICAgICBsZXQga2VwdElkeCA9IGdyZWF0ZXJFcXVhbChyYW5kb21Vbmlmb3JtKG5vaXNlU2hhcGUpLCB0aGlzLnJhdGUpO1xyXG5cclxuICAgICAgICAgIGtlcHRJZHggPSBLLmNhc3Qoa2VwdElkeCwgJ2Zsb2F0MzInKTsgIC8vIGdldCBkZWZhdWx0IGR0eXBlLlxyXG5cclxuICAgICAgICAgIC8vIEdldCBhZmZpbmUgdHJhbnNmb3JtYXRpb24gcGFyYW1zLlxyXG4gICAgICAgICAgY29uc3QgYSA9ICgoMSAtIHRoaXMucmF0ZSkgKiAoMSArIHRoaXMucmF0ZSAqIGFscGhhUCAqKiAyKSkgKiogLTAuNTtcclxuICAgICAgICAgIGNvbnN0IGIgPSAtYSAqIGFscGhhUCAqIHRoaXMucmF0ZTtcclxuXHJcbiAgICAgICAgICAvLyBBcHBseSBtYXNrLlxyXG4gICAgICAgICAgY29uc3QgeCA9IGFkZChtdWwoaW5wdXQsIGtlcHRJZHgpLCBtdWwoYWRkKGtlcHRJZHgsIC0xKSwgYWxwaGFQKSk7XHJcblxyXG4gICAgICAgICAgcmV0dXJuIGFkZChtdWwoeCwgYSksIGIpO1xyXG4gICAgICAgIH07XHJcbiAgICAgICAgcmV0dXJuIEsuaW5UcmFpblBoYXNlKFxyXG4gICAgICAgICAgICBkcm9wcGVkSW5wdXRzLCAoKSA9PiBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyksXHJcbiAgICAgICAgICAgIGt3YXJnc1sndHJhaW5pbmcnXSB8fCBmYWxzZSk7XHJcbiAgICAgIH1cclxuICAgICAgcmV0dXJuIGlucHV0cztcclxuICAgIH0pO1xyXG4gIH1cclxufVxyXG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQWxwaGFEcm9wb3V0KTtcclxuIl19
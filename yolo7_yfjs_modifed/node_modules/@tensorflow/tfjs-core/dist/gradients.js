/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { ENGINE } from './engine';
import { Tensor, Variable } from './tensor';
import { convertToTensor, convertToTensorArray } from './tensor_util_env';
import * as util from './util';
/**
 * Provided `f(x)`, returns another function `g(x, dy?)`, which gives the
 * gradient of `f(x)` with respect to `x`.
 *
 * If `dy` is provided, the gradient of `f(x).mul(dy).sum()` with respect to
 * `x` is computed instead. `f(x)` must take a single tensor `x` and return a
 * single tensor `y`. If `f()` takes multiple inputs, use `tf.grads` instead.
 *
 * ```js
 * // f(x) = x ^ 2
 * const f = x => x.square();
 * // f'(x) = 2x
 * const g = tf.grad(f);
 *
 * const x = tf.tensor1d([2, 3]);
 * g(x).print();
 * ```
 *
 * ```js
 * // f(x) = x ^ 3
 * const f = x => x.pow(tf.scalar(3, 'int32'));
 * // f'(x) = 3x ^ 2
 * const g = tf.grad(f);
 * // f''(x) = 6x
 * const gg = tf.grad(g);
 *
 * const x = tf.tensor1d([2, 3]);
 * gg(x).print();
 * ```
 *
 * @param f The function f(x), to compute gradient for.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function grad(f) {
    util.assert(util.isFunction(f), () => 'The f passed in grad(f) must be a function');
    return (x, dy) => {
        // x can be of any dtype, thus null as the last argument.
        const $x = convertToTensor(x, 'x', 'tf.grad', 'string_or_numeric');
        const $dy = (dy != null) ? convertToTensor(dy, 'dy', 'tf.grad') : null;
        return ENGINE.tidy(() => {
            const { value, grads } = ENGINE.gradients(() => f($x), [$x], $dy);
            if ($dy != null) {
                util.assertShapesMatch(value.shape, $dy.shape, 'The shape of dy passed in grad(f)(x, dy) must match the shape ' +
                    'returned by f(x)');
            }
            checkGrads(grads);
            return grads[0];
        });
    };
}
/**
 * Provided `f(x1, x2,...)`, returns another function `g([x1, x2,...], dy?)`,
 * which gives an array of gradients of `f()` with respect to each input
 * [`x1`,`x2`,...].
 *
 * If `dy` is passed when calling `g()`, the gradient of
 * `f(x1,...).mul(dy).sum()` with respect to each input is computed instead.
 * The provided `f` must take one or more tensors and return a single tensor
 * `y`. If `f()` takes a single input, we recommend using `tf.grad` instead.
 *
 * ```js
 * // f(a, b) = a * b
 * const f = (a, b) => a.mul(b);
 * // df / da = b, df / db = a
 * const g = tf.grads(f);
 *
 * const a = tf.tensor1d([2, 3]);
 * const b = tf.tensor1d([-2, -3]);
 * const [da, db] = g([a, b]);
 * console.log('da');
 * da.print();
 * console.log('db');
 * db.print();
 * ```
 *
 * @param f The function `f(x1, x2,...)` to compute gradients for.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function grads(f) {
    util.assert(util.isFunction(f), () => 'The f passed in grads(f) must be a function');
    return (args, dy) => {
        util.assert(Array.isArray(args), () => 'The args passed in grads(f)(args) must be an array ' +
            'of `Tensor`s or `TensorLike`s');
        // args can be of any dtype, thus null as the last argument.
        const $args = convertToTensorArray(args, 'args', 'tf.grads', 'string_or_numeric');
        const $dy = (dy != null) ? convertToTensor(dy, 'dy', 'tf.grads') : null;
        return ENGINE.tidy(() => {
            const { value, grads } = ENGINE.gradients(() => f(...$args), $args, $dy);
            if ($dy != null) {
                util.assertShapesMatch(value.shape, $dy.shape, 'The shape of dy passed in grads(f)([x1,...], dy) must ' +
                    'match the shape returned by f([x1,...])');
            }
            checkGrads(grads);
            return grads;
        });
    };
}
/**
 * Like `tf.grad`, but also returns the value of `f()`. Useful when `f()`
 * returns a metric you want to show.
 *
 * The result is a rich object with the following properties:
 * - grad: The gradient of `f(x)` w.r.t `x` (result of `tf.grad`).
 * - value: The value returned by `f(x)`.
 *
 * ```js
 * // f(x) = x ^ 2
 * const f = x => x.square();
 * // f'(x) = 2x
 * const g = tf.valueAndGrad(f);
 *
 * const x = tf.tensor1d([2, 3]);
 * const {value, grad} = g(x);
 *
 * console.log('value');
 * value.print();
 * console.log('grad');
 * grad.print();
 * ```
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function valueAndGrad(f) {
    util.assert(util.isFunction(f), () => 'The f passed in valueAndGrad(f) must be a function');
    return (x, dy) => {
        util.assert(x instanceof Tensor, () => 'The x passed in valueAndGrad(f)(x) must be a tensor');
        util.assert(dy == null || dy instanceof Tensor, () => 'The dy passed in valueAndGrad(f)(x, dy) must be a tensor');
        const { grads, value } = ENGINE.gradients(() => f(x), [x], dy);
        checkGrads(grads);
        return { grad: grads[0], value };
    };
}
/**
 * Like `tf.grads`, but returns also the value of `f()`. Useful when `f()`
 * returns a metric you want to show.
 *
 * The result is a rich object with the following properties:
 * - grads: The gradients of `f()` w.r.t each input (result of `tf.grads`).
 * - value: The value returned by `f(x)`.
 *
 * ```js
 * // f(a, b) = a * b
 * const f = (a, b) => a.mul(b);
 * // df/da = b, df/db = a
 * const g = tf.valueAndGrads(f);
 *
 * const a = tf.tensor1d([2, 3]);
 * const b = tf.tensor1d([-2, -3]);
 * const {value, grads} = g([a, b]);
 *
 * const [da, db] = grads;
 *
 * console.log('value');
 * value.print();
 *
 * console.log('da');
 * da.print();
 * console.log('db');
 * db.print();
 * ```
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function valueAndGrads(f) {
    util.assert(util.isFunction(f), () => 'The f passed in valueAndGrads(f) must be a function');
    return (args, dy) => {
        util.assert(Array.isArray(args) && args.every(arg => arg instanceof Tensor), () => 'The args passed in valueAndGrads(f)(args) must be array of ' +
            'tensors');
        util.assert(dy == null || dy instanceof Tensor, () => 'The dy passed in valueAndGrads(f)(args, dy) must be a tensor');
        const res = ENGINE.gradients(() => f(...args), args, dy);
        if (dy != null) {
            util.assertShapesMatch(res.value.shape, dy.shape, 'The shape of dy passed in valueAndGrads(f)([x1,...], dy) must ' +
                'match the shape returned by f([x1,...])');
        }
        checkGrads(res.grads);
        return res;
    };
}
/**
 * Computes and returns the gradient of f(x) with respect to the list of
 * trainable variables provided by `varList`. If no list is provided, it
 * defaults to all trainable variables.
 *
 * ```js
 * const a = tf.variable(tf.tensor1d([3, 4]));
 * const b = tf.variable(tf.tensor1d([5, 6]));
 * const x = tf.tensor1d([1, 2]);
 *
 * // f(a, b) = a * x ^ 2 + b * x
 * const f = () => a.mul(x.square()).add(b.mul(x)).sum();
 * // df/da = x ^ 2, df/db = x
 * const {value, grads} = tf.variableGrads(f);
 *
 * Object.keys(grads).forEach(varName => grads[varName].print());
 * ```
 *
 * @param f The function to execute. f() should return a scalar.
 * @param varList The list of variables to compute the gradients with respect
 *     to. Defaults to all trainable variables.
 * @returns An object with the following keys and values:
 *   - `value`: The value of the function `f`.
 *   - `grads`: A map from the names of the variables to the gradients.
 *     If the `varList` argument is provided explicitly and contains a subset of
 *     non-trainable variables, this map in the return value will contain keys
 *     that map the names of the non-trainable variables to `null`.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function variableGrads(f, varList) {
    util.assert(util.isFunction(f), () => 'The f passed in variableGrads(f) must be a function');
    util.assert(varList == null ||
        Array.isArray(varList) && varList.every(v => v instanceof Variable), () => 'The varList passed in variableGrads(f, varList) must be an array ' +
        'of variables');
    const specifiedVarList = varList != null;
    if (!specifiedVarList) {
        // Get all of the trainable variables.
        varList = [];
        for (const varName in ENGINE.registeredVariables) {
            varList.push(ENGINE.registeredVariables[varName]);
        }
    }
    const specifiedNonTrainable = specifiedVarList ? varList.filter(variable => !variable.trainable) : null;
    // Prune non-trainable variables.
    const originalVarCount = varList.length;
    varList = varList.filter(variable => variable.trainable);
    util.assert(varList.length > 0, () => `variableGrads() expects at least one of the input variables to ` +
        `be trainable, but none of the ${originalVarCount} variables is ` +
        `trainable.`);
    const allowNoGradients = true;
    const { value, grads } = ENGINE.gradients(f, varList, null, allowNoGradients);
    util.assert(grads.some(g => g != null), () => 'Cannot find a connection between any variable and the result of ' +
        'the loss function y=f(x). Please make sure the operations that ' +
        'use variables are inside the function f passed to minimize().');
    util.assert(value.rank === 0, () => `The f passed in variableGrads(f) must return a scalar, but it ` +
        `returned a rank-${value.rank} tensor`);
    const namedGrads = {};
    varList.forEach((v, i) => {
        if (grads[i] != null) {
            namedGrads[v.name] = grads[i];
        }
    });
    if (specifiedNonTrainable != null) {
        // If varList is explicitly provided and contains non-trainable values,
        // add them to the returned gradients with `null` values.
        specifiedNonTrainable.forEach(v => namedGrads[v.name] = null);
    }
    return { value, grads: namedGrads };
}
/**
 * Overrides the gradient computation of a function `f`.
 *
 * Takes a function
 * `f(...inputs, save) => {value: Tensor, gradFunc: (dy, saved) => Tensor[]}`
 * and returns another function `g(...inputs)` which takes the same inputs as
 * `f`. When called, `g` returns `f().value`. In backward mode, custom gradients
 * with respect to each input of `f` are computed using `f().gradFunc`.
 *
 * The `save` function passsed to `f` should be used for saving tensors needed
 * in the gradient. And the `saved` passed to the `gradFunc` is a
 * `NamedTensorMap`, which contains those saved tensor.
 *
 * ```js
 * const customOp = tf.customGrad((x, save) => {
 *   // Save x to make sure it's available later for the gradient.
 *   save([x]);
 *   // Override gradient of our custom x ^ 2 op to be dy * abs(x);
 *   return {
 *     value: x.square(),
 *     // Note `saved.x` which points to the `x` we saved earlier.
 *     gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
 *   };
 * });
 *
 * const x = tf.tensor1d([-1, -2, 3]);
 * const dx = tf.grad(x => customOp(x));
 *
 * console.log(`f(x):`);
 * customOp(x).print();
 * console.log(`f'(x):`);
 * dx(x).print();
 * ```
 *
 * @param f The function to evaluate in forward mode, which should return
 *     `{value: Tensor, gradFunc: (dy, saved) => Tensor[]}`, where `gradFunc`
 *     returns the custom gradients of `f` with respect to its inputs.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function customGrad(f) {
    return ENGINE.customGrad(f);
}
function checkGrads(grads) {
    const numNullGradients = grads.filter(g => g == null).length;
    if (numNullGradients > 0) {
        throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`);
    }
}
export { customGrad, variableGrads, valueAndGrad, valueAndGrads, grad, grads, };
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3JhZGllbnRzLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9ncmFkaWVudHMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFxQixNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDcEQsT0FBTyxFQUFTLE1BQU0sRUFBRSxRQUFRLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFFbEQsT0FBTyxFQUFDLGVBQWUsRUFBRSxvQkFBb0IsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBRXhFLE9BQU8sS0FBSyxJQUFJLE1BQU0sUUFBUSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FpQ0c7QUFDSCxTQUFTLElBQUksQ0FBQyxDQUF3QjtJQUVwQyxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUMsNENBQTRDLENBQUMsQ0FBQztJQUM1RSxPQUFPLENBQUMsQ0FBb0IsRUFBRSxFQUFzQixFQUFVLEVBQUU7UUFDOUQseURBQXlEO1FBQ3pELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFNBQVMsRUFBRSxtQkFBbUIsQ0FBQyxDQUFDO1FBQ25FLE1BQU0sR0FBRyxHQUNMLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxlQUFlLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDO1FBQy9ELE9BQU8sTUFBTSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDdEIsTUFBTSxFQUFDLEtBQUssRUFBRSxLQUFLLEVBQUMsR0FBRyxNQUFNLENBQUMsU0FBUyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFB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LEVBQUMsQ0FBQztBQUNwQyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXVDRztBQUNILFNBQVMsVUFBVSxDQUFtQixDQUF3QjtJQUU1RCxPQUFPLE1BQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDOUIsQ0FBQztBQUVELFNBQVMsVUFBVSxDQUFDLEtBQWU7SUFDakMsTUFBTSxnQkFBZ0IsR0FBRyxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxDQUFDLE1BQU0sQ0FBQztJQUM3RCxJQUFJLGdCQUFnQixHQUFHLENBQUMsRUFBRTtRQUN4QixNQUFNLElBQUksS0FBSyxDQUNYO29FQUM0RCxDQUFDLENBQUM7S0FDbkU7QUFDSCxDQUFDO0FBRUQsT0FBTyxFQUNMLFVBQVUsRUFDVixhQUFhLEVBQ2IsWUFBWSxFQUNaLGFBQWEsRUFDYixJQUFJLEVBQ0osS0FBSyxHQUNOLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Q3VzdG9tR3JhZGllbnRGdW5jLCBFTkdJTkV9IGZyb20gJy4vZW5naW5lJztcbmltcG9ydCB7U2NhbGFyLCBUZW5zb3IsIFZhcmlhYmxlfSBmcm9tICcuL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvciwgY29udmVydFRvVGVuc29yQXJyYXl9IGZyb20gJy4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4vdXRpbCc7XG5cbi8qKlxuICogUHJvdmlkZWQgYGYoeClgLCByZXR1cm5zIGFub3RoZXIgZnVuY3Rpb24gYGcoeCwgZHk/KWAsIHdoaWNoIGdpdmVzIHRoZVxuICogZ3JhZGllbnQgb2YgYGYoeClgIHdpdGggcmVzcGVjdCB0byBgeGAuXG4gKlxuICogSWYgYGR5YCBpcyBwcm92aWRlZCwgdGhlIGdyYWRpZW50IG9mIGBmKHgpLm11bChkeSkuc3VtKClgIHdpdGggcmVzcGVjdCB0b1xuICogYHhgIGlzIGNvbXB1dGVkIGluc3RlYWQuIGBmKHgpYCBtdXN0IHRha2UgYSBzaW5nbGUgdGVuc29yIGB4YCBhbmQgcmV0dXJuIGFcbiAqIHNpbmdsZSB0ZW5zb3IgYHlgLiBJZiBgZigpYCB0YWtlcyBtdWx0aXBsZSBpbnB1dHMsIHVzZSBgdGYuZ3JhZHNgIGluc3RlYWQuXG4gKlxuICogYGBganNcbiAqIC8vIGYoeCkgPSB4IF4gMlxuICogY29uc3QgZiA9IHggPT4geC5zcXVhcmUoKTtcbiAqIC8vIGYnKHgpID0gMnhcbiAqIGNvbnN0IGcgPSB0Zi5ncmFkKGYpO1xuICpcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMiwgM10pO1xuICogZyh4KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogYGBganNcbiAqIC8vIGYoeCkgPSB4IF4gM1xuICogY29uc3QgZiA9IHggPT4geC5wb3codGYuc2NhbGFyKDMsICdpbnQzMicpKTtcbiAqIC8vIGYnKHgpID0gM3ggXiAyXG4gKiBjb25zdCBnID0gdGYuZ3JhZChmKTtcbiAqIC8vIGYnJyh4KSA9IDZ4XG4gKiBjb25zdCBnZyA9IHRmLmdyYWQoZyk7XG4gKlxuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsyLCAzXSk7XG4gKiBnZyh4KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIGYgVGhlIGZ1bmN0aW9uIGYoeCksIHRvIGNvbXB1dGUgZ3JhZGllbnQgZm9yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUcmFpbmluZycsIHN1YmhlYWRpbmc6ICdHcmFkaWVudHMnfVxuICovXG5mdW5jdGlvbiBncmFkKGY6ICh4OiBUZW5zb3IpID0+IFRlbnNvcik6IChcbiAgICB4OiBUZW5zb3JMaWtlfFRlbnNvciwgZHk/OiBUZW5zb3JMaWtlfFRlbnNvcikgPT4gVGVuc29yIHtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB1dGlsLmlzRnVuY3Rpb24oZiksICgpID0+ICdUaGUgZiBwYXNzZWQgaW4gZ3JhZChmKSBtdXN0IGJlIGEgZnVuY3Rpb24nKTtcbiAgcmV0dXJuICh4OiBUZW5zb3JMaWtlfFRlbnNvciwgZHk/OiBUZW5zb3JMaWtlfFRlbnNvcik6IFRlbnNvciA9PiB7XG4gICAgLy8geCBjYW4gYmUgb2YgYW55IGR0eXBlLCB0aHVzIG51bGwgYXMgdGhlIGxhc3QgYXJndW1lbnQuXG4gICAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAndGYuZ3JhZCcsICdzdHJpbmdfb3JfbnVtZXJpYycpO1xuICAgIGNvbnN0ICRkeTogVGVuc29yID1cbiAgICAgICAgKGR5ICE9IG51bGwpID8gY29udmVydFRvVGVuc29yKGR5LCAnZHknLCAndGYuZ3JhZCcpIDogbnVsbDtcbiAgICByZXR1cm4gRU5HSU5FLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3Qge3ZhbHVlLCBncmFkc30gPSBFTkdJTkUuZ3JhZGllbnRzKCgpID0+IGYoJHgpLCBbJHhdLCAkZHkpO1xuICAgICAgaWYgKCRkeSAhPSBudWxsKSB7XG4gICAgICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgICAgICAgICB2YWx1ZS5zaGFwZSwgJGR5LnNoYXBlLFxuICAgICAgICAgICAgJ1RoZSBzaGFwZSBvZiBkeSBwYXNzZWQgaW4gZ3JhZChmKSh4LCBkeSkgbXVzdCBtYXRjaCB0aGUgc2hhcGUgJyArXG4gICAgICAgICAgICAgICAgJ3JldHVybmVkIGJ5IGYoeCknKTtcbiAgICAgIH1cbiAgICAgIGNoZWNrR3JhZHMoZ3JhZHMpO1xuICAgICAgcmV0dXJuIGdyYWRzWzBdO1xuICAgIH0pO1xuICB9O1xufVxuXG4vKipcbiAqIFByb3ZpZGVkIGBmKHgxLCB4MiwuLi4pYCwgcmV0dXJucyBhbm90aGVyIGZ1bmN0aW9uIGBnKFt4MSwgeDIsLi4uXSwgZHk/KWAsXG4gKiB3aGljaCBnaXZlcyBhbiBhcnJheSBvZiBncmFkaWVudHMgb2YgYGYoKWAgd2l0aCByZXNwZWN0IHRvIGVhY2ggaW5wdXRcbiAqIFtgeDFgLGB4MmAsLi4uXS5cbiAqXG4gKiBJZiBgZHlgIGlzIHBhc3NlZCB3aGVuIGNhbGxpbmcgYGcoKWAsIHRoZSBncmFkaWVudCBvZlxuICogYGYoeDEsLi4uKS5tdWwoZHkpLnN1bSgpYCB3aXRoIHJlc3BlY3QgdG8gZWFjaCBpbnB1dCBpcyBjb21wdXRlZCBpbnN0ZWFkLlxuICogVGhlIHByb3ZpZGVkIGBmYCBtdXN0IHRha2Ugb25lIG9yIG1vcmUgdGVuc29ycyBhbmQgcmV0dXJuIGEgc2luZ2xlIHRlbnNvclxuICogYHlgLiBJZiBgZigpYCB0YWtlcyBhIHNpbmdsZSBpbnB1dCwgd2UgcmVjb21tZW5kIHVzaW5nIGB0Zi5ncmFkYCBpbnN0ZWFkLlxuICpcbiAqIGBgYGpzXG4gKiAvLyBmKGEsIGIpID0gYSAqIGJcbiAqIGNvbnN0IGYgPSAoYSwgYikgPT4gYS5tdWwoYik7XG4gKiAvLyBkZiAvIGRhID0gYiwgZGYgLyBkYiA9IGFcbiAqIGNvbnN0IGcgPSB0Zi5ncmFkcyhmKTtcbiAqXG4gKiBjb25zdCBhID0gdGYudGVuc29yMWQoWzIsIDNdKTtcbiAqIGNvbnN0IGIgPSB0Zi50ZW5zb3IxZChbLTIsIC0zXSk7XG4gKiBjb25zdCBbZGEsIGRiXSA9IGcoW2EsIGJdKTtcbiAqIGNvbnNvbGUubG9nKCdkYScpO1xuICogZGEucHJpbnQoKTtcbiAqIGNvbnNvbGUubG9nKCdkYicpO1xuICogZGIucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBmIFRoZSBmdW5jdGlvbiBgZih4MSwgeDIsLi4uKWAgdG8gY29tcHV0ZSBncmFkaWVudHMgZm9yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUcmFpbmluZycsIHN1YmhlYWRpbmc6ICdHcmFkaWVudHMnfVxuICovXG5mdW5jdGlvbiBncmFkcyhmOiAoLi4uYXJnczogVGVuc29yW10pID0+IFRlbnNvcik6IChcbiAgICBhcmdzOiBBcnJheTxUZW5zb3J8VGVuc29yTGlrZT4sIGR5PzogVGVuc29yfFRlbnNvckxpa2UpID0+IFRlbnNvcltdIHtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB1dGlsLmlzRnVuY3Rpb24oZiksICgpID0+ICdUaGUgZiBwYXNzZWQgaW4gZ3JhZHMoZikgbXVzdCBiZSBhIGZ1bmN0aW9uJyk7XG4gIHJldHVybiAoYXJnczogQXJyYXk8VGVuc29yfFRlbnNvckxpa2U+LCBkeT86IFRlbnNvcnxUZW5zb3JMaWtlKTogVGVuc29yW10gPT4ge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBBcnJheS5pc0FycmF5KGFyZ3MpLFxuICAgICAgICAoKSA9PiAnVGhlIGFyZ3MgcGFzc2VkIGluIGdyYWRzKGYpKGFyZ3MpIG11c3QgYmUgYW4gYXJyYXkgJyArXG4gICAgICAgICAgICAnb2YgYFRlbnNvcmBzIG9yIGBUZW5zb3JMaWtlYHMnKTtcbiAgICAvLyBhcmdzIGNhbiBiZSBvZiBhbnkgZHR5cGUsIHRodXMgbnVsbCBhcyB0aGUgbGFzdCBhcmd1bWVudC5cbiAgICBjb25zdCAkYXJncyA9XG4gICAgICAgIGNvbnZlcnRUb1RlbnNvckFycmF5KGFyZ3MsICdhcmdzJywgJ3RmLmdyYWRzJywgJ3N0cmluZ19vcl9udW1lcmljJyk7XG4gICAgY29uc3QgJGR5OiBUZW5zb3IgPVxuICAgICAgICAoZHkgIT0gbnVsbCkgPyBjb252ZXJ0VG9UZW5zb3IoZHksICdkeScsICd0Zi5ncmFkcycpIDogbnVsbDtcbiAgICByZXR1cm4gRU5HSU5FLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3Qge3ZhbHVlLCBncmFkc30gPSBFTkdJTkUuZ3JhZGllbnRzKCgpID0+IGYoLi4uJGFyZ3MpLCAkYXJncywgJGR5KTtcbiAgICAgIGlmICgkZHkgIT0gbnVsbCkge1xuICAgICAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKFxuICAgICAgICAgICAgdmFsdWUuc2hhcGUsICRkeS5zaGFwZSxcbiAgICAgICAgICAgICdUaGUgc2hhcGUgb2YgZHkgcGFzc2VkIGluIGdyYWRzKGYpKFt4MSwuLi5dLCBkeSkgbXVzdCAnICtcbiAgICAgICAgICAgICAgICAnbWF0Y2ggdGhlIHNoYXBlIHJldHVybmVkIGJ5IGYoW3gxLC4uLl0pJyk7XG4gICAgICB9XG4gICAgICBjaGVja0dyYWRzKGdyYWRzKTtcbiAgICAgIHJldHVybiBncmFkcztcbiAgICB9KTtcbiAgfTtcbn1cblxuLyoqXG4gKiBMaWtlIGB0Zi5ncmFkYCwgYnV0IGFsc28gcmV0dXJucyB0aGUgdmFsdWUgb2YgYGYoKWAuIFVzZWZ1bCB3aGVuIGBmKClgXG4gKiByZXR1cm5zIGEgbWV0cmljIHlvdSB3YW50IHRvIHNob3cuXG4gKlxuICogVGhlIHJlc3VsdCBpcyBhIHJpY2ggb2JqZWN0IHdpdGggdGhlIGZvbGxvd2luZyBwcm9wZXJ0aWVzOlxuICogLSBncmFkOiBUaGUgZ3JhZGllbnQgb2YgYGYoeClgIHcuci50IGB4YCAocmVzdWx0IG9mIGB0Zi5ncmFkYCkuXG4gKiAtIHZhbHVlOiBUaGUgdmFsdWUgcmV0dXJuZWQgYnkgYGYoeClgLlxuICpcbiAqIGBgYGpzXG4gKiAvLyBmKHgpID0geCBeIDJcbiAqIGNvbnN0IGYgPSB4ID0+IHguc3F1YXJlKCk7XG4gKiAvLyBmJyh4KSA9IDJ4XG4gKiBjb25zdCBnID0gdGYudmFsdWVBbmRHcmFkKGYpO1xuICpcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMiwgM10pO1xuICogY29uc3Qge3ZhbHVlLCBncmFkfSA9IGcoeCk7XG4gKlxuICogY29uc29sZS5sb2coJ3ZhbHVlJyk7XG4gKiB2YWx1ZS5wcmludCgpO1xuICogY29uc29sZS5sb2coJ2dyYWQnKTtcbiAqIGdyYWQucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUcmFpbmluZycsIHN1YmhlYWRpbmc6ICdHcmFkaWVudHMnfVxuICovXG5mdW5jdGlvbiB2YWx1ZUFuZEdyYWQ8SSBleHRlbmRzIFRlbnNvciwgTyBleHRlbmRzIFRlbnNvcj4oZjogKHg6IEkpID0+IE8pOiAoXG4gICAgeDogSSwgZHk/OiBPKSA9PiB7XG4gIHZhbHVlOiBPO1xuICBncmFkOiBJO1xufSB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgdXRpbC5pc0Z1bmN0aW9uKGYpLFxuICAgICAgKCkgPT4gJ1RoZSBmIHBhc3NlZCBpbiB2YWx1ZUFuZEdyYWQoZikgbXVzdCBiZSBhIGZ1bmN0aW9uJyk7XG4gIHJldHVybiAoeDogSSwgZHk/OiBPKSA9PiB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHggaW5zdGFuY2VvZiBUZW5zb3IsXG4gICAgICAgICgpID0+ICdUaGUgeCBwYXNzZWQgaW4gdmFsdWVBbmRHcmFkKGYpKHgpIG11c3QgYmUgYSB0ZW5zb3InKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZHkgPT0gbnVsbCB8fCBkeSBpbnN0YW5jZW9mIFRlbnNvcixcbiAgICAgICAgKCkgPT4gJ1RoZSBkeSBwYXNzZWQgaW4gdmFsdWVBbmRHcmFkKGYpKHgsIGR5KSBtdXN0IGJlIGEgdGVuc29yJyk7XG4gICAgY29uc3Qge2dyYWRzLCB2YWx1ZX0gPSBFTkdJTkUuZ3JhZGllbnRzKCgpID0+IGYoeCksIFt4XSwgZHkpO1xuICAgIGNoZWNrR3JhZHMoZ3JhZHMpO1xuICAgIHJldHVybiB7Z3JhZDogZ3JhZHNbMF0gYXMgSSwgdmFsdWV9O1xuICB9O1xufVxuXG4vKipcbiAqIExpa2UgYHRmLmdyYWRzYCwgYnV0IHJldHVybnMgYWxzbyB0aGUgdmFsdWUgb2YgYGYoKWAuIFVzZWZ1bCB3aGVuIGBmKClgXG4gKiByZXR1cm5zIGEgbWV0cmljIHlvdSB3YW50IHRvIHNob3cuXG4gKlxuICogVGhlIHJlc3VsdCBpcyBhIHJpY2ggb2JqZWN0IHdpdGggdGhlIGZvbGxvd2luZyBwcm9wZXJ0aWVzOlxuICogLSBncmFkczogVGhlIGdyYWRpZW50cyBvZiBgZigpYCB3LnIudCBlYWNoIGlucHV0IChyZXN1bHQgb2YgYHRmLmdyYWRzYCkuXG4gKiAtIHZhbHVlOiBUaGUgdmFsdWUgcmV0dXJuZWQgYnkgYGYoeClgLlxuICpcbiAqIGBgYGpzXG4gKiAvLyBmKGEsIGIpID0gYSAqIGJcbiAqIGNvbnN0IGYgPSAoYSwgYikgPT4gYS5tdWwoYik7XG4gKiAvLyBkZi9kYSA9IGIsIGRmL2RiID0gYVxuICogY29uc3QgZyA9IHRmLnZhbHVlQW5kR3JhZHMoZik7XG4gKlxuICogY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsyLCAzXSk7XG4gKiBjb25zdCBiID0gdGYudGVuc29yMWQoWy0yLCAtM10pO1xuICogY29uc3Qge3ZhbHVlLCBncmFkc30gPSBnKFthLCBiXSk7XG4gKlxuICogY29uc3QgW2RhLCBkYl0gPSBncmFkcztcbiAqXG4gKiBjb25zb2xlLmxvZygndmFsdWUnKTtcbiAqIHZhbHVlLnByaW50KCk7XG4gKlxuICogY29uc29sZS5sb2coJ2RhJyk7XG4gKiBkYS5wcmludCgpO1xuICogY29uc29sZS5sb2coJ2RiJyk7XG4gKiBkYi5wcmludCgpO1xuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RyYWluaW5nJywgc3ViaGVhZGluZzogJ0dyYWRpZW50cyd9XG4gKi9cbmZ1bmN0aW9uIHZhbHVlQW5kR3JhZHM8TyBleHRlbmRzIFRlbnNvcj4oZjogKC4uLmFyZ3M6IFRlbnNvcltdKSA9PiBPKTogKFxuICAgIGFyZ3M6IFRlbnNvcltdLCBkeT86IE8pID0+IHtcbiAgZ3JhZHM6IFRlbnNvcltdO1xuICB2YWx1ZTogTztcbn0ge1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNGdW5jdGlvbihmKSxcbiAgICAgICgpID0+ICdUaGUgZiBwYXNzZWQgaW4gdmFsdWVBbmRHcmFkcyhmKSBtdXN0IGJlIGEgZnVuY3Rpb24nKTtcbiAgcmV0dXJuIChhcmdzOiBUZW5zb3JbXSwgZHk/OiBPKSA9PiB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIEFycmF5LmlzQXJyYXkoYXJncykgJiYgYXJncy5ldmVyeShhcmcgPT4gYXJnIGluc3RhbmNlb2YgVGVuc29yKSxcbiAgICAgICAgKCkgPT4gJ1RoZSBhcmdzIHBhc3NlZCBpbiB2YWx1ZUFuZEdyYWRzKGYpKGFyZ3MpIG11c3QgYmUgYXJyYXkgb2YgJyArXG4gICAgICAgICAgICAndGVuc29ycycpO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBkeSA9PSBudWxsIHx8IGR5IGluc3RhbmNlb2YgVGVuc29yLFxuICAgICAgICAoKSA9PiAnVGhlIGR5IHBhc3NlZCBpbiB2YWx1ZUFuZEdyYWRzKGYpKGFyZ3MsIGR5KSBtdXN0IGJlIGEgdGVuc29yJyk7XG4gICAgY29uc3QgcmVzID0gRU5HSU5FLmdyYWRpZW50cygoKSA9PiBmKC4uLmFyZ3MpLCBhcmdzLCBkeSk7XG4gICAgaWYgKGR5ICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgICAgICAgcmVzLnZhbHVlLnNoYXBlLCBkeS5zaGFwZSxcbiAgICAgICAgICAnVGhlIHNoYXBlIG9mIGR5IHBhc3NlZCBpbiB2YWx1ZUFuZEdyYWRzKGYpKFt4MSwuLi5dLCBkeSkgbXVzdCAnICtcbiAgICAgICAgICAgICAgJ21hdGNoIHRoZSBzaGFwZSByZXR1cm5lZCBieSBmKFt4MSwuLi5dKScpO1xuICAgIH1cbiAgICBjaGVja0dyYWRzKHJlcy5ncmFkcyk7XG4gICAgcmV0dXJuIHJlcztcbiAgfTtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyBhbmQgcmV0dXJucyB0aGUgZ3JhZGllbnQgb2YgZih4KSB3aXRoIHJlc3BlY3QgdG8gdGhlIGxpc3Qgb2ZcbiAqIHRyYWluYWJsZSB2YXJpYWJsZXMgcHJvdmlkZWQgYnkgYHZhckxpc3RgLiBJZiBubyBsaXN0IGlzIHByb3ZpZGVkLCBpdFxuICogZGVmYXVsdHMgdG8gYWxsIHRyYWluYWJsZSB2YXJpYWJsZXMuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi52YXJpYWJsZSh0Zi50ZW5zb3IxZChbMywgNF0pKTtcbiAqIGNvbnN0IGIgPSB0Zi52YXJpYWJsZSh0Zi50ZW5zb3IxZChbNSwgNl0pKTtcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMl0pO1xuICpcbiAqIC8vIGYoYSwgYikgPSBhICogeCBeIDIgKyBiICogeFxuICogY29uc3QgZiA9ICgpID0+IGEubXVsKHguc3F1YXJlKCkpLmFkZChiLm11bCh4KSkuc3VtKCk7XG4gKiAvLyBkZi9kYSA9IHggXiAyLCBkZi9kYiA9IHhcbiAqIGNvbnN0IHt2YWx1ZSwgZ3JhZHN9ID0gdGYudmFyaWFibGVHcmFkcyhmKTtcbiAqXG4gKiBPYmplY3Qua2V5cyhncmFkcykuZm9yRWFjaCh2YXJOYW1lID0+IGdyYWRzW3Zhck5hbWVdLnByaW50KCkpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIGYgVGhlIGZ1bmN0aW9uIHRvIGV4ZWN1dGUuIGYoKSBzaG91bGQgcmV0dXJuIGEgc2NhbGFyLlxuICogQHBhcmFtIHZhckxpc3QgVGhlIGxpc3Qgb2YgdmFyaWFibGVzIHRvIGNvbXB1dGUgdGhlIGdyYWRpZW50cyB3aXRoIHJlc3BlY3RcbiAqICAgICB0by4gRGVmYXVsdHMgdG8gYWxsIHRyYWluYWJsZSB2YXJpYWJsZXMuXG4gKiBAcmV0dXJucyBBbiBvYmplY3Qgd2l0aCB0aGUgZm9sbG93aW5nIGtleXMgYW5kIHZhbHVlczpcbiAqICAgLSBgdmFsdWVgOiBUaGUgdmFsdWUgb2YgdGhlIGZ1bmN0aW9uIGBmYC5cbiAqICAgLSBgZ3JhZHNgOiBBIG1hcCBmcm9tIHRoZSBuYW1lcyBvZiB0aGUgdmFyaWFibGVzIHRvIHRoZSBncmFkaWVudHMuXG4gKiAgICAgSWYgdGhlIGB2YXJMaXN0YCBhcmd1bWVudCBpcyBwcm92aWRlZCBleHBsaWNpdGx5IGFuZCBjb250YWlucyBhIHN1YnNldCBvZlxuICogICAgIG5vbi10cmFpbmFibGUgdmFyaWFibGVzLCB0aGlzIG1hcCBpbiB0aGUgcmV0dXJuIHZhbHVlIHdpbGwgY29udGFpbiBrZXlzXG4gKiAgICAgdGhhdCBtYXAgdGhlIG5hbWVzIG9mIHRoZSBub24tdHJhaW5hYmxlIHZhcmlhYmxlcyB0byBgbnVsbGAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RyYWluaW5nJywgc3ViaGVhZGluZzogJ0dyYWRpZW50cyd9XG4gKi9cbmZ1bmN0aW9uIHZhcmlhYmxlR3JhZHMoZjogKCkgPT4gU2NhbGFyLCB2YXJMaXN0PzogVmFyaWFibGVbXSk6XG4gICAge3ZhbHVlOiBTY2FsYXIsIGdyYWRzOiBOYW1lZFRlbnNvck1hcH0ge1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNGdW5jdGlvbihmKSxcbiAgICAgICgpID0+ICdUaGUgZiBwYXNzZWQgaW4gdmFyaWFibGVHcmFkcyhmKSBtdXN0IGJlIGEgZnVuY3Rpb24nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB2YXJMaXN0ID09IG51bGwgfHxcbiAgICAgICAgICBBcnJheS5pc0FycmF5KHZhckxpc3QpICYmIHZhckxpc3QuZXZlcnkodiA9PiB2IGluc3RhbmNlb2YgVmFyaWFibGUpLFxuICAgICAgKCkgPT5cbiAgICAgICAgICAnVGhlIHZhckxpc3QgcGFzc2VkIGluIHZhcmlhYmxlR3JhZHMoZiwgdmFyTGlzdCkgbXVzdCBiZSBhbiBhcnJheSAnICtcbiAgICAgICAgICAnb2YgdmFyaWFibGVzJyk7XG5cbiAgY29uc3Qgc3BlY2lmaWVkVmFyTGlzdCA9IHZhckxpc3QgIT0gbnVsbDtcbiAgaWYgKCFzcGVjaWZpZWRWYXJMaXN0KSB7XG4gICAgLy8gR2V0IGFsbCBvZiB0aGUgdHJhaW5hYmxlIHZhcmlhYmxlcy5cbiAgICB2YXJMaXN0ID0gW107XG4gICAgZm9yIChjb25zdCB2YXJOYW1lIGluIEVOR0lORS5yZWdpc3RlcmVkVmFyaWFibGVzKSB7XG4gICAgICB2YXJMaXN0LnB1c2goRU5HSU5FLnJlZ2lzdGVyZWRWYXJpYWJsZXNbdmFyTmFtZV0pO1xuICAgIH1cbiAgfVxuXG4gIGNvbnN0IHNwZWNpZmllZE5vblRyYWluYWJsZTogVmFyaWFibGVbXSA9XG4gICAgICBzcGVjaWZpZWRWYXJMaXN0ID8gdmFyTGlzdC5maWx0ZXIodmFyaWFibGUgPT4gIXZhcmlhYmxlLnRyYWluYWJsZSkgOiBudWxsO1xuXG4gIC8vIFBydW5lIG5vbi10cmFpbmFibGUgdmFyaWFibGVzLlxuICBjb25zdCBvcmlnaW5hbFZhckNvdW50ID0gdmFyTGlzdC5sZW5ndGg7XG4gIHZhckxpc3QgPSB2YXJMaXN0LmZpbHRlcih2YXJpYWJsZSA9PiB2YXJpYWJsZS50cmFpbmFibGUpO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHZhckxpc3QubGVuZ3RoID4gMCxcbiAgICAgICgpID0+IGB2YXJpYWJsZUdyYWRzKCkgZXhwZWN0cyBhdCBsZWFzdCBvbmUgb2YgdGhlIGlucHV0IHZhcmlhYmxlcyB0byBgICtcbiAgICAgICAgICBgYmUgdHJhaW5hYmxlLCBidXQgbm9uZSBvZiB0aGUgJHtvcmlnaW5hbFZhckNvdW50fSB2YXJpYWJsZXMgaXMgYCArXG4gICAgICAgICAgYHRyYWluYWJsZS5gKTtcblxuICBjb25zdCBhbGxvd05vR3JhZGllbnRzID0gdHJ1ZTtcbiAgY29uc3Qge3ZhbHVlLCBncmFkc30gPSBFTkdJTkUuZ3JhZGllbnRzKGYsIHZhckxpc3QsIG51bGwsIGFsbG93Tm9HcmFkaWVudHMpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgZ3JhZHMuc29tZShnID0+IGcgIT0gbnVsbCksXG4gICAgICAoKSA9PiAnQ2Fubm90IGZpbmQgYSBjb25uZWN0aW9uIGJldHdlZW4gYW55IHZhcmlhYmxlIGFuZCB0aGUgcmVzdWx0IG9mICcgK1xuICAgICAgICAgICd0aGUgbG9zcyBmdW5jdGlvbiB5PWYoeCkuIFBsZWFzZSBtYWtlIHN1cmUgdGhlIG9wZXJhdGlvbnMgdGhhdCAnICtcbiAgICAgICAgICAndXNlIHZhcmlhYmxlcyBhcmUgaW5zaWRlIHRoZSBmdW5jdGlvbiBmIHBhc3NlZCB0byBtaW5pbWl6ZSgpLicpO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHZhbHVlLnJhbmsgPT09IDAsXG4gICAgICAoKSA9PiBgVGhlIGYgcGFzc2VkIGluIHZhcmlhYmxlR3JhZHMoZikgbXVzdCByZXR1cm4gYSBzY2FsYXIsIGJ1dCBpdCBgICtcbiAgICAgICAgICBgcmV0dXJuZWQgYSByYW5rLSR7dmFsdWUucmFua30gdGVuc29yYCk7XG5cbiAgY29uc3QgbmFtZWRHcmFkczogTmFtZWRUZW5zb3JNYXAgPSB7fTtcbiAgdmFyTGlzdC5mb3JFYWNoKCh2LCBpKSA9PiB7XG4gICAgaWYgKGdyYWRzW2ldICE9IG51bGwpIHtcbiAgICAgIG5hbWVkR3JhZHNbdi5uYW1lXSA9IGdyYWRzW2ldO1xuICAgIH1cbiAgfSk7XG4gIGlmIChzcGVjaWZpZWROb25UcmFpbmFibGUgIT0gbnVsbCkge1xuICAgIC8vIElmIHZhckxpc3QgaXMgZXhwbGljaXRseSBwcm92aWRlZCBhbmQgY29udGFpbnMgbm9uLXRyYWluYWJsZSB2YWx1ZXMsXG4gICAgLy8gYWRkIHRoZW0gdG8gdGhlIHJldHVybmVkIGdyYWRpZW50cyB3aXRoIGBudWxsYCB2YWx1ZXMuXG4gICAgc3BlY2lmaWVkTm9uVHJhaW5hYmxlLmZvckVhY2godiA9PiBuYW1lZEdyYWRzW3YubmFtZV0gPSBudWxsKTtcbiAgfVxuICByZXR1cm4ge3ZhbHVlLCBncmFkczogbmFtZWRHcmFkc307XG59XG5cbi8qKlxuICogT3ZlcnJpZGVzIHRoZSBncmFkaWVudCBjb21wdXRhdGlvbiBvZiBhIGZ1bmN0aW9uIGBmYC5cbiAqXG4gKiBUYWtlcyBhIGZ1bmN0aW9uXG4gKiBgZiguLi5pbnB1dHMsIHNhdmUpID0+IHt2YWx1ZTogVGVuc29yLCBncmFkRnVuYzogKGR5LCBzYXZlZCkgPT4gVGVuc29yW119YFxuICogYW5kIHJldHVybnMgYW5vdGhlciBmdW5jdGlvbiBgZyguLi5pbnB1dHMpYCB3aGljaCB0YWtlcyB0aGUgc2FtZSBpbnB1dHMgYXNcbiAqIGBmYC4gV2hlbiBjYWxsZWQsIGBnYCByZXR1cm5zIGBmKCkudmFsdWVgLiBJbiBiYWNrd2FyZCBtb2RlLCBjdXN0b20gZ3JhZGllbnRzXG4gKiB3aXRoIHJlc3BlY3QgdG8gZWFjaCBpbnB1dCBvZiBgZmAgYXJlIGNvbXB1dGVkIHVzaW5nIGBmKCkuZ3JhZEZ1bmNgLlxuICpcbiAqIFRoZSBgc2F2ZWAgZnVuY3Rpb24gcGFzc3NlZCB0byBgZmAgc2hvdWxkIGJlIHVzZWQgZm9yIHNhdmluZyB0ZW5zb3JzIG5lZWRlZFxuICogaW4gdGhlIGdyYWRpZW50LiBBbmQgdGhlIGBzYXZlZGAgcGFzc2VkIHRvIHRoZSBgZ3JhZEZ1bmNgIGlzIGFcbiAqIGBOYW1lZFRlbnNvck1hcGAsIHdoaWNoIGNvbnRhaW5zIHRob3NlIHNhdmVkIHRlbnNvci5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgY3VzdG9tT3AgPSB0Zi5jdXN0b21HcmFkKCh4LCBzYXZlKSA9PiB7XG4gKiAgIC8vIFNhdmUgeCB0byBtYWtlIHN1cmUgaXQncyBhdmFpbGFibGUgbGF0ZXIgZm9yIHRoZSBncmFkaWVudC5cbiAqICAgc2F2ZShbeF0pO1xuICogICAvLyBPdmVycmlkZSBncmFkaWVudCBvZiBvdXIgY3VzdG9tIHggXiAyIG9wIHRvIGJlIGR5ICogYWJzKHgpO1xuICogICByZXR1cm4ge1xuICogICAgIHZhbHVlOiB4LnNxdWFyZSgpLFxuICogICAgIC8vIE5vdGUgYHNhdmVkLnhgIHdoaWNoIHBvaW50cyB0byB0aGUgYHhgIHdlIHNhdmVkIGVhcmxpZXIuXG4gKiAgICAgZ3JhZEZ1bmM6IChkeSwgc2F2ZWQpID0+IFtkeS5tdWwoc2F2ZWRbMF0uYWJzKCkpXVxuICogICB9O1xuICogfSk7XG4gKlxuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFstMSwgLTIsIDNdKTtcbiAqIGNvbnN0IGR4ID0gdGYuZ3JhZCh4ID0+IGN1c3RvbU9wKHgpKTtcbiAqXG4gKiBjb25zb2xlLmxvZyhgZih4KTpgKTtcbiAqIGN1c3RvbU9wKHgpLnByaW50KCk7XG4gKiBjb25zb2xlLmxvZyhgZicoeCk6YCk7XG4gKiBkeCh4KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIGYgVGhlIGZ1bmN0aW9uIHRvIGV2YWx1YXRlIGluIGZvcndhcmQgbW9kZSwgd2hpY2ggc2hvdWxkIHJldHVyblxuICogICAgIGB7dmFsdWU6IFRlbnNvciwgZ3JhZEZ1bmM6IChkeSwgc2F2ZWQpID0+IFRlbnNvcltdfWAsIHdoZXJlIGBncmFkRnVuY2BcbiAqICAgICByZXR1cm5zIHRoZSBjdXN0b20gZ3JhZGllbnRzIG9mIGBmYCB3aXRoIHJlc3BlY3QgdG8gaXRzIGlucHV0cy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVHJhaW5pbmcnLCBzdWJoZWFkaW5nOiAnR3JhZGllbnRzJ31cbiAqL1xuZnVuY3Rpb24gY3VzdG9tR3JhZDxUIGV4dGVuZHMgVGVuc29yPihmOiBDdXN0b21HcmFkaWVudEZ1bmM8VD4pOlxuICAgICguLi5hcmdzOiBUZW5zb3JbXSkgPT4gVCB7XG4gIHJldHVybiBFTkdJTkUuY3VzdG9tR3JhZChmKTtcbn1cblxuZnVuY3Rpb24gY2hlY2tHcmFkcyhncmFkczogVGVuc29yW10pIHtcbiAgY29uc3QgbnVtTnVsbEdyYWRpZW50cyA9IGdyYWRzLmZpbHRlcihnID0+IGcgPT0gbnVsbCkubGVuZ3RoO1xuICBpZiAobnVtTnVsbEdyYWRpZW50cyA+IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBDYW5ub3QgY29tcHV0ZSBncmFkaWVudCBvZiB5PWYoeCkgd2l0aCByZXNwZWN0IHRvIHguIE1ha2Ugc3VyZSB0aGF0XG4gICAgdGhlIGYgeW91IHBhc3NlZCBlbmNsb3NlcyBhbGwgb3BlcmF0aW9ucyB0aGF0IGxlYWQgZnJvbSB4IHRvIHkuYCk7XG4gIH1cbn1cblxuZXhwb3J0IHtcbiAgY3VzdG9tR3JhZCxcbiAgdmFyaWFibGVHcmFkcyxcbiAgdmFsdWVBbmRHcmFkLFxuICB2YWx1ZUFuZEdyYWRzLFxuICBncmFkLFxuICBncmFkcyxcbn07XG4iXX0=
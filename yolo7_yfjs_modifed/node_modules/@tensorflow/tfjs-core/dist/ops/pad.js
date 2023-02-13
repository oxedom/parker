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
import { ENGINE } from '../engine';
import { PadV2 } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Pads a `tf.Tensor` with a given value and paddings.
 *
 * This operation implements `CONSTANT` mode. For `REFLECT` and `SYMMETRIC`,
 * refer to `tf.mirrorPad`
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that `paddings` is of given length.
 *   - `tf.pad1d`
 *   - `tf.pad2d`
 *   - `tf.pad3d`
 *   - `tf.pad4d`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.pad([[1, 2]]).print();
 * ```
 * @param x The tensor to pad.
 * @param paddings An array of length `R` (the rank of the tensor), where
 * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
 * specifying how much to pad along each dimension of the tensor.
 * @param constantValue The pad value to use. Defaults to 0.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function pad_(x, paddings, constantValue = 0) {
    const $x = convertToTensor(x, 'x', 'pad');
    if ($x.rank === 0) {
        throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
    }
    const attrs = { paddings, constantValue };
    const inputs = { x: $x };
    return ENGINE.runKernel(PadV2, inputs, attrs);
}
export const pad = op({ pad_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGFkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvcGFkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEtBQUssRUFBMEIsTUFBTSxpQkFBaUIsQ0FBQztBQUkvRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBd0JHO0FBQ0gsU0FBUyxJQUFJLENBQ1QsQ0FBZSxFQUFFLFFBQWlDLEVBQUUsYUFBYSxHQUFHLENBQUM7SUFDdkUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDMUMsSUFBSSxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNqQixNQUFNLElBQUksS0FBSyxDQUFDLG9EQUFvRCxDQUFDLENBQUM7S0FDdkU7SUFFRCxNQUFNLEtBQUssR0FBZSxFQUFDLFFBQVEsRUFBRSxhQUFhLEVBQUMsQ0FBQztJQUNwRCxNQUFNLE1BQU0sR0FBZ0IsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDcEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixLQUFLLEVBQUUsTUFBbUMsRUFDMUMsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLEVBQUMsSUFBSSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1BhZFYyLCBQYWRWMkF0dHJzLCBQYWRWMklucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIFBhZHMgYSBgdGYuVGVuc29yYCB3aXRoIGEgZ2l2ZW4gdmFsdWUgYW5kIHBhZGRpbmdzLlxuICpcbiAqIFRoaXMgb3BlcmF0aW9uIGltcGxlbWVudHMgYENPTlNUQU5UYCBtb2RlLiBGb3IgYFJFRkxFQ1RgIGFuZCBgU1lNTUVUUklDYCxcbiAqIHJlZmVyIHRvIGB0Zi5taXJyb3JQYWRgXG4gKlxuICogQWxzbyBhdmFpbGFibGUgYXJlIHN0cmljdGVyIHJhbmstc3BlY2lmaWMgbWV0aG9kcyB3aXRoIHRoZSBzYW1lIHNpZ25hdHVyZVxuICogYXMgdGhpcyBtZXRob2QgdGhhdCBhc3NlcnQgdGhhdCBgcGFkZGluZ3NgIGlzIG9mIGdpdmVuIGxlbmd0aC5cbiAqICAgLSBgdGYucGFkMWRgXG4gKiAgIC0gYHRmLnBhZDJkYFxuICogICAtIGB0Zi5wYWQzZGBcbiAqICAgLSBgdGYucGFkNGRgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgMywgNF0pO1xuICogeC5wYWQoW1sxLCAyXV0pLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSB4IFRoZSB0ZW5zb3IgdG8gcGFkLlxuICogQHBhcmFtIHBhZGRpbmdzIEFuIGFycmF5IG9mIGxlbmd0aCBgUmAgKHRoZSByYW5rIG9mIHRoZSB0ZW5zb3IpLCB3aGVyZVxuICogZWFjaCBlbGVtZW50IGlzIGEgbGVuZ3RoLTIgdHVwbGUgb2YgaW50cyBgW3BhZEJlZm9yZSwgcGFkQWZ0ZXJdYCxcbiAqIHNwZWNpZnlpbmcgaG93IG11Y2ggdG8gcGFkIGFsb25nIGVhY2ggZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuXG4gKiBAcGFyYW0gY29uc3RhbnRWYWx1ZSBUaGUgcGFkIHZhbHVlIHRvIHVzZS4gRGVmYXVsdHMgdG8gMC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdUcmFuc2Zvcm1hdGlvbnMnfVxuICovXG5mdW5jdGlvbiBwYWRfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgcGFkZGluZ3M6IEFycmF5PFtudW1iZXIsIG51bWJlcl0+LCBjb25zdGFudFZhbHVlID0gMCk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdwYWQnKTtcbiAgaWYgKCR4LnJhbmsgPT09IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ3BhZChzY2FsYXIpIGlzIG5vdCBkZWZpbmVkLiBQYXNzIG5vbi1zY2FsYXIgdG8gcGFkJyk7XG4gIH1cblxuICBjb25zdCBhdHRyczogUGFkVjJBdHRycyA9IHtwYWRkaW5ncywgY29uc3RhbnRWYWx1ZX07XG4gIGNvbnN0IGlucHV0czogUGFkVjJJbnB1dHMgPSB7eDogJHh9O1xuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIFBhZFYyLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IHBhZCA9IG9wKHtwYWRffSk7XG4iXX0=
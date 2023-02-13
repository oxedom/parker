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
import { Max } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the maximum of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If `axes` has no entries, all dimensions are reduced, and an
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.max().print();  // or tf.max(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.max(axis).print();  // or tf.max(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function max_(x, axis = null, keepDims = false) {
    const $x = convertToTensor(x, 'x', 'max');
    const inputs = { x: $x };
    const attrs = { reductionIndices: axis, keepDims };
    return ENGINE.runKernel(Max, inputs, attrs);
}
export const max = op({ max_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF4LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvbWF4LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEdBQUcsRUFBc0IsTUFBTSxpQkFBaUIsQ0FBQztBQUl6RCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTRCRztBQUNILFNBQVMsSUFBSSxDQUNULENBQW9CLEVBQUUsT0FBd0IsSUFBSSxFQUFFLFFBQVEsR0FBRyxLQUFLO0lBQ3RFLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBRTFDLE1BQU0sTUFBTSxHQUFjLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ2xDLE1BQU0sS0FBSyxHQUFhLEVBQUMsZ0JBQWdCLEVBQUUsSUFBSSxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBRTNELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsR0FBRyxFQUFFLE1BQThCLEVBQUUsS0FBMkIsQ0FBQyxDQUFDO0FBQ3hFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLEVBQUMsSUFBSSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge01heCwgTWF4QXR0cnMsIE1heElucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBtYXhpbXVtIG9mIGVsZW1lbnRzIGFjcm9zcyBkaW1lbnNpb25zIG9mIGEgYHRmLlRlbnNvcmAuXG4gKlxuICogUmVkdWNlcyB0aGUgaW5wdXQgYWxvbmcgdGhlIGRpbWVuc2lvbnMgZ2l2ZW4gaW4gYGF4ZXNgLiBVbmxlc3MgYGtlZXBEaW1zYFxuICogaXMgdHJ1ZSwgdGhlIHJhbmsgb2YgdGhlIGB0Zi5UZW5zb3JgIGlzIHJlZHVjZWQgYnkgMSBmb3IgZWFjaCBlbnRyeSBpblxuICogYGF4ZXNgLiBJZiBga2VlcERpbXNgIGlzIHRydWUsIHRoZSByZWR1Y2VkIGRpbWVuc2lvbnMgYXJlIHJldGFpbmVkIHdpdGhcbiAqIGxlbmd0aCAxLiBJZiBgYXhlc2AgaGFzIG5vIGVudHJpZXMsIGFsbCBkaW1lbnNpb25zIGFyZSByZWR1Y2VkLCBhbmQgYW5cbiAqIGB0Zi5UZW5zb3JgIHdpdGggYSBzaW5nbGUgZWxlbWVudCBpcyByZXR1cm5lZC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKlxuICogeC5tYXgoKS5wcmludCgpOyAgLy8gb3IgdGYubWF4KHgpXG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCA0XSwgWzIsIDJdKTtcbiAqXG4gKiBjb25zdCBheGlzID0gMTtcbiAqIHgubWF4KGF4aXMpLnByaW50KCk7ICAvLyBvciB0Zi5tYXgoeCwgYXhpcylcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IuXG4gKiBAcGFyYW0gYXhpcyBUaGUgZGltZW5zaW9uKHMpIHRvIHJlZHVjZS4gQnkgZGVmYXVsdCBpdCByZWR1Y2VzXG4gKiAgICAgYWxsIGRpbWVuc2lvbnMuXG4gKiBAcGFyYW0ga2VlcERpbXMgSWYgdHJ1ZSwgcmV0YWlucyByZWR1Y2VkIGRpbWVuc2lvbnMgd2l0aCBzaXplIDEuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnUmVkdWN0aW9uJ31cbiAqL1xuZnVuY3Rpb24gbWF4XzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICB4OiBUZW5zb3J8VGVuc29yTGlrZSwgYXhpczogbnVtYmVyfG51bWJlcltdID0gbnVsbCwga2VlcERpbXMgPSBmYWxzZSk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdtYXgnKTtcblxuICBjb25zdCBpbnB1dHM6IE1heElucHV0cyA9IHt4OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBNYXhBdHRycyA9IHtyZWR1Y3Rpb25JbmRpY2VzOiBheGlzLCBrZWVwRGltc307XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBNYXgsIGlucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCwgYXR0cnMgYXMge30gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IG1heCA9IG9wKHttYXhffSk7XG4iXX0=
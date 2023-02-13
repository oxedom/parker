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
import { All } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the logical and of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If `axes` has no entries, all dimensions are reduced, and an
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 1, 1], 'bool');
 *
 * x.all().print();  // or tf.all(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
 *
 * const axis = 1;
 * x.all(axis).print();  // or tf.all(x, axis)
 * ```
 *
 * @param x The input tensor. Must be of dtype bool.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function all_(x, axis = null, keepDims = false) {
    const $x = convertToTensor(x, 'x', 'all', 'bool');
    const inputs = { x: $x };
    const attrs = { axis, keepDims };
    return ENGINE.runKernel(All, inputs, attrs);
}
export const all = op({ all_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWxsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvYWxsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEdBQUcsRUFBc0IsTUFBTSxpQkFBaUIsQ0FBQztBQUl6RCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTRCRztBQUNILFNBQVMsSUFBSSxDQUNULENBQW9CLEVBQUUsT0FBd0IsSUFBSSxFQUFFLFFBQVEsR0FBRyxLQUFLO0lBQ3RFLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLEtBQUssRUFBRSxNQUFNLENBQUMsQ0FBQztJQUVsRCxNQUFNLE1BQU0sR0FBYyxFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUMsQ0FBQztJQUNsQyxNQUFNLEtBQUssR0FBYSxFQUFDLElBQUksRUFBRSxRQUFRLEVBQUMsQ0FBQztJQUV6QyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLEdBQUcsRUFBRSxNQUE4QixFQUFFLEtBQTJCLENBQUMsQ0FBQztBQUN4RSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxFQUFDLElBQUksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtBbGwsIEFsbEF0dHJzLCBBbGxJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgbG9naWNhbCBhbmQgb2YgZWxlbWVudHMgYWNyb3NzIGRpbWVuc2lvbnMgb2YgYSBgdGYuVGVuc29yYC5cbiAqXG4gKiBSZWR1Y2VzIHRoZSBpbnB1dCBhbG9uZyB0aGUgZGltZW5zaW9ucyBnaXZlbiBpbiBgYXhlc2AuIFVubGVzcyBga2VlcERpbXNgXG4gKiBpcyB0cnVlLCB0aGUgcmFuayBvZiB0aGUgYHRmLlRlbnNvcmAgaXMgcmVkdWNlZCBieSAxIGZvciBlYWNoIGVudHJ5IGluXG4gKiBgYXhlc2AuIElmIGBrZWVwRGltc2AgaXMgdHJ1ZSwgdGhlIHJlZHVjZWQgZGltZW5zaW9ucyBhcmUgcmV0YWluZWQgd2l0aFxuICogbGVuZ3RoIDEuIElmIGBheGVzYCBoYXMgbm8gZW50cmllcywgYWxsIGRpbWVuc2lvbnMgYXJlIHJlZHVjZWQsIGFuZCBhblxuICogYHRmLlRlbnNvcmAgd2l0aCBhIHNpbmdsZSBlbGVtZW50IGlzIHJldHVybmVkLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDEsIDFdLCAnYm9vbCcpO1xuICpcbiAqIHguYWxsKCkucHJpbnQoKTsgIC8vIG9yIHRmLmFsbCh4KVxuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbMSwgMSwgMCwgMF0sIFsyLCAyXSwgJ2Jvb2wnKTtcbiAqXG4gKiBjb25zdCBheGlzID0gMTtcbiAqIHguYWxsKGF4aXMpLnByaW50KCk7ICAvLyBvciB0Zi5hbGwoeCwgYXhpcylcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IuIE11c3QgYmUgb2YgZHR5cGUgYm9vbC5cbiAqIEBwYXJhbSBheGlzIFRoZSBkaW1lbnNpb24ocykgdG8gcmVkdWNlLiBCeSBkZWZhdWx0IGl0IHJlZHVjZXNcbiAqICAgICBhbGwgZGltZW5zaW9ucy5cbiAqIEBwYXJhbSBrZWVwRGltcyBJZiB0cnVlLCByZXRhaW5zIHJlZHVjZWQgZGltZW5zaW9ucyB3aXRoIHNpemUgMS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdSZWR1Y3Rpb24nfVxuICovXG5mdW5jdGlvbiBhbGxfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBheGlzOiBudW1iZXJ8bnVtYmVyW10gPSBudWxsLCBrZWVwRGltcyA9IGZhbHNlKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2FsbCcsICdib29sJyk7XG5cbiAgY29uc3QgaW5wdXRzOiBBbGxJbnB1dHMgPSB7eDogJHh9O1xuICBjb25zdCBhdHRyczogQWxsQXR0cnMgPSB7YXhpcywga2VlcERpbXN9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQWxsLCBpbnB1dHMgYXMge30gYXMgTmFtZWRUZW5zb3JNYXAsIGF0dHJzIGFzIHt9IGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBhbGwgPSBvcCh7YWxsX30pO1xuIl19
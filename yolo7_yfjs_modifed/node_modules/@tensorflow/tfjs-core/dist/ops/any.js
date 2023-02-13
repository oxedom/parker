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
import { Any } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the logical or of elements across dimensions of a `tf.Tensor`.
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
 * x.any().print();  // or tf.any(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
 *
 * const axis = 1;
 * x.any(axis).print();  // or tf.any(x, axis)
 * ```
 *
 * @param x The input tensor. Must be of dtype bool.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function any_(x, axis = null, keepDims = false) {
    const $x = convertToTensor(x, 'x', 'any', 'bool');
    const inputs = { x: $x };
    const attrs = { axis, keepDims };
    return ENGINE.runKernel(Any, inputs, attrs);
}
// tslint:disable-next-line:variable-name
export const any = op({ any_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYW55LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvYW55LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEdBQUcsRUFBc0IsTUFBTSxpQkFBaUIsQ0FBQztBQUl6RCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTRCRztBQUNILFNBQVMsSUFBSSxDQUNULENBQW9CLEVBQUUsT0FBd0IsSUFBSSxFQUFFLFFBQVEsR0FBRyxLQUFLO0lBQ3RFLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLEtBQUssRUFBRSxNQUFNLENBQUMsQ0FBQztJQUVsRCxNQUFNLE1BQU0sR0FBYyxFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUMsQ0FBQztJQUNsQyxNQUFNLEtBQUssR0FBYSxFQUFDLElBQUksRUFBRSxRQUFRLEVBQUMsQ0FBQztJQUV6QyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLEdBQUcsRUFBRSxNQUE4QixFQUFFLEtBQTJCLENBQUMsQ0FBQztBQUN4RSxDQUFDO0FBRUQseUNBQXlDO0FBQ3pDLE1BQU0sQ0FBQyxNQUFNLEdBQUcsR0FBRyxFQUFFLENBQUMsRUFBQyxJQUFJLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7QW55LCBBbnlBdHRycywgQW55SW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIGxvZ2ljYWwgb3Igb2YgZWxlbWVudHMgYWNyb3NzIGRpbWVuc2lvbnMgb2YgYSBgdGYuVGVuc29yYC5cbiAqXG4gKiBSZWR1Y2VzIHRoZSBpbnB1dCBhbG9uZyB0aGUgZGltZW5zaW9ucyBnaXZlbiBpbiBgYXhlc2AuIFVubGVzcyBga2VlcERpbXNgXG4gKiBpcyB0cnVlLCB0aGUgcmFuayBvZiB0aGUgYHRmLlRlbnNvcmAgaXMgcmVkdWNlZCBieSAxIGZvciBlYWNoIGVudHJ5IGluXG4gKiBgYXhlc2AuIElmIGBrZWVwRGltc2AgaXMgdHJ1ZSwgdGhlIHJlZHVjZWQgZGltZW5zaW9ucyBhcmUgcmV0YWluZWQgd2l0aFxuICogbGVuZ3RoIDEuIElmIGBheGVzYCBoYXMgbm8gZW50cmllcywgYWxsIGRpbWVuc2lvbnMgYXJlIHJlZHVjZWQsIGFuZCBhblxuICogYHRmLlRlbnNvcmAgd2l0aCBhIHNpbmdsZSBlbGVtZW50IGlzIHJldHVybmVkLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDEsIDFdLCAnYm9vbCcpO1xuICpcbiAqIHguYW55KCkucHJpbnQoKTsgIC8vIG9yIHRmLmFueSh4KVxuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbMSwgMSwgMCwgMF0sIFsyLCAyXSwgJ2Jvb2wnKTtcbiAqXG4gKiBjb25zdCBheGlzID0gMTtcbiAqIHguYW55KGF4aXMpLnByaW50KCk7ICAvLyBvciB0Zi5hbnkoeCwgYXhpcylcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IuIE11c3QgYmUgb2YgZHR5cGUgYm9vbC5cbiAqIEBwYXJhbSBheGlzIFRoZSBkaW1lbnNpb24ocykgdG8gcmVkdWNlLiBCeSBkZWZhdWx0IGl0IHJlZHVjZXNcbiAqICAgICBhbGwgZGltZW5zaW9ucy5cbiAqIEBwYXJhbSBrZWVwRGltcyBJZiB0cnVlLCByZXRhaW5zIHJlZHVjZWQgZGltZW5zaW9ucyB3aXRoIHNpemUgMS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdSZWR1Y3Rpb24nfVxuICovXG5mdW5jdGlvbiBhbnlfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBheGlzOiBudW1iZXJ8bnVtYmVyW10gPSBudWxsLCBrZWVwRGltcyA9IGZhbHNlKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2FueScsICdib29sJyk7XG5cbiAgY29uc3QgaW5wdXRzOiBBbnlJbnB1dHMgPSB7eDogJHh9O1xuICBjb25zdCBhdHRyczogQW55QXR0cnMgPSB7YXhpcywga2VlcERpbXN9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQW55LCBpbnB1dHMgYXMge30gYXMgTmFtZWRUZW5zb3JNYXAsIGF0dHJzIGFzIHt9IGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTp2YXJpYWJsZS1uYW1lXG5leHBvcnQgY29uc3QgYW55ID0gb3Aoe2FueV99KTtcbiJdfQ==
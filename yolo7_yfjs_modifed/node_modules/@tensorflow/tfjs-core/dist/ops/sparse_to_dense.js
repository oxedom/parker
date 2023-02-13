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
import { ENGINE } from '../engine';
import { SparseToDense } from '../kernel_names';
import * as sparse_to_dense from '../ops/sparse_to_dense_util';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Converts a sparse representation into a dense tensor.
 *
 * Builds an array dense with shape outputShape such that:
 *
 * // If sparseIndices is scalar
 * dense[i] = (i == sparseIndices ? sparseValues : defaultValue)
 *
 * // If sparseIndices is a vector, then for each i
 * dense[sparseIndices[i]] = sparseValues[i]
 *
 * // If sparseIndices is an n by d matrix, then for each i in [0, n)
 * dense[sparseIndices[i][0], ..., sparseIndices[i][d-1]] = sparseValues[i]
 * All other values in dense are set to defaultValue. If sparseValues is a
 * scalar, all sparse indices are set to this single value.
 *
 * If indices are repeated the final value is summed over all values for those
 * indices.
 *
 * ```js
 * const indices = tf.tensor1d([4, 5, 6, 1, 2, 3], 'int32');
 * const values = tf.tensor1d([10, 11, 12, 13, 14, 15], 'float32');
 * const shape = [8];
 * tf.sparseToDense(indices, values, shape).print();
 * ```
 *
 * @param sparseIndices A 0-D, 1-D, or 2-D Tensor of type int32.
 * sparseIndices[i] contains the complete index where sparseValues[i] will be
 * placed.
 * @param sparseValues A 0-D or 1-D Tensor. Values
 * corresponding to each row of sparseIndices, or a scalar value to be used for
 * all sparse indices.
 * @param outputShape Shape of the dense output tensor. the type is inferred.
 * @param defaultValue Scalar. Value to set for indices not specified in
 * sparseIndices. Defaults to zero.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function sparseToDense_(sparseIndices, sparseValues, outputShape, defaultValue = 0) {
    const $sparseIndices = convertToTensor(sparseIndices, 'sparseIndices', 'sparseToDense', 'int32');
    const $sparseValues = convertToTensor(sparseValues, 'sparseValues', 'sparseToDense');
    const $defaultValue = convertToTensor(defaultValue, 'defaultValue', 'sparseToDense', $sparseValues.dtype);
    sparse_to_dense.validateInput($sparseIndices, $sparseValues, outputShape, $defaultValue);
    const inputs = {
        sparseIndices: $sparseIndices,
        sparseValues: $sparseValues,
        defaultValue: $defaultValue
    };
    const attrs = { outputShape };
    return ENGINE.runKernel(SparseToDense, inputs, attrs);
}
export const sparseToDense = op({ sparseToDense_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhcnNlX3RvX2RlbnNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvc3BhcnNlX3RvX2RlbnNlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLGFBQWEsRUFBMEMsTUFBTSxpQkFBaUIsQ0FBQztBQUV2RixPQUFPLEtBQUssZUFBZSxNQUFNLDZCQUE2QixDQUFDO0FBRy9ELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBcUNHO0FBQ0gsU0FBUyxjQUFjLENBQ25CLGFBQWdDLEVBQUUsWUFBK0IsRUFDakUsV0FBd0IsRUFBRSxlQUFrQyxDQUFDO0lBQy9ELE1BQU0sY0FBYyxHQUNoQixlQUFlLENBQUMsYUFBYSxFQUFFLGVBQWUsRUFBRSxlQUFlLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDOUUsTUFBTSxhQUFhLEdBQ2YsZUFBZSxDQUFDLFlBQVksRUFBRSxjQUFjLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDbkUsTUFBTSxhQUFhLEdBQUcsZUFBZSxDQUNqQyxZQUFZLEVBQUUsY0FBYyxFQUFFLGVBQWUsRUFBRSxhQUFhLENBQUMsS0FBSyxDQUFDLENBQUM7SUFFeEUsZUFBZSxDQUFDLGFBQWEsQ0FDekIsY0FBYyxFQUFFLGFBQWEsRUFBRSxXQUFXLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFFL0QsTUFBTSxNQUFNLEdBQXdCO1FBQ2xDLGFBQWEsRUFBRSxjQUFjO1FBQzdCLFlBQVksRUFBRSxhQUFhO1FBQzNCLFlBQVksRUFBRSxhQUFhO0tBQzVCLENBQUM7SUFFRixNQUFNLEtBQUssR0FBdUIsRUFBQyxXQUFXLEVBQUMsQ0FBQztJQUVoRCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLGFBQWEsRUFBRSxNQUE4QixFQUM3QyxLQUEyQixDQUFDLENBQUM7QUFDbkMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBRyxFQUFFLENBQUMsRUFBQyxjQUFjLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7U3BhcnNlVG9EZW5zZSwgU3BhcnNlVG9EZW5zZUF0dHJzLCBTcGFyc2VUb0RlbnNlSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQgKiBhcyBzcGFyc2VfdG9fZGVuc2UgZnJvbSAnLi4vb3BzL3NwYXJzZV90b19kZW5zZV91dGlsJztcbmltcG9ydCB7U2NhbGFyLCBUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1JhbmssIFNjYWxhckxpa2UsIFNoYXBlTWFwLCBUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb252ZXJ0cyBhIHNwYXJzZSByZXByZXNlbnRhdGlvbiBpbnRvIGEgZGVuc2UgdGVuc29yLlxuICpcbiAqIEJ1aWxkcyBhbiBhcnJheSBkZW5zZSB3aXRoIHNoYXBlIG91dHB1dFNoYXBlIHN1Y2ggdGhhdDpcbiAqXG4gKiAvLyBJZiBzcGFyc2VJbmRpY2VzIGlzIHNjYWxhclxuICogZGVuc2VbaV0gPSAoaSA9PSBzcGFyc2VJbmRpY2VzID8gc3BhcnNlVmFsdWVzIDogZGVmYXVsdFZhbHVlKVxuICpcbiAqIC8vIElmIHNwYXJzZUluZGljZXMgaXMgYSB2ZWN0b3IsIHRoZW4gZm9yIGVhY2ggaVxuICogZGVuc2Vbc3BhcnNlSW5kaWNlc1tpXV0gPSBzcGFyc2VWYWx1ZXNbaV1cbiAqXG4gKiAvLyBJZiBzcGFyc2VJbmRpY2VzIGlzIGFuIG4gYnkgZCBtYXRyaXgsIHRoZW4gZm9yIGVhY2ggaSBpbiBbMCwgbilcbiAqIGRlbnNlW3NwYXJzZUluZGljZXNbaV1bMF0sIC4uLiwgc3BhcnNlSW5kaWNlc1tpXVtkLTFdXSA9IHNwYXJzZVZhbHVlc1tpXVxuICogQWxsIG90aGVyIHZhbHVlcyBpbiBkZW5zZSBhcmUgc2V0IHRvIGRlZmF1bHRWYWx1ZS4gSWYgc3BhcnNlVmFsdWVzIGlzIGFcbiAqIHNjYWxhciwgYWxsIHNwYXJzZSBpbmRpY2VzIGFyZSBzZXQgdG8gdGhpcyBzaW5nbGUgdmFsdWUuXG4gKlxuICogSWYgaW5kaWNlcyBhcmUgcmVwZWF0ZWQgdGhlIGZpbmFsIHZhbHVlIGlzIHN1bW1lZCBvdmVyIGFsbCB2YWx1ZXMgZm9yIHRob3NlXG4gKiBpbmRpY2VzLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbmRpY2VzID0gdGYudGVuc29yMWQoWzQsIDUsIDYsIDEsIDIsIDNdLCAnaW50MzInKTtcbiAqIGNvbnN0IHZhbHVlcyA9IHRmLnRlbnNvcjFkKFsxMCwgMTEsIDEyLCAxMywgMTQsIDE1XSwgJ2Zsb2F0MzInKTtcbiAqIGNvbnN0IHNoYXBlID0gWzhdO1xuICogdGYuc3BhcnNlVG9EZW5zZShpbmRpY2VzLCB2YWx1ZXMsIHNoYXBlKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHNwYXJzZUluZGljZXMgQSAwLUQsIDEtRCwgb3IgMi1EIFRlbnNvciBvZiB0eXBlIGludDMyLlxuICogc3BhcnNlSW5kaWNlc1tpXSBjb250YWlucyB0aGUgY29tcGxldGUgaW5kZXggd2hlcmUgc3BhcnNlVmFsdWVzW2ldIHdpbGwgYmVcbiAqIHBsYWNlZC5cbiAqIEBwYXJhbSBzcGFyc2VWYWx1ZXMgQSAwLUQgb3IgMS1EIFRlbnNvci4gVmFsdWVzXG4gKiBjb3JyZXNwb25kaW5nIHRvIGVhY2ggcm93IG9mIHNwYXJzZUluZGljZXMsIG9yIGEgc2NhbGFyIHZhbHVlIHRvIGJlIHVzZWQgZm9yXG4gKiBhbGwgc3BhcnNlIGluZGljZXMuXG4gKiBAcGFyYW0gb3V0cHV0U2hhcGUgU2hhcGUgb2YgdGhlIGRlbnNlIG91dHB1dCB0ZW5zb3IuIHRoZSB0eXBlIGlzIGluZmVycmVkLlxuICogQHBhcmFtIGRlZmF1bHRWYWx1ZSBTY2FsYXIuIFZhbHVlIHRvIHNldCBmb3IgaW5kaWNlcyBub3Qgc3BlY2lmaWVkIGluXG4gKiBzcGFyc2VJbmRpY2VzLiBEZWZhdWx0cyB0byB6ZXJvLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ05vcm1hbGl6YXRpb24nfVxuICovXG5mdW5jdGlvbiBzcGFyc2VUb0RlbnNlXzxSIGV4dGVuZHMgUmFuaz4oXG4gICAgc3BhcnNlSW5kaWNlczogVGVuc29yfFRlbnNvckxpa2UsIHNwYXJzZVZhbHVlczogVGVuc29yfFRlbnNvckxpa2UsXG4gICAgb3V0cHV0U2hhcGU6IFNoYXBlTWFwW1JdLCBkZWZhdWx0VmFsdWU6IFNjYWxhcnxTY2FsYXJMaWtlID0gMCk6IFRlbnNvcjxSPiB7XG4gIGNvbnN0ICRzcGFyc2VJbmRpY2VzID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihzcGFyc2VJbmRpY2VzLCAnc3BhcnNlSW5kaWNlcycsICdzcGFyc2VUb0RlbnNlJywgJ2ludDMyJyk7XG4gIGNvbnN0ICRzcGFyc2VWYWx1ZXMgPVxuICAgICAgY29udmVydFRvVGVuc29yKHNwYXJzZVZhbHVlcywgJ3NwYXJzZVZhbHVlcycsICdzcGFyc2VUb0RlbnNlJyk7XG4gIGNvbnN0ICRkZWZhdWx0VmFsdWUgPSBjb252ZXJ0VG9UZW5zb3IoXG4gICAgICBkZWZhdWx0VmFsdWUsICdkZWZhdWx0VmFsdWUnLCAnc3BhcnNlVG9EZW5zZScsICRzcGFyc2VWYWx1ZXMuZHR5cGUpO1xuXG4gIHNwYXJzZV90b19kZW5zZS52YWxpZGF0ZUlucHV0KFxuICAgICAgJHNwYXJzZUluZGljZXMsICRzcGFyc2VWYWx1ZXMsIG91dHB1dFNoYXBlLCAkZGVmYXVsdFZhbHVlKTtcblxuICBjb25zdCBpbnB1dHM6IFNwYXJzZVRvRGVuc2VJbnB1dHMgPSB7XG4gICAgc3BhcnNlSW5kaWNlczogJHNwYXJzZUluZGljZXMsXG4gICAgc3BhcnNlVmFsdWVzOiAkc3BhcnNlVmFsdWVzLFxuICAgIGRlZmF1bHRWYWx1ZTogJGRlZmF1bHRWYWx1ZVxuICB9O1xuXG4gIGNvbnN0IGF0dHJzOiBTcGFyc2VUb0RlbnNlQXR0cnMgPSB7b3V0cHV0U2hhcGV9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgU3BhcnNlVG9EZW5zZSwgaW5wdXRzIGFzIHt9IGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMge30gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IHNwYXJzZVRvRGVuc2UgPSBvcCh7c3BhcnNlVG9EZW5zZV99KTtcbiJdfQ==
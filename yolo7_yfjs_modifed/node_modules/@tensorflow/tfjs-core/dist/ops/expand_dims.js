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
import { ExpandDims } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
 * into the tensor's shape.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const axis = 1;
 * x.expandDims(axis).print();
 * ```
 *
 * @param x The input tensor whose dimensions to be expanded.
 * @param axis The dimension index at which to insert shape of `1`. Defaults
 *     to 0 (the first dimension).
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function expandDims_(x, axis = 0) {
    const $x = convertToTensor(x, 'x', 'expandDims', 'string_or_numeric');
    util.assert(axis <= $x.rank, () => 'Axis must be <= rank of the tensor');
    const inputs = { input: $x };
    const attrs = { dim: axis };
    return ENGINE.runKernel(ExpandDims, inputs, attrs);
}
export const expandDims = op({ expandDims_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXhwYW5kX2RpbXMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9leHBhbmRfZGltcy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxVQUFVLEVBQW9DLE1BQU0saUJBQWlCLENBQUM7QUFJOUUsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsU0FBUyxXQUFXLENBQW1CLENBQW9CLEVBQUUsSUFBSSxHQUFHLENBQUM7SUFDbkUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsWUFBWSxFQUFFLG1CQUFtQixDQUFDLENBQUM7SUFFdEUsSUFBSSxDQUFDLE1BQU0sQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLElBQUksRUFBRSxHQUFHLEVBQUUsQ0FBQyxvQ0FBb0MsQ0FBQyxDQUFDO0lBRXpFLE1BQU0sTUFBTSxHQUFxQixFQUFDLEtBQUssRUFBRSxFQUFFLEVBQUMsQ0FBQztJQUM3QyxNQUFNLEtBQUssR0FBb0IsRUFBQyxHQUFHLEVBQUUsSUFBSSxFQUFDLENBQUM7SUFFM0MsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixVQUFVLEVBQUUsTUFBOEIsRUFBRSxLQUEyQixDQUFDLENBQUM7QUFDL0UsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFVBQVUsR0FBRyxFQUFFLENBQUMsRUFBQyxXQUFXLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7RXhwYW5kRGltcywgRXhwYW5kRGltc0F0dHJzLCBFeHBhbmREaW1zSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIFJldHVybnMgYSBgdGYuVGVuc29yYCB0aGF0IGhhcyBleHBhbmRlZCByYW5rLCBieSBpbnNlcnRpbmcgYSBkaW1lbnNpb25cbiAqIGludG8gdGhlIHRlbnNvcidzIHNoYXBlLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDMsIDRdKTtcbiAqIGNvbnN0IGF4aXMgPSAxO1xuICogeC5leHBhbmREaW1zKGF4aXMpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yIHdob3NlIGRpbWVuc2lvbnMgdG8gYmUgZXhwYW5kZWQuXG4gKiBAcGFyYW0gYXhpcyBUaGUgZGltZW5zaW9uIGluZGV4IGF0IHdoaWNoIHRvIGluc2VydCBzaGFwZSBvZiBgMWAuIERlZmF1bHRzXG4gKiAgICAgdG8gMCAodGhlIGZpcnN0IGRpbWVuc2lvbikuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnVHJhbnNmb3JtYXRpb25zJ31cbiAqL1xuZnVuY3Rpb24gZXhwYW5kRGltc188VCBleHRlbmRzIFRlbnNvcj4oeDogVGVuc29yfFRlbnNvckxpa2UsIGF4aXMgPSAwKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2V4cGFuZERpbXMnLCAnc3RyaW5nX29yX251bWVyaWMnKTtcblxuICB1dGlsLmFzc2VydChheGlzIDw9ICR4LnJhbmssICgpID0+ICdBeGlzIG11c3QgYmUgPD0gcmFuayBvZiB0aGUgdGVuc29yJyk7XG5cbiAgY29uc3QgaW5wdXRzOiBFeHBhbmREaW1zSW5wdXRzID0ge2lucHV0OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBFeHBhbmREaW1zQXR0cnMgPSB7ZGltOiBheGlzfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIEV4cGFuZERpbXMsIGlucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCwgYXR0cnMgYXMge30gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGV4cGFuZERpbXMgPSBvcCh7ZXhwYW5kRGltc199KTtcbiJdfQ==
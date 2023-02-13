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
import { decodeString, encodeString } from '../util';
// Utilities needed by backend consumers of tf-core.
export * from '../ops/axis_util';
export * from '../ops/broadcast_util';
export * from '../ops/concat_util';
export * from '../ops/conv_util';
export * from '../ops/fused_util';
export * from '../ops/reduce_util';
import * as slice_util from '../ops/slice_util';
export { slice_util };
export { upcastType } from '../types';
export * from '../ops/rotate_util';
export * from '../ops/array_ops_util';
export * from '../ops/gather_nd_util';
export * from '../ops/scatter_nd_util';
export * from '../ops/selu_util';
export * from '../ops/fused_util';
export * from '../ops/erf_util';
export * from '../log';
export * from '../backends/complex_util';
export * from '../backends/einsum_util';
export * from '../ops/split_util';
export * from '../ops/sparse/sparse_fill_empty_rows_util';
export * from '../ops/sparse/sparse_reshape_util';
export * from '../ops/sparse/sparse_segment_reduction_util';
import * as segment_util from '../ops/segment_util';
export { segment_util };
export function fromUint8ToStringArray(vals) {
    try {
        // Decode the bytes into string.
        return vals.map(val => decodeString(val));
    }
    catch (err) {
        throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${err}`);
    }
}
export function fromStringArrayToUint8(strings) {
    return strings.map(s => encodeString(s));
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF91dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9iYWNrZW5kcy9iYWNrZW5kX3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBRSxZQUFZLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFbkQsb0RBQW9EO0FBQ3BELGNBQWMsa0JBQWtCLENBQUM7QUFDakMsY0FBYyx1QkFBdUIsQ0FBQztBQUN0QyxjQUFjLG9CQUFvQixDQUFDO0FBQ25DLGNBQWMsa0JBQWtCLENBQUM7QUFDakMsY0FBYyxtQkFBbUIsQ0FBQztBQUVsQyxjQUFjLG9CQUFvQixDQUFDO0FBRW5DLE9BQU8sS0FBSyxVQUFVLE1BQU0sbUJBQW1CLENBQUM7QUFDaEQsT0FBTyxFQUFDLFVBQVUsRUFBQyxDQUFDO0FBRXBCLE9BQU8sRUFBNEIsVUFBVSxFQUFZLE1BQU0sVUFBVSxDQUFDO0FBRTFFLGNBQWMsb0JBQW9CLENBQUM7QUFDbkMsY0FBYyx1QkFBdUIsQ0FBQztBQUN0QyxjQUFjLHVCQUF1QixDQUFDO0FBQ3RDLGNBQWMsd0JBQXdCLENBQUM7QUFDdkMsY0FBYyxrQkFBa0IsQ0FBQztBQUNqQyxjQUFjLG1CQUFtQixDQUFDO0FBQ2xDLGNBQWMsaUJBQWlCLENBQUM7QUFDaEMsY0FBYyxRQUFRLENBQUM7QUFDdkIsY0FBYywwQkFBMEIsQ0FBQztBQUN6QyxjQUFjLHlCQUF5QixDQUFDO0FBQ3hDLGNBQWMsbUJBQW1CLENBQUM7QUFDbEMsY0FBYywyQ0FBMkMsQ0FBQztBQUMxRCxjQUFjLG1DQUFtQyxDQUFDO0FBQ2xELGNBQWMsNkNBQTZDLENBQUM7QUFFNUQsT0FBTyxLQUFLLFlBQVksTUFBTSxxQkFBcUIsQ0FBQztBQUNwRCxPQUFPLEVBQUMsWUFBWSxFQUFDLENBQUM7QUFFdEIsTUFBTSxVQUFVLHNCQUFzQixDQUFDLElBQWtCO0lBQ3ZELElBQUk7UUFDRixnQ0FBZ0M7UUFDaEMsT0FBTyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsWUFBWSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7S0FDM0M7SUFBQyxPQUFPLEdBQUcsRUFBRTtRQUNaLE1BQU0sSUFBSSxLQUFLLENBQ1gsNERBQTRELEdBQUcsRUFBRSxDQUFDLENBQUM7S0FDeEU7QUFDSCxDQUFDO0FBRUQsTUFBTSxVQUFVLHNCQUFzQixDQUFDLE9BQWlCO0lBQ3RELE9BQU8sT0FBTyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQzNDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7ZGVjb2RlU3RyaW5nLCBlbmNvZGVTdHJpbmd9IGZyb20gJy4uL3V0aWwnO1xuXG4vLyBVdGlsaXRpZXMgbmVlZGVkIGJ5IGJhY2tlbmQgY29uc3VtZXJzIG9mIHRmLWNvcmUuXG5leHBvcnQgKiBmcm9tICcuLi9vcHMvYXhpc191dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9icm9hZGNhc3RfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvY29uY2F0X3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2NvbnZfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvZnVzZWRfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvZnVzZWRfdHlwZXMnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL3JlZHVjZV91dGlsJztcblxuaW1wb3J0ICogYXMgc2xpY2VfdXRpbCBmcm9tICcuLi9vcHMvc2xpY2VfdXRpbCc7XG5leHBvcnQge3NsaWNlX3V0aWx9O1xuXG5leHBvcnQge0JhY2tlbmRWYWx1ZXMsIFR5cGVkQXJyYXksIHVwY2FzdFR5cGUsIFBpeGVsRGF0YX0gZnJvbSAnLi4vdHlwZXMnO1xuZXhwb3J0IHtNZW1vcnlJbmZvLCBUaW1pbmdJbmZvfSBmcm9tICcuLi9lbmdpbmUnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL3JvdGF0ZV91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9hcnJheV9vcHNfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvZ2F0aGVyX25kX3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL3NjYXR0ZXJfbmRfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvc2VsdV91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9mdXNlZF91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9lcmZfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9sb2cnO1xuZXhwb3J0ICogZnJvbSAnLi4vYmFja2VuZHMvY29tcGxleF91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL2JhY2tlbmRzL2VpbnN1bV91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9zcGxpdF91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9zcGFyc2Uvc3BhcnNlX2ZpbGxfZW1wdHlfcm93c191dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9zcGFyc2Uvc3BhcnNlX3Jlc2hhcGVfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvc3BhcnNlL3NwYXJzZV9zZWdtZW50X3JlZHVjdGlvbl91dGlsJztcblxuaW1wb3J0ICogYXMgc2VnbWVudF91dGlsIGZyb20gJy4uL29wcy9zZWdtZW50X3V0aWwnO1xuZXhwb3J0IHtzZWdtZW50X3V0aWx9O1xuXG5leHBvcnQgZnVuY3Rpb24gZnJvbVVpbnQ4VG9TdHJpbmdBcnJheSh2YWxzOiBVaW50OEFycmF5W10pIHtcbiAgdHJ5IHtcbiAgICAvLyBEZWNvZGUgdGhlIGJ5dGVzIGludG8gc3RyaW5nLlxuICAgIHJldHVybiB2YWxzLm1hcCh2YWwgPT4gZGVjb2RlU3RyaW5nKHZhbCkpO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBGYWlsZWQgdG8gZGVjb2RlIGVuY29kZWQgc3RyaW5nIGJ5dGVzIGludG8gdXRmLTgsIGVycm9yOiAke2Vycn1gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZnJvbVN0cmluZ0FycmF5VG9VaW50OChzdHJpbmdzOiBzdHJpbmdbXSkge1xuICByZXR1cm4gc3RyaW5ncy5tYXAocyA9PiBlbmNvZGVTdHJpbmcocykpO1xufVxuIl19
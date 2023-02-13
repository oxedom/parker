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
import { Range } from '../kernel_names';
/**
 * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
 *
 * The tensor is a is half-open interval meaning it includes start, but
 * excludes stop. Decrementing ranges and negative step values are also
 * supported.sv
 *
 *
 * ```js
 * tf.range(0, 9, 2).print();
 * ```
 *
 * @param start An integer start value
 * @param stop An integer stop value
 * @param step An integer increment (will default to 1 or -1)
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function range(start, stop, step = 1, dtype = 'float32') {
    if (step === 0) {
        throw new Error('Cannot have a step of zero');
    }
    const attrs = { start, stop, step, dtype };
    return ENGINE.runKernel(Range, {} /* inputs */, attrs);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFuZ2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9yYW5nZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxLQUFLLEVBQWEsTUFBTSxpQkFBaUIsQ0FBQztBQUlsRDs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBa0JHO0FBQ0gsTUFBTSxVQUFVLEtBQUssQ0FDakIsS0FBYSxFQUFFLElBQVksRUFBRSxJQUFJLEdBQUcsQ0FBQyxFQUNyQyxRQUEyQixTQUFTO0lBQ3RDLElBQUksSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNkLE1BQU0sSUFBSSxLQUFLLENBQUMsNEJBQTRCLENBQUMsQ0FBQztLQUMvQztJQUVELE1BQU0sS0FBSyxHQUFlLEVBQUMsS0FBSyxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFDLENBQUM7SUFFckQsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsWUFBWSxFQUFFLEtBQTJCLENBQUMsQ0FBQztBQUMvRSxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7UmFuZ2UsIFJhbmdlQXR0cnN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yMUR9IGZyb20gJy4uL3RlbnNvcic7XG5cbi8qKlxuICogQ3JlYXRlcyBhIG5ldyBgdGYuVGVuc29yMURgIGZpbGxlZCB3aXRoIHRoZSBudW1iZXJzIGluIHRoZSByYW5nZSBwcm92aWRlZC5cbiAqXG4gKiBUaGUgdGVuc29yIGlzIGEgaXMgaGFsZi1vcGVuIGludGVydmFsIG1lYW5pbmcgaXQgaW5jbHVkZXMgc3RhcnQsIGJ1dFxuICogZXhjbHVkZXMgc3RvcC4gRGVjcmVtZW50aW5nIHJhbmdlcyBhbmQgbmVnYXRpdmUgc3RlcCB2YWx1ZXMgYXJlIGFsc29cbiAqIHN1cHBvcnRlZC5zdlxuICpcbiAqXG4gKiBgYGBqc1xuICogdGYucmFuZ2UoMCwgOSwgMikucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBzdGFydCBBbiBpbnRlZ2VyIHN0YXJ0IHZhbHVlXG4gKiBAcGFyYW0gc3RvcCBBbiBpbnRlZ2VyIHN0b3AgdmFsdWVcbiAqIEBwYXJhbSBzdGVwIEFuIGludGVnZXIgaW5jcmVtZW50ICh3aWxsIGRlZmF1bHQgdG8gMSBvciAtMSlcbiAqIEBwYXJhbSBkdHlwZSBUaGUgZGF0YSB0eXBlIG9mIHRoZSBvdXRwdXQgdGVuc29yLiBEZWZhdWx0cyB0byAnZmxvYXQzMicuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICovXG5leHBvcnQgZnVuY3Rpb24gcmFuZ2UoXG4gICAgc3RhcnQ6IG51bWJlciwgc3RvcDogbnVtYmVyLCBzdGVwID0gMSxcbiAgICBkdHlwZTogJ2Zsb2F0MzInfCdpbnQzMicgPSAnZmxvYXQzMicpOiBUZW5zb3IxRCB7XG4gIGlmIChzdGVwID09PSAwKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdDYW5ub3QgaGF2ZSBhIHN0ZXAgb2YgemVybycpO1xuICB9XG5cbiAgY29uc3QgYXR0cnM6IFJhbmdlQXR0cnMgPSB7c3RhcnQsIHN0b3AsIHN0ZXAsIGR0eXBlfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChSYW5nZSwge30gLyogaW5wdXRzICovLCBhdHRycyBhcyB7fSBhcyBOYW1lZEF0dHJNYXApO1xufVxuIl19
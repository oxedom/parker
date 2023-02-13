/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
const CUSTOM_OPS = {};
/**
 * Register an Op for graph model executor. This allow you to register
 * TensorFlow custom op or override existing op.
 *
 * Here is an example of registering a new MatMul Op.
 * ```js
 * const customMatmul = (node) =>
 *    tf.matMul(
 *        node.inputs[0], node.inputs[1],
 *        node.attrs['transpose_a'], node.attrs['transpose_b']);
 *
 * tf.registerOp('MatMul', customMatmul);
 * ```
 * The inputs and attrs of the node object is based on the TensorFlow op
 * registry.
 *
 * @param name The Tensorflow Op name.
 * @param opFunc An op function which is called with the current graph node
 * during execution and needs to return a tensor or a list of tensors. The node
 * has the following attributes:
 *    - attr: A map from attribute name to its value
 *    - inputs: A list of input tensors
 *
 * @doc {heading: 'Models', subheading: 'Op Registry'}
 */
export function registerOp(name, opFunc) {
    const opMapper = {
        tfOpName: name,
        category: 'custom',
        inputs: [],
        attrs: [],
        customExecutor: opFunc
    };
    CUSTOM_OPS[name] = opMapper;
}
/**
 * Retrieve the OpMapper object for the registered op.
 *
 * @param name The Tensorflow Op name.
 *
 * @doc {heading: 'Models', subheading: 'Op Registry'}
 */
export function getRegisteredOp(name) {
    return CUSTOM_OPS[name];
}
/**
 * Deregister the Op for graph model executor.
 *
 * @param name The Tensorflow Op name.
 *
 * @doc {heading: 'Models', subheading: 'Op Registry'}
 */
export function deregisterOp(name) {
    delete CUSTOM_OPS[name];
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVnaXN0ZXIuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvbnZlcnRlci9zcmMvb3BlcmF0aW9ucy9jdXN0b21fb3AvcmVnaXN0ZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQ0E7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUgsTUFBTSxVQUFVLEdBQThCLEVBQUUsQ0FBQztBQUVqRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBd0JHO0FBQ0gsTUFBTSxVQUFVLFVBQVUsQ0FBQyxJQUFZLEVBQUUsTUFBa0I7SUFDekQsTUFBTSxRQUFRLEdBQWE7UUFDekIsUUFBUSxFQUFFLElBQUk7UUFDZCxRQUFRLEVBQUUsUUFBUTtRQUNsQixNQUFNLEVBQUUsRUFBRTtRQUNWLEtBQUssRUFBRSxFQUFFO1FBQ1QsY0FBYyxFQUFFLE1BQU07S0FDdkIsQ0FBQztJQUVGLFVBQVUsQ0FBQyxJQUFJLENBQUMsR0FBRyxRQUFRLENBQUM7QUFDOUIsQ0FBQztBQUVEOzs7Ozs7R0FNRztBQUNILE1BQU0sVUFBVSxlQUFlLENBQUMsSUFBWTtJQUMxQyxPQUFPLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUMxQixDQUFDO0FBRUQ7Ozs7OztHQU1HO0FBQ0gsTUFBTSxVQUFVLFlBQVksQ0FBQyxJQUFZO0lBQ3ZDLE9BQU8sVUFBVSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQzFCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyJcbi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtPcEV4ZWN1dG9yLCBPcE1hcHBlcn0gZnJvbSAnLi4vdHlwZXMnO1xuXG5jb25zdCBDVVNUT01fT1BTOiB7W2tleTogc3RyaW5nXTogT3BNYXBwZXJ9ID0ge307XG5cbi8qKlxuICogUmVnaXN0ZXIgYW4gT3AgZm9yIGdyYXBoIG1vZGVsIGV4ZWN1dG9yLiBUaGlzIGFsbG93IHlvdSB0byByZWdpc3RlclxuICogVGVuc29yRmxvdyBjdXN0b20gb3Agb3Igb3ZlcnJpZGUgZXhpc3Rpbmcgb3AuXG4gKlxuICogSGVyZSBpcyBhbiBleGFtcGxlIG9mIHJlZ2lzdGVyaW5nIGEgbmV3IE1hdE11bCBPcC5cbiAqIGBgYGpzXG4gKiBjb25zdCBjdXN0b21NYXRtdWwgPSAobm9kZSkgPT5cbiAqICAgIHRmLm1hdE11bChcbiAqICAgICAgICBub2RlLmlucHV0c1swXSwgbm9kZS5pbnB1dHNbMV0sXG4gKiAgICAgICAgbm9kZS5hdHRyc1sndHJhbnNwb3NlX2EnXSwgbm9kZS5hdHRyc1sndHJhbnNwb3NlX2InXSk7XG4gKlxuICogdGYucmVnaXN0ZXJPcCgnTWF0TXVsJywgY3VzdG9tTWF0bXVsKTtcbiAqIGBgYFxuICogVGhlIGlucHV0cyBhbmQgYXR0cnMgb2YgdGhlIG5vZGUgb2JqZWN0IGlzIGJhc2VkIG9uIHRoZSBUZW5zb3JGbG93IG9wXG4gKiByZWdpc3RyeS5cbiAqXG4gKiBAcGFyYW0gbmFtZSBUaGUgVGVuc29yZmxvdyBPcCBuYW1lLlxuICogQHBhcmFtIG9wRnVuYyBBbiBvcCBmdW5jdGlvbiB3aGljaCBpcyBjYWxsZWQgd2l0aCB0aGUgY3VycmVudCBncmFwaCBub2RlXG4gKiBkdXJpbmcgZXhlY3V0aW9uIGFuZCBuZWVkcyB0byByZXR1cm4gYSB0ZW5zb3Igb3IgYSBsaXN0IG9mIHRlbnNvcnMuIFRoZSBub2RlXG4gKiBoYXMgdGhlIGZvbGxvd2luZyBhdHRyaWJ1dGVzOlxuICogICAgLSBhdHRyOiBBIG1hcCBmcm9tIGF0dHJpYnV0ZSBuYW1lIHRvIGl0cyB2YWx1ZVxuICogICAgLSBpbnB1dHM6IEEgbGlzdCBvZiBpbnB1dCB0ZW5zb3JzXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdPcCBSZWdpc3RyeSd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiByZWdpc3Rlck9wKG5hbWU6IHN0cmluZywgb3BGdW5jOiBPcEV4ZWN1dG9yKSB7XG4gIGNvbnN0IG9wTWFwcGVyOiBPcE1hcHBlciA9IHtcbiAgICB0Zk9wTmFtZTogbmFtZSxcbiAgICBjYXRlZ29yeTogJ2N1c3RvbScsXG4gICAgaW5wdXRzOiBbXSxcbiAgICBhdHRyczogW10sXG4gICAgY3VzdG9tRXhlY3V0b3I6IG9wRnVuY1xuICB9O1xuXG4gIENVU1RPTV9PUFNbbmFtZV0gPSBvcE1hcHBlcjtcbn1cblxuLyoqXG4gKiBSZXRyaWV2ZSB0aGUgT3BNYXBwZXIgb2JqZWN0IGZvciB0aGUgcmVnaXN0ZXJlZCBvcC5cbiAqXG4gKiBAcGFyYW0gbmFtZSBUaGUgVGVuc29yZmxvdyBPcCBuYW1lLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnT3AgUmVnaXN0cnknfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0UmVnaXN0ZXJlZE9wKG5hbWU6IHN0cmluZyk6IE9wTWFwcGVyIHtcbiAgcmV0dXJuIENVU1RPTV9PUFNbbmFtZV07XG59XG5cbi8qKlxuICogRGVyZWdpc3RlciB0aGUgT3AgZm9yIGdyYXBoIG1vZGVsIGV4ZWN1dG9yLlxuICpcbiAqIEBwYXJhbSBuYW1lIFRoZSBUZW5zb3JmbG93IE9wIG5hbWUuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdPcCBSZWdpc3RyeSd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBkZXJlZ2lzdGVyT3AobmFtZTogc3RyaW5nKSB7XG4gIGRlbGV0ZSBDVVNUT01fT1BTW25hbWVdO1xufVxuIl19
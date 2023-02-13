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
import { Atan2 } from '@tensorflow/tfjs-core';
import { binaryKernelFunc, CHECK_NAN_SNIPPET_BINARY, CHECK_NAN_SNIPPET_BINARY_PACKED } from '../kernel_utils/kernel_funcs_utils';
const ATAN2 = CHECK_NAN_SNIPPET_BINARY + `
  return atan(a, b);
`;
const ATAN2_PACKED = `
  vec4 result = atan(a, b);
  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
  ` +
    CHECK_NAN_SNIPPET_BINARY_PACKED + `
  return result;
`;
export const atan2 = binaryKernelFunc({ opSnippet: ATAN2, packedOpSnippet: ATAN2_PACKED });
export const atan2Config = {
    kernelName: Atan2,
    backendName: 'webgl',
    kernelFunc: atan2,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQXRhbjIuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ2wvc3JjL2tlcm5lbHMvQXRhbjIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRzVDLE9BQU8sRUFBQyxnQkFBZ0IsRUFBRSx3QkFBd0IsRUFBRSwrQkFBK0IsRUFBQyxNQUFNLG9DQUFvQyxDQUFDO0FBRS9ILE1BQU0sS0FBSyxHQUFHLHdCQUF3QixHQUFHOztDQUV4QyxDQUFDO0FBRUYsTUFBTSxZQUFZLEdBQUc7OztHQUdsQjtJQUNDLCtCQUErQixHQUFHOztDQUVyQyxDQUFDO0FBRUYsTUFBTSxDQUFDLE1BQU0sS0FBSyxHQUNkLGdCQUFnQixDQUFDLEVBQUMsU0FBUyxFQUFFLEtBQUssRUFBRSxlQUFlLEVBQUUsWUFBWSxFQUFDLENBQUMsQ0FBQztBQUV4RSxNQUFNLENBQUMsTUFBTSxXQUFXLEdBQWlCO0lBQ3ZDLFVBQVUsRUFBRSxLQUFLO0lBQ2pCLFdBQVcsRUFBRSxPQUFPO0lBQ3BCLFVBQVUsRUFBRSxLQUFLO0NBQ2xCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7QXRhbjJ9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge0tlcm5lbENvbmZpZ30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtiaW5hcnlLZXJuZWxGdW5jLCBDSEVDS19OQU5fU05JUFBFVF9CSU5BUlksIENIRUNLX05BTl9TTklQUEVUX0JJTkFSWV9QQUNLRUR9IGZyb20gJy4uL2tlcm5lbF91dGlscy9rZXJuZWxfZnVuY3NfdXRpbHMnO1xuXG5jb25zdCBBVEFOMiA9IENIRUNLX05BTl9TTklQUEVUX0JJTkFSWSArIGBcbiAgcmV0dXJuIGF0YW4oYSwgYik7XG5gO1xuXG5jb25zdCBBVEFOMl9QQUNLRUQgPSBgXG4gIHZlYzQgcmVzdWx0ID0gYXRhbihhLCBiKTtcbiAgdmVjNCBpc05hTiA9IG1pbih2ZWM0KGlzbmFuKGEpKSArIHZlYzQoaXNuYW4oYikpLCB2ZWM0KDEuMCkpO1xuICBgICtcbiAgICBDSEVDS19OQU5fU05JUFBFVF9CSU5BUllfUEFDS0VEICsgYFxuICByZXR1cm4gcmVzdWx0O1xuYDtcblxuZXhwb3J0IGNvbnN0IGF0YW4yID1cbiAgICBiaW5hcnlLZXJuZWxGdW5jKHtvcFNuaXBwZXQ6IEFUQU4yLCBwYWNrZWRPcFNuaXBwZXQ6IEFUQU4yX1BBQ0tFRH0pO1xuXG5leHBvcnQgY29uc3QgYXRhbjJDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQXRhbjIsXG4gIGJhY2tlbmROYW1lOiAnd2ViZ2wnLFxuICBrZXJuZWxGdW5jOiBhdGFuMixcbn07XG4iXX0=
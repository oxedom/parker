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
import { Pow } from '@tensorflow/tfjs-core';
import { CHECK_NAN_SNIPPET } from '../binaryop_packed_gpu';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
const POW = `
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`;
const POW_PACKED = `
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  vec4 isNaN = vec4(lessThan(a, vec4(0.0))) * vec4(lessThan(floor(b), b));
  ` +
    CHECK_NAN_SNIPPET + `
  return result;
`;
export const pow = binaryKernelFunc({ opSnippet: POW, packedOpSnippet: POW_PACKED });
export const powConfig = {
    kernelName: Pow,
    backendName: 'webgl',
    kernelFunc: pow
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUG93LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL1Bvdy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLEdBQUcsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXBFLE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQ3pELE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLG9DQUFvQyxDQUFDO0FBRXBFLE1BQU0sR0FBRyxHQUFHOzs7Ozs7Ozs7Q0FTWCxDQUFDO0FBRUYsTUFBTSxVQUFVLEdBQUc7Ozs7Ozs7Ozs7Ozs7O0dBY2hCO0lBQ0MsaUJBQWlCLEdBQUc7O0NBRXZCLENBQUM7QUFFRixNQUFNLENBQUMsTUFBTSxHQUFHLEdBQ1osZ0JBQWdCLENBQUMsRUFBQyxTQUFTLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBRSxVQUFVLEVBQUMsQ0FBQyxDQUFDO0FBRXBFLE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBaUI7SUFDckMsVUFBVSxFQUFFLEdBQUc7SUFDZixXQUFXLEVBQUUsT0FBTztJQUNwQixVQUFVLEVBQUUsR0FBdUI7Q0FDcEMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFBvd30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtDSEVDS19OQU5fU05JUFBFVH0gZnJvbSAnLi4vYmluYXJ5b3BfcGFja2VkX2dwdSc7XG5pbXBvcnQge2JpbmFyeUtlcm5lbEZ1bmN9IGZyb20gJy4uL2tlcm5lbF91dGlscy9rZXJuZWxfZnVuY3NfdXRpbHMnO1xuXG5jb25zdCBQT1cgPSBgXG4gIGlmKGEgPCAwLjAgJiYgZmxvb3IoYikgPCBiKXtcbiAgICByZXR1cm4gTkFOO1xuICB9XG4gIGlmIChiID09IDAuMCkge1xuICAgIHJldHVybiAxLjA7XG4gIH1cbiAgcmV0dXJuIChyb3VuZChtb2QoYiwgMi4wKSkgIT0gMSkgP1xuICAgICAgcG93KGFicyhhKSwgYikgOiBzaWduKGEpICogcG93KGFicyhhKSwgYik7XG5gO1xuXG5jb25zdCBQT1dfUEFDS0VEID0gYFxuICAvLyBpc01vZFJvdW5kMSBoYXMgMSBmb3IgY29tcG9uZW50cyB3aXRoIHJvdW5kKG1vZChiLCAyLjApKSA9PSAxLCAwIG90aGVyd2lzZS5cbiAgdmVjNCBpc01vZFJvdW5kMSA9IHZlYzQoZXF1YWwocm91bmQobW9kKGIsIDIuMCkpLCBpdmVjNCgxKSkpO1xuICB2ZWM0IG11bHRpcGxpZXIgPSBzaWduKGEpICogaXNNb2RSb3VuZDEgKyAodmVjNCgxLjApIC0gaXNNb2RSb3VuZDEpO1xuICB2ZWM0IHJlc3VsdCA9IG11bHRpcGxpZXIgKiBwb3coYWJzKGEpLCBiKTtcblxuICAvLyBFbnN1cmUgdGhhdCBhXjAgPSAxLCBpbmNsdWRpbmcgMF4wID0gMSBhcyB0aGlzIGNvcnJlc3BvbmQgdG8gVEYgYW5kIEpTXG4gIGJ2ZWM0IGlzRXhwWmVybyA9IGVxdWFsKGIsIHZlYzQoMC4wKSk7XG4gIHJlc3VsdC5yID0gaXNFeHBaZXJvLnIgPyAxLjAgOiByZXN1bHQucjtcbiAgcmVzdWx0LmcgPSBpc0V4cFplcm8uZyA/IDEuMCA6IHJlc3VsdC5nO1xuICByZXN1bHQuYiA9IGlzRXhwWmVyby5iID8gMS4wIDogcmVzdWx0LmI7XG4gIHJlc3VsdC5hID0gaXNFeHBaZXJvLmEgPyAxLjAgOiByZXN1bHQuYTtcblxuICB2ZWM0IGlzTmFOID0gdmVjNChsZXNzVGhhbihhLCB2ZWM0KDAuMCkpKSAqIHZlYzQobGVzc1RoYW4oZmxvb3IoYiksIGIpKTtcbiAgYCArXG4gICAgQ0hFQ0tfTkFOX1NOSVBQRVQgKyBgXG4gIHJldHVybiByZXN1bHQ7XG5gO1xuXG5leHBvcnQgY29uc3QgcG93ID1cbiAgICBiaW5hcnlLZXJuZWxGdW5jKHtvcFNuaXBwZXQ6IFBPVywgcGFja2VkT3BTbmlwcGV0OiBQT1dfUEFDS0VEfSk7XG5cbmV4cG9ydCBjb25zdCBwb3dDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogUG93LFxuICBiYWNrZW5kTmFtZTogJ3dlYmdsJyxcbiAga2VybmVsRnVuYzogcG93IGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=
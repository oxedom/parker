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
import { Mod } from '@tensorflow/tfjs-core';
import { CHECK_NAN_SNIPPET } from '../binaryop_packed_gpu';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
const MOD = `if (b == 0.0) return NAN;
  return mod(a, b);`;
const MOD_PACKED = `
  vec4 result = mod(a, b);
  vec4 isNaN = vec4(equal(b, vec4(0.0)));
  ` +
    CHECK_NAN_SNIPPET + `
  return result;
`;
export const mod = binaryKernelFunc({
    opSnippet: MOD,
    packedOpSnippet: MOD_PACKED,
});
export const modConfig = {
    kernelName: Mod,
    backendName: 'webgl',
    kernelFunc: mod
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTW9kLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL01vZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLEdBQUcsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXBFLE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQ3pELE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLG9DQUFvQyxDQUFDO0FBRXBFLE1BQU0sR0FBRyxHQUFHO29CQUNRLENBQUM7QUFFckIsTUFBTSxVQUFVLEdBQUc7OztHQUdoQjtJQUNDLGlCQUFpQixHQUFHOztDQUV2QixDQUFDO0FBRUYsTUFBTSxDQUFDLE1BQU0sR0FBRyxHQUFHLGdCQUFnQixDQUFDO0lBQ2xDLFNBQVMsRUFBRSxHQUFHO0lBQ2QsZUFBZSxFQUFFLFVBQVU7Q0FDNUIsQ0FBQyxDQUFDO0FBRUgsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUFpQjtJQUNyQyxVQUFVLEVBQUUsR0FBRztJQUNmLFdBQVcsRUFBRSxPQUFPO0lBQ3BCLFVBQVUsRUFBRSxHQUF1QjtDQUNwQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgS2VybmVsRnVuYywgTW9kfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0NIRUNLX05BTl9TTklQUEVUfSBmcm9tICcuLi9iaW5hcnlvcF9wYWNrZWRfZ3B1JztcbmltcG9ydCB7YmluYXJ5S2VybmVsRnVuY30gZnJvbSAnLi4va2VybmVsX3V0aWxzL2tlcm5lbF9mdW5jc191dGlscyc7XG5cbmNvbnN0IE1PRCA9IGBpZiAoYiA9PSAwLjApIHJldHVybiBOQU47XG4gIHJldHVybiBtb2QoYSwgYik7YDtcblxuY29uc3QgTU9EX1BBQ0tFRCA9IGBcbiAgdmVjNCByZXN1bHQgPSBtb2QoYSwgYik7XG4gIHZlYzQgaXNOYU4gPSB2ZWM0KGVxdWFsKGIsIHZlYzQoMC4wKSkpO1xuICBgICtcbiAgICBDSEVDS19OQU5fU05JUFBFVCArIGBcbiAgcmV0dXJuIHJlc3VsdDtcbmA7XG5cbmV4cG9ydCBjb25zdCBtb2QgPSBiaW5hcnlLZXJuZWxGdW5jKHtcbiAgb3BTbmlwcGV0OiBNT0QsXG4gIHBhY2tlZE9wU25pcHBldDogTU9EX1BBQ0tFRCxcbn0pO1xuXG5leHBvcnQgY29uc3QgbW9kQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IE1vZCxcbiAgYmFja2VuZE5hbWU6ICd3ZWJnbCcsXG4gIGtlcm5lbEZ1bmM6IG1vZCBhcyB7fSBhcyBLZXJuZWxGdW5jXG59O1xuIl19
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
import { Minimum } from '@tensorflow/tfjs-core';
import { CHECK_NAN_SNIPPET } from '../binaryop_gpu';
import { CHECK_NAN_SNIPPET as CHECK_NAN_SNIPPET_PACKED } from '../binaryop_packed_gpu';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { minimumImplCPU } from '../kernel_utils/shared';
const MINIMUM = CHECK_NAN_SNIPPET + `
  return min(a, b);
`;
const MINIMUM_PACKED = `
  vec4 result = vec4(min(a, b));
  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
  ` +
    CHECK_NAN_SNIPPET_PACKED + `
  return result;
`;
export const minimum = binaryKernelFunc({
    opSnippet: MINIMUM,
    packedOpSnippet: MINIMUM_PACKED,
    cpuKernelImpl: minimumImplCPU
});
export const minimumConfig = {
    kernelName: Minimum,
    backendName: 'webgl',
    kernelFunc: minimum
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTWluaW11bS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMva2VybmVscy9NaW5pbXVtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBMkIsT0FBTyxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFeEUsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFDbEQsT0FBTyxFQUFDLGlCQUFpQixJQUFJLHdCQUF3QixFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDckYsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sb0NBQW9DLENBQUM7QUFDcEUsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBRXRELE1BQU0sT0FBTyxHQUFHLGlCQUFpQixHQUFHOztDQUVuQyxDQUFDO0FBRUYsTUFBTSxjQUFjLEdBQUc7OztHQUdwQjtJQUNDLHdCQUF3QixHQUFHOztDQUU5QixDQUFDO0FBRUYsTUFBTSxDQUFDLE1BQU0sT0FBTyxHQUFHLGdCQUFnQixDQUFDO0lBQ3RDLFNBQVMsRUFBRSxPQUFPO0lBQ2xCLGVBQWUsRUFBRSxjQUFjO0lBQy9CLGFBQWEsRUFBRSxjQUFjO0NBQzlCLENBQUMsQ0FBQztBQUVILE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBaUI7SUFDekMsVUFBVSxFQUFFLE9BQU87SUFDbkIsV0FBVyxFQUFFLE9BQU87SUFDcEIsVUFBVSxFQUFFLE9BQTJCO0NBQ3hDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBNaW5pbXVtfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0NIRUNLX05BTl9TTklQUEVUfSBmcm9tICcuLi9iaW5hcnlvcF9ncHUnO1xuaW1wb3J0IHtDSEVDS19OQU5fU05JUFBFVCBhcyBDSEVDS19OQU5fU05JUFBFVF9QQUNLRUR9IGZyb20gJy4uL2JpbmFyeW9wX3BhY2tlZF9ncHUnO1xuaW1wb3J0IHtiaW5hcnlLZXJuZWxGdW5jfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMva2VybmVsX2Z1bmNzX3V0aWxzJztcbmltcG9ydCB7bWluaW11bUltcGxDUFV9IGZyb20gJy4uL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuXG5jb25zdCBNSU5JTVVNID0gQ0hFQ0tfTkFOX1NOSVBQRVQgKyBgXG4gIHJldHVybiBtaW4oYSwgYik7XG5gO1xuXG5jb25zdCBNSU5JTVVNX1BBQ0tFRCA9IGBcbiAgdmVjNCByZXN1bHQgPSB2ZWM0KG1pbihhLCBiKSk7XG4gIHZlYzQgaXNOYU4gPSBtaW4odmVjNChpc25hbihhKSkgKyB2ZWM0KGlzbmFuKGIpKSwgdmVjNCgxLjApKTtcbiAgYCArXG4gICAgQ0hFQ0tfTkFOX1NOSVBQRVRfUEFDS0VEICsgYFxuICByZXR1cm4gcmVzdWx0O1xuYDtcblxuZXhwb3J0IGNvbnN0IG1pbmltdW0gPSBiaW5hcnlLZXJuZWxGdW5jKHtcbiAgb3BTbmlwcGV0OiBNSU5JTVVNLFxuICBwYWNrZWRPcFNuaXBwZXQ6IE1JTklNVU1fUEFDS0VELFxuICBjcHVLZXJuZWxJbXBsOiBtaW5pbXVtSW1wbENQVVxufSk7XG5cbmV4cG9ydCBjb25zdCBtaW5pbXVtQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IE1pbmltdW0sXG4gIGJhY2tlbmROYW1lOiAnd2ViZ2wnLFxuICBrZXJuZWxGdW5jOiBtaW5pbXVtIGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=
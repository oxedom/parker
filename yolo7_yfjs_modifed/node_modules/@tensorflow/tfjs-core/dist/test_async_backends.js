#!/usr/bin/env node
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
/**
 * This file tests that we don't have any dataSyncs in the unconstrainted tests
 * so that we can run backends that have async init and async data reads against
 * our exported test files.
 */
// Use require here to workaround this being a circular dependency.
// This should only be done in tests.
// tslint:disable-next-line: no-require-imports
require('@tensorflow/tfjs-backend-cpu');
import './index';
import './public/chained_ops/register_all_chained_ops';
import './register_all_gradients';
import { setTestEnvs } from './jasmine_util';
import { registerBackend, engine } from './globals';
import { KernelBackend } from './backends/backend';
import { getKernelsForBackend, registerKernel } from './kernel_registry';
// tslint:disable-next-line:no-require-imports
const jasmine = require('jasmine');
process.on('unhandledRejection', e => {
    throw e;
});
class AsyncCPUBackend extends KernelBackend {
}
const asyncBackend = new AsyncCPUBackend();
// backend is cast as any so that we can access methods through bracket
// notation.
const backend = engine().findBackend('cpu');
const proxyBackend = new Proxy(asyncBackend, {
    get(target, name, receiver) {
        if (name === 'readSync') {
            throw new Error(`Found dataSync() in a unit test. This is disabled so unit tests ` +
                `can run in backends that only support async data. Please use ` +
                `.data() in unit tests or if you truly are testing dataSync(), ` +
                `constrain your test with SYNC_BACKEND_ENVS`);
        }
        //@ts-ignore;
        const origSymbol = backend[name];
        if (typeof origSymbol === 'function') {
            // tslint:disable-next-line:no-any
            return (...args) => {
                return origSymbol.apply(backend, args);
            };
        }
        else {
            return origSymbol;
        }
    }
});
const proxyBackendName = 'test-async-cpu';
// The registration is async on purpose, so we know our testing infra works
// with backends that have async init (e.g. WASM and WebGPU).
registerBackend(proxyBackendName, async () => proxyBackend);
// All the kernels are registered under the 'cpu' name, so we need to
// register them also under the proxy backend name.
const kernels = getKernelsForBackend('cpu');
kernels.forEach(({ kernelName, kernelFunc, setupFunc }) => {
    registerKernel({ kernelName, backendName: proxyBackendName, kernelFunc, setupFunc });
});
setTestEnvs([{
        name: proxyBackendName,
        backendName: proxyBackendName,
        isDataSync: false,
    }]);
const runner = new jasmine();
runner.loadConfig({ spec_files: ['tfjs-core/src/**/**_test.js'], random: false });
runner.execute();
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVzdF9hc3luY19iYWNrZW5kcy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvdGVzdF9hc3luY19iYWNrZW5kcy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiO0FBQ0E7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0g7Ozs7R0FJRztBQUVILG1FQUFtRTtBQUNuRSxxQ0FBcUM7QUFDckMsK0NBQStDO0FBQy9DLE9BQU8sQ0FBQyw4QkFBOEIsQ0FBQyxDQUFDO0FBQ3hDLE9BQU8sU0FBUyxDQUFDO0FBQ2pCLE9BQU8sK0NBQStDLENBQUM7QUFDdkQsT0FBTywwQkFBMEIsQ0FBQztBQUNsQyxPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDM0MsT0FBTyxFQUFDLGVBQWUsRUFBRSxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEQsT0FBTyxFQUFDLGFBQWEsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBQ2pELE9BQU8sRUFBQyxvQkFBb0IsRUFBRSxjQUFjLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUV2RSw4Q0FBOEM7QUFDOUMsTUFBTSxPQUFPLEdBQUcsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDO0FBRW5DLE9BQU8sQ0FBQyxFQUFFLENBQUMsb0JBQW9CLEVBQUUsQ0FBQyxDQUFDLEVBQUU7SUFDbkMsTUFBTSxDQUFDLENBQUM7QUFDVixDQUFDLENBQUMsQ0FBQztBQUVILE1BQU0sZUFBZ0IsU0FBUSxhQUFhO0NBQUc7QUFDOUMsTUFBTSxZQUFZLEdBQUcsSUFBSSxlQUFlLEVBQUUsQ0FBQztBQUUzQyx1RUFBdUU7QUFDdkUsWUFBWTtBQUNaLE1BQU0sT0FBTyxHQUFrQixNQUFNLEVBQUUsQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDM0QsTUFBTSxZQUFZLEdBQUcsSUFBSSxLQUFLLENBQUMsWUFBWSxFQUFFO0lBQzNDLEdBQUcsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFFBQVE7UUFDeEIsSUFBSSxJQUFJLEtBQUssVUFBVSxFQUFFO1lBQ3ZCLE1BQU0sSUFBSSxLQUFLLENBQ1gsa0VBQWtFO2dCQUNsRSwrREFBK0Q7Z0JBQy9ELGdFQUFnRTtnQkFDaEUsNENBQTRDLENBQUMsQ0FBQztTQUNuRDtRQUNELGFBQWE7UUFDYixNQUFNLFVBQVUsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDakMsSUFBSSxPQUFPLFVBQVUsS0FBSyxVQUFVLEVBQUU7WUFDcEMsa0NBQWtDO1lBQ2xDLE9BQU8sQ0FBQyxHQUFHLElBQVcsRUFBRSxFQUFFO2dCQUN4QixPQUFPLFVBQVUsQ0FBQyxLQUFLLENBQUMsT0FBTyxFQUFFLElBQUksQ0FBQyxDQUFDO1lBQ3pDLENBQUMsQ0FBQztTQUNIO2FBQU07WUFDTCxPQUFPLFVBQVUsQ0FBQztTQUNuQjtJQUNILENBQUM7Q0FDRixDQUFDLENBQUM7QUFFSCxNQUFNLGdCQUFnQixHQUFHLGdCQUFnQixDQUFDO0FBRTFDLDJFQUEyRTtBQUMzRSw2REFBNkQ7QUFDN0QsZUFBZSxDQUFDLGdCQUFnQixFQUFFLEtBQUssSUFBSSxFQUFFLENBQUMsWUFBWSxDQUFDLENBQUM7QUFFNUQscUVBQXFFO0FBQ3JFLG1EQUFtRDtBQUNuRCxNQUFNLE9BQU8sR0FBRyxvQkFBb0IsQ0FBQyxLQUFLLENBQUMsQ0FBQztBQUM1QyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsRUFBQyxVQUFVLEVBQUUsVUFBVSxFQUFFLFNBQVMsRUFBQyxFQUFFLEVBQUU7SUFDdEQsY0FBYyxDQUNWLEVBQUMsVUFBVSxFQUFFLFdBQVcsRUFBRSxnQkFBZ0IsRUFBRSxVQUFVLEVBQUUsU0FBUyxFQUFDLENBQUMsQ0FBQztBQUMxRSxDQUFDLENBQUMsQ0FBQztBQUVILFdBQVcsQ0FBQyxDQUFDO1FBQ1gsSUFBSSxFQUFFLGdCQUFnQjtRQUN0QixXQUFXLEVBQUUsZ0JBQWdCO1FBQzdCLFVBQVUsRUFBRSxLQUFLO0tBQ2xCLENBQUMsQ0FBQyxDQUFDO0FBRUosTUFBTSxNQUFNLEdBQUcsSUFBSSxPQUFPLEVBQUUsQ0FBQztBQUU3QixNQUFNLENBQUMsVUFBVSxDQUFDLEVBQUMsVUFBVSxFQUFFLENBQUMsNkJBQTZCLENBQUMsRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFDLENBQUMsQ0FBQztBQUNoRixNQUFNLENBQUMsT0FBTyxFQUFFLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIjIS91c3IvYmluL2VudiBub2RlXG4vKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG4vKipcbiAqIFRoaXMgZmlsZSB0ZXN0cyB0aGF0IHdlIGRvbid0IGhhdmUgYW55IGRhdGFTeW5jcyBpbiB0aGUgdW5jb25zdHJhaW50ZWQgdGVzdHNcbiAqIHNvIHRoYXQgd2UgY2FuIHJ1biBiYWNrZW5kcyB0aGF0IGhhdmUgYXN5bmMgaW5pdCBhbmQgYXN5bmMgZGF0YSByZWFkcyBhZ2FpbnN0XG4gKiBvdXIgZXhwb3J0ZWQgdGVzdCBmaWxlcy5cbiAqL1xuXG4vLyBVc2UgcmVxdWlyZSBoZXJlIHRvIHdvcmthcm91bmQgdGhpcyBiZWluZyBhIGNpcmN1bGFyIGRlcGVuZGVuY3kuXG4vLyBUaGlzIHNob3VsZCBvbmx5IGJlIGRvbmUgaW4gdGVzdHMuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXJlcXVpcmUtaW1wb3J0c1xucmVxdWlyZSgnQHRlbnNvcmZsb3cvdGZqcy1iYWNrZW5kLWNwdScpO1xuaW1wb3J0ICcuL2luZGV4JztcbmltcG9ydCAnLi9wdWJsaWMvY2hhaW5lZF9vcHMvcmVnaXN0ZXJfYWxsX2NoYWluZWRfb3BzJztcbmltcG9ydCAnLi9yZWdpc3Rlcl9hbGxfZ3JhZGllbnRzJztcbmltcG9ydCB7c2V0VGVzdEVudnN9IGZyb20gJy4vamFzbWluZV91dGlsJztcbmltcG9ydCB7cmVnaXN0ZXJCYWNrZW5kLCBlbmdpbmV9IGZyb20gJy4vZ2xvYmFscyc7XG5pbXBvcnQge0tlcm5lbEJhY2tlbmR9IGZyb20gJy4vYmFja2VuZHMvYmFja2VuZCc7XG5pbXBvcnQge2dldEtlcm5lbHNGb3JCYWNrZW5kLCByZWdpc3Rlcktlcm5lbH0gZnJvbSAnLi9rZXJuZWxfcmVnaXN0cnknO1xuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tcmVxdWlyZS1pbXBvcnRzXG5jb25zdCBqYXNtaW5lID0gcmVxdWlyZSgnamFzbWluZScpO1xuXG5wcm9jZXNzLm9uKCd1bmhhbmRsZWRSZWplY3Rpb24nLCBlID0+IHtcbiAgdGhyb3cgZTtcbn0pO1xuXG5jbGFzcyBBc3luY0NQVUJhY2tlbmQgZXh0ZW5kcyBLZXJuZWxCYWNrZW5kIHt9XG5jb25zdCBhc3luY0JhY2tlbmQgPSBuZXcgQXN5bmNDUFVCYWNrZW5kKCk7XG5cbi8vIGJhY2tlbmQgaXMgY2FzdCBhcyBhbnkgc28gdGhhdCB3ZSBjYW4gYWNjZXNzIG1ldGhvZHMgdGhyb3VnaCBicmFja2V0XG4vLyBub3RhdGlvbi5cbmNvbnN0IGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQgPSBlbmdpbmUoKS5maW5kQmFja2VuZCgnY3B1Jyk7XG5jb25zdCBwcm94eUJhY2tlbmQgPSBuZXcgUHJveHkoYXN5bmNCYWNrZW5kLCB7XG4gIGdldCh0YXJnZXQsIG5hbWUsIHJlY2VpdmVyKSB7XG4gICAgaWYgKG5hbWUgPT09ICdyZWFkU3luYycpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgRm91bmQgZGF0YVN5bmMoKSBpbiBhIHVuaXQgdGVzdC4gVGhpcyBpcyBkaXNhYmxlZCBzbyB1bml0IHRlc3RzIGAgK1xuICAgICAgICAgIGBjYW4gcnVuIGluIGJhY2tlbmRzIHRoYXQgb25seSBzdXBwb3J0IGFzeW5jIGRhdGEuIFBsZWFzZSB1c2UgYCArXG4gICAgICAgICAgYC5kYXRhKCkgaW4gdW5pdCB0ZXN0cyBvciBpZiB5b3UgdHJ1bHkgYXJlIHRlc3RpbmcgZGF0YVN5bmMoKSwgYCArXG4gICAgICAgICAgYGNvbnN0cmFpbiB5b3VyIHRlc3Qgd2l0aCBTWU5DX0JBQ0tFTkRfRU5WU2ApO1xuICAgIH1cbiAgICAvL0B0cy1pZ25vcmU7XG4gICAgY29uc3Qgb3JpZ1N5bWJvbCA9IGJhY2tlbmRbbmFtZV07XG4gICAgaWYgKHR5cGVvZiBvcmlnU3ltYm9sID09PSAnZnVuY3Rpb24nKSB7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICByZXR1cm4gKC4uLmFyZ3M6IGFueVtdKSA9PiB7XG4gICAgICAgIHJldHVybiBvcmlnU3ltYm9sLmFwcGx5KGJhY2tlbmQsIGFyZ3MpO1xuICAgICAgfTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIG9yaWdTeW1ib2w7XG4gICAgfVxuICB9XG59KTtcblxuY29uc3QgcHJveHlCYWNrZW5kTmFtZSA9ICd0ZXN0LWFzeW5jLWNwdSc7XG5cbi8vIFRoZSByZWdpc3RyYXRpb24gaXMgYXN5bmMgb24gcHVycG9zZSwgc28gd2Uga25vdyBvdXIgdGVzdGluZyBpbmZyYSB3b3Jrc1xuLy8gd2l0aCBiYWNrZW5kcyB0aGF0IGhhdmUgYXN5bmMgaW5pdCAoZS5nLiBXQVNNIGFuZCBXZWJHUFUpLlxucmVnaXN0ZXJCYWNrZW5kKHByb3h5QmFja2VuZE5hbWUsIGFzeW5jICgpID0+IHByb3h5QmFja2VuZCk7XG5cbi8vIEFsbCB0aGUga2VybmVscyBhcmUgcmVnaXN0ZXJlZCB1bmRlciB0aGUgJ2NwdScgbmFtZSwgc28gd2UgbmVlZCB0b1xuLy8gcmVnaXN0ZXIgdGhlbSBhbHNvIHVuZGVyIHRoZSBwcm94eSBiYWNrZW5kIG5hbWUuXG5jb25zdCBrZXJuZWxzID0gZ2V0S2VybmVsc0ZvckJhY2tlbmQoJ2NwdScpO1xua2VybmVscy5mb3JFYWNoKCh7a2VybmVsTmFtZSwga2VybmVsRnVuYywgc2V0dXBGdW5jfSkgPT4ge1xuICByZWdpc3Rlcktlcm5lbChcbiAgICAgIHtrZXJuZWxOYW1lLCBiYWNrZW5kTmFtZTogcHJveHlCYWNrZW5kTmFtZSwga2VybmVsRnVuYywgc2V0dXBGdW5jfSk7XG59KTtcblxuc2V0VGVzdEVudnMoW3tcbiAgbmFtZTogcHJveHlCYWNrZW5kTmFtZSxcbiAgYmFja2VuZE5hbWU6IHByb3h5QmFja2VuZE5hbWUsXG4gIGlzRGF0YVN5bmM6IGZhbHNlLFxufV0pO1xuXG5jb25zdCBydW5uZXIgPSBuZXcgamFzbWluZSgpO1xuXG5ydW5uZXIubG9hZENvbmZpZyh7c3BlY19maWxlczogWyd0ZmpzLWNvcmUvc3JjLyoqLyoqX3Rlc3QuanMnXSwgcmFuZG9tOiBmYWxzZX0pO1xucnVubmVyLmV4ZWN1dGUoKTtcbiJdfQ==
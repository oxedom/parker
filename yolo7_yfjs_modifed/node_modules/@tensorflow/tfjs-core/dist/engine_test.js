/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import { ENGINE } from './engine';
import * as tf from './index';
import { ALL_ENVS, describeWithFlags, TestKernelBackend } from './jasmine_util';
import { expectArraysClose } from './test_util';
describe('Backend registration', () => {
    beforeAll(() => {
        // Silences backend registration warnings.
        spyOn(console, 'warn');
    });
    let registeredBackends = [];
    let registerBackend;
    beforeEach(() => {
        // Registering a backend changes global state (engine), so we wrap
        // registration to automatically remove registered backend at the end
        // of each test.
        registerBackend = (name, factory, priority) => {
            registeredBackends.push(name);
            return tf.registerBackend(name, factory, priority);
        };
        ENGINE.reset();
    });
    afterEach(() => {
        // Remove all registered backends at the end of each test.
        registeredBackends.forEach(name => {
            if (tf.findBackendFactory(name) != null) {
                tf.removeBackend(name);
            }
        });
        registeredBackends = [];
    });
    it('removeBackend disposes the backend and removes the factory', () => {
        let backend;
        const factory = () => {
            const newBackend = new TestKernelBackend();
            if (backend == null) {
                backend = newBackend;
                spyOn(backend, 'dispose').and.callThrough();
            }
            return newBackend;
        };
        registerBackend('test-backend', factory);
        expect(tf.findBackend('test-backend') != null).toBe(true);
        expect(tf.findBackend('test-backend')).toBe(backend);
        expect(tf.findBackendFactory('test-backend')).toBe(factory);
        tf.removeBackend('test-backend');
        expect(tf.findBackend('test-backend') == null).toBe(true);
        expect(tf.findBackend('test-backend')).toBe(null);
        expect(backend.dispose.calls.count()).toBe(1);
        expect(tf.findBackendFactory('test-backend')).toBe(null);
    });
    it('findBackend initializes the backend', () => {
        let backend;
        const factory = () => {
            const newBackend = new TestKernelBackend();
            if (backend == null) {
                backend = newBackend;
            }
            return newBackend;
        };
        registerBackend('custom-cpu', factory);
        expect(tf.findBackend('custom-cpu') != null).toBe(true);
        expect(tf.findBackend('custom-cpu')).toBe(backend);
        expect(tf.findBackendFactory('custom-cpu')).toBe(factory);
    });
    it('custom backend registration', () => {
        let backend;
        const priority = 103;
        registerBackend('custom-cpu', () => {
            const newBackend = new TestKernelBackend();
            if (backend == null) {
                backend = newBackend;
            }
            return newBackend;
        }, priority);
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(backend);
    });
    it('high priority backend registration fails, falls back', () => {
        let lowPriorityBackend;
        const lowPriority = 103;
        const highPriority = 104;
        registerBackend('custom-low-priority', () => {
            lowPriorityBackend = new TestKernelBackend();
            return lowPriorityBackend;
        }, lowPriority);
        registerBackend('custom-high-priority', () => {
            throw new Error(`High priority backend fails`);
        }, highPriority);
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(lowPriorityBackend);
        expect(tf.getBackend()).toBe('custom-low-priority');
    });
    it('low priority and high priority backends, setBackend low priority', () => {
        let lowPriorityBackend;
        let highPriorityBackend;
        const lowPriority = 103;
        const highPriority = 104;
        registerBackend('custom-low-priority', () => {
            lowPriorityBackend = new TestKernelBackend();
            return lowPriorityBackend;
        }, lowPriority);
        registerBackend('custom-high-priority', () => {
            highPriorityBackend = new TestKernelBackend();
            return highPriorityBackend;
        }, highPriority);
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(highPriorityBackend);
        expect(tf.getBackend()).toBe('custom-high-priority');
        tf.setBackend('custom-low-priority');
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(lowPriorityBackend);
        expect(tf.getBackend()).toBe('custom-low-priority');
    });
    it('default custom background null', () => {
        expect(tf.findBackend('custom')).toBeNull();
    });
    it('allow custom backend', () => {
        const backend = new TestKernelBackend();
        const success = registerBackend('custom', () => backend);
        expect(success).toBeTruthy();
        expect(tf.findBackend('custom')).toEqual(backend);
    });
    it('sync backend with await ready works', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('sync', () => testBackend);
        tf.setBackend('sync');
        expect(tf.getBackend()).toEqual('sync');
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
    });
    it('sync backend without await ready works', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('sync', () => testBackend);
        tf.setBackend('sync');
        expect(tf.getBackend()).toEqual('sync');
        expect(tf.backend()).toEqual(testBackend);
    });
    it('async backend with await ready works', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        });
        tf.setBackend('async');
        expect(tf.getBackend()).toEqual('async');
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
    });
    it('async backend without await ready does not work', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        });
        tf.setBackend('async');
        expect(tf.getBackend()).toEqual('async');
        expect(() => tf.backend())
            .toThrowError(/Backend 'async' has not yet been initialized./);
    });
    it('tf.square() fails if user does not await ready on async backend', async () => {
        registerBackend('async', async () => {
            await tf.nextFrame();
            return new TestKernelBackend();
        });
        tf.setBackend('async');
        expect(() => tf.square(2))
            .toThrowError(/Backend 'async' has not yet been initialized/);
    });
    it('tf.square() works when user awaits ready on async backend', async () => {
        registerBackend('async', async () => {
            await tf.nextFrame();
            return new TestKernelBackend();
        });
        tf.setBackend('async');
        await tf.ready();
        expect(() => tf.square(2)).toThrowError(/'write' not yet implemented/);
    });
    it('Registering async2 (higher priority) fails, async1 becomes active', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async1', async () => {
            await tf.nextFrame();
            return testBackend;
        }, 100 /* priority */);
        registerBackend('async2', async () => {
            await tf.nextFrame();
            throw new Error('failed to create async2');
        }, 101 /* priority */);
        // Await for the library to find the best backend that succesfully
        // initializes.
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
        expect(tf.getBackend()).toBe('async1');
    });
    it('Registering sync as higher priority and async as lower priority', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('sync', () => testBackend, 101 /* priority */);
        registerBackend('async', async () => {
            await tf.nextFrame();
            return new TestKernelBackend();
        }, 100 /* priority */);
        // No need to await for ready() since the highest priority one is sync.
        expect(tf.backend()).toEqual(testBackend);
        expect(tf.getBackend()).toBe('sync');
    });
    it('async as higher priority and sync as lower priority with await ready', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        }, 101 /* priority */);
        registerBackend('sync', () => new TestKernelBackend(), 100 /* priority */);
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
        expect(tf.getBackend()).toBe('async');
    });
    it('async as higher priority and sync as lower priority w/o await ready', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        }, 101 /* priority */);
        registerBackend('sync', () => new TestKernelBackend(), 100 /* priority */);
        expect(() => tf.backend())
            .toThrowError(/The highest priority backend 'async' has not yet been/);
    });
    it('Registering and setting a backend that fails to register', async () => {
        registerBackend('async', async () => {
            await tf.nextFrame();
            throw new Error('failed to create async');
        });
        const success = tf.setBackend('async');
        expect(tf.getBackend()).toBe('async');
        expect(() => tf.backend())
            .toThrowError(/Backend 'async' has not yet been initialized/);
        expect(await success).toBe(false);
    });
});
describeWithFlags('memory', ALL_ENVS, () => {
    it('Sum(float)', async () => {
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
        const sum = tf.tidy(() => {
            const a = tf.tensor1d([1, 2, 3, 4]);
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4 * 4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expectArraysClose(await sum.data(), [1 + 2 + 3 + 4]);
    });
    it('Sum(bool)', async () => {
        const sum = tf.tidy(() => {
            const a = tf.tensor1d([true, true, false, true], 'bool');
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expect(sum.dtype).toBe('int32');
        expectArraysClose(await sum.data(), [1 + 1 + 0 + 1]);
    });
    it('Sum(int32)', async () => {
        const sum = tf.tidy(() => {
            const a = tf.tensor1d([1, 1, 0, 1], 'int32');
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4 * 4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expect(sum.dtype).toBe('int32');
        expectArraysClose(await sum.data(), [1 + 1 + 0 + 1]);
    });
    it('string tensor', () => {
        const a = tf.tensor([['a', 'bb'], ['c', 'd']]);
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(5); // 5 letters, each 1 byte in utf8.
        a.dispose();
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
    });
    it('unreliable is true for string tensors', () => {
        tf.tensor('a');
        const mem = tf.memory();
        expect(mem.unreliable).toBe(true);
        const expectedReason = 'Memory usage by string tensors is approximate ' +
            '(2 bytes per character)';
        expect(mem.reasons.indexOf(expectedReason) >= 0).toBe(true);
    });
});
describeWithFlags('profile', ALL_ENVS, () => {
    it('squaring', async () => {
        const profile = await tf.profile(() => {
            const x = tf.tensor1d([1, 2, 3]);
            let x2 = x.square();
            x2.dispose();
            x2 = x.square();
            x2.dispose();
            return x;
        });
        const result = profile.result;
        expect(profile.newBytes).toBe(12);
        expect(profile.peakBytes).toBe(24);
        expect(profile.newTensors).toBe(1);
        expectArraysClose(await result.data(), [1, 2, 3]);
        expect(profile.kernels.length).toBe(2);
        // Test the types for `kernelTimeMs` and `extraInfo` to confirm the promises
        // are resolved.
        expect(profile.kernels[0].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[0].extraInfo instanceof Promise).toBe(false);
        expect(profile.kernels[1].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[1].extraInfo instanceof Promise).toBe(false);
        // The specific values of `kernelTimeMs` and `extraInfo` are tested in the
        // tests of Profiler.profileKernel, so their values are not tested here.
        expect(profile.kernels[0]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[0].kernelTimeMs,
            'extraInfo': profile.kernels[0].extraInfo
        });
        expect(profile.kernels[1]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[1].kernelTimeMs,
            'extraInfo': profile.kernels[1].extraInfo
        });
    });
    it('squaring without disposing', async () => {
        const profile = await tf.profile(() => {
            const x = tf.tensor1d([1, 2, 3]);
            const x2 = x.square();
            return x2;
        });
        const result = profile.result;
        expect(profile.newBytes).toBe(24);
        expect(profile.peakBytes).toBe(24);
        expect(profile.newTensors).toBe(2);
        expectArraysClose(await result.data(), [1, 4, 9]);
        expect(profile.kernels.length).toBe(1);
        expect(profile.kernels[0].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[0].extraInfo instanceof Promise).toBe(false);
        expect(profile.kernels[0]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[0].kernelTimeMs,
            'extraInfo': profile.kernels[0].extraInfo
        });
    });
    it('squaring in async query', async () => {
        const profile = await tf.profile(async () => {
            await new Promise(resolve => setTimeout(resolve, 1));
            const x = tf.tensor1d([1, 2, 3]);
            const x2 = x.square();
            x2.dispose();
            return x;
        });
        const result = profile.result;
        expect(profile.newBytes).toBe(12);
        expect(profile.peakBytes).toBe(24);
        expect(profile.newTensors).toBe(1);
        expectArraysClose(await result.data(), [1, 2, 3]);
        expect(profile.kernels.length).toBe(1);
        expect(profile.kernels[0].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[0].extraInfo instanceof Promise).toBe(false);
        expect(profile.kernels[0]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[0].kernelTimeMs,
            'extraInfo': profile.kernels[0].extraInfo
        });
    });
    it('reports correct kernelNames', async () => {
        const profile = await tf.profile(() => {
            const x = tf.tensor1d([1, 2, 3]);
            const x2 = x.square();
            const x3 = x2.abs();
            return x3;
        });
        expect(profile.kernelNames).toEqual(jasmine.arrayWithExactContents([
            'Square', 'Abs'
        ]));
    });
});
describeWithFlags('disposeVariables', ALL_ENVS, () => {
    it('reuse same name variable', () => {
        tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        tf.tensor1d([1, 2, 3]).variable(true, 'v2');
        expect(() => {
            tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        }).toThrowError();
        tf.disposeVariables();
        tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        tf.tensor1d([1, 2, 3]).variable(true, 'v2');
    });
});
/**
 * The following test constraints to the CPU environment because it needs a
 * concrete backend to exist. This test will work for any backend, but currently
 * this is the simplest backend to test against.
 */
describeWithFlags('Switching cpu backends', { predicate: testEnv => testEnv.backendName === 'cpu' }, () => {
    beforeEach(() => {
        tf.registerBackend('cpu1', tf.findBackendFactory('cpu'));
        tf.registerBackend('cpu2', tf.findBackendFactory('cpu'));
    });
    afterEach(() => {
        tf.removeBackend('cpu1');
        tf.removeBackend('cpu2');
    });
    it('Move data from cpu1 to cpu2 backend', async () => {
        tf.setBackend('cpu1');
        // This scalar lives in cpu1.
        const a = tf.scalar(5);
        tf.setBackend('cpu2');
        // This scalar lives in cpu2.
        const b = tf.scalar(3);
        expect(tf.memory().numDataBuffers).toBe(2);
        expect(tf.memory().numTensors).toBe(2);
        expect(tf.memory().numBytes).toBe(8);
        // Make sure you can read both tensors.
        expectArraysClose(await a.data(), [5]);
        expectArraysClose(await b.data(), [3]);
        // Switch back to cpu1.
        tf.setBackend('cpu1');
        // Again make sure you can read both tensors.
        expectArraysClose(await a.data(), [5]);
        expectArraysClose(await b.data(), [3]);
        tf.dispose([a, b]);
        expect(tf.memory().numDataBuffers).toBe(0);
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
    });
    it('can execute op with data from mixed backends', async () => {
        const kernelFunc = tf.getKernel('Add', 'cpu').kernelFunc;
        tf.registerKernel({ kernelName: 'Add', backendName: 'cpu1', kernelFunc });
        tf.registerKernel({ kernelName: 'Add', backendName: 'cpu2', kernelFunc });
        tf.setBackend('cpu1');
        // This scalar lives in cpu1.
        const a = tf.scalar(5);
        tf.setBackend('cpu2');
        // This scalar lives in cpu2.
        const b = tf.scalar(3);
        // Verify that ops can execute with mixed backend data.
        ENGINE.startScope();
        tf.setBackend('cpu1');
        expectArraysClose(await tf.add(a, b).data(), [8]);
        tf.setBackend('cpu2');
        expectArraysClose(await tf.add(a, b).data(), [8]);
        ENGINE.endScope();
        tf.dispose([a, b]);
    });
});
describeWithFlags('Detects memory leaks in kernels', ALL_ENVS, () => {
    const backendName = 'test-mem';
    const kernelName = 'MyKernel';
    const kernelNameComplex = 'Kernel-complex';
    it('Detects memory leak in a kernel', () => {
        let dataIdsCount = 0;
        tf.registerBackend(backendName, () => {
            return {
                id: 1,
                dispose: () => null,
                disposeData: (dataId) => null,
                numDataIds: () => dataIdsCount
            };
        });
        const kernelWithMemLeak = () => {
            dataIdsCount += 2;
            return { dataId: {}, shape: [], dtype: 'float32' };
        };
        tf.registerKernel({ kernelName, backendName, kernelFunc: kernelWithMemLeak });
        tf.setBackend(backendName);
        expect(() => tf.engine().runKernel(kernelName, {}, {}))
            .toThrowError(/Backend 'test-mem' has an internal memory leak \(1 data ids\)/);
        tf.removeBackend(backendName);
        tf.unregisterKernel(kernelName, backendName);
    });
    it('No mem leak in a kernel with multiple outputs', () => {
        let dataIdsCount = 0;
        tf.registerBackend(backendName, () => {
            return {
                id: 1,
                dispose: () => null,
                disposeData: (dataId) => null,
                numDataIds: () => dataIdsCount
            };
        });
        tf.setBackend(backendName);
        const kernelWith3Outputs = () => {
            dataIdsCount += 3;
            const t = { dataId: {}, shape: [], dtype: 'float32' };
            return [t, t, t];
        };
        tf.registerKernel({ kernelName, backendName, kernelFunc: kernelWith3Outputs });
        const res = tf.engine().runKernel(kernelName, {}, {});
        expect(Array.isArray(res)).toBe(true);
        expect(res.length).toBe(3);
        const kernelWithComplexOutputs = () => {
            dataIdsCount += 3;
            return { dataId: {}, shape: [], dtype: 'complex64' };
        };
        tf.registerKernel({
            kernelName: kernelNameComplex,
            backendName,
            kernelFunc: kernelWithComplexOutputs
        });
        const res2 = tf.engine().runKernel(kernelNameComplex, {}, {});
        expect(res2.shape).toEqual([]);
        expect(res2.dtype).toEqual('complex64');
        tf.removeBackend(backendName);
        tf.unregisterKernel(kernelName, backendName);
        tf.unregisterKernel(kernelNameComplex, backendName);
    });
});
// NOTE: This describe is purposefully not a describeWithFlags so that we
// test tensor allocation where no scopes have been created.
describe('Memory allocation outside a test scope', () => {
    it('constructing a tensor works', async () => {
        const backendName = 'test-backend';
        tf.registerBackend(backendName, () => {
            let storedValues = null;
            return {
                id: 1,
                floatPrecision: () => 32,
                write: (values, shape, dtype) => {
                    const dataId = {};
                    storedValues = values;
                    return dataId;
                },
                read: async (dataId) => storedValues,
                dispose: () => null,
                disposeData: (dataId) => null
            };
        });
        tf.setBackend(backendName);
        const a = tf.tensor1d([1, 2, 3]);
        expectArraysClose(await a.data(), [1, 2, 3]);
        a.dispose();
        tf.removeBackend(backendName);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW5naW5lX3Rlc3QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2VuZ2luZV90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxLQUFLLEVBQUUsTUFBTSxTQUFTLENBQUM7QUFFOUIsT0FBTyxFQUFDLFFBQVEsRUFBRSxpQkFBaUIsRUFBRSxpQkFBaUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBRzlFLE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUc5QyxRQUFRLENBQUMsc0JBQXNCLEVBQUUsR0FBRyxFQUFFO0lBQ3BDLFNBQVMsQ0FBQyxHQUFHLEVBQUU7UUFDYiwwQ0FBMEM7UUFDMUMsS0FBSyxDQUFDLE9BQU8sRUFBRSxNQUFNLENBQUMsQ0FBQztJQUN6QixDQUFDLENBQUMsQ0FBQztJQUVILElBQUksa0JBQWtCLEdBQWEsRUFBRSxDQUFDO0lBQ3RDLElBQUksZUFBMEMsQ0FBQztJQUUvQyxVQUFVLENBQUMsR0FBRyxFQUFFO1FBQ2Qsa0VBQWtFO1FBQ2xFLHFFQUFxRTtRQUNyRSxnQkFBZ0I7UUFDaEIsZUFBZSxHQUFHLENBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxRQUFRLEVBQUUsRUFBRTtZQUM1QyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDOUIsT0FBTyxFQUFFLENBQUMsZUFBZSxDQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFDckQsQ0FBQyxDQUFDO1FBRUYsTUFBTSxDQUFDLEtBQUssRUFBRSxDQUFDO0lBQ2pCLENBQUMsQ0FBQyxDQUFDO0lBRUgsU0FBUyxDQUFDLEdBQUcsRUFBRTtRQUNiLDBEQUEwRDtRQUMxRCxrQkFBa0IsQ0FBQyxPQUFPLENBQUMsSUFB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QUFDLFFBQVEsRUFBRSxRQUFRLEVBQUUsR0FBRyxFQUFFO0lBQ3pDLEVBQUUsQ0FBQyxZQUFZLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDMUIsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckMsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDdkIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDcEMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBQ3pDLE9BQU8sQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ2pCLENBQUMsQ0FBQyxDQUFDO1FBQ0gsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckMsaUJBQWlCLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZELENBQUMsQ0FBQyxDQUFDO0lB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QUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM3QyxDQUFDLENBQUMsT0FBTyxFQUFFLENBQUM7UUFFWixFQUFFLENBQUMsYUFBYSxDQUFDLFdBQVcsQ0FBQyxDQUFDO0lBQ2hDLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQmFja2VuZH0gZnJvbSAnLi9iYWNrZW5kcy9iYWNrZW5kJztcbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuL2VuZ2luZSc7XG5pbXBvcnQgKiBhcyB0ZiBmcm9tICcuL2luZGV4JztcbmltcG9ydCB7S2VybmVsRnVuY30gZnJvbSAnLi9pbmRleCc7XG5pbXBvcnQge0FMTF9FTlZTLCBkZXNjcmliZVdpdGhGbGFncywgVGVzdEtlcm5lbEJhY2tlbmR9IGZyb20gJy4vamFzbWluZV91dGlsJztcbmltcG9ydCB7VGVuc29ySW5mb30gZnJvbSAnLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4vdGVuc29yJztcbmltcG9ydCB7ZXhwZWN0QXJyYXlzQ2xvc2V9IGZyb20gJy4vdGVzdF91dGlsJztcbmltcG9ydCB7QmFja2VuZFZhbHVlcywgRGF0YVR5cGV9IGZyb20gJy4vdHlwZXMnO1xuXG5kZXNjcmliZSgnQmFja2VuZCByZWdpc3RyYXRpb24nLCAoKSA9PiB7XG4gIGJlZm9yZUFsbCgoKSA9PiB7XG4gICAgLy8gU2lsZW5jZXMgYmFja2VuZCByZWdpc3RyYXRpb24gd2FybmluZ3MuXG4gICAgc3B5T24oY29uc29sZSwgJ3dhcm4nKTtcbiAgfSk7XG5cbiAgbGV0IHJlZ2lzdGVyZWRCYWNrZW5kczogc3RyaW5nW10gPSBbXTtcbiAgbGV0IHJlZ2lzdGVyQmFja2VuZDogdHlwZW9mIHRmLnJlZ2lzdGVyQmFja2VuZDtcblxuICBiZWZvcmVFYWNoKCgpID0+IHtcbiAgICAvLyBSZWdpc3RlcmluZyBhIGJhY2tlbmQgY2hhbmdlcyBnbG9iYWwgc3RhdGUgKGVuZ2luZSksIHNvIHdlIHdyYXBcbiAgICAvLyByZWdpc3RyYXRpb24gdG8gYXV0b21hdGljYWxseSByZW1vdmUgcmVnaXN0ZXJlZCBiYWNrZW5kIGF0IHRoZSBlbmRcbiAgICAvLyBvZiBlYWNoIHRlc3QuXG4gICAgcmVnaXN0ZXJCYWNrZW5kID0gKG5hbWUsIGZhY3RvcnksIHByaW9yaXR5KSA9PiB7XG4gICAgICByZWdpc3RlcmVkQmFja2VuZHMucHVzaChuYW1lKTtcbiAgICAgIHJldHVybiB0Zi5yZWdpc3RlckJhY2tlbmQobmFtZSwgZmFjdG9yeSwgcHJpb3JpdHkpO1xuICAgIH07XG5cbiAgICBFTkdJTkUucmVzZXQoKTtcbiAgfSk7XG5cbiAgYWZ0ZXJFYWNoKCgpID0+IHtcbiAgICAvLyBSZW1vdmUgYWxsIHJlZ2lzdGVyZWQgYmFja2VuZHMgYXQgdGhlIGVuZCBvZiBlYWNoIHRlc3QuXG4gICAgcmVnaXN0ZXJlZEJhY2tlbmRzLmZvckVhY2gobmFtZSA9PiB7XG4gICAgICBpZiAodGYuZmluZEJhY2tlbmRGYWN0b3J5KG5hbWUpICE9IG51bGwpIHtcbiAgICAgICAgdGYucmVtb3ZlQmFja2VuZChuYW1lKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgICByZWdpc3RlcmVkQmFja2VuZHMgPSBbXTtcbiAgfSk7XG5cbiAgaXQoJ3JlbW92ZUJhY2tlbmQgZGlzcG9zZXMgdGhlIGJhY2tlbmQgYW5kIHJlbW92ZXMgdGhlIGZhY3RvcnknLCAoKSA9PiB7XG4gICAgbGV0IGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQ7XG4gICAgY29uc3QgZmFjdG9yeSA9ICgpID0+IHtcbiAgICAgIGNvbnN0IG5ld0JhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICAgIGlmIChiYWNrZW5kID09IG51bGwpIHtcbiAgICAgICAgYmFja2VuZCA9IG5ld0JhY2tlbmQ7XG4gICAgICAgIHNweU9uKGJhY2tlbmQsICdkaXNwb3NlJykuYW5kLmNhbGxUaHJvdWdoKCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gbmV3QmFja2VuZDtcbiAgICB9O1xuXG4gICAgcmVnaXN0ZXJCYWNrZW5kKCd0ZXN0LWJhY2tlbmQnLCBmYWN0b3J5KTtcblxuICAgIGV4cGVjdCh0Zi5maW5kQmFja2VuZCgndGVzdC1iYWNrZW5kJykgIT0gbnVsbCkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodGYuZmluZEJhY2tlbmQoJ3Rlc3QtYmFja2VuZCcpKS50b0JlKGJhY2tlbmQpO1xuICAgIGV4cGVjdCh0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ3Rlc3QtYmFja2VuZCcpKS50b0JlKGZhY3RvcnkpO1xuXG4gICAgdGYucmVtb3ZlQmFja2VuZCgndGVzdC1iYWNrZW5kJyk7XG5cbiAgICBleHBlY3QodGYuZmluZEJhY2tlbmQoJ3Rlc3QtYmFja2VuZCcpID09IG51bGwpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kKCd0ZXN0LWJhY2tlbmQnKSkudG9CZShudWxsKTtcbiAgICBleHBlY3QoKGJhY2tlbmQuZGlzcG9zZSBhcyBqYXNtaW5lLlNweSkuY2FsbHMuY291bnQoKSkudG9CZSgxKTtcbiAgICBleHBlY3QodGYuZmluZEJhY2tlbmRGYWN0b3J5KCd0ZXN0LWJhY2tlbmQnKSkudG9CZShudWxsKTtcbiAgfSk7XG5cbiAgaXQoJ2ZpbmRCYWNrZW5kIGluaXRpYWxpemVzIHRoZSBiYWNrZW5kJywgKCkgPT4ge1xuICAgIGxldCBiYWNrZW5kOiBLZXJuZWxCYWNrZW5kO1xuICAgIGNvbnN0IGZhY3RvcnkgPSAoKSA9PiB7XG4gICAgICBjb25zdCBuZXdCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICBpZiAoYmFja2VuZCA9PSBudWxsKSB7XG4gICAgICAgIGJhY2tlbmQgPSBuZXdCYWNrZW5kO1xuICAgICAgfVxuICAgICAgcmV0dXJuIG5ld0JhY2tlbmQ7XG4gICAgfTtcbiAgICByZWdpc3RlckJhY2tlbmQoJ2N1c3RvbS1jcHUnLCBmYWN0b3J5KTtcblxuICAgIGV4cGVjdCh0Zi5maW5kQmFja2VuZCgnY3VzdG9tLWNwdScpICE9IG51bGwpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kKCdjdXN0b20tY3B1JykpLnRvQmUoYmFja2VuZCk7XG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kRmFjdG9yeSgnY3VzdG9tLWNwdScpKS50b0JlKGZhY3RvcnkpO1xuICB9KTtcblxuICBpdCgnY3VzdG9tIGJhY2tlbmQgcmVnaXN0cmF0aW9uJywgKCkgPT4ge1xuICAgIGxldCBiYWNrZW5kOiBLZXJuZWxCYWNrZW5kO1xuICAgIGNvbnN0IHByaW9yaXR5ID0gMTAzO1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnY3VzdG9tLWNwdScsICgpID0+IHtcbiAgICAgIGNvbnN0IG5ld0JhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICAgIGlmIChiYWNrZW5kID09IG51bGwpIHtcbiAgICAgICAgYmFja2VuZCA9IG5ld0JhY2tlbmQ7XG4gICAgICB9XG4gICAgICByZXR1cm4gbmV3QmFja2VuZDtcbiAgICB9LCBwcmlvcml0eSk7XG5cbiAgICBleHBlY3QodGYuYmFja2VuZCgpICE9IG51bGwpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHRmLmJhY2tlbmQoKSkudG9CZShiYWNrZW5kKTtcbiAgfSk7XG5cbiAgaXQoJ2hpZ2ggcHJpb3JpdHkgYmFja2VuZCByZWdpc3RyYXRpb24gZmFpbHMsIGZhbGxzIGJhY2snLCAoKSA9PiB7XG4gICAgbGV0IGxvd1ByaW9yaXR5QmFja2VuZDogS2VybmVsQmFja2VuZDtcbiAgICBjb25zdCBsb3dQcmlvcml0eSA9IDEwMztcbiAgICBjb25zdCBoaWdoUHJpb3JpdHkgPSAxMDQ7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdjdXN0b20tbG93LXByaW9yaXR5JywgKCkgPT4ge1xuICAgICAgbG93UHJpb3JpdHlCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICByZXR1cm4gbG93UHJpb3JpdHlCYWNrZW5kO1xuICAgIH0sIGxvd1ByaW9yaXR5KTtcbiAgICByZWdpc3RlckJhY2tlbmQoJ2N1c3RvbS1oaWdoLXByaW9yaXR5JywgKCkgPT4ge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBIaWdoIHByaW9yaXR5IGJhY2tlbmQgZmFpbHNgKTtcbiAgICB9LCBoaWdoUHJpb3JpdHkpO1xuXG4gICAgZXhwZWN0KHRmLmJhY2tlbmQoKSAhPSBudWxsKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvQmUobG93UHJpb3JpdHlCYWNrZW5kKTtcbiAgICBleHBlY3QodGYuZ2V0QmFja2VuZCgpKS50b0JlKCdjdXN0b20tbG93LXByaW9yaXR5Jyk7XG4gIH0pO1xuXG4gIGl0KCdsb3cgcHJpb3JpdHkgYW5kIGhpZ2ggcHJpb3JpdHkgYmFja2VuZHMsIHNldEJhY2tlbmQgbG93IHByaW9yaXR5JywgKCkgPT4ge1xuICAgIGxldCBsb3dQcmlvcml0eUJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQ7XG4gICAgbGV0IGhpZ2hQcmlvcml0eUJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQ7XG4gICAgY29uc3QgbG93UHJpb3JpdHkgPSAxMDM7XG4gICAgY29uc3QgaGlnaFByaW9yaXR5ID0gMTA0O1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnY3VzdG9tLWxvdy1wcmlvcml0eScsICgpID0+IHtcbiAgICAgIGxvd1ByaW9yaXR5QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgcmV0dXJuIGxvd1ByaW9yaXR5QmFja2VuZDtcbiAgICB9LCBsb3dQcmlvcml0eSk7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdjdXN0b20taGlnaC1wcmlvcml0eScsICgpID0+IHtcbiAgICAgIGhpZ2hQcmlvcml0eUJhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICAgIHJldHVybiBoaWdoUHJpb3JpdHlCYWNrZW5kO1xuICAgIH0sIGhpZ2hQcmlvcml0eSk7XG5cbiAgICBleHBlY3QodGYuYmFja2VuZCgpICE9IG51bGwpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHRmLmJhY2tlbmQoKSkudG9CZShoaWdoUHJpb3JpdHlCYWNrZW5kKTtcbiAgICBleHBlY3QodGYuZ2V0QmFja2VuZCgpKS50b0JlKCdjdXN0b20taGlnaC1wcmlvcml0eScpO1xuXG4gICAgdGYuc2V0QmFja2VuZCgnY3VzdG9tLWxvdy1wcmlvcml0eScpO1xuXG4gICAgZXhwZWN0KHRmLmJhY2tlbmQoKSAhPSBudWxsKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvQmUobG93UHJpb3JpdHlCYWNrZW5kKTtcbiAgICBleHBlY3QodGYuZ2V0QmFja2VuZCgpKS50b0JlKCdjdXN0b20tbG93LXByaW9yaXR5Jyk7XG4gIH0pO1xuXG4gIGl0KCdkZWZhdWx0IGN1c3RvbSBiYWNrZ3JvdW5kIG51bGwnLCAoKSA9PiB7XG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kKCdjdXN0b20nKSkudG9CZU51bGwoKTtcbiAgfSk7XG5cbiAgaXQoJ2FsbG93IGN1c3RvbSBiYWNrZW5kJywgKCkgPT4ge1xuICAgIGNvbnN0IGJhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICBjb25zdCBzdWNjZXNzID0gcmVnaXN0ZXJCYWNrZW5kKCdjdXN0b20nLCAoKSA9PiBiYWNrZW5kKTtcbiAgICBleHBlY3Qoc3VjY2VzcykudG9CZVRydXRoeSgpO1xuICAgIGV4cGVjdCh0Zi5maW5kQmFja2VuZCgnY3VzdG9tJykpLnRvRXF1YWwoYmFja2VuZCk7XG4gIH0pO1xuXG4gIGl0KCdzeW5jIGJhY2tlbmQgd2l0aCBhd2FpdCByZWFkeSB3b3JrcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB0ZXN0QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnc3luYycsICgpID0+IHRlc3RCYWNrZW5kKTtcbiAgICB0Zi5zZXRCYWNrZW5kKCdzeW5jJyk7XG5cbiAgICBleHBlY3QodGYuZ2V0QmFja2VuZCgpKS50b0VxdWFsKCdzeW5jJyk7XG4gICAgYXdhaXQgdGYucmVhZHkoKTtcbiAgICBleHBlY3QodGYuYmFja2VuZCgpKS50b0VxdWFsKHRlc3RCYWNrZW5kKTtcbiAgfSk7XG5cbiAgaXQoJ3N5bmMgYmFja2VuZCB3aXRob3V0IGF3YWl0IHJlYWR5IHdvcmtzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHRlc3RCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdzeW5jJywgKCkgPT4gdGVzdEJhY2tlbmQpO1xuICAgIHRmLnNldEJhY2tlbmQoJ3N5bmMnKTtcblxuICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvRXF1YWwoJ3N5bmMnKTtcbiAgICBleHBlY3QodGYuYmFja2VuZCgpKS50b0VxdWFsKHRlc3RCYWNrZW5kKTtcbiAgfSk7XG5cbiAgaXQoJ2FzeW5jIGJhY2tlbmQgd2l0aCBhd2FpdCByZWFkeSB3b3JrcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB0ZXN0QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnYXN5bmMnLCBhc3luYyAoKSA9PiB7XG4gICAgICBhd2FpdCB0Zi5uZXh0RnJhbWUoKTtcbiAgICAgIHJldHVybiB0ZXN0QmFja2VuZDtcbiAgICB9KTtcbiAgICB0Zi5zZXRCYWNrZW5kKCdhc3luYycpO1xuXG4gICAgZXhwZWN0KHRmLmdldEJhY2tlbmQoKSkudG9FcXVhbCgnYXN5bmMnKTtcbiAgICBhd2FpdCB0Zi5yZWFkeSgpO1xuICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvRXF1YWwodGVzdEJhY2tlbmQpO1xuICB9KTtcblxuICBpdCgnYXN5bmMgYmFja2VuZCB3aXRob3V0IGF3YWl0IHJlYWR5IGRvZXMgbm90IHdvcmsnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgdGVzdEJhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICByZWdpc3RlckJhY2tlbmQoJ2FzeW5jJywgYXN5bmMgKCkgPT4ge1xuICAgICAgYXdhaXQgdGYubmV4dEZyYW1lKCk7XG4gICAgICByZXR1cm4gdGVzdEJhY2tlbmQ7XG4gICAgfSk7XG4gICAgdGYuc2V0QmFja2VuZCgnYXN5bmMnKTtcblxuICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvRXF1YWwoJ2FzeW5jJyk7XG4gICAgZXhwZWN0KCgpID0+IHRmLmJhY2tlbmQoKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigvQmFja2VuZCAnYXN5bmMnIGhhcyBub3QgeWV0IGJlZW4gaW5pdGlhbGl6ZWQuLyk7XG4gIH0pO1xuXG4gIGl0KCd0Zi5zcXVhcmUoKSBmYWlscyBpZiB1c2VyIGRvZXMgbm90IGF3YWl0IHJlYWR5IG9uIGFzeW5jIGJhY2tlbmQnLFxuICAgICBhc3luYyAoKSA9PiB7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKCdhc3luYycsIGFzeW5jICgpID0+IHtcbiAgICAgICAgIGF3YWl0IHRmLm5leHRGcmFtZSgpO1xuICAgICAgICAgcmV0dXJuIG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgIH0pO1xuICAgICAgIHRmLnNldEJhY2tlbmQoJ2FzeW5jJyk7XG4gICAgICAgZXhwZWN0KCgpID0+IHRmLnNxdWFyZSgyKSlcbiAgICAgICAgICAgLnRvVGhyb3dFcnJvcigvQmFja2VuZCAnYXN5bmMnIGhhcyBub3QgeWV0IGJlZW4gaW5pdGlhbGl6ZWQvKTtcbiAgICAgfSk7XG5cbiAgaXQoJ3RmLnNxdWFyZSgpIHdvcmtzIHdoZW4gdXNlciBhd2FpdHMgcmVhZHkgb24gYXN5bmMgYmFja2VuZCcsIGFzeW5jICgpID0+IHtcbiAgICByZWdpc3RlckJhY2tlbmQoJ2FzeW5jJywgYXN5bmMgKCkgPT4ge1xuICAgICAgYXdhaXQgdGYubmV4dEZyYW1lKCk7XG4gICAgICByZXR1cm4gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgfSk7XG4gICAgdGYuc2V0QmFja2VuZCgnYXN5bmMnKTtcbiAgICBhd2FpdCB0Zi5yZWFkeSgpO1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5zcXVhcmUoMikpLnRvVGhyb3dFcnJvcigvJ3dyaXRlJyBub3QgeWV0IGltcGxlbWVudGVkLyk7XG4gIH0pO1xuXG4gIGl0KCdSZWdpc3RlcmluZyBhc3luYzIgKGhpZ2hlciBwcmlvcml0eSkgZmFpbHMsIGFzeW5jMSBiZWNvbWVzIGFjdGl2ZScsXG4gICAgIGFzeW5jICgpID0+IHtcbiAgICAgICBjb25zdCB0ZXN0QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgIHJlZ2lzdGVyQmFja2VuZCgnYXN5bmMxJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICAgYXdhaXQgdGYubmV4dEZyYW1lKCk7XG4gICAgICAgICByZXR1cm4gdGVzdEJhY2tlbmQ7XG4gICAgICAgfSwgMTAwIC8qIHByaW9yaXR5ICovKTtcbiAgICAgICByZWdpc3RlckJhY2tlbmQoJ2FzeW5jMicsIGFzeW5jICgpID0+IHtcbiAgICAgICAgIGF3YWl0IHRmLm5leHRGcmFtZSgpO1xuICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdmYWlsZWQgdG8gY3JlYXRlIGFzeW5jMicpO1xuICAgICAgIH0sIDEwMSAvKiBwcmlvcml0eSAqLyk7XG5cbiAgICAgICAvLyBBd2FpdCBmb3IgdGhlIGxpYnJhcnkgdG8gZmluZCB0aGUgYmVzdCBiYWNrZW5kIHRoYXQgc3VjY2VzZnVsbHlcbiAgICAgICAvLyBpbml0aWFsaXplcy5cbiAgICAgICBhd2FpdCB0Zi5yZWFkeSgpO1xuICAgICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvRXF1YWwodGVzdEJhY2tlbmQpO1xuICAgICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvQmUoJ2FzeW5jMScpO1xuICAgICB9KTtcblxuICBpdCgnUmVnaXN0ZXJpbmcgc3luYyBhcyBoaWdoZXIgcHJpb3JpdHkgYW5kIGFzeW5jIGFzIGxvd2VyIHByaW9yaXR5JyxcbiAgICAgYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IHRlc3RCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKCdzeW5jJywgKCkgPT4gdGVzdEJhY2tlbmQsIDEwMSAvKiBwcmlvcml0eSAqLyk7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKCdhc3luYycsIGFzeW5jICgpID0+IHtcbiAgICAgICAgIGF3YWl0IHRmLm5leHRGcmFtZSgpO1xuICAgICAgICAgcmV0dXJuIG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgIH0sIDEwMCAvKiBwcmlvcml0eSAqLyk7XG5cbiAgICAgICAvLyBObyBuZWVkIHRvIGF3YWl0IGZvciByZWFkeSgpIHNpbmNlIHRoZSBoaWdoZXN0IHByaW9yaXR5IG9uZSBpcyBzeW5jLlxuICAgICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvRXF1YWwodGVzdEJhY2tlbmQpO1xuICAgICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvQmUoJ3N5bmMnKTtcbiAgICAgfSk7XG5cbiAgaXQoJ2FzeW5jIGFzIGhpZ2hlciBwcmlvcml0eSBhbmQgc3luYyBhcyBsb3dlciBwcmlvcml0eSB3aXRoIGF3YWl0IHJlYWR5JyxcbiAgICAgYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IHRlc3RCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKCdhc3luYycsIGFzeW5jICgpID0+IHtcbiAgICAgICAgIGF3YWl0IHRmLm5leHRGcmFtZSgpO1xuICAgICAgICAgcmV0dXJuIHRlc3RCYWNrZW5kO1xuICAgICAgIH0sIDEwMSAvKiBwcmlvcml0eSAqLyk7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKFxuICAgICAgICAgICAnc3luYycsICgpID0+IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpLCAxMDAgLyogcHJpb3JpdHkgKi8pO1xuXG4gICAgICAgYXdhaXQgdGYucmVhZHkoKTtcbiAgICAgICBleHBlY3QodGYuYmFja2VuZCgpKS50b0VxdWFsKHRlc3RCYWNrZW5kKTtcbiAgICAgICBleHBlY3QodGYuZ2V0QmFja2VuZCgpKS50b0JlKCdhc3luYycpO1xuICAgICB9KTtcblxuICBpdCgnYXN5bmMgYXMgaGlnaGVyIHByaW9yaXR5IGFuZCBzeW5jIGFzIGxvd2VyIHByaW9yaXR5IHcvbyBhd2FpdCByZWFkeScsXG4gICAgIGFzeW5jICgpID0+IHtcbiAgICAgICBjb25zdCB0ZXN0QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgIHJlZ2lzdGVyQmFja2VuZCgnYXN5bmMnLCBhc3luYyAoKSA9PiB7XG4gICAgICAgICBhd2FpdCB0Zi5uZXh0RnJhbWUoKTtcbiAgICAgICAgIHJldHVybiB0ZXN0QmFja2VuZDtcbiAgICAgICB9LCAxMDEgLyogcHJpb3JpdHkgKi8pO1xuICAgICAgIHJlZ2lzdGVyQmFja2VuZChcbiAgICAgICAgICAgJ3N5bmMnLCAoKSA9PiBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKSwgMTAwIC8qIHByaW9yaXR5ICovKTtcblxuICAgICAgIGV4cGVjdCgoKSA9PiB0Zi5iYWNrZW5kKCkpXG4gICAgICAgICAgIC50b1Rocm93RXJyb3IoXG4gICAgICAgICAgICAgICAvVGhlIGhpZ2hlc3QgcHJpb3JpdHkgYmFja2VuZCAnYXN5bmMnIGhhcyBub3QgeWV0IGJlZW4vKTtcbiAgICAgfSk7XG5cbiAgaXQoJ1JlZ2lzdGVyaW5nIGFuZCBzZXR0aW5nIGEgYmFja2VuZCB0aGF0IGZhaWxzIHRvIHJlZ2lzdGVyJywgYXN5bmMgKCkgPT4ge1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnYXN5bmMnLCBhc3luYyAoKSA9PiB7XG4gICAgICBhd2FpdCB0Zi5uZXh0RnJhbWUoKTtcbiAgICAgIHRocm93IG5ldyBFcnJvcignZmFpbGVkIHRvIGNyZWF0ZSBhc3luYycpO1xuICAgIH0pO1xuICAgIGNvbnN0IHN1Y2Nlc3MgPSB0Zi5zZXRCYWNrZW5kKCdhc3luYycpO1xuICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvQmUoJ2FzeW5jJyk7XG4gICAgZXhwZWN0KCgpID0+IHRmLmJhY2tlbmQoKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigvQmFja2VuZCAnYXN5bmMnIGhhcyBub3QgeWV0IGJlZW4gaW5pdGlhbGl6ZWQvKTtcbiAgICBleHBlY3QoYXdhaXQgc3VjY2VzcykudG9CZShmYWxzZSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdtZW1vcnknLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnU3VtKGZsb2F0KScsIGFzeW5jICgpID0+IHtcbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgwKTtcbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtQnl0ZXMpLnRvQmUoMCk7XG4gICAgY29uc3Qgc3VtID0gdGYudGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBhID0gdGYudGVuc29yMWQoWzEsIDIsIDMsIDRdKTtcbiAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDEpO1xuICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDQgKiA0KTtcbiAgICAgIHJldHVybiBhLnN1bSgpO1xuICAgIH0pO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDEpO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1CeXRlcykudG9CZSg0KTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBzdW0uZGF0YSgpLCBbMSArIDIgKyAzICsgNF0pO1xuICB9KTtcblxuICBpdCgnU3VtKGJvb2wpJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHN1bSA9IHRmLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgYSA9IHRmLnRlbnNvcjFkKFt0cnVlLCB0cnVlLCBmYWxzZSwgdHJ1ZV0sICdib29sJyk7XG4gICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgxKTtcbiAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1CeXRlcykudG9CZSg0KTtcbiAgICAgIHJldHVybiBhLnN1bSgpO1xuICAgIH0pO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDEpO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1CeXRlcykudG9CZSg0KTtcbiAgICBleHBlY3Qoc3VtLmR0eXBlKS50b0JlKCdpbnQzMicpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHN1bS5kYXRhKCksIFsxICsgMSArIDAgKyAxXSk7XG4gIH0pO1xuXG4gIGl0KCdTdW0oaW50MzIpJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHN1bSA9IHRmLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCAxLCAwLCAxXSwgJ2ludDMyJyk7XG4gICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgxKTtcbiAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1CeXRlcykudG9CZSg0ICogNCk7XG4gICAgICByZXR1cm4gYS5zdW0oKTtcbiAgICB9KTtcbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgxKTtcbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtQnl0ZXMpLnRvQmUoNCk7XG4gICAgZXhwZWN0KHN1bS5kdHlwZSkudG9CZSgnaW50MzInKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBzdW0uZGF0YSgpLCBbMSArIDEgKyAwICsgMV0pO1xuICB9KTtcblxuICBpdCgnc3RyaW5nIHRlbnNvcicsICgpID0+IHtcbiAgICBjb25zdCBhID0gdGYudGVuc29yKFtbJ2EnLCAnYmInXSwgWydjJywgJ2QnXV0pO1xuXG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMSk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDUpOyAgLy8gNSBsZXR0ZXJzLCBlYWNoIDEgYnl0ZSBpbiB1dGY4LlxuXG4gICAgYS5kaXNwb3NlKCk7XG5cbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgwKTtcbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtQnl0ZXMpLnRvQmUoMCk7XG4gIH0pO1xuXG4gIGl0KCd1bnJlbGlhYmxlIGlzIHRydWUgZm9yIHN0cmluZyB0ZW5zb3JzJywgKCkgPT4ge1xuICAgIHRmLnRlbnNvcignYScpO1xuICAgIGNvbnN0IG1lbSA9IHRmLm1lbW9yeSgpO1xuICAgIGV4cGVjdChtZW0udW5yZWxpYWJsZSkudG9CZSh0cnVlKTtcbiAgICBjb25zdCBleHBlY3RlZFJlYXNvbiA9ICdNZW1vcnkgdXNhZ2UgYnkgc3RyaW5nIHRlbnNvcnMgaXMgYXBwcm94aW1hdGUgJyArXG4gICAgICAgICcoMiBieXRlcyBwZXIgY2hhcmFjdGVyKSc7XG4gICAgZXhwZWN0KG1lbS5yZWFzb25zLmluZGV4T2YoZXhwZWN0ZWRSZWFzb24pID49IDApLnRvQmUodHJ1ZSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdwcm9maWxlJywgQUxMX0VOVlMsICgpID0+IHtcbiAgaXQoJ3NxdWFyaW5nJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHByb2ZpbGUgPSBhd2FpdCB0Zi5wcm9maWxlKCgpID0+IHtcbiAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICAgICAgbGV0IHgyID0geC5zcXVhcmUoKTtcbiAgICAgIHgyLmRpc3Bvc2UoKTtcbiAgICAgIHgyID0geC5zcXVhcmUoKTtcbiAgICAgIHgyLmRpc3Bvc2UoKTtcbiAgICAgIHJldHVybiB4O1xuICAgIH0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gcHJvZmlsZS5yZXN1bHQgYXMgVGVuc29yO1xuXG4gICAgZXhwZWN0KHByb2ZpbGUubmV3Qnl0ZXMpLnRvQmUoMTIpO1xuICAgIGV4cGVjdChwcm9maWxlLnBlYWtCeXRlcykudG9CZSgyNCk7XG4gICAgZXhwZWN0KHByb2ZpbGUubmV3VGVuc29ycykudG9CZSgxKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBbMSwgMiwgM10pO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHMubGVuZ3RoKS50b0JlKDIpO1xuXG4gICAgLy8gVGVzdCB0aGUgdHlwZXMgZm9yIGBrZXJuZWxUaW1lTXNgIGFuZCBgZXh0cmFJbmZvYCB0byBjb25maXJtIHRoZSBwcm9taXNlc1xuICAgIC8vIGFyZSByZXNvbHZlZC5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdLmtlcm5lbFRpbWVNcyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHNbMF0uZXh0cmFJbmZvIGluc3RhbmNlb2YgUHJvbWlzZSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVsc1sxXS5rZXJuZWxUaW1lTXMgaW5zdGFuY2VvZiBQcm9taXNlKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzFdLmV4dHJhSW5mbyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuXG4gICAgLy8gVGhlIHNwZWNpZmljIHZhbHVlcyBvZiBga2VybmVsVGltZU1zYCBhbmQgYGV4dHJhSW5mb2AgYXJlIHRlc3RlZCBpbiB0aGVcbiAgICAvLyB0ZXN0cyBvZiBQcm9maWxlci5wcm9maWxlS2VybmVsLCBzbyB0aGVpciB2YWx1ZXMgYXJlIG5vdCB0ZXN0ZWQgaGVyZS5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdKS50b0VxdWFsKHtcbiAgICAgICduYW1lJzogJ1NxdWFyZScsXG4gICAgICAnYnl0ZXNBZGRlZCc6IDEyLFxuICAgICAgJ3RvdGFsQnl0ZXNTbmFwc2hvdCc6IDI0LFxuICAgICAgJ3RlbnNvcnNBZGRlZCc6IDEsXG4gICAgICAndG90YWxUZW5zb3JzU25hcHNob3QnOiAyLFxuICAgICAgJ2lucHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAnb3V0cHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAna2VybmVsVGltZU1zJzogcHJvZmlsZS5rZXJuZWxzWzBdLmtlcm5lbFRpbWVNcyxcbiAgICAgICdleHRyYUluZm8nOiBwcm9maWxlLmtlcm5lbHNbMF0uZXh0cmFJbmZvXG4gICAgfSk7XG5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzFdKS50b0VxdWFsKHtcbiAgICAgICduYW1lJzogJ1NxdWFyZScsXG4gICAgICAnYnl0ZXNBZGRlZCc6IDEyLFxuICAgICAgJ3RvdGFsQnl0ZXNTbmFwc2hvdCc6IDI0LFxuICAgICAgJ3RlbnNvcnNBZGRlZCc6IDEsXG4gICAgICAndG90YWxUZW5zb3JzU25hcHNob3QnOiAyLFxuICAgICAgJ2lucHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAnb3V0cHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAna2VybmVsVGltZU1zJzogcHJvZmlsZS5rZXJuZWxzWzFdLmtlcm5lbFRpbWVNcyxcbiAgICAgICdleHRyYUluZm8nOiBwcm9maWxlLmtlcm5lbHNbMV0uZXh0cmFJbmZvXG4gICAgfSk7XG4gIH0pO1xuXG4gIGl0KCdzcXVhcmluZyB3aXRob3V0IGRpc3Bvc2luZycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBwcm9maWxlID0gYXdhaXQgdGYucHJvZmlsZSgoKSA9PiB7XG4gICAgICBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAgICAgIGNvbnN0IHgyID0geC5zcXVhcmUoKTtcbiAgICAgIHJldHVybiB4MjtcbiAgICB9KTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHByb2ZpbGUucmVzdWx0IGFzIFRlbnNvcjtcblxuICAgIGV4cGVjdChwcm9maWxlLm5ld0J5dGVzKS50b0JlKDI0KTtcbiAgICBleHBlY3QocHJvZmlsZS5wZWFrQnl0ZXMpLnRvQmUoMjQpO1xuICAgIGV4cGVjdChwcm9maWxlLm5ld1RlbnNvcnMpLnRvQmUoMik7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgWzEsIDQsIDldKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzLmxlbmd0aCkudG9CZSgxKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdLmtlcm5lbFRpbWVNcyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHNbMF0uZXh0cmFJbmZvIGluc3RhbmNlb2YgUHJvbWlzZSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVsc1swXSkudG9FcXVhbCh7XG4gICAgICAnbmFtZSc6ICdTcXVhcmUnLFxuICAgICAgJ2J5dGVzQWRkZWQnOiAxMixcbiAgICAgICd0b3RhbEJ5dGVzU25hcHNob3QnOiAyNCxcbiAgICAgICd0ZW5zb3JzQWRkZWQnOiAxLFxuICAgICAgJ3RvdGFsVGVuc29yc1NuYXBzaG90JzogMixcbiAgICAgICdpbnB1dFNoYXBlcyc6IFtbM11dLFxuICAgICAgJ291dHB1dFNoYXBlcyc6IFtbM11dLFxuICAgICAgJ2tlcm5lbFRpbWVNcyc6IHByb2ZpbGUua2VybmVsc1swXS5rZXJuZWxUaW1lTXMsXG4gICAgICAnZXh0cmFJbmZvJzogcHJvZmlsZS5rZXJuZWxzWzBdLmV4dHJhSW5mb1xuICAgIH0pO1xuICB9KTtcblxuICBpdCgnc3F1YXJpbmcgaW4gYXN5bmMgcXVlcnknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgcHJvZmlsZSA9IGF3YWl0IHRmLnByb2ZpbGUoYXN5bmMgKCkgPT4ge1xuICAgICAgYXdhaXQgbmV3IFByb21pc2UocmVzb2x2ZSA9PiBzZXRUaW1lb3V0KHJlc29sdmUsIDEpKTtcbiAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICAgICAgY29uc3QgeDIgPSB4LnNxdWFyZSgpO1xuICAgICAgeDIuZGlzcG9zZSgpO1xuICAgICAgcmV0dXJuIHg7XG4gICAgfSk7XG5cbiAgICBjb25zdCByZXN1bHQgPSBwcm9maWxlLnJlc3VsdCBhcyBUZW5zb3I7XG5cbiAgICBleHBlY3QocHJvZmlsZS5uZXdCeXRlcykudG9CZSgxMik7XG4gICAgZXhwZWN0KHByb2ZpbGUucGVha0J5dGVzKS50b0JlKDI0KTtcbiAgICBleHBlY3QocHJvZmlsZS5uZXdUZW5zb3JzKS50b0JlKDEpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFsxLCAyLCAzXSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVscy5sZW5ndGgpLnRvQmUoMSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVsc1swXS5rZXJuZWxUaW1lTXMgaW5zdGFuY2VvZiBQcm9taXNlKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdLmV4dHJhSW5mbyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHNbMF0pLnRvRXF1YWwoe1xuICAgICAgJ25hbWUnOiAnU3F1YXJlJyxcbiAgICAgICdieXRlc0FkZGVkJzogMTIsXG4gICAgICAndG90YWxCeXRlc1NuYXBzaG90JzogMjQsXG4gICAgICAndGVuc29yc0FkZGVkJzogMSxcbiAgICAgICd0b3RhbFRlbnNvcnNTbmFwc2hvdCc6IDIsXG4gICAgICAnaW5wdXRTaGFwZXMnOiBbWzNdXSxcbiAgICAgICdvdXRwdXRTaGFwZXMnOiBbWzNdXSxcbiAgICAgICdrZXJuZWxUaW1lTXMnOiBwcm9maWxlLmtlcm5lbHNbMF0ua2VybmVsVGltZU1zLFxuICAgICAgJ2V4dHJhSW5mbyc6IHByb2ZpbGUua2VybmVsc1swXS5leHRyYUluZm9cbiAgICB9KTtcbiAgfSk7XG5cbiAgaXQoJ3JlcG9ydHMgY29ycmVjdCBrZXJuZWxOYW1lcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBwcm9maWxlID0gYXdhaXQgdGYucHJvZmlsZSgoKSA9PiB7XG4gICAgICBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAgICAgIGNvbnN0IHgyID0geC5zcXVhcmUoKTtcbiAgICAgIGNvbnN0IHgzID0geDIuYWJzKCk7XG4gICAgICByZXR1cm4geDM7XG4gICAgfSk7XG5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxOYW1lcykudG9FcXVhbChqYXNtaW5lLmFycmF5V2l0aEV4YWN0Q29udGVudHMoW1xuICAgICAgJ1NxdWFyZScsICdBYnMnXG4gICAgXSkpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZVdpdGhGbGFncygnZGlzcG9zZVZhcmlhYmxlcycsIEFMTF9FTlZTLCAoKSA9PiB7XG4gIGl0KCdyZXVzZSBzYW1lIG5hbWUgdmFyaWFibGUnLCAoKSA9PiB7XG4gICAgdGYudGVuc29yMWQoWzEsIDIsIDNdKS52YXJpYWJsZSh0cnVlLCAndjEnKTtcbiAgICB0Zi50ZW5zb3IxZChbMSwgMiwgM10pLnZhcmlhYmxlKHRydWUsICd2MicpO1xuICAgIGV4cGVjdCgoKSA9PiB7XG4gICAgICB0Zi50ZW5zb3IxZChbMSwgMiwgM10pLnZhcmlhYmxlKHRydWUsICd2MScpO1xuICAgIH0pLnRvVGhyb3dFcnJvcigpO1xuICAgIHRmLmRpc3Bvc2VWYXJpYWJsZXMoKTtcbiAgICB0Zi50ZW5zb3IxZChbMSwgMiwgM10pLnZhcmlhYmxlKHRydWUsICd2MScpO1xuICAgIHRmLnRlbnNvcjFkKFsxLCAyLCAzXSkudmFyaWFibGUodHJ1ZSwgJ3YyJyk7XG4gIH0pO1xufSk7XG5cbi8qKlxuICogVGhlIGZvbGxvd2luZyB0ZXN0IGNvbnN0cmFpbnRzIHRvIHRoZSBDUFUgZW52aXJvbm1lbnQgYmVjYXVzZSBpdCBuZWVkcyBhXG4gKiBjb25jcmV0ZSBiYWNrZW5kIHRvIGV4aXN0LiBUaGlzIHRlc3Qgd2lsbCB3b3JrIGZvciBhbnkgYmFja2VuZCwgYnV0IGN1cnJlbnRseVxuICogdGhpcyBpcyB0aGUgc2ltcGxlc3QgYmFja2VuZCB0byB0ZXN0IGFnYWluc3QuXG4gKi9cbmRlc2NyaWJlV2l0aEZsYWdzKFxuICAgICdTd2l0Y2hpbmcgY3B1IGJhY2tlbmRzJyxcbiAgICB7cHJlZGljYXRlOiB0ZXN0RW52ID0+IHRlc3RFbnYuYmFja2VuZE5hbWUgPT09ICdjcHUnfSwgKCkgPT4ge1xuICAgICAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgICAgIHRmLnJlZ2lzdGVyQmFja2VuZCgnY3B1MScsIHRmLmZpbmRCYWNrZW5kRmFjdG9yeSgnY3B1JykpO1xuICAgICAgICB0Zi5yZWdpc3RlckJhY2tlbmQoJ2NwdTInLCB0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ2NwdScpKTtcbiAgICAgIH0pO1xuXG4gICAgICBhZnRlckVhY2goKCkgPT4ge1xuICAgICAgICB0Zi5yZW1vdmVCYWNrZW5kKCdjcHUxJyk7XG4gICAgICAgIHRmLnJlbW92ZUJhY2tlbmQoJ2NwdTInKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnTW92ZSBkYXRhIGZyb20gY3B1MSB0byBjcHUyIGJhY2tlbmQnLCBhc3luYyAoKSA9PiB7XG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgICAgLy8gVGhpcyBzY2FsYXIgbGl2ZXMgaW4gY3B1MS5cbiAgICAgICAgY29uc3QgYSA9IHRmLnNjYWxhcig1KTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCdjcHUyJyk7XG4gICAgICAgIC8vIFRoaXMgc2NhbGFyIGxpdmVzIGluIGNwdTIuXG4gICAgICAgIGNvbnN0IGIgPSB0Zi5zY2FsYXIoMyk7XG5cbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bURhdGFCdWZmZXJzKS50b0JlKDIpO1xuICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgyKTtcbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDgpO1xuXG4gICAgICAgIC8vIE1ha2Ugc3VyZSB5b3UgY2FuIHJlYWQgYm90aCB0ZW5zb3JzLlxuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBhLmRhdGEoKSwgWzVdKTtcbiAgICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgYi5kYXRhKCksIFszXSk7XG5cbiAgICAgICAgLy8gU3dpdGNoIGJhY2sgdG8gY3B1MS5cbiAgICAgICAgdGYuc2V0QmFja2VuZCgnY3B1MScpO1xuICAgICAgICAvLyBBZ2FpbiBtYWtlIHN1cmUgeW91IGNhbiByZWFkIGJvdGggdGVuc29ycy5cbiAgICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgYS5kYXRhKCksIFs1XSk7XG4gICAgICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGIuZGF0YSgpLCBbM10pO1xuXG4gICAgICAgIHRmLmRpc3Bvc2UoW2EsIGJdKTtcblxuICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtRGF0YUJ1ZmZlcnMpLnRvQmUoMCk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtQnl0ZXMpLnRvQmUoMCk7XG4gICAgICB9KTtcblxuICAgICAgaXQoJ2NhbiBleGVjdXRlIG9wIHdpdGggZGF0YSBmcm9tIG1peGVkIGJhY2tlbmRzJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICBjb25zdCBrZXJuZWxGdW5jID0gdGYuZ2V0S2VybmVsKCdBZGQnLCAnY3B1Jykua2VybmVsRnVuYztcbiAgICAgICAgdGYucmVnaXN0ZXJLZXJuZWwoe2tlcm5lbE5hbWU6ICdBZGQnLCBiYWNrZW5kTmFtZTogJ2NwdTEnLCBrZXJuZWxGdW5jfSk7XG4gICAgICAgIHRmLnJlZ2lzdGVyS2VybmVsKHtrZXJuZWxOYW1lOiAnQWRkJywgYmFja2VuZE5hbWU6ICdjcHUyJywga2VybmVsRnVuY30pO1xuXG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgICAgLy8gVGhpcyBzY2FsYXIgbGl2ZXMgaW4gY3B1MS5cbiAgICAgICAgY29uc3QgYSA9IHRmLnNjYWxhcig1KTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCdjcHUyJyk7XG4gICAgICAgIC8vIFRoaXMgc2NhbGFyIGxpdmVzIGluIGNwdTIuXG4gICAgICAgIGNvbnN0IGIgPSB0Zi5zY2FsYXIoMyk7XG5cbiAgICAgICAgLy8gVmVyaWZ5IHRoYXQgb3BzIGNhbiBleGVjdXRlIHdpdGggbWl4ZWQgYmFja2VuZCBkYXRhLlxuICAgICAgICBFTkdJTkUuc3RhcnRTY29wZSgpO1xuICAgICAgICB0Zi5zZXRCYWNrZW5kKCdjcHUxJyk7XG4gICAgICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHRmLmFkZChhLCBiKS5kYXRhKCksIFs4XSk7XG5cbiAgICAgICAgdGYuc2V0QmFja2VuZCgnY3B1MicpO1xuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGQoYSwgYikuZGF0YSgpLCBbOF0pO1xuICAgICAgICBFTkdJTkUuZW5kU2NvcGUoKTtcblxuICAgICAgICB0Zi5kaXNwb3NlKFthLCBiXSk7XG4gICAgICB9KTtcbiAgICB9KTtcblxuLyoqXG4gKiBUaGUgZm9sbG93aW5nIHVuaXQgdGVzdCBpcyBhIHNwZWNpYWwgaW50ZWdyYXRpb24tc3R5bGUgdGVzdCB0aGF0IGFzc3VtZXNcbiAqIHRoaW5ncyBhYm91dCBDUFUgJiBXZWJHTCBiYWNrZW5kcyBiZWluZyByZWdpc3RlcmVkLiBUaGlzIHRlc3RzIGRvZXNuJ3QgbGl2ZVxuICogaW4gdGhlIGJhY2tlbmQgZGlyZWN0b3J5IGJlY2F1c2UgaXQgaXMgdGVzdGluZyBlbmdpbmUgcmF0aGVyIHRoYW5cbiAqIGJhY2tlbmQtc3BlY2lmaWMgZGV0YWlscyBidXQgbmVlZHMgYSByZWFsIGJhY2tlbmQgdG8gZXhpc3QuIFRoaXMgdGVzdCB3aWxsXG4gKiBmYWlsIGlmIHRoZSBDUFUgYmFja2VuZHMgaXMgbm90IHJlZ2lzdGVyZWQuIFRoaXMgaXMgaW50ZW50aW9uYWwsIHdlIHNob3VsZFxuICogaGF2ZSBjb3ZlcmFnZSBmb3Igd2hlbiB0aGVzZSBiYWNrZW5kcyBhcmUgZW5hYmxlZCBhbmQgZW5zdXJlIHRoZXkgd29yayB3aXRoXG4gKiB0aGUgZW5naW5lLlxuICovXG4vLyBUT0RPKCM1NjMyKTogUmUtZW5hYmxlIHRoZXNlIHRlc3RzXG4vKlxuZGVzY3JpYmVXaXRoRmxhZ3MoXG4gICAgJ1N3aXRjaGluZyBXZWJHTCArIENQVSBiYWNrZW5kcycsIHtcbiAgICAgIHByZWRpY2F0ZTogdGVzdEVudiA9PiB0ZXN0RW52LmJhY2tlbmROYW1lID09PSAnd2ViZ2wnICYmXG4gICAgICAgICAgRU5HSU5FLmJhY2tlbmROYW1lcygpLmluZGV4T2YoJ3dlYmdsJykgIT09IC0xICYmXG4gICAgICAgICAgRU5HSU5FLmJhY2tlbmROYW1lcygpLmluZGV4T2YoJ2NwdScpICE9PSAtMVxuICAgIH0sXG4gICAgKCkgPT4ge1xuICAgICAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgICAgIHRmLnJlZ2lzdGVyQmFja2VuZCgnd2ViZ2wxJywgdGYuZmluZEJhY2tlbmRGYWN0b3J5KCd3ZWJnbCcpKTtcbiAgICAgICAgdGYucmVnaXN0ZXJCYWNrZW5kKCd3ZWJnbDInLCB0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ3dlYmdsJykpO1xuICAgICAgICB0Zi5yZWdpc3RlckJhY2tlbmQoJ2NwdTEnLCB0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ2NwdScpKTtcbiAgICAgIH0pO1xuXG4gICAgICBhZnRlckVhY2goKCkgPT4ge1xuICAgICAgICB0Zi5yZW1vdmVCYWNrZW5kKCd3ZWJnbDEnKTtcbiAgICAgICAgdGYucmVtb3ZlQmFja2VuZCgnd2ViZ2wyJyk7XG4gICAgICAgIHRmLnJlbW92ZUJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnY2FuIGV4ZWN1dGUgb3Agd2l0aCBkYXRhIGZyb20gbWl4ZWQgYmFja2VuZHMnLCBhc3luYyAoKSA9PiB7XG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMScpO1xuICAgICAgICBjb25zdCBhID0gdGYuc2NhbGFyKDUpO1xuXG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMicpO1xuICAgICAgICBjb25zdCBiID0gdGYuc2NhbGFyKDMpO1xuXG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgICAgY29uc3QgYyA9IHRmLnNjYWxhcigyKTtcblxuICAgICAgICAvLyBWZXJpZnkgdGhhdCBvcHMgY2FuIGV4ZWN1dGUgd2l0aCBtaXhlZCBiYWNrZW5kIGRhdGEuXG4gICAgICAgIEVOR0lORS5zdGFydFNjb3BlKCk7XG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMScpO1xuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGROKFthLCBiLCBjXSkuZGF0YSgpLCBbMTBdKTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCd3ZWJnbDInKTtcbiAgICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgdGYuYWRkTihbYSwgYiwgY10pLmRhdGEoKSwgWzEwXSk7XG5cbiAgICAgICAgdGYuc2V0QmFja2VuZCgnY3B1MScpO1xuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGROKFthLCBiLCBjXSkuZGF0YSgpLCBbMTBdKTtcbiAgICAgICAgRU5HSU5FLmVuZFNjb3BlKCk7XG5cbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMyk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1EYXRhQnVmZmVycykudG9CZSgzKTtcblxuICAgICAgICB0Zi5kaXNwb3NlKFthLCBiLCBjXSk7XG5cbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMCk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1EYXRhQnVmZmVycykudG9CZSgwKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnZnJvbVBpeGVscyB3aXRoIG1peGVkIGJhY2tlbmRzIHdvcmtzJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICB0Zi5zZXRCYWNrZW5kKCd3ZWJnbDEnKTtcbiAgICAgICAgY29uc3QgYSA9IHRmLmJyb3dzZXIuZnJvbVBpeGVscyhcbiAgICAgICAgICAgIG5ldyBJbWFnZURhdGEobmV3IFVpbnQ4Q2xhbXBlZEFycmF5KFsxLCAyLCAzLCA0XSksIDEsIDEpKTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCd3ZWJnbDInKTtcbiAgICAgICAgY29uc3QgYiA9IHRmLmJyb3dzZXIuZnJvbVBpeGVscyhcbiAgICAgICAgICAgIG5ldyBJbWFnZURhdGEobmV3IFVpbnQ4Q2xhbXBlZEFycmF5KFs1LCA2LCA3LCA4XSksIDEsIDEpKTtcblxuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGQoYSwgYikuZGF0YSgpLCBbNiwgOCwgMTBdKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnc2luZ2xlIHRpZHkgbXVsdGlwbGUgYmFja2VuZHMnLCAoKSA9PiB7XG4gICAgICAgIGNvbnN0IGtlcm5lbEZ1bmMgPSB0Zi5nZXRLZXJuZWwoJ1NxdWFyZScsICd3ZWJnbCcpLmtlcm5lbEZ1bmM7XG4gICAgICAgIHRmLnJlZ2lzdGVyS2VybmVsKFxuICAgICAgICAgICAge2tlcm5lbE5hbWU6ICdTcXVhcmUnLCBiYWNrZW5kTmFtZTogJ3dlYmdsMScsIGtlcm5lbEZ1bmN9KTtcbiAgICAgICAgdGYucmVnaXN0ZXJLZXJuZWwoXG4gICAgICAgICAgICB7a2VybmVsTmFtZTogJ1NxdWFyZScsIGJhY2tlbmROYW1lOiAnd2ViZ2wyJywga2VybmVsRnVuY30pO1xuXG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuXG4gICAgICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMScpO1xuICAgICAgICAgIGNvbnN0IGEgPSB0Zi5zY2FsYXIoMSk7XG4gICAgICAgICAgYS5zcXVhcmUoKTsgIC8vIFVwbG9hZHMgdG8gR1BVLlxuXG4gICAgICAgICAgdGYuc2V0QmFja2VuZCgnd2ViZ2wyJyk7XG4gICAgICAgICAgY29uc3QgYiA9IHRmLnNjYWxhcigxKTtcbiAgICAgICAgICBiLnNxdWFyZSgpOyAgLy8gVXBsb2FkcyB0byBHUFUuXG5cbiAgICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSg0KTtcbiAgICAgICAgfSk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuXG4gICAgICAgIHRmLnVucmVnaXN0ZXJLZXJuZWwoJ1NxdWFyZScsICd3ZWJnbDEnKTtcbiAgICAgICAgdGYudW5yZWdpc3Rlcktlcm5lbCgnU3F1YXJlJywgJ3dlYmdsMicpO1xuICAgICAgfSk7XG4gICAgfSk7XG4qL1xuaW50ZXJmYWNlIFRlc3RTdG9yYWdlIGV4dGVuZHMgS2VybmVsQmFja2VuZCB7XG4gIGlkOiBudW1iZXI7XG59XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdEZXRlY3RzIG1lbW9yeSBsZWFrcyBpbiBrZXJuZWxzJywgQUxMX0VOVlMsICgpID0+IHtcbiAgY29uc3QgYmFja2VuZE5hbWUgPSAndGVzdC1tZW0nO1xuICBjb25zdCBrZXJuZWxOYW1lID0gJ015S2VybmVsJztcbiAgY29uc3Qga2VybmVsTmFtZUNvbXBsZXggPSAnS2VybmVsLWNvbXBsZXgnO1xuXG4gIGl0KCdEZXRlY3RzIG1lbW9yeSBsZWFrIGluIGEga2VybmVsJywgKCkgPT4ge1xuICAgIGxldCBkYXRhSWRzQ291bnQgPSAwO1xuICAgIHRmLnJlZ2lzdGVyQmFja2VuZChiYWNrZW5kTmFtZSwgKCkgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaWQ6IDEsXG4gICAgICAgIGRpc3Bvc2U6ICgpID0+IG51bGwsXG4gICAgICAgIGRpc3Bvc2VEYXRhOiAoZGF0YUlkOiB7fSkgPT4gbnVsbCxcbiAgICAgICAgbnVtRGF0YUlkczogKCkgPT4gZGF0YUlkc0NvdW50XG4gICAgICB9IGFzIFRlc3RTdG9yYWdlO1xuICAgIH0pO1xuXG4gICAgY29uc3Qga2VybmVsV2l0aE1lbUxlYWs6IEtlcm5lbEZ1bmMgPSAoKSA9PiB7XG4gICAgICBkYXRhSWRzQ291bnQgKz0gMjtcbiAgICAgIHJldHVybiB7ZGF0YUlkOiB7fSwgc2hhcGU6IFtdLCBkdHlwZTogJ2Zsb2F0MzInfTtcbiAgICB9O1xuICAgIHRmLnJlZ2lzdGVyS2VybmVsKHtrZXJuZWxOYW1lLCBiYWNrZW5kTmFtZSwga2VybmVsRnVuYzoga2VybmVsV2l0aE1lbUxlYWt9KTtcblxuICAgIHRmLnNldEJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5lbmdpbmUoKS5ydW5LZXJuZWwoa2VybmVsTmFtZSwge30sIHt9KSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcihcbiAgICAgICAgICAgIC9CYWNrZW5kICd0ZXN0LW1lbScgaGFzIGFuIGludGVybmFsIG1lbW9yeSBsZWFrIFxcKDEgZGF0YSBpZHNcXCkvKTtcblxuICAgIHRmLnJlbW92ZUJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgIHRmLnVucmVnaXN0ZXJLZXJuZWwoa2VybmVsTmFtZSwgYmFja2VuZE5hbWUpO1xuICB9KTtcblxuICBpdCgnTm8gbWVtIGxlYWsgaW4gYSBrZXJuZWwgd2l0aCBtdWx0aXBsZSBvdXRwdXRzJywgKCkgPT4ge1xuICAgIGxldCBkYXRhSWRzQ291bnQgPSAwO1xuICAgIHRmLnJlZ2lzdGVyQmFja2VuZChiYWNrZW5kTmFtZSwgKCkgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaWQ6IDEsXG4gICAgICAgIGRpc3Bvc2U6ICgpID0+IG51bGwsXG4gICAgICAgIGRpc3Bvc2VEYXRhOiAoZGF0YUlkOiB7fSkgPT4gbnVsbCxcbiAgICAgICAgbnVtRGF0YUlkczogKCkgPT4gZGF0YUlkc0NvdW50XG4gICAgICB9IGFzIFRlc3RTdG9yYWdlO1xuICAgIH0pO1xuICAgIHRmLnNldEJhY2tlbmQoYmFja2VuZE5hbWUpO1xuXG4gICAgY29uc3Qga2VybmVsV2l0aDNPdXRwdXRzOiBLZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgZGF0YUlkc0NvdW50ICs9IDM7XG4gICAgICBjb25zdCB0OiBUZW5zb3JJbmZvID0ge2RhdGFJZDoge30sIHNoYXBlOiBbXSwgZHR5cGU6ICdmbG9hdDMyJ307XG4gICAgICByZXR1cm4gW3QsIHQsIHRdO1xuICAgIH07XG4gICAgdGYucmVnaXN0ZXJLZXJuZWwoXG4gICAgICAgIHtrZXJuZWxOYW1lLCBiYWNrZW5kTmFtZSwga2VybmVsRnVuYzoga2VybmVsV2l0aDNPdXRwdXRzfSk7XG5cbiAgICBjb25zdCByZXMgPSB0Zi5lbmdpbmUoKS5ydW5LZXJuZWwoa2VybmVsTmFtZSwge30sIHt9KTtcbiAgICBleHBlY3QoQXJyYXkuaXNBcnJheShyZXMpKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCgocmVzIGFzIEFycmF5PHt9PikubGVuZ3RoKS50b0JlKDMpO1xuXG4gICAgY29uc3Qga2VybmVsV2l0aENvbXBsZXhPdXRwdXRzOiBLZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgZGF0YUlkc0NvdW50ICs9IDM7XG4gICAgICByZXR1cm4ge2RhdGFJZDoge30sIHNoYXBlOiBbXSwgZHR5cGU6ICdjb21wbGV4NjQnfTtcbiAgICB9O1xuICAgIHRmLnJlZ2lzdGVyS2VybmVsKHtcbiAgICAgIGtlcm5lbE5hbWU6IGtlcm5lbE5hbWVDb21wbGV4LFxuICAgICAgYmFja2VuZE5hbWUsXG4gICAgICBrZXJuZWxGdW5jOiBrZXJuZWxXaXRoQ29tcGxleE91dHB1dHNcbiAgICB9KTtcblxuICAgIGNvbnN0IHJlczIgPSB0Zi5lbmdpbmUoKS5ydW5LZXJuZWwoa2VybmVsTmFtZUNvbXBsZXgsIHt9LCB7fSkgYXMgVGVuc29ySW5mbztcbiAgICBleHBlY3QocmVzMi5zaGFwZSkudG9FcXVhbChbXSk7XG4gICAgZXhwZWN0KHJlczIuZHR5cGUpLnRvRXF1YWwoJ2NvbXBsZXg2NCcpO1xuXG4gICAgdGYucmVtb3ZlQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gICAgdGYudW5yZWdpc3Rlcktlcm5lbChrZXJuZWxOYW1lLCBiYWNrZW5kTmFtZSk7XG4gICAgdGYudW5yZWdpc3Rlcktlcm5lbChrZXJuZWxOYW1lQ29tcGxleCwgYmFja2VuZE5hbWUpO1xuICB9KTtcbn0pO1xuXG4vLyBOT1RFOiBUaGlzIGRlc2NyaWJlIGlzIHB1cnBvc2VmdWxseSBub3QgYSBkZXNjcmliZVdpdGhGbGFncyBzbyB0aGF0IHdlXG4vLyB0ZXN0IHRlbnNvciBhbGxvY2F0aW9uIHdoZXJlIG5vIHNjb3BlcyBoYXZlIGJlZW4gY3JlYXRlZC5cbmRlc2NyaWJlKCdNZW1vcnkgYWxsb2NhdGlvbiBvdXRzaWRlIGEgdGVzdCBzY29wZScsICgpID0+IHtcbiAgaXQoJ2NvbnN0cnVjdGluZyBhIHRlbnNvciB3b3JrcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBiYWNrZW5kTmFtZSA9ICd0ZXN0LWJhY2tlbmQnO1xuICAgIHRmLnJlZ2lzdGVyQmFja2VuZChiYWNrZW5kTmFtZSwgKCkgPT4ge1xuICAgICAgbGV0IHN0b3JlZFZhbHVlczogQmFja2VuZFZhbHVlcyA9IG51bGw7XG4gICAgICByZXR1cm4ge1xuICAgICAgICBpZDogMSxcbiAgICAgICAgZmxvYXRQcmVjaXNpb246ICgpID0+IDMyLFxuICAgICAgICB3cml0ZTogKHZhbHVlczogQmFja2VuZFZhbHVlcywgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUpID0+IHtcbiAgICAgICAgICBjb25zdCBkYXRhSWQgPSB7fTtcbiAgICAgICAgICBzdG9yZWRWYWx1ZXMgPSB2YWx1ZXM7XG4gICAgICAgICAgcmV0dXJuIGRhdGFJZDtcbiAgICAgICAgfSxcbiAgICAgICAgcmVhZDogYXN5bmMgKGRhdGFJZDogb2JqZWN0KSA9PiBzdG9yZWRWYWx1ZXMsXG4gICAgICAgIGRpc3Bvc2U6ICgpID0+IG51bGwsXG4gICAgICAgIGRpc3Bvc2VEYXRhOiAoZGF0YUlkOiB7fSkgPT4gbnVsbFxuICAgICAgfSBhcyBUZXN0U3RvcmFnZTtcbiAgICB9KTtcbiAgICB0Zi5zZXRCYWNrZW5kKGJhY2tlbmROYW1lKTtcblxuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGEuZGF0YSgpLCBbMSwgMiwgM10pO1xuICAgIGEuZGlzcG9zZSgpO1xuXG4gICAgdGYucmVtb3ZlQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gIH0pO1xufSk7XG4iXX0=
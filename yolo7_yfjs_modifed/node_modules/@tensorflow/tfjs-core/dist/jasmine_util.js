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
// We use the pattern below (as opposed to require('jasmine') to create the
// jasmine module in order to avoid loading node specific modules which may
// be ignored in browser environments but cannot be ignored in react-native
// due to the pre-bundling of dependencies that it must do.
// tslint:disable-next-line:no-require-imports
const jasmineRequire = require('jasmine-core/lib/jasmine-core/jasmine.js');
const jasmineCore = jasmineRequire.core(jasmineRequire);
import { KernelBackend } from './backends/backend';
import { ENGINE } from './engine';
import { env } from './environment';
import { purgeLocalStorageArtifacts } from './io/local_storage';
import { isPromise } from './util_base';
Error.stackTraceLimit = Infinity;
jasmineCore.DEFAULT_TIMEOUT_INTERVAL = 20000;
export const NODE_ENVS = {
    predicate: () => env().platformName === 'node'
};
export const CHROME_ENVS = {
    flags: { 'IS_CHROME': true }
};
export const BROWSER_ENVS = {
    predicate: () => env().platformName === 'browser'
};
export const SYNC_BACKEND_ENVS = {
    predicate: (testEnv) => testEnv.isDataSync === true
};
export const HAS_WORKER = {
    predicate: () => typeof (Worker) !== 'undefined' &&
        typeof (Blob) !== 'undefined' && typeof (URL) !== 'undefined'
};
export const HAS_NODE_WORKER = {
    predicate: () => {
        let hasWorker = true;
        try {
            require.resolve('worker_threads');
        }
        catch (_a) {
            hasWorker = false;
        }
        return typeof (process) !== 'undefined' && hasWorker;
    }
};
export const ALL_ENVS = {};
// Tests whether the current environment satisfies the set of constraints.
export function envSatisfiesConstraints(env, testEnv, constraints) {
    if (constraints == null) {
        return true;
    }
    if (constraints.flags != null) {
        for (const flagName in constraints.flags) {
            const flagValue = constraints.flags[flagName];
            if (env.get(flagName) !== flagValue) {
                return false;
            }
        }
    }
    if (constraints.predicate != null && !constraints.predicate(testEnv)) {
        return false;
    }
    return true;
}
/**
 * Add test filtering logic to Jasmine's specFilter hook.
 *
 * @param testFilters Used for include a test suite, with the ability
 *     to selectively exclude some of the tests.
 *     Either `include` or `startsWith` must exist for a `TestFilter`.
 *     Tests that have the substrings specified by the include or startsWith
 *     will be included in the test run, unless one of the substrings specified
 *     by `excludes` appears in the name.
 * @param customInclude Function to programatically include a test.
 *     If this function returns true, a test will immediately run. Otherwise,
 *     `testFilters` is used for fine-grained filtering.
 *
 * If a test is not handled by `testFilters` or `customInclude`, the test will
 * be excluded in the test run.
 */
export function setupTestFilters(testFilters, customInclude) {
    const env = jasmine.getEnv();
    // Account for --grep flag passed to karma by saving the existing specFilter.
    const grepFilter = env.specFilter;
    /**
     * Filter method that returns boolean, if a given test should run or be
     * ignored based on its name. The exclude list has priority over the
     * include list. Thus, if a test matches both the exclude and the include
     * list, it will be exluded.
     */
    // tslint:disable-next-line: no-any
    env.specFilter = (spec) => {
        // Filter out tests if the --grep flag is passed.
        if (!grepFilter(spec)) {
            return false;
        }
        const name = spec.getFullName();
        if (customInclude(name)) {
            return true;
        }
        // Include tests of a test suite unless tests are in excludes list.
        for (let i = 0; i < testFilters.length; ++i) {
            const testFilter = testFilters[i];
            if ((testFilter.include != null &&
                name.indexOf(testFilter.include) > -1) ||
                (testFilter.startsWith != null &&
                    name.startsWith(testFilter.startsWith))) {
                if (testFilter.excludes != null) {
                    for (let j = 0; j < testFilter.excludes.length; j++) {
                        if (name.indexOf(testFilter.excludes[j]) > -1) {
                            return false;
                        }
                    }
                }
                return true;
            }
        }
        // Otherwise ignore the test.
        return false;
    };
}
export function parseTestEnvFromKarmaFlags(args, registeredTestEnvs) {
    let flags;
    let testEnvName;
    args.forEach((arg, i) => {
        if (arg === '--flags') {
            flags = JSON.parse(args[i + 1]);
        }
        else if (arg === '--testEnv') {
            testEnvName = args[i + 1];
        }
    });
    const testEnvNames = registeredTestEnvs.map(env => env.name).join(', ');
    if (flags != null && testEnvName == null) {
        throw new Error('--testEnv flag is required when --flags is present. ' +
            `Available values are [${testEnvNames}].`);
    }
    if (testEnvName == null) {
        return null;
    }
    let testEnv;
    registeredTestEnvs.forEach(env => {
        if (env.name === testEnvName) {
            testEnv = env;
        }
    });
    if (testEnv == null) {
        throw new Error(`Test environment with name ${testEnvName} not ` +
            `found. Available test environment names are ` +
            `${testEnvNames}`);
    }
    if (flags != null) {
        testEnv.flags = flags;
    }
    return testEnv;
}
export function describeWithFlags(name, constraints, tests) {
    if (TEST_ENVS.length === 0) {
        throw new Error(`Found no test environments. This is likely due to test environment ` +
            `registries never being imported or test environment registries ` +
            `being registered too late.`);
    }
    TEST_ENVS.forEach(testEnv => {
        env().setFlags(testEnv.flags);
        if (envSatisfiesConstraints(env(), testEnv, constraints)) {
            const testName = name + ' ' + testEnv.name + ' ' + JSON.stringify(testEnv.flags || {});
            executeTests(testName, tests, testEnv);
        }
    });
}
export const TEST_ENVS = [];
// Whether a call to setTestEnvs has been called so we turn off
// registration. This allows command line overriding or programmatic
// overriding of the default registrations.
let testEnvSet = false;
export function setTestEnvs(testEnvs) {
    testEnvSet = true;
    TEST_ENVS.length = 0;
    TEST_ENVS.push(...testEnvs);
}
export function registerTestEnv(testEnv) {
    // When using an explicit call to setTestEnvs, turn off registration of
    // test environments because the explicit call will set the test
    // environments.
    if (testEnvSet) {
        return;
    }
    TEST_ENVS.push(testEnv);
}
function executeTests(testName, tests, testEnv) {
    describe(testName, () => {
        beforeAll(async () => {
            ENGINE.reset();
            if (testEnv.flags != null) {
                env().setFlags(testEnv.flags);
            }
            env().set('IS_TEST', true);
            // Await setting the new backend since it can have async init.
            await ENGINE.setBackend(testEnv.backendName);
        });
        beforeEach(() => {
            ENGINE.startScope();
        });
        afterEach(() => {
            ENGINE.endScope();
            ENGINE.disposeVariables();
        });
        afterAll(() => {
            ENGINE.reset();
        });
        tests(testEnv);
    });
}
export class TestKernelBackend extends KernelBackend {
    dispose() { }
}
let lock = Promise.resolve();
/**
 * Wraps a Jasmine spec's test function so it is run exclusively to others that
 * use runWithLock.
 *
 * @param spec The function that runs the spec. Must return a promise or call
 *     `done()`.
 *
 */
export function runWithLock(spec) {
    return () => {
        lock = lock.then(async () => {
            let done;
            const donePromise = new Promise((resolve, reject) => {
                done = (() => {
                    resolve();
                });
                done.fail = (message) => {
                    reject(message);
                };
            });
            purgeLocalStorageArtifacts();
            const result = spec(done);
            if (isPromise(result)) {
                await result;
            }
            else {
                await donePromise;
            }
        });
        return lock;
    };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiamFzbWluZV91dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9qYXNtaW5lX3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsMkVBQTJFO0FBQzNFLDJFQUEyRTtBQUMzRSwyRUFBMkU7QUFDM0UsMkRBQTJEO0FBQzNELDhDQUE4QztBQUM5QyxNQUFNLGNBQWMsR0FBRyxPQUFPLENBQUMsMENBQTBDLENBQUMsQ0FBQztBQUMzRSxNQUFNLFdBQVcsR0FBRyxjQUFjLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDO0FBQ3hELE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUNqRCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxHQUFHLEVBQXFCLE1BQU0sZUFBZSxDQUFDO0FBQ3RELE9BQU8sRUFBQywwQkFBMEIsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBQzlELE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFdEMsS0FBSyxDQUFDLGVBQWUsR0FBRyxRQUFRLENBQUM7QUFDakMsV0FBVyxDQUFDLHdCQUF3QixHQUFHLEtBQUssQ0FBQztBQU83QyxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQWdCO0lBQ3BDLFNBQVMsRUFBRSxHQUFHLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxZQUFZLEtBQUssTUFBTTtDQUMvQyxDQUFDO0FBQ0YsTUFBTSxDQUFDLE1BQU0sV0FBVyxHQUFnQjtJQUN0QyxLQUFLLEVBQUUsRUFBQyxXQUFXLEVBQUUsSUFBSSxFQUFDO0NBQzNCLENBQUM7QUFDRixNQUFNLENBQUMsTUFBTSxZQUFZLEdBQWdCO0lBQ3ZDLFNBQVMsRUFBRSxHQUFHLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxZQUFZLEtBQUssU0FBUztDQUNsRCxD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QUFDO1lBQ0osQ0FBQyxDQUFDLENBQUM7WUFFSCwwQkFBMEIsRUFBRSxDQUFDO1lBQzdCLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUUxQixJQUFJLFNBQVMsQ0FBQyxNQUFNLENBQUMsRUFBRTtnQkFDckIsTUFBTSxNQUFNLENBQUM7YUFDZDtpQkFBTTtnQkFDTCxNQUFNLFdBQVcsQ0FBQzthQUNuQjtRQUNILENBQUMsQ0FBQyxDQUFDO1FBQ0gsT0FBTyxJQUFJLENBQUM7SUFDZCxDQUFDLENBQUM7QUFDSixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vLyBXZSB1c2UgdGhlIHBhdHRlcm4gYmVsb3cgKGFzIG9wcG9zZWQgdG8gcmVxdWlyZSgnamFzbWluZScpIHRvIGNyZWF0ZSB0aGVcbi8vIGphc21pbmUgbW9kdWxlIGluIG9yZGVyIHRvIGF2b2lkIGxvYWRpbmcgbm9kZSBzcGVjaWZpYyBtb2R1bGVzIHdoaWNoIG1heVxuLy8gYmUgaWdub3JlZCBpbiBicm93c2VyIGVudmlyb25tZW50cyBidXQgY2Fubm90IGJlIGlnbm9yZWQgaW4gcmVhY3QtbmF0aXZlXG4vLyBkdWUgdG8gdGhlIHByZS1idW5kbGluZyBvZiBkZXBlbmRlbmNpZXMgdGhhdCBpdCBtdXN0IGRvLlxuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLXJlcXVpcmUtaW1wb3J0c1xuY29uc3QgamFzbWluZVJlcXVpcmUgPSByZXF1aXJlKCdqYXNtaW5lLWNvcmUvbGliL2phc21pbmUtY29yZS9qYXNtaW5lLmpzJyk7XG5jb25zdCBqYXNtaW5lQ29yZSA9IGphc21pbmVSZXF1aXJlLmNvcmUoamFzbWluZVJlcXVpcmUpO1xuaW1wb3J0IHtLZXJuZWxCYWNrZW5kfSBmcm9tICcuL2JhY2tlbmRzL2JhY2tlbmQnO1xuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4vZW5naW5lJztcbmltcG9ydCB7ZW52LCBFbnZpcm9ubWVudCwgRmxhZ3N9IGZyb20gJy4vZW52aXJvbm1lbnQnO1xuaW1wb3J0IHtwdXJnZUxvY2FsU3RvcmFnZUFydGlmYWN0c30gZnJvbSAnLi9pby9sb2NhbF9zdG9yYWdlJztcbmltcG9ydCB7aXNQcm9taXNlfSBmcm9tICcuL3V0aWxfYmFzZSc7XG5cbkVycm9yLnN0YWNrVHJhY2VMaW1pdCA9IEluZmluaXR5O1xuamFzbWluZUNvcmUuREVGQVVMVF9USU1FT1VUX0lOVEVSVkFMID0gMjAwMDA7XG5cbmV4cG9ydCB0eXBlIENvbnN0cmFpbnRzID0ge1xuICBmbGFncz86IEZsYWdzLFxuICBwcmVkaWNhdGU/OiAodGVzdEVudjogVGVzdEVudikgPT4gYm9vbGVhbixcbn07XG5cbmV4cG9ydCBjb25zdCBOT0RFX0VOVlM6IENvbnN0cmFpbnRzID0ge1xuICBwcmVkaWNhdGU6ICgpID0+IGVudigpLnBsYXRmb3JtTmFtZSA9PT0gJ25vZGUnXG59O1xuZXhwb3J0IGNvbnN0IENIUk9NRV9FTlZTOiBDb25zdHJhaW50cyA9IHtcbiAgZmxhZ3M6IHsnSVNfQ0hST01FJzogdHJ1ZX1cbn07XG5leHBvcnQgY29uc3QgQlJPV1NFUl9FTlZTOiBDb25zdHJhaW50cyA9IHtcbiAgcHJlZGljYXRlOiAoKSA9PiBlbnYoKS5wbGF0Zm9ybU5hbWUgPT09ICdicm93c2VyJ1xufTtcblxuZXhwb3J0IGNvbnN0IFNZTkNfQkFDS0VORF9FTlZTOiBDb25zdHJhaW50cyA9IHtcbiAgcHJlZGljYXRlOiAodGVzdEVudjogVGVzdEVudikgPT4gdGVzdEVudi5pc0RhdGFTeW5jID09PSB0cnVlXG59O1xuXG5leHBvcnQgY29uc3QgSEFTX1dPUktFUiA9IHtcbiAgcHJlZGljYXRlOiAoKSA9PiB0eXBlb2YgKFdvcmtlcikgIT09ICd1bmRlZmluZWQnICYmXG4gICAgICB0eXBlb2YgKEJsb2IpICE9PSAndW5kZWZpbmVkJyAmJiB0eXBlb2YgKFVSTCkgIT09ICd1bmRlZmluZWQnXG59O1xuXG5leHBvcnQgY29uc3QgSEFTX05PREVfV09SS0VSID0ge1xuICBwcmVkaWNhdGU6ICgpID0+IHtcbiAgICBsZXQgaGFzV29ya2VyID0gdHJ1ZTtcbiAgICB0cnkge1xuICAgICAgcmVxdWlyZS5yZXNvbHZlKCd3b3JrZXJfdGhyZWFkcycpO1xuICAgIH0gY2F0Y2gge1xuICAgICAgaGFzV29ya2VyID0gZmFsc2U7XG4gICAgfVxuICAgIHJldHVybiB0eXBlb2YgKHByb2Nlc3MpICE9PSAndW5kZWZpbmVkJyAmJiBoYXNXb3JrZXI7XG4gIH1cbn07XG5cbmV4cG9ydCBjb25zdCBBTExfRU5WUzogQ29uc3RyYWludHMgPSB7fTtcblxuLy8gVGVzdHMgd2hldGhlciB0aGUgY3VycmVudCBlbnZpcm9ubWVudCBzYXRpc2ZpZXMgdGhlIHNldCBvZiBjb25zdHJhaW50cy5cbmV4cG9ydCBmdW5jdGlvbiBlbnZTYXRpc2ZpZXNDb25zdHJhaW50cyhcbiAgICBlbnY6IEVudmlyb25tZW50LCB0ZXN0RW52OiBUZXN0RW52LCBjb25zdHJhaW50czogQ29uc3RyYWludHMpOiBib29sZWFuIHtcbiAgaWYgKGNvbnN0cmFpbnRzID09IG51bGwpIHtcbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIGlmIChjb25zdHJhaW50cy5mbGFncyAhPSBudWxsKSB7XG4gICAgZm9yIChjb25zdCBmbGFnTmFtZSBpbiBjb25zdHJhaW50cy5mbGFncykge1xuICAgICAgY29uc3QgZmxhZ1ZhbHVlID0gY29uc3RyYWludHMuZmxhZ3NbZmxhZ05hbWVdO1xuICAgICAgaWYgKGVudi5nZXQoZmxhZ05hbWUpICE9PSBmbGFnVmFsdWUpIHtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfVxuICAgIH1cbiAgfVxuICBpZiAoY29uc3RyYWludHMucHJlZGljYXRlICE9IG51bGwgJiYgIWNvbnN0cmFpbnRzLnByZWRpY2F0ZSh0ZXN0RW52KSkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuICByZXR1cm4gdHJ1ZTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBUZXN0RmlsdGVyIHtcbiAgaW5jbHVkZT86IHN0cmluZztcbiAgc3RhcnRzV2l0aD86IHN0cmluZztcbiAgZXhjbHVkZXM/OiBzdHJpbmdbXTtcbn1cblxuLyoqXG4gKiBBZGQgdGVzdCBmaWx0ZXJpbmcgbG9naWMgdG8gSmFzbWluZSdzIHNwZWNGaWx0ZXIgaG9vay5cbiAqXG4gKiBAcGFyYW0gdGVzdEZpbHRlcnMgVXNlZCBmb3IgaW5jbHVkZSBhIHRlc3Qgc3VpdGUsIHdpdGggdGhlIGFiaWxpdHlcbiAqICAgICB0byBzZWxlY3RpdmVseSBleGNsdWRlIHNvbWUgb2YgdGhlIHRlc3RzLlxuICogICAgIEVpdGhlciBgaW5jbHVkZWAgb3IgYHN0YXJ0c1dpdGhgIG11c3QgZXhpc3QgZm9yIGEgYFRlc3RGaWx0ZXJgLlxuICogICAgIFRlc3RzIHRoYXQgaGF2ZSB0aGUgc3Vic3RyaW5ncyBzcGVjaWZpZWQgYnkgdGhlIGluY2x1ZGUgb3Igc3RhcnRzV2l0aFxuICogICAgIHdpbGwgYmUgaW5jbHVkZWQgaW4gdGhlIHRlc3QgcnVuLCB1bmxlc3Mgb25lIG9mIHRoZSBzdWJzdHJpbmdzIHNwZWNpZmllZFxuICogICAgIGJ5IGBleGNsdWRlc2AgYXBwZWFycyBpbiB0aGUgbmFtZS5cbiAqIEBwYXJhbSBjdXN0b21JbmNsdWRlIEZ1bmN0aW9uIHRvIHByb2dyYW1hdGljYWxseSBpbmNsdWRlIGEgdGVzdC5cbiAqICAgICBJZiB0aGlzIGZ1bmN0aW9uIHJldHVybnMgdHJ1ZSwgYSB0ZXN0IHdpbGwgaW1tZWRpYXRlbHkgcnVuLiBPdGhlcndpc2UsXG4gKiAgICAgYHRlc3RGaWx0ZXJzYCBpcyB1c2VkIGZvciBmaW5lLWdyYWluZWQgZmlsdGVyaW5nLlxuICpcbiAqIElmIGEgdGVzdCBpcyBub3QgaGFuZGxlZCBieSBgdGVzdEZpbHRlcnNgIG9yIGBjdXN0b21JbmNsdWRlYCwgdGhlIHRlc3Qgd2lsbFxuICogYmUgZXhjbHVkZWQgaW4gdGhlIHRlc3QgcnVuLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc2V0dXBUZXN0RmlsdGVycyhcbiAgICB0ZXN0RmlsdGVyczogVGVzdEZpbHRlcltdLCBjdXN0b21JbmNsdWRlOiAobmFtZTogc3RyaW5nKSA9PiBib29sZWFuKSB7XG4gIGNvbnN0IGVudiA9IGphc21pbmUuZ2V0RW52KCk7XG5cbiAgLy8gQWNjb3VudCBmb3IgLS1ncmVwIGZsYWcgcGFzc2VkIHRvIGthcm1hIGJ5IHNhdmluZyB0aGUgZXhpc3Rpbmcgc3BlY0ZpbHRlci5cbiAgY29uc3QgZ3JlcEZpbHRlciA9IGVudi5zcGVjRmlsdGVyO1xuXG4gIC8qKlxuICAgKiBGaWx0ZXIgbWV0aG9kIHRoYXQgcmV0dXJucyBib29sZWFuLCBpZiBhIGdpdmVuIHRlc3Qgc2hvdWxkIHJ1biBvciBiZVxuICAgKiBpZ25vcmVkIGJhc2VkIG9uIGl0cyBuYW1lLiBUaGUgZXhjbHVkZSBsaXN0IGhhcyBwcmlvcml0eSBvdmVyIHRoZVxuICAgKiBpbmNsdWRlIGxpc3QuIFRodXMsIGlmIGEgdGVzdCBtYXRjaGVzIGJvdGggdGhlIGV4Y2x1ZGUgYW5kIHRoZSBpbmNsdWRlXG4gICAqIGxpc3QsIGl0IHdpbGwgYmUgZXhsdWRlZC5cbiAgICovXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tYW55XG4gIGVudi5zcGVjRmlsdGVyID0gKHNwZWM6IGFueSkgPT4ge1xuICAgIC8vIEZpbHRlciBvdXQgdGVzdHMgaWYgdGhlIC0tZ3JlcCBmbGFnIGlzIHBhc3NlZC5cbiAgICBpZiAoIWdyZXBGaWx0ZXIoc3BlYykpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG5cbiAgICBjb25zdCBuYW1lID0gc3BlYy5nZXRGdWxsTmFtZSgpO1xuXG4gICAgaWYgKGN1c3RvbUluY2x1ZGUobmFtZSkpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIC8vIEluY2x1ZGUgdGVzdHMgb2YgYSB0ZXN0IHN1aXRlIHVubGVzcyB0ZXN0cyBhcmUgaW4gZXhjbHVkZXMgbGlzdC5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRlc3RGaWx0ZXJzLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCB0ZXN0RmlsdGVyID0gdGVzdEZpbHRlcnNbaV07XG4gICAgICBpZiAoKHRlc3RGaWx0ZXIuaW5jbHVkZSAhPSBudWxsICYmXG4gICAgICAgICAgIG5hbWUuaW5kZXhPZih0ZXN0RmlsdGVyLmluY2x1ZGUpID4gLTEpIHx8XG4gICAgICAgICAgKHRlc3RGaWx0ZXIuc3RhcnRzV2l0aCAhPSBudWxsICYmXG4gICAgICAgICAgIG5hbWUuc3RhcnRzV2l0aCh0ZXN0RmlsdGVyLnN0YXJ0c1dpdGgpKSkge1xuICAgICAgICBpZiAodGVzdEZpbHRlci5leGNsdWRlcyAhPSBudWxsKSB7XG4gICAgICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCB0ZXN0RmlsdGVyLmV4Y2x1ZGVzLmxlbmd0aDsgaisrKSB7XG4gICAgICAgICAgICBpZiAobmFtZS5pbmRleE9mKHRlc3RGaWx0ZXIuZXhjbHVkZXNbal0pID4gLTEpIHtcbiAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBPdGhlcndpc2UgaWdub3JlIHRoZSB0ZXN0LlxuICAgIHJldHVybiBmYWxzZTtcbiAgfTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHBhcnNlVGVzdEVudkZyb21LYXJtYUZsYWdzKFxuICAgIGFyZ3M6IHN0cmluZ1tdLCByZWdpc3RlcmVkVGVzdEVudnM6IFRlc3RFbnZbXSk6IFRlc3RFbnYge1xuICBsZXQgZmxhZ3M6IEZsYWdzO1xuICBsZXQgdGVzdEVudk5hbWU6IHN0cmluZztcblxuICBhcmdzLmZvckVhY2goKGFyZywgaSkgPT4ge1xuICAgIGlmIChhcmcgPT09ICctLWZsYWdzJykge1xuICAgICAgZmxhZ3MgPSBKU09OLnBhcnNlKGFyZ3NbaSArIDFdKTtcbiAgICB9IGVsc2UgaWYgKGFyZyA9PT0gJy0tdGVzdEVudicpIHtcbiAgICAgIHRlc3RFbnZOYW1lID0gYXJnc1tpICsgMV07XG4gICAgfVxuICB9KTtcblxuICBjb25zdCB0ZXN0RW52TmFtZXMgPSByZWdpc3RlcmVkVGVzdEVudnMubWFwKGVudiA9PiBlbnYubmFtZSkuam9pbignLCAnKTtcbiAgaWYgKGZsYWdzICE9IG51bGwgJiYgdGVzdEVudk5hbWUgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJy0tdGVzdEVudiBmbGFnIGlzIHJlcXVpcmVkIHdoZW4gLS1mbGFncyBpcyBwcmVzZW50LiAnICtcbiAgICAgICAgYEF2YWlsYWJsZSB2YWx1ZXMgYXJlIFske3Rlc3RFbnZOYW1lc31dLmApO1xuICB9XG4gIGlmICh0ZXN0RW52TmFtZSA9PSBudWxsKSB7XG4gICAgcmV0dXJuIG51bGw7XG4gIH1cblxuICBsZXQgdGVzdEVudjogVGVzdEVudjtcbiAgcmVnaXN0ZXJlZFRlc3RFbnZzLmZvckVhY2goZW52ID0+IHtcbiAgICBpZiAoZW52Lm5hbWUgPT09IHRlc3RFbnZOYW1lKSB7XG4gICAgICB0ZXN0RW52ID0gZW52O1xuICAgIH1cbiAgfSk7XG4gIGlmICh0ZXN0RW52ID09IG51bGwpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBUZXN0IGVudmlyb25tZW50IHdpdGggbmFtZSAke3Rlc3RFbnZOYW1lfSBub3QgYCArXG4gICAgICAgIGBmb3VuZC4gQXZhaWxhYmxlIHRlc3QgZW52aXJvbm1lbnQgbmFtZXMgYXJlIGAgK1xuICAgICAgICBgJHt0ZXN0RW52TmFtZXN9YCk7XG4gIH1cbiAgaWYgKGZsYWdzICE9IG51bGwpIHtcbiAgICB0ZXN0RW52LmZsYWdzID0gZmxhZ3M7XG4gIH1cblxuICByZXR1cm4gdGVzdEVudjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRlc2NyaWJlV2l0aEZsYWdzKFxuICAgIG5hbWU6IHN0cmluZywgY29uc3RyYWludHM6IENvbnN0cmFpbnRzLCB0ZXN0czogKGVudjogVGVzdEVudikgPT4gdm9pZCkge1xuICBpZiAoVEVTVF9FTlZTLmxlbmd0aCA9PT0gMCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYEZvdW5kIG5vIHRlc3QgZW52aXJvbm1lbnRzLiBUaGlzIGlzIGxpa2VseSBkdWUgdG8gdGVzdCBlbnZpcm9ubWVudCBgICtcbiAgICAgICAgYHJlZ2lzdHJpZXMgbmV2ZXIgYmVpbmcgaW1wb3J0ZWQgb3IgdGVzdCBlbnZpcm9ubWVudCByZWdpc3RyaWVzIGAgK1xuICAgICAgICBgYmVpbmcgcmVnaXN0ZXJlZCB0b28gbGF0ZS5gKTtcbiAgfVxuXG4gIFRFU1RfRU5WUy5mb3JFYWNoKHRlc3RFbnYgPT4ge1xuICAgIGVudigpLnNldEZsYWdzKHRlc3RFbnYuZmxhZ3MpO1xuICAgIGlmIChlbnZTYXRpc2ZpZXNDb25zdHJhaW50cyhlbnYoKSwgdGVzdEVudiwgY29uc3RyYWludHMpKSB7XG4gICAgICBjb25zdCB0ZXN0TmFtZSA9XG4gICAgICAgICAgbmFtZSArICcgJyArIHRlc3RFbnYubmFtZSArICcgJyArIEpTT04uc3RyaW5naWZ5KHRlc3RFbnYuZmxhZ3MgfHwge30pO1xuICAgICAgZXhlY3V0ZVRlc3RzKHRlc3ROYW1lLCB0ZXN0cywgdGVzdEVudik7XG4gICAgfVxuICB9KTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBUZXN0RW52IHtcbiAgbmFtZTogc3RyaW5nO1xuICBiYWNrZW5kTmFtZTogc3RyaW5nO1xuICBmbGFncz86IEZsYWdzO1xuICBpc0RhdGFTeW5jPzogYm9vbGVhbjtcbn1cblxuZXhwb3J0IGNvbnN0IFRFU1RfRU5WUzogVGVzdEVudltdID0gW107XG5cbi8vIFdoZXRoZXIgYSBjYWxsIHRvIHNldFRlc3RFbnZzIGhhcyBiZWVuIGNhbGxlZCBzbyB3ZSB0dXJuIG9mZlxuLy8gcmVnaXN0cmF0aW9uLiBUaGlzIGFsbG93cyBjb21tYW5kIGxpbmUgb3ZlcnJpZGluZyBvciBwcm9ncmFtbWF0aWNcbi8vIG92ZXJyaWRpbmcgb2YgdGhlIGRlZmF1bHQgcmVnaXN0cmF0aW9ucy5cbmxldCB0ZXN0RW52U2V0ID0gZmFsc2U7XG5leHBvcnQgZnVuY3Rpb24gc2V0VGVzdEVudnModGVzdEVudnM6IFRlc3RFbnZbXSkge1xuICB0ZXN0RW52U2V0ID0gdHJ1ZTtcbiAgVEVTVF9FTlZTLmxlbmd0aCA9IDA7XG4gIFRFU1RfRU5WUy5wdXNoKC4uLnRlc3RFbnZzKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHJlZ2lzdGVyVGVzdEVudih0ZXN0RW52OiBUZXN0RW52KSB7XG4gIC8vIFdoZW4gdXNpbmcgYW4gZXhwbGljaXQgY2FsbCB0byBzZXRUZXN0RW52cywgdHVybiBvZmYgcmVnaXN0cmF0aW9uIG9mXG4gIC8vIHRlc3QgZW52aXJvbm1lbnRzIGJlY2F1c2UgdGhlIGV4cGxpY2l0IGNhbGwgd2lsbCBzZXQgdGhlIHRlc3RcbiAgLy8gZW52aXJvbm1lbnRzLlxuICBpZiAodGVzdEVudlNldCkge1xuICAgIHJldHVybjtcbiAgfVxuICBURVNUX0VOVlMucHVzaCh0ZXN0RW52KTtcbn1cblxuZnVuY3Rpb24gZXhlY3V0ZVRlc3RzKFxuICAgIHRlc3ROYW1lOiBzdHJpbmcsIHRlc3RzOiAoZW52OiBUZXN0RW52KSA9PiB2b2lkLCB0ZXN0RW52OiBUZXN0RW52KSB7XG4gIGRlc2NyaWJlKHRlc3ROYW1lLCAoKSA9PiB7XG4gICAgYmVmb3JlQWxsKGFzeW5jICgpID0+IHtcbiAgICAgIEVOR0lORS5yZXNldCgpO1xuICAgICAgaWYgKHRlc3RFbnYuZmxhZ3MgIT0gbnVsbCkge1xuICAgICAgICBlbnYoKS5zZXRGbGFncyh0ZXN0RW52LmZsYWdzKTtcbiAgICAgIH1cbiAgICAgIGVudigpLnNldCgnSVNfVEVTVCcsIHRydWUpO1xuICAgICAgLy8gQXdhaXQgc2V0dGluZyB0aGUgbmV3IGJhY2tlbmQgc2luY2UgaXQgY2FuIGhhdmUgYXN5bmMgaW5pdC5cbiAgICAgIGF3YWl0IEVOR0lORS5zZXRCYWNrZW5kKHRlc3RFbnYuYmFja2VuZE5hbWUpO1xuICAgIH0pO1xuXG4gICAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgICBFTkdJTkUuc3RhcnRTY29wZSgpO1xuICAgIH0pO1xuXG4gICAgYWZ0ZXJFYWNoKCgpID0+IHtcbiAgICAgIEVOR0lORS5lbmRTY29wZSgpO1xuICAgICAgRU5HSU5FLmRpc3Bvc2VWYXJpYWJsZXMoKTtcbiAgICB9KTtcblxuICAgIGFmdGVyQWxsKCgpID0+IHtcbiAgICAgIEVOR0lORS5yZXNldCgpO1xuICAgIH0pO1xuXG4gICAgdGVzdHModGVzdEVudik7XG4gIH0pO1xufVxuXG5leHBvcnQgY2xhc3MgVGVzdEtlcm5lbEJhY2tlbmQgZXh0ZW5kcyBLZXJuZWxCYWNrZW5kIHtcbiAgZGlzcG9zZSgpOiB2b2lkIHt9XG59XG5cbmxldCBsb2NrID0gUHJvbWlzZS5yZXNvbHZlKCk7XG5cbi8qKlxuICogV3JhcHMgYSBKYXNtaW5lIHNwZWMncyB0ZXN0IGZ1bmN0aW9uIHNvIGl0IGlzIHJ1biBleGNsdXNpdmVseSB0byBvdGhlcnMgdGhhdFxuICogdXNlIHJ1bldpdGhMb2NrLlxuICpcbiAqIEBwYXJhbSBzcGVjIFRoZSBmdW5jdGlvbiB0aGF0IHJ1bnMgdGhlIHNwZWMuIE11c3QgcmV0dXJuIGEgcHJvbWlzZSBvciBjYWxsXG4gKiAgICAgYGRvbmUoKWAuXG4gKlxuICovXG5leHBvcnQgZnVuY3Rpb24gcnVuV2l0aExvY2soc3BlYzogKGRvbmU/OiBEb25lRm4pID0+IFByb21pc2U8dm9pZD58IHZvaWQpIHtcbiAgcmV0dXJuICgpID0+IHtcbiAgICBsb2NrID0gbG9jay50aGVuKGFzeW5jICgpID0+IHtcbiAgICAgIGxldCBkb25lOiBEb25lRm47XG4gICAgICBjb25zdCBkb25lUHJvbWlzZSA9IG5ldyBQcm9taXNlPHZvaWQ+KChyZXNvbHZlLCByZWplY3QpID0+IHtcbiAgICAgICAgZG9uZSA9ICgoKSA9PiB7XG4gICAgICAgICAgICAgICAgIHJlc29sdmUoKTtcbiAgICAgICAgICAgICAgIH0pIGFzIERvbmVGbjtcbiAgICAgICAgZG9uZS5mYWlsID0gKG1lc3NhZ2U/KSA9PiB7XG4gICAgICAgICAgcmVqZWN0KG1lc3NhZ2UpO1xuICAgICAgICB9O1xuICAgICAgfSk7XG5cbiAgICAgIHB1cmdlTG9jYWxTdG9yYWdlQXJ0aWZhY3RzKCk7XG4gICAgICBjb25zdCByZXN1bHQgPSBzcGVjKGRvbmUpO1xuXG4gICAgICBpZiAoaXNQcm9taXNlKHJlc3VsdCkpIHtcbiAgICAgICAgYXdhaXQgcmVzdWx0O1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgYXdhaXQgZG9uZVByb21pc2U7XG4gICAgICB9XG4gICAgfSk7XG4gICAgcmV0dXJuIGxvY2s7XG4gIH07XG59XG4iXX0=
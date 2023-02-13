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
/**
 * IOHandler implementations based on HTTP requests in the web browser.
 *
 * Uses [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 */
import { env } from '../environment';
import { assert } from '../util';
import { concatenateArrayBuffers, getModelArtifactsForJSON, getModelArtifactsInfoForJSON, getModelJSONForModelArtifacts } from './io_utils';
import { IORouterRegistry } from './router_registry';
import { loadWeightsAsArrayBuffer } from './weights_loader';
const OCTET_STREAM_MIME_TYPE = 'application/octet-stream';
const JSON_TYPE = 'application/json';
export class HTTPRequest {
    constructor(path, loadOptions) {
        this.DEFAULT_METHOD = 'POST';
        if (loadOptions == null) {
            loadOptions = {};
        }
        this.weightPathPrefix = loadOptions.weightPathPrefix;
        this.onProgress = loadOptions.onProgress;
        this.weightUrlConverter = loadOptions.weightUrlConverter;
        if (loadOptions.fetchFunc != null) {
            assert(typeof loadOptions.fetchFunc === 'function', () => 'Must pass a function that matches the signature of ' +
                '`fetch` (see ' +
                'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)');
            this.fetch = loadOptions.fetchFunc;
        }
        else {
            this.fetch = env().platform.fetch;
        }
        assert(path != null && path.length > 0, () => 'URL path for http must not be null, undefined or ' +
            'empty.');
        if (Array.isArray(path)) {
            assert(path.length === 2, () => 'URL paths for http must have a length of 2, ' +
                `(actual length is ${path.length}).`);
        }
        this.path = path;
        if (loadOptions.requestInit != null &&
            loadOptions.requestInit.body != null) {
            throw new Error('requestInit is expected to have no pre-existing body, but has one.');
        }
        this.requestInit = loadOptions.requestInit || {};
    }
    async save(modelArtifacts) {
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
            throw new Error('BrowserHTTPRequest.save() does not support saving model topology ' +
                'in binary formats yet.');
        }
        const init = Object.assign({ method: this.DEFAULT_METHOD }, this.requestInit);
        init.body = new FormData();
        const weightsManifest = [{
                paths: ['./model.weights.bin'],
                weights: modelArtifacts.weightSpecs,
            }];
        const modelTopologyAndWeightManifest = getModelJSONForModelArtifacts(modelArtifacts, weightsManifest);
        init.body.append('model.json', new Blob([JSON.stringify(modelTopologyAndWeightManifest)], { type: JSON_TYPE }), 'model.json');
        if (modelArtifacts.weightData != null) {
            init.body.append('model.weights.bin', new Blob([modelArtifacts.weightData], { type: OCTET_STREAM_MIME_TYPE }), 'model.weights.bin');
        }
        const response = await this.fetch(this.path, init);
        if (response.ok) {
            return {
                modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts),
                responses: [response],
            };
        }
        else {
            throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ` +
                `${response.status}.`);
        }
    }
    /**
     * Load model artifacts via HTTP request(s).
     *
     * See the documentation to `tf.io.http` for details on the saved
     * artifacts.
     *
     * @returns The loaded model artifacts (if loading succeeds).
     */
    async load() {
        const modelConfigRequest = await this.fetch(this.path, this.requestInit);
        if (!modelConfigRequest.ok) {
            throw new Error(`Request to ${this.path} failed with status code ` +
                `${modelConfigRequest.status}. Please verify this URL points to ` +
                `the model JSON of the model to load.`);
        }
        let modelJSON;
        try {
            modelJSON = await modelConfigRequest.json();
        }
        catch (e) {
            let message = `Failed to parse model JSON of response from ${this.path}.`;
            // TODO(nsthorat): Remove this after some time when we're comfortable that
            // .pb files are mostly gone.
            if (this.path.endsWith('.pb')) {
                message += ' Your path contains a .pb file extension. ' +
                    'Support for .pb models have been removed in TensorFlow.js 1.0 ' +
                    'in favor of .json models. You can re-convert your Python ' +
                    'TensorFlow model using the TensorFlow.js 1.0 conversion scripts ' +
                    'or you can convert your.pb models with the \'pb2json\'' +
                    'NPM script in the tensorflow/tfjs-converter repository.';
            }
            else {
                message += ' Please make sure the server is serving valid ' +
                    'JSON for this request.';
            }
            throw new Error(message);
        }
        // We do not allow both modelTopology and weightsManifest to be missing.
        const modelTopology = modelJSON.modelTopology;
        const weightsManifest = modelJSON.weightsManifest;
        if (modelTopology == null && weightsManifest == null) {
            throw new Error(`The JSON from HTTP path ${this.path} contains neither model ` +
                `topology or manifest for weights.`);
        }
        return getModelArtifactsForJSON(modelJSON, (weightsManifest) => this.loadWeights(weightsManifest));
    }
    async loadWeights(weightsManifest) {
        const weightPath = Array.isArray(this.path) ? this.path[1] : this.path;
        const [prefix, suffix] = parseUrl(weightPath);
        const pathPrefix = this.weightPathPrefix || prefix;
        const weightSpecs = [];
        for (const entry of weightsManifest) {
            weightSpecs.push(...entry.weights);
        }
        const fetchURLs = [];
        const urlPromises = [];
        for (const weightsGroup of weightsManifest) {
            for (const path of weightsGroup.paths) {
                if (this.weightUrlConverter != null) {
                    urlPromises.push(this.weightUrlConverter(path));
                }
                else {
                    fetchURLs.push(pathPrefix + path + suffix);
                }
            }
        }
        if (this.weightUrlConverter) {
            fetchURLs.push(...await Promise.all(urlPromises));
        }
        const buffers = await loadWeightsAsArrayBuffer(fetchURLs, {
            requestInit: this.requestInit,
            fetchFunc: this.fetch,
            onProgress: this.onProgress
        });
        return [weightSpecs, concatenateArrayBuffers(buffers)];
    }
}
HTTPRequest.URL_SCHEME_REGEX = /^https?:\/\//;
/**
 * Extract the prefix and suffix of the url, where the prefix is the path before
 * the last file, and suffix is the search params after the last file.
 * ```
 * const url = 'http://tfhub.dev/model/1/tensorflowjs_model.pb?tfjs-format=file'
 * [prefix, suffix] = parseUrl(url)
 * // prefix = 'http://tfhub.dev/model/1/'
 * // suffix = '?tfjs-format=file'
 * ```
 * @param url the model url to be parsed.
 */
export function parseUrl(url) {
    const lastSlash = url.lastIndexOf('/');
    const lastSearchParam = url.lastIndexOf('?');
    const prefix = url.substring(0, lastSlash);
    const suffix = lastSearchParam > lastSlash ? url.substring(lastSearchParam) : '';
    return [prefix + '/', suffix];
}
export function isHTTPScheme(url) {
    return url.match(HTTPRequest.URL_SCHEME_REGEX) != null;
}
export const httpRouter = (url, loadOptions) => {
    if (typeof fetch === 'undefined' &&
        (loadOptions == null || loadOptions.fetchFunc == null)) {
        // `http` uses `fetch` or `node-fetch`, if one wants to use it in
        // an environment that is not the browser or node they have to setup a
        // global fetch polyfill.
        return null;
    }
    else {
        let isHTTP = true;
        if (Array.isArray(url)) {
            isHTTP = url.every(urlItem => isHTTPScheme(urlItem));
        }
        else {
            isHTTP = isHTTPScheme(url);
        }
        if (isHTTP) {
            return http(url, loadOptions);
        }
    }
    return null;
};
IORouterRegistry.registerSaveRouter(httpRouter);
IORouterRegistry.registerLoadRouter(httpRouter);
/**
 * Creates an IOHandler subtype that sends model artifacts to HTTP server.
 *
 * An HTTP request of the `multipart/form-data` mime type will be sent to the
 * `path` URL. The form data includes artifacts that represent the topology
 * and/or weights of the model. In the case of Keras-style `tf.Model`, two
 * blobs (files) exist in form-data:
 *   - A JSON file consisting of `modelTopology` and `weightsManifest`.
 *   - A binary weights file consisting of the concatenated weight values.
 * These files are in the same format as the one generated by
 * [tfjs_converter](https://js.tensorflow.org/tutorials/import-keras.html).
 *
 * The following code snippet exemplifies the client-side code that uses this
 * function:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save(tf.io.http(
 *     'http://model-server:5000/upload', {requestInit: {method: 'PUT'}}));
 * console.log(saveResult);
 * ```
 *
 * If the default `POST` method is to be used, without any custom parameters
 * such as headers, you can simply pass an HTTP or HTTPS URL to `model.save`:
 *
 * ```js
 * const saveResult = await model.save('http://model-server:5000/upload');
 * ```
 *
 * The following GitHub Gist
 * https://gist.github.com/dsmilkov/1b6046fd6132d7408d5257b0976f7864
 * implements a server based on [flask](https://github.com/pallets/flask) that
 * can receive the request. Upon receiving the model artifacts via the requst,
 * this particular server reconsistutes instances of [Keras
 * Models](https://keras.io/models/model/) in memory.
 *
 *
 * @param path A URL path to the model.
 *   Can be an absolute HTTP path (e.g.,
 *   'http://localhost:8000/model-upload)') or a relative path (e.g.,
 *   './model-upload').
 * @param requestInit Request configurations to be used when sending
 *    HTTP request to server using `fetch`. It can contain fields such as
 *    `method`, `credentials`, `headers`, `mode`, etc. See
 *    https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
 *    for more information. `requestInit` must not have a body, because the
 * body will be set by TensorFlow.js. File blobs representing the model
 * topology (filename: 'model.json') and the weights of the model (filename:
 * 'model.weights.bin') will be appended to the body. If `requestInit` has a
 * `body`, an Error will be thrown.
 * @param loadOptions Optional configuration for the loading. It includes the
 *   following fields:
 *   - weightPathPrefix Optional, this specifies the path prefix for weight
 *     files, by default this is calculated from the path param.
 *   - fetchFunc Optional, custom `fetch` function. E.g., in Node.js,
 *     the `fetch` from node-fetch can be used here.
 *   - onProgress Optional, progress callback function, fired periodically
 *     before the load is completed.
 * @returns An instance of `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
export function http(path, loadOptions) {
    return new HTTPRequest(path, loadOptions);
}
/**
 * Deprecated. Use `tf.io.http`.
 * @param path
 * @param loadOptions
 */
export function browserHTTPRequest(path, loadOptions) {
    return http(path, loadOptions);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaHR0cC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvaW8vaHR0cC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSDs7OztHQUlHO0FBRUgsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBRW5DLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDL0IsT0FBTyxFQUFDLHVCQUF1QixFQUFFLHdCQUF3QixFQUFFLDRCQUE0QixFQUFFLDZCQUE2QixFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQzFJLE9BQU8sRUFBVyxnQkFBZ0IsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBRTdELE9BQU8sRUFBQyx3QkFBd0IsRUFBQyxNQUFNLGtCQUFrQixDQUFDO0FBRTFELE1BQU0sc0JBQXNCLEdBQUcsMEJBQTBCLENBQUM7QUFDMUQsTUFBTSxTQUFTLEdBQUcsa0JBQWtCLENBQUM7QUFDckMsTUFBTSxPQUFPLFdBQVc7SUFjdEIsWUFBWSxJQUFZLEVBQUUsV0FBeUI7UUFQMUMsbUJBQWMsR0FBRyxNQUFNLENBQUM7UUFRL0IsSUFBSSxXQUFXLElBQUksSUFBSSxFQUFFO1lBQ3ZCLFdBQVcsR0FBRyxFQUFFLENBQUM7U0FDbEI7UUFDRCxJQUFJLENBQUMsZ0JBQWdCLEdBQUcsV0FBVyxDQUFDLGdCQUFnQixDQUFDO1FBQ3JELElBQUksQ0FBQyxVQUFVLEdBQUcsV0FBVyxDQUFDLFVBQVUsQ0FBQztRQUN6QyxJQUFJLENBQUMsa0JBQWtCLEdBQUcsV0FBVyxDQUFDLGtCQUFrQixDQUFDO1FBRXpELElBQUksV0FBVyxDQUFDLFNBQVMsSUFBSSxJQUFJLEVBQUU7WUFDakMsTUFBTSxDQUNGLE9BQU8sV0FBVyxDQUFDLFNBQVMsS0FBSyxVQUFVLEVBQzNDLEdBQUcsRUFBRSxDQUFDLHFEQUFxRDtnQkFDdkQsZUFBZTtnQkFDZiw2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QUM7QUFDakMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyoqXG4gKiBJT0hhbmRsZXIgaW1wbGVtZW50YXRpb25zIGJhc2VkIG9uIEhUVFAgcmVxdWVzdHMgaW4gdGhlIHdlYiBicm93c2VyLlxuICpcbiAqIFVzZXMgW2BmZXRjaGBdKGh0dHBzOi8vZGV2ZWxvcGVyLm1vemlsbGEub3JnL2VuLVVTL2RvY3MvV2ViL0FQSS9GZXRjaF9BUEkpLlxuICovXG5cbmltcG9ydCB7ZW52fSBmcm9tICcuLi9lbnZpcm9ubWVudCc7XG5cbmltcG9ydCB7YXNzZXJ0fSBmcm9tICcuLi91dGlsJztcbmltcG9ydCB7Y29uY2F0ZW5hdGVBcnJheUJ1ZmZlcnMsIGdldE1vZGVsQXJ0aWZhY3RzRm9ySlNPTiwgZ2V0TW9kZWxBcnRpZmFjdHNJbmZvRm9ySlNPTiwgZ2V0TW9kZWxKU09ORm9yTW9kZWxBcnRpZmFjdHN9IGZyb20gJy4vaW9fdXRpbHMnO1xuaW1wb3J0IHtJT1JvdXRlciwgSU9Sb3V0ZXJSZWdpc3RyeX0gZnJvbSAnLi9yb3V0ZXJfcmVnaXN0cnknO1xuaW1wb3J0IHtJT0hhbmRsZXIsIExvYWRPcHRpb25zLCBNb2RlbEFydGlmYWN0cywgTW9kZWxKU09OLCBPblByb2dyZXNzQ2FsbGJhY2ssIFNhdmVSZXN1bHQsIFdlaWdodHNNYW5pZmVzdENvbmZpZywgV2VpZ2h0c01hbmlmZXN0RW50cnl9IGZyb20gJy4vdHlwZXMnO1xuaW1wb3J0IHtsb2FkV2VpZ2h0c0FzQXJyYXlCdWZmZXJ9IGZyb20gJy4vd2VpZ2h0c19sb2FkZXInO1xuXG5jb25zdCBPQ1RFVF9TVFJFQU1fTUlNRV9UWVBFID0gJ2FwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSc7XG5jb25zdCBKU09OX1RZUEUgPSAnYXBwbGljYXRpb24vanNvbic7XG5leHBvcnQgY2xhc3MgSFRUUFJlcXVlc3QgaW1wbGVtZW50cyBJT0hhbmRsZXIge1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcGF0aDogc3RyaW5nO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcmVxdWVzdEluaXQ6IFJlcXVlc3RJbml0O1xuXG4gIHByaXZhdGUgcmVhZG9ubHkgZmV0Y2g6IEZ1bmN0aW9uO1xuICBwcml2YXRlIHJlYWRvbmx5IHdlaWdodFVybENvbnZlcnRlcjogKHdlaWdodE5hbWU6IHN0cmluZykgPT4gUHJvbWlzZTxzdHJpbmc+O1xuXG4gIHJlYWRvbmx5IERFRkFVTFRfTUVUSE9EID0gJ1BPU1QnO1xuXG4gIHN0YXRpYyByZWFkb25seSBVUkxfU0NIRU1FX1JFR0VYID0gL15odHRwcz86XFwvXFwvLztcblxuICBwcml2YXRlIHJlYWRvbmx5IHdlaWdodFBhdGhQcmVmaXg6IHN0cmluZztcbiAgcHJpdmF0ZSByZWFkb25seSBvblByb2dyZXNzOiBPblByb2dyZXNzQ2FsbGJhY2s7XG5cbiAgY29uc3RydWN0b3IocGF0aDogc3RyaW5nLCBsb2FkT3B0aW9ucz86IExvYWRPcHRpb25zKSB7XG4gICAgaWYgKGxvYWRPcHRpb25zID09IG51bGwpIHtcbiAgICAgIGxvYWRPcHRpb25zID0ge307XG4gICAgfVxuICAgIHRoaXMud2VpZ2h0UGF0aFByZWZpeCA9IGxvYWRPcHRpb25zLndlaWdodFBhdGhQcmVmaXg7XG4gICAgdGhpcy5vblByb2dyZXNzID0gbG9hZE9wdGlvbnMub25Qcm9ncmVzcztcbiAgICB0aGlzLndlaWdodFVybENvbnZlcnRlciA9IGxvYWRPcHRpb25zLndlaWdodFVybENvbnZlcnRlcjtcblxuICAgIGlmIChsb2FkT3B0aW9ucy5mZXRjaEZ1bmMgIT0gbnVsbCkge1xuICAgICAgYXNzZXJ0KFxuICAgICAgICAgIHR5cGVvZiBsb2FkT3B0aW9ucy5mZXRjaEZ1bmMgPT09ICdmdW5jdGlvbicsXG4gICAgICAgICAgKCkgPT4gJ011c3QgcGFzcyBhIGZ1bmN0aW9uIHRoYXQgbWF0Y2hlcyB0aGUgc2lnbmF0dXJlIG9mICcgK1xuICAgICAgICAgICAgICAnYGZldGNoYCAoc2VlICcgK1xuICAgICAgICAgICAgICAnaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL0ZldGNoX0FQSSknKTtcbiAgICAgIHRoaXMuZmV0Y2ggPSBsb2FkT3B0aW9ucy5mZXRjaEZ1bmM7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuZmV0Y2ggPSBlbnYoKS5wbGF0Zm9ybS5mZXRjaDtcbiAgICB9XG5cbiAgICBhc3NlcnQoXG4gICAgICAgIHBhdGggIT0gbnVsbCAmJiBwYXRoLmxlbmd0aCA+IDAsXG4gICAgICAgICgpID0+ICdVUkwgcGF0aCBmb3IgaHR0cCBtdXN0IG5vdCBiZSBudWxsLCB1bmRlZmluZWQgb3IgJyArXG4gICAgICAgICAgICAnZW1wdHkuJyk7XG5cbiAgICBpZiAoQXJyYXkuaXNBcnJheShwYXRoKSkge1xuICAgICAgYXNzZXJ0KFxuICAgICAgICAgIHBhdGgubGVuZ3RoID09PSAyLFxuICAgICAgICAgICgpID0+ICdVUkwgcGF0aHMgZm9yIGh0dHAgbXVzdCBoYXZlIGEgbGVuZ3RoIG9mIDIsICcgK1xuICAgICAgICAgICAgICBgKGFjdHVhbCBsZW5ndGggaXMgJHtwYXRoLmxlbmd0aH0pLmApO1xuICAgIH1cbiAgICB0aGlzLnBhdGggPSBwYXRoO1xuXG4gICAgaWYgKGxvYWRPcHRpb25zLnJlcXVlc3RJbml0ICE9IG51bGwgJiZcbiAgICAgICAgbG9hZE9wdGlvbnMucmVxdWVzdEluaXQuYm9keSAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ3JlcXVlc3RJbml0IGlzIGV4cGVjdGVkIHRvIGhhdmUgbm8gcHJlLWV4aXN0aW5nIGJvZHksIGJ1dCBoYXMgb25lLicpO1xuICAgIH1cbiAgICB0aGlzLnJlcXVlc3RJbml0ID0gbG9hZE9wdGlvbnMucmVxdWVzdEluaXQgfHwge307XG4gIH1cblxuICBhc3luYyBzYXZlKG1vZGVsQXJ0aWZhY3RzOiBNb2RlbEFydGlmYWN0cyk6IFByb21pc2U8U2F2ZVJlc3VsdD4ge1xuICAgIGlmIChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5IGluc3RhbmNlb2YgQXJyYXlCdWZmZXIpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQnJvd3NlckhUVFBSZXF1ZXN0LnNhdmUoKSBkb2VzIG5vdCBzdXBwb3J0IHNhdmluZyBtb2RlbCB0b3BvbG9neSAnICtcbiAgICAgICAgICAnaW4gYmluYXJ5IGZvcm1hdHMgeWV0LicpO1xuICAgIH1cblxuICAgIGNvbnN0IGluaXQgPSBPYmplY3QuYXNzaWduKHttZXRob2Q6IHRoaXMuREVGQVVMVF9NRVRIT0R9LCB0aGlzLnJlcXVlc3RJbml0KTtcbiAgICBpbml0LmJvZHkgPSBuZXcgRm9ybURhdGEoKTtcblxuICAgIGNvbnN0IHdlaWdodHNNYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgIHBhdGhzOiBbJy4vbW9kZWwud2VpZ2h0cy5iaW4nXSxcbiAgICAgIHdlaWdodHM6IG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzLFxuICAgIH1dO1xuICAgIGNvbnN0IG1vZGVsVG9wb2xvZ3lBbmRXZWlnaHRNYW5pZmVzdDogTW9kZWxKU09OID1cbiAgICAgICAgZ2V0TW9kZWxKU09ORm9yTW9kZWxBcnRpZmFjdHMobW9kZWxBcnRpZmFjdHMsIHdlaWdodHNNYW5pZmVzdCk7XG5cbiAgICBpbml0LmJvZHkuYXBwZW5kKFxuICAgICAgICAnbW9kZWwuanNvbicsXG4gICAgICAgIG5ldyBCbG9iKFxuICAgICAgICAgICAgW0pTT04uc3RyaW5naWZ5KG1vZGVsVG9wb2xvZ3lBbmRXZWlnaHRNYW5pZmVzdCldLFxuICAgICAgICAgICAge3R5cGU6IEpTT05fVFlQRX0pLFxuICAgICAgICAnbW9kZWwuanNvbicpO1xuXG4gICAgaWYgKG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEgIT0gbnVsbCkge1xuICAgICAgaW5pdC5ib2R5LmFwcGVuZChcbiAgICAgICAgICAnbW9kZWwud2VpZ2h0cy5iaW4nLFxuICAgICAgICAgIG5ldyBCbG9iKFttb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhXSwge3R5cGU6IE9DVEVUX1NUUkVBTV9NSU1FX1RZUEV9KSxcbiAgICAgICAgICAnbW9kZWwud2VpZ2h0cy5iaW4nKTtcbiAgICB9XG5cbiAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IHRoaXMuZmV0Y2godGhpcy5wYXRoLCBpbml0KTtcblxuICAgIGlmIChyZXNwb25zZS5vaykge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgbW9kZWxBcnRpZmFjdHNJbmZvOiBnZXRNb2RlbEFydGlmYWN0c0luZm9Gb3JKU09OKG1vZGVsQXJ0aWZhY3RzKSxcbiAgICAgICAgcmVzcG9uc2VzOiBbcmVzcG9uc2VdLFxuICAgICAgfTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBCcm93c2VySFRUUFJlcXVlc3Quc2F2ZSgpIGZhaWxlZCBkdWUgdG8gSFRUUCByZXNwb25zZSBzdGF0dXMgYCArXG4gICAgICAgICAgYCR7cmVzcG9uc2Uuc3RhdHVzfS5gKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogTG9hZCBtb2RlbCBhcnRpZmFjdHMgdmlhIEhUVFAgcmVxdWVzdChzKS5cbiAgICpcbiAgICogU2VlIHRoZSBkb2N1bWVudGF0aW9uIHRvIGB0Zi5pby5odHRwYCBmb3IgZGV0YWlscyBvbiB0aGUgc2F2ZWRcbiAgICogYXJ0aWZhY3RzLlxuICAgKlxuICAgKiBAcmV0dXJucyBUaGUgbG9hZGVkIG1vZGVsIGFydGlmYWN0cyAoaWYgbG9hZGluZyBzdWNjZWVkcykuXG4gICAqL1xuICBhc3luYyBsb2FkKCk6IFByb21pc2U8TW9kZWxBcnRpZmFjdHM+IHtcbiAgICBjb25zdCBtb2RlbENvbmZpZ1JlcXVlc3QgPSBhd2FpdCB0aGlzLmZldGNoKHRoaXMucGF0aCwgdGhpcy5yZXF1ZXN0SW5pdCk7XG5cbiAgICBpZiAoIW1vZGVsQ29uZmlnUmVxdWVzdC5vaykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBSZXF1ZXN0IHRvICR7dGhpcy5wYXRofSBmYWlsZWQgd2l0aCBzdGF0dXMgY29kZSBgICtcbiAgICAgICAgICBgJHttb2RlbENvbmZpZ1JlcXVlc3Quc3RhdHVzfS4gUGxlYXNlIHZlcmlmeSB0aGlzIFVSTCBwb2ludHMgdG8gYCArXG4gICAgICAgICAgYHRoZSBtb2RlbCBKU09OIG9mIHRoZSBtb2RlbCB0byBsb2FkLmApO1xuICAgIH1cbiAgICBsZXQgbW9kZWxKU09OOiBNb2RlbEpTT047XG4gICAgdHJ5IHtcbiAgICAgIG1vZGVsSlNPTiA9IGF3YWl0IG1vZGVsQ29uZmlnUmVxdWVzdC5qc29uKCk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgbGV0IG1lc3NhZ2UgPSBgRmFpbGVkIHRvIHBhcnNlIG1vZGVsIEpTT04gb2YgcmVzcG9uc2UgZnJvbSAke3RoaXMucGF0aH0uYDtcbiAgICAgIC8vIFRPRE8obnN0aG9yYXQpOiBSZW1vdmUgdGhpcyBhZnRlciBzb21lIHRpbWUgd2hlbiB3ZSdyZSBjb21mb3J0YWJsZSB0aGF0XG4gICAgICAvLyAucGIgZmlsZXMgYXJlIG1vc3RseSBnb25lLlxuICAgICAgaWYgKHRoaXMucGF0aC5lbmRzV2l0aCgnLnBiJykpIHtcbiAgICAgICAgbWVzc2FnZSArPSAnIFlvdXIgcGF0aCBjb250YWlucyBhIC5wYiBmaWxlIGV4dGVuc2lvbi4gJyArXG4gICAgICAgICAgICAnU3VwcG9ydCBmb3IgLnBiIG1vZGVscyBoYXZlIGJlZW4gcmVtb3ZlZCBpbiBUZW5zb3JGbG93LmpzIDEuMCAnICtcbiAgICAgICAgICAgICdpbiBmYXZvciBvZiAuanNvbiBtb2RlbHMuIFlvdSBjYW4gcmUtY29udmVydCB5b3VyIFB5dGhvbiAnICtcbiAgICAgICAgICAgICdUZW5zb3JGbG93IG1vZGVsIHVzaW5nIHRoZSBUZW5zb3JGbG93LmpzIDEuMCBjb252ZXJzaW9uIHNjcmlwdHMgJyArXG4gICAgICAgICAgICAnb3IgeW91IGNhbiBjb252ZXJ0IHlvdXIucGIgbW9kZWxzIHdpdGggdGhlIFxcJ3BiMmpzb25cXCcnICtcbiAgICAgICAgICAgICdOUE0gc2NyaXB0IGluIHRoZSB0ZW5zb3JmbG93L3RmanMtY29udmVydGVyIHJlcG9zaXRvcnkuJztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIG1lc3NhZ2UgKz0gJyBQbGVhc2UgbWFrZSBzdXJlIHRoZSBzZXJ2ZXIgaXMgc2VydmluZyB2YWxpZCAnICtcbiAgICAgICAgICAgICdKU09OIGZvciB0aGlzIHJlcXVlc3QuJztcbiAgICAgIH1cbiAgICAgIHRocm93IG5ldyBFcnJvcihtZXNzYWdlKTtcbiAgICB9XG5cbiAgICAvLyBXZSBkbyBub3QgYWxsb3cgYm90aCBtb2RlbFRvcG9sb2d5IGFuZCB3ZWlnaHRzTWFuaWZlc3QgdG8gYmUgbWlzc2luZy5cbiAgICBjb25zdCBtb2RlbFRvcG9sb2d5ID0gbW9kZWxKU09OLm1vZGVsVG9wb2xvZ3k7XG4gICAgY29uc3Qgd2VpZ2h0c01hbmlmZXN0ID0gbW9kZWxKU09OLndlaWdodHNNYW5pZmVzdDtcbiAgICBpZiAobW9kZWxUb3BvbG9neSA9PSBudWxsICYmIHdlaWdodHNNYW5pZmVzdCA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYFRoZSBKU09OIGZyb20gSFRUUCBwYXRoICR7dGhpcy5wYXRofSBjb250YWlucyBuZWl0aGVyIG1vZGVsIGAgK1xuICAgICAgICAgIGB0b3BvbG9neSBvciBtYW5pZmVzdCBmb3Igd2VpZ2h0cy5gKTtcbiAgICB9XG5cbiAgICByZXR1cm4gZ2V0TW9kZWxBcnRpZmFjdHNGb3JKU09OKFxuICAgICAgICBtb2RlbEpTT04sICh3ZWlnaHRzTWFuaWZlc3QpID0+IHRoaXMubG9hZFdlaWdodHMod2VpZ2h0c01hbmlmZXN0KSk7XG4gIH1cblxuICBwcml2YXRlIGFzeW5jIGxvYWRXZWlnaHRzKHdlaWdodHNNYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnKTpcbiAgICAgIFByb21pc2U8W1dlaWdodHNNYW5pZmVzdEVudHJ5W10sIEFycmF5QnVmZmVyXT4ge1xuICAgIGNvbnN0IHdlaWdodFBhdGggPSBBcnJheS5pc0FycmF5KHRoaXMucGF0aCkgPyB0aGlzLnBhdGhbMV0gOiB0aGlzLnBhdGg7XG4gICAgY29uc3QgW3ByZWZpeCwgc3VmZml4XSA9IHBhcnNlVXJsKHdlaWdodFBhdGgpO1xuICAgIGNvbnN0IHBhdGhQcmVmaXggPSB0aGlzLndlaWdodFBhdGhQcmVmaXggfHwgcHJlZml4O1xuXG4gICAgY29uc3Qgd2VpZ2h0U3BlY3MgPSBbXTtcbiAgICBmb3IgKGNvbnN0IGVudHJ5IG9mIHdlaWdodHNNYW5pZmVzdCkge1xuICAgICAgd2VpZ2h0U3BlY3MucHVzaCguLi5lbnRyeS53ZWlnaHRzKTtcbiAgICB9XG5cbiAgICBjb25zdCBmZXRjaFVSTHM6IHN0cmluZ1tdID0gW107XG4gICAgY29uc3QgdXJsUHJvbWlzZXM6IEFycmF5PFByb21pc2U8c3RyaW5nPj4gPSBbXTtcbiAgICBmb3IgKGNvbnN0IHdlaWdodHNHcm91cCBvZiB3ZWlnaHRzTWFuaWZlc3QpIHtcbiAgICAgIGZvciAoY29uc3QgcGF0aCBvZiB3ZWlnaHRzR3JvdXAucGF0aHMpIHtcbiAgICAgICAgaWYgKHRoaXMud2VpZ2h0VXJsQ29udmVydGVyICE9IG51bGwpIHtcbiAgICAgICAgICB1cmxQcm9taXNlcy5wdXNoKHRoaXMud2VpZ2h0VXJsQ29udmVydGVyKHBhdGgpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBmZXRjaFVSTHMucHVzaChwYXRoUHJlZml4ICsgcGF0aCArIHN1ZmZpeCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICBpZiAodGhpcy53ZWlnaHRVcmxDb252ZXJ0ZXIpIHtcbiAgICAgIGZldGNoVVJMcy5wdXNoKC4uLmF3YWl0IFByb21pc2UuYWxsKHVybFByb21pc2VzKSk7XG4gICAgfVxuXG4gICAgY29uc3QgYnVmZmVycyA9IGF3YWl0IGxvYWRXZWlnaHRzQXNBcnJheUJ1ZmZlcihmZXRjaFVSTHMsIHtcbiAgICAgIHJlcXVlc3RJbml0OiB0aGlzLnJlcXVlc3RJbml0LFxuICAgICAgZmV0Y2hGdW5jOiB0aGlzLmZldGNoLFxuICAgICAgb25Qcm9ncmVzczogdGhpcy5vblByb2dyZXNzXG4gICAgfSk7XG4gICAgcmV0dXJuIFt3ZWlnaHRTcGVjcywgY29uY2F0ZW5hdGVBcnJheUJ1ZmZlcnMoYnVmZmVycyldO1xuICB9XG59XG5cbi8qKlxuICogRXh0cmFjdCB0aGUgcHJlZml4IGFuZCBzdWZmaXggb2YgdGhlIHVybCwgd2hlcmUgdGhlIHByZWZpeCBpcyB0aGUgcGF0aCBiZWZvcmVcbiAqIHRoZSBsYXN0IGZpbGUsIGFuZCBzdWZmaXggaXMgdGhlIHNlYXJjaCBwYXJhbXMgYWZ0ZXIgdGhlIGxhc3QgZmlsZS5cbiAqIGBgYFxuICogY29uc3QgdXJsID0gJ2h0dHA6Ly90Zmh1Yi5kZXYvbW9kZWwvMS90ZW5zb3JmbG93anNfbW9kZWwucGI/dGZqcy1mb3JtYXQ9ZmlsZSdcbiAqIFtwcmVmaXgsIHN1ZmZpeF0gPSBwYXJzZVVybCh1cmwpXG4gKiAvLyBwcmVmaXggPSAnaHR0cDovL3RmaHViLmRldi9tb2RlbC8xLydcbiAqIC8vIHN1ZmZpeCA9ICc/dGZqcy1mb3JtYXQ9ZmlsZSdcbiAqIGBgYFxuICogQHBhcmFtIHVybCB0aGUgbW9kZWwgdXJsIHRvIGJlIHBhcnNlZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHBhcnNlVXJsKHVybDogc3RyaW5nKTogW3N0cmluZywgc3RyaW5nXSB7XG4gIGNvbnN0IGxhc3RTbGFzaCA9IHVybC5sYXN0SW5kZXhPZignLycpO1xuICBjb25zdCBsYXN0U2VhcmNoUGFyYW0gPSB1cmwubGFzdEluZGV4T2YoJz8nKTtcbiAgY29uc3QgcHJlZml4ID0gdXJsLnN1YnN0cmluZygwLCBsYXN0U2xhc2gpO1xuICBjb25zdCBzdWZmaXggPVxuICAgICAgbGFzdFNlYXJjaFBhcmFtID4gbGFzdFNsYXNoID8gdXJsLnN1YnN0cmluZyhsYXN0U2VhcmNoUGFyYW0pIDogJyc7XG4gIHJldHVybiBbcHJlZml4ICsgJy8nLCBzdWZmaXhdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNIVFRQU2NoZW1lKHVybDogc3RyaW5nKTogYm9vbGVhbiB7XG4gIHJldHVybiB1cmwubWF0Y2goSFRUUFJlcXVlc3QuVVJMX1NDSEVNRV9SRUdFWCkgIT0gbnVsbDtcbn1cblxuZXhwb3J0IGNvbnN0IGh0dHBSb3V0ZXI6IElPUm91dGVyID1cbiAgICAodXJsOiBzdHJpbmcsIGxvYWRPcHRpb25zPzogTG9hZE9wdGlvbnMpID0+IHtcbiAgICAgIGlmICh0eXBlb2YgZmV0Y2ggPT09ICd1bmRlZmluZWQnICYmXG4gICAgICAgICAgKGxvYWRPcHRpb25zID09IG51bGwgfHwgbG9hZE9wdGlvbnMuZmV0Y2hGdW5jID09IG51bGwpKSB7XG4gICAgICAgIC8vIGBodHRwYCB1c2VzIGBmZXRjaGAgb3IgYG5vZGUtZmV0Y2hgLCBpZiBvbmUgd2FudHMgdG8gdXNlIGl0IGluXG4gICAgICAgIC8vIGFuIGVudmlyb25tZW50IHRoYXQgaXMgbm90IHRoZSBicm93c2VyIG9yIG5vZGUgdGhleSBoYXZlIHRvIHNldHVwIGFcbiAgICAgICAgLy8gZ2xvYmFsIGZldGNoIHBvbHlmaWxsLlxuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGxldCBpc0hUVFAgPSB0cnVlO1xuICAgICAgICBpZiAoQXJyYXkuaXNBcnJheSh1cmwpKSB7XG4gICAgICAgICAgaXNIVFRQID0gdXJsLmV2ZXJ5KHVybEl0ZW0gPT4gaXNIVFRQU2NoZW1lKHVybEl0ZW0pKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBpc0hUVFAgPSBpc0hUVFBTY2hlbWUodXJsKTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoaXNIVFRQKSB7XG4gICAgICAgICAgcmV0dXJuIGh0dHAodXJsLCBsb2FkT3B0aW9ucyk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIHJldHVybiBudWxsO1xuICAgIH07XG5JT1JvdXRlclJlZ2lzdHJ5LnJlZ2lzdGVyU2F2ZVJvdXRlcihodHRwUm91dGVyKTtcbklPUm91dGVyUmVnaXN0cnkucmVnaXN0ZXJMb2FkUm91dGVyKGh0dHBSb3V0ZXIpO1xuXG4vKipcbiAqIENyZWF0ZXMgYW4gSU9IYW5kbGVyIHN1YnR5cGUgdGhhdCBzZW5kcyBtb2RlbCBhcnRpZmFjdHMgdG8gSFRUUCBzZXJ2ZXIuXG4gKlxuICogQW4gSFRUUCByZXF1ZXN0IG9mIHRoZSBgbXVsdGlwYXJ0L2Zvcm0tZGF0YWAgbWltZSB0eXBlIHdpbGwgYmUgc2VudCB0byB0aGVcbiAqIGBwYXRoYCBVUkwuIFRoZSBmb3JtIGRhdGEgaW5jbHVkZXMgYXJ0aWZhY3RzIHRoYXQgcmVwcmVzZW50IHRoZSB0b3BvbG9neVxuICogYW5kL29yIHdlaWdodHMgb2YgdGhlIG1vZGVsLiBJbiB0aGUgY2FzZSBvZiBLZXJhcy1zdHlsZSBgdGYuTW9kZWxgLCB0d29cbiAqIGJsb2JzIChmaWxlcykgZXhpc3QgaW4gZm9ybS1kYXRhOlxuICogICAtIEEgSlNPTiBmaWxlIGNvbnNpc3Rpbmcgb2YgYG1vZGVsVG9wb2xvZ3lgIGFuZCBgd2VpZ2h0c01hbmlmZXN0YC5cbiAqICAgLSBBIGJpbmFyeSB3ZWlnaHRzIGZpbGUgY29uc2lzdGluZyBvZiB0aGUgY29uY2F0ZW5hdGVkIHdlaWdodCB2YWx1ZXMuXG4gKiBUaGVzZSBmaWxlcyBhcmUgaW4gdGhlIHNhbWUgZm9ybWF0IGFzIHRoZSBvbmUgZ2VuZXJhdGVkIGJ5XG4gKiBbdGZqc19jb252ZXJ0ZXJdKGh0dHBzOi8vanMudGVuc29yZmxvdy5vcmcvdHV0b3JpYWxzL2ltcG9ydC1rZXJhcy5odG1sKS5cbiAqXG4gKiBUaGUgZm9sbG93aW5nIGNvZGUgc25pcHBldCBleGVtcGxpZmllcyB0aGUgY2xpZW50LXNpZGUgY29kZSB0aGF0IHVzZXMgdGhpc1xuICogZnVuY3Rpb246XG4gKlxuICogYGBganNcbiAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICogbW9kZWwuYWRkKFxuICogICAgIHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMDBdLCBhY3RpdmF0aW9uOiAnc2lnbW9pZCd9KSk7XG4gKlxuICogY29uc3Qgc2F2ZVJlc3VsdCA9IGF3YWl0IG1vZGVsLnNhdmUodGYuaW8uaHR0cChcbiAqICAgICAnaHR0cDovL21vZGVsLXNlcnZlcjo1MDAwL3VwbG9hZCcsIHtyZXF1ZXN0SW5pdDoge21ldGhvZDogJ1BVVCd9fSkpO1xuICogY29uc29sZS5sb2coc2F2ZVJlc3VsdCk7XG4gKiBgYGBcbiAqXG4gKiBJZiB0aGUgZGVmYXVsdCBgUE9TVGAgbWV0aG9kIGlzIHRvIGJlIHVzZWQsIHdpdGhvdXQgYW55IGN1c3RvbSBwYXJhbWV0ZXJzXG4gKiBzdWNoIGFzIGhlYWRlcnMsIHlvdSBjYW4gc2ltcGx5IHBhc3MgYW4gSFRUUCBvciBIVFRQUyBVUkwgdG8gYG1vZGVsLnNhdmVgOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBzYXZlUmVzdWx0ID0gYXdhaXQgbW9kZWwuc2F2ZSgnaHR0cDovL21vZGVsLXNlcnZlcjo1MDAwL3VwbG9hZCcpO1xuICogYGBgXG4gKlxuICogVGhlIGZvbGxvd2luZyBHaXRIdWIgR2lzdFxuICogaHR0cHM6Ly9naXN0LmdpdGh1Yi5jb20vZHNtaWxrb3YvMWI2MDQ2ZmQ2MTMyZDc0MDhkNTI1N2IwOTc2Zjc4NjRcbiAqIGltcGxlbWVudHMgYSBzZXJ2ZXIgYmFzZWQgb24gW2ZsYXNrXShodHRwczovL2dpdGh1Yi5jb20vcGFsbGV0cy9mbGFzaykgdGhhdFxuICogY2FuIHJlY2VpdmUgdGhlIHJlcXVlc3QuIFVwb24gcmVjZWl2aW5nIHRoZSBtb2RlbCBhcnRpZmFjdHMgdmlhIHRoZSByZXF1c3QsXG4gKiB0aGlzIHBhcnRpY3VsYXIgc2VydmVyIHJlY29uc2lzdHV0ZXMgaW5zdGFuY2VzIG9mIFtLZXJhc1xuICogTW9kZWxzXShodHRwczovL2tlcmFzLmlvL21vZGVscy9tb2RlbC8pIGluIG1lbW9yeS5cbiAqXG4gKlxuICogQHBhcmFtIHBhdGggQSBVUkwgcGF0aCB0byB0aGUgbW9kZWwuXG4gKiAgIENhbiBiZSBhbiBhYnNvbHV0ZSBIVFRQIHBhdGggKGUuZy4sXG4gKiAgICdodHRwOi8vbG9jYWxob3N0OjgwMDAvbW9kZWwtdXBsb2FkKScpIG9yIGEgcmVsYXRpdmUgcGF0aCAoZS5nLixcbiAqICAgJy4vbW9kZWwtdXBsb2FkJykuXG4gKiBAcGFyYW0gcmVxdWVzdEluaXQgUmVxdWVzdCBjb25maWd1cmF0aW9ucyB0byBiZSB1c2VkIHdoZW4gc2VuZGluZ1xuICogICAgSFRUUCByZXF1ZXN0IHRvIHNlcnZlciB1c2luZyBgZmV0Y2hgLiBJdCBjYW4gY29udGFpbiBmaWVsZHMgc3VjaCBhc1xuICogICAgYG1ldGhvZGAsIGBjcmVkZW50aWFsc2AsIGBoZWFkZXJzYCwgYG1vZGVgLCBldGMuIFNlZVxuICogICAgaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL1JlcXVlc3QvUmVxdWVzdFxuICogICAgZm9yIG1vcmUgaW5mb3JtYXRpb24uIGByZXF1ZXN0SW5pdGAgbXVzdCBub3QgaGF2ZSBhIGJvZHksIGJlY2F1c2UgdGhlXG4gKiBib2R5IHdpbGwgYmUgc2V0IGJ5IFRlbnNvckZsb3cuanMuIEZpbGUgYmxvYnMgcmVwcmVzZW50aW5nIHRoZSBtb2RlbFxuICogdG9wb2xvZ3kgKGZpbGVuYW1lOiAnbW9kZWwuanNvbicpIGFuZCB0aGUgd2VpZ2h0cyBvZiB0aGUgbW9kZWwgKGZpbGVuYW1lOlxuICogJ21vZGVsLndlaWdodHMuYmluJykgd2lsbCBiZSBhcHBlbmRlZCB0byB0aGUgYm9keS4gSWYgYHJlcXVlc3RJbml0YCBoYXMgYVxuICogYGJvZHlgLCBhbiBFcnJvciB3aWxsIGJlIHRocm93bi5cbiAqIEBwYXJhbSBsb2FkT3B0aW9ucyBPcHRpb25hbCBjb25maWd1cmF0aW9uIGZvciB0aGUgbG9hZGluZy4gSXQgaW5jbHVkZXMgdGhlXG4gKiAgIGZvbGxvd2luZyBmaWVsZHM6XG4gKiAgIC0gd2VpZ2h0UGF0aFByZWZpeCBPcHRpb25hbCwgdGhpcyBzcGVjaWZpZXMgdGhlIHBhdGggcHJlZml4IGZvciB3ZWlnaHRcbiAqICAgICBmaWxlcywgYnkgZGVmYXVsdCB0aGlzIGlzIGNhbGN1bGF0ZWQgZnJvbSB0aGUgcGF0aCBwYXJhbS5cbiAqICAgLSBmZXRjaEZ1bmMgT3B0aW9uYWwsIGN1c3RvbSBgZmV0Y2hgIGZ1bmN0aW9uLiBFLmcuLCBpbiBOb2RlLmpzLFxuICogICAgIHRoZSBgZmV0Y2hgIGZyb20gbm9kZS1mZXRjaCBjYW4gYmUgdXNlZCBoZXJlLlxuICogICAtIG9uUHJvZ3Jlc3MgT3B0aW9uYWwsIHByb2dyZXNzIGNhbGxiYWNrIGZ1bmN0aW9uLCBmaXJlZCBwZXJpb2RpY2FsbHlcbiAqICAgICBiZWZvcmUgdGhlIGxvYWQgaXMgY29tcGxldGVkLlxuICogQHJldHVybnMgQW4gaW5zdGFuY2Ugb2YgYElPSGFuZGxlcmAuXG4gKlxuICogQGRvYyB7XG4gKiAgIGhlYWRpbmc6ICdNb2RlbHMnLFxuICogICBzdWJoZWFkaW5nOiAnTG9hZGluZycsXG4gKiAgIG5hbWVzcGFjZTogJ2lvJyxcbiAqICAgaWdub3JlQ0k6IHRydWVcbiAqIH1cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGh0dHAocGF0aDogc3RyaW5nLCBsb2FkT3B0aW9ucz86IExvYWRPcHRpb25zKTogSU9IYW5kbGVyIHtcbiAgcmV0dXJuIG5ldyBIVFRQUmVxdWVzdChwYXRoLCBsb2FkT3B0aW9ucyk7XG59XG5cbi8qKlxuICogRGVwcmVjYXRlZC4gVXNlIGB0Zi5pby5odHRwYC5cbiAqIEBwYXJhbSBwYXRoXG4gKiBAcGFyYW0gbG9hZE9wdGlvbnNcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGJyb3dzZXJIVFRQUmVxdWVzdChcbiAgICBwYXRoOiBzdHJpbmcsIGxvYWRPcHRpb25zPzogTG9hZE9wdGlvbnMpOiBJT0hhbmRsZXIge1xuICByZXR1cm4gaHR0cChwYXRoLCBsb2FkT3B0aW9ucyk7XG59XG4iXX0=
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
 * Classes and functions for model management across multiple storage mediums.
 *
 * Supported client actions:
 * - Listing models on all registered storage mediums.
 * - Remove model by URL from any registered storage mediums, by using URL
 *   string.
 * - Moving or copying model from one path to another in the same medium or from
 *   one medium to another, by using URL strings.
 */
import { assert } from '../util';
import { IORouterRegistry } from './router_registry';
const URL_SCHEME_SUFFIX = '://';
export class ModelStoreManagerRegistry {
    constructor() {
        this.managers = {};
    }
    static getInstance() {
        if (ModelStoreManagerRegistry.instance == null) {
            ModelStoreManagerRegistry.instance = new ModelStoreManagerRegistry();
        }
        return ModelStoreManagerRegistry.instance;
    }
    /**
     * Register a save-handler router.
     *
     * @param saveRouter A function that maps a URL-like string onto an instance
     * of `IOHandler` with the `save` method defined or `null`.
     */
    static registerManager(scheme, manager) {
        assert(scheme != null, () => 'scheme must not be undefined or null.');
        if (scheme.endsWith(URL_SCHEME_SUFFIX)) {
            scheme = scheme.slice(0, scheme.indexOf(URL_SCHEME_SUFFIX));
        }
        assert(scheme.length > 0, () => 'scheme must not be an empty string.');
        const registry = ModelStoreManagerRegistry.getInstance();
        assert(registry.managers[scheme] == null, () => `A model store manager is already registered for scheme '${scheme}'.`);
        registry.managers[scheme] = manager;
    }
    static getManager(scheme) {
        const manager = this.getInstance().managers[scheme];
        if (manager == null) {
            throw new Error(`Cannot find model manager for scheme '${scheme}'`);
        }
        return manager;
    }
    static getSchemes() {
        return Object.keys(this.getInstance().managers);
    }
}
/**
 * Helper method for parsing a URL string into a scheme and a path.
 *
 * @param url E.g., 'localstorage://my-model'
 * @returns A dictionary with two fields: scheme and path.
 *   Scheme: e.g., 'localstorage' in the example above.
 *   Path: e.g., 'my-model' in the example above.
 */
function parseURL(url) {
    if (url.indexOf(URL_SCHEME_SUFFIX) === -1) {
        throw new Error(`The url string provided does not contain a scheme. ` +
            `Supported schemes are: ` +
            `${ModelStoreManagerRegistry.getSchemes().join(',')}`);
    }
    return {
        scheme: url.split(URL_SCHEME_SUFFIX)[0],
        path: url.split(URL_SCHEME_SUFFIX)[1],
    };
}
async function cloneModelInternal(sourceURL, destURL, deleteSource = false) {
    assert(sourceURL !== destURL, () => `Old path and new path are the same: '${sourceURL}'`);
    const loadHandlers = IORouterRegistry.getLoadHandlers(sourceURL);
    assert(loadHandlers.length > 0, () => `Copying failed because no load handler is found for source URL ${sourceURL}.`);
    assert(loadHandlers.length < 2, () => `Copying failed because more than one (${loadHandlers.length}) ` +
        `load handlers for source URL ${sourceURL}.`);
    const loadHandler = loadHandlers[0];
    const saveHandlers = IORouterRegistry.getSaveHandlers(destURL);
    assert(saveHandlers.length > 0, () => `Copying failed because no save handler is found for destination ` +
        `URL ${destURL}.`);
    assert(saveHandlers.length < 2, () => `Copying failed because more than one (${loadHandlers.length}) ` +
        `save handlers for destination URL ${destURL}.`);
    const saveHandler = saveHandlers[0];
    const sourceScheme = parseURL(sourceURL).scheme;
    const sourcePath = parseURL(sourceURL).path;
    const sameMedium = sourceScheme === parseURL(sourceURL).scheme;
    const modelArtifacts = await loadHandler.load();
    // If moving within the same storage medium, remove the old model as soon as
    // the loading is done. Without doing this, it is possible that the combined
    // size of the two models will cause the cloning to fail.
    if (deleteSource && sameMedium) {
        await ModelStoreManagerRegistry.getManager(sourceScheme)
            .removeModel(sourcePath);
    }
    const saveResult = await saveHandler.save(modelArtifacts);
    // If moving between mediums, the deletion is done after the save succeeds.
    // This guards against the case in which saving to the destination medium
    // fails.
    if (deleteSource && !sameMedium) {
        await ModelStoreManagerRegistry.getManager(sourceScheme)
            .removeModel(sourcePath);
    }
    return saveResult.modelArtifactsInfo;
}
/**
 * List all models stored in registered storage mediums.
 *
 * For a web browser environment, the registered mediums are Local Storage and
 * IndexedDB.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @returns A `Promise` of a dictionary mapping URLs of existing models to
 * their model artifacts info. URLs include medium-specific schemes, e.g.,
 *   'indexeddb://my/model/1'. Model artifacts info include type of the
 * model's topology, byte sizes of the topology, weights, etc.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function listModels() {
    const schemes = ModelStoreManagerRegistry.getSchemes();
    const out = {};
    for (const scheme of schemes) {
        const schemeOut = await ModelStoreManagerRegistry.getManager(scheme).listModels();
        for (const path in schemeOut) {
            const url = scheme + URL_SCHEME_SUFFIX + path;
            out[url] = schemeOut[path];
        }
    }
    return out;
}
/**
 * Remove a model specified by URL from a reigstered storage medium.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @param url A URL to a stored model, with a scheme prefix, e.g.,
 *   'localstorage://my-model-1', 'indexeddb://my/model/2'.
 * @returns ModelArtifactsInfo of the deleted model (if and only if deletion
 *   is successful).
 * @throws Error if deletion fails, e.g., if no model exists at `path`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function removeModel(url) {
    const schemeAndPath = parseURL(url);
    const manager = ModelStoreManagerRegistry.getManager(schemeAndPath.scheme);
    return manager.removeModel(schemeAndPath.path);
}
/**
 * Copy a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Copying within a storage medium, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Copying between two storage mediums, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Copy the model, from Local Storage to IndexedDB.
 * await tf.io.copyModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove both models.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of copying.
 * @param destURL Destination URL of copying.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if copying fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function copyModel(sourceURL, destURL) {
    const deleteSource = false;
    return cloneModelInternal(sourceURL, destURL, deleteSource);
}
/**
 * Move a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Moving within a storage medium, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Moving between two storage mediums, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Move the model, from Local Storage to IndexedDB.
 * await tf.io.moveModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove the moved model.
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of moving.
 * @param destURL Destination URL of moving.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if moving fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function moveModel(sourceURL, destURL) {
    const deleteSource = true;
    return cloneModelInternal(sourceURL, destURL, deleteSource);
}
export { moveModel, copyModel, removeModel, listModels };
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibW9kZWxfbWFuYWdlbWVudC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvaW8vbW9kZWxfbWFuYWdlbWVudC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSDs7Ozs7Ozs7O0dBU0c7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRS9CLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBR25ELE1BQU0saUJBQWlCLEdBQUcsS0FBSyxDQUFDO0FBRWhDLE1BQU0sT0FBTyx5QkFBeUI7SUFNcEM7UUFDRSxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQztJQUNyQixDQUFDO0lBRU8sTUFBTSxDQUFDLFdBQVc7UUFDeEIsSUFBSSx5QkFBeUIsQ0FBQyxRQUFRLElBQUksSUFBSSxFQUFFO1lBQzlDLHlCQUF5QixDQUFDLFFBQVEsR0FBRyxJQUFJLHlCQUF5QixFQUFFLENBQUM7U0FDdEU7UUFDRCxPQUFPLHlCQUF5QixDQUFDLFFBQVEsQ0FBQztJQUM1QyxDQUFDO0lBRUQ7Ozs7O09BS0c7SUFDSCxNQUFNLENBQUMsZUFBZSxDQUFDLE1BQWMsRUFBRSxPQUEwQjtRQUMvRCxNQUFNLENBQUMsTUFBTSxJQUFJLElBQUksRUFBRSxHQUFHLEVBQUUsQ0FBQyx1Q0FBdUMsQ0FBQyxDQUFDO1FBQ3RFLElBQUksTUFBTSxDQUFDLFFBQVEsQ0FBQyxpQkFBaUIsQ0FBQyxFQUFFO1lBQ3RDLE1BQU0sR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxNQUFNLENBQUMsT0FBTyxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQztTQUM3RDtRQUNELE1BQU0sQ0FBQyxNQUFNLENBQUMsTUFBTSxHQUFHLENBQUMsRUFBRSxHQUFHLEVBQUUsQ0FBQyxxQ0FBcUMsQ0FBQyxDQUFDO1FBQ3ZFLE1BQU0s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LENBQUMsR0FBRyxTQUFTLENBQUMsSUFBSSxDQUFDLENBQUM7U0FDNUI7S0FDRjtJQUNELE9BQU8sR0FBRyxDQUFDO0FBQ2IsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQWdDRztBQUNILEtBQUssVUFBVSxXQUFXLENBQUMsR0FBVztJQUNwQyxNQUFNLGFBQWEsR0FBRyxRQUFRLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDcEMsTUFBTSxPQUFPLEdBQUcseUJBQXlCLENBQUMsVUFBVSxDQUFDLGFBQWEsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUMzRSxPQUFPLE9BQU8sQ0FBQyxXQUFXLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQ2pELENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQThDRztBQUNILEtBQUssVUFBVSxTQUFTLENBQ3BCLFNBQWlCLEVBQUUsT0FBZTtJQUNwQyxNQUFNLFlBQVksR0FBRyxLQUFLLENBQUM7SUFDM0IsT0FBTyxrQkFBa0IsQ0FBQyxTQUFTLEVBQUUsT0FBTyxFQUFFLFlBQVksQ0FBQyxDQUFDO0FBQzlELENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBNkNHO0FBQ0gsS0FBSyxVQUFVLFNBQVMsQ0FDcEIsU0FBaUIsRUFBRSxPQUFlO0lBQ3BDLE1BQU0sWUFBWSxHQUFHLElBQUksQ0FBQztJQUMxQixPQUFPLGtCQUFrQixDQUFDLFNBQVMsRUFBRSxPQUFPLEVBQUUsWUFBWSxDQUFDLENBQUM7QUFDOUQsQ0FBQztBQUVELE9BQU8sRUFBQyxTQUFTLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBRSxVQUFVLEVBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyoqXG4gKiBDbGFzc2VzIGFuZCBmdW5jdGlvbnMgZm9yIG1vZGVsIG1hbmFnZW1lbnQgYWNyb3NzIG11bHRpcGxlIHN0b3JhZ2UgbWVkaXVtcy5cbiAqXG4gKiBTdXBwb3J0ZWQgY2xpZW50IGFjdGlvbnM6XG4gKiAtIExpc3RpbmcgbW9kZWxzIG9uIGFsbCByZWdpc3RlcmVkIHN0b3JhZ2UgbWVkaXVtcy5cbiAqIC0gUmVtb3ZlIG1vZGVsIGJ5IFVSTCBmcm9tIGFueSByZWdpc3RlcmVkIHN0b3JhZ2UgbWVkaXVtcywgYnkgdXNpbmcgVVJMXG4gKiAgIHN0cmluZy5cbiAqIC0gTW92aW5nIG9yIGNvcHlpbmcgbW9kZWwgZnJvbSBvbmUgcGF0aCB0byBhbm90aGVyIGluIHRoZSBzYW1lIG1lZGl1bSBvciBmcm9tXG4gKiAgIG9uZSBtZWRpdW0gdG8gYW5vdGhlciwgYnkgdXNpbmcgVVJMIHN0cmluZ3MuXG4gKi9cblxuaW1wb3J0IHthc3NlcnR9IGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge0lPUm91dGVyUmVnaXN0cnl9IGZyb20gJy4vcm91dGVyX3JlZ2lzdHJ5JztcbmltcG9ydCB7TW9kZWxBcnRpZmFjdHNJbmZvLCBNb2RlbFN0b3JlTWFuYWdlcn0gZnJvbSAnLi90eXBlcyc7XG5cbmNvbnN0IFVSTF9TQ0hFTUVfU1VGRklYID0gJzovLyc7XG5cbmV4cG9ydCBjbGFzcyBNb2RlbFN0b3JlTWFuYWdlclJlZ2lzdHJ5IHtcbiAgLy8gU2luZ2xldG9uIGluc3RhbmNlLlxuICBwcml2YXRlIHN0YXRpYyBpbnN0YW5jZTogTW9kZWxTdG9yZU1hbmFnZXJSZWdpc3RyeTtcblxuICBwcml2YXRlIG1hbmFnZXJzOiB7W3NjaGVtZTogc3RyaW5nXTogTW9kZWxTdG9yZU1hbmFnZXJ9O1xuXG4gIHByaXZhdGUgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy5tYW5hZ2VycyA9IHt9O1xuICB9XG5cbiAgcHJpdmF0ZSBzdGF0aWMgZ2V0SW5zdGFuY2UoKTogTW9kZWxTdG9yZU1hbmFnZXJSZWdpc3RyeSB7XG4gICAgaWYgKE1vZGVsU3RvcmVNYW5hZ2VyUmVnaXN0cnkuaW5zdGFuY2UgPT0gbnVsbCkge1xuICAgICAgTW9kZWxTdG9yZU1hbmFnZXJSZWdpc3RyeS5pbnN0YW5jZSA9IG5ldyBNb2RlbFN0b3JlTWFuYWdlclJlZ2lzdHJ5KCk7XG4gICAgfVxuICAgIHJldHVybiBNb2RlbFN0b3JlTWFuYWdlclJlZ2lzdHJ5Lmluc3RhbmNlO1xuICB9XG5cbiAgLyoqXG4gICAqIFJlZ2lzdGVyIGEgc2F2ZS1oYW5kbGVyIHJvdXRlci5cbiAgICpcbiAgICogQHBhcmFtIHNhdmVSb3V0ZXIgQSBmdW5jdGlvbiB0aGF0IG1hcHMgYSBVUkwtbGlrZSBzdHJpbmcgb250byBhbiBpbnN0YW5jZVxuICAgKiBvZiBgSU9IYW5kbGVyYCB3aXRoIHRoZSBgc2F2ZWAgbWV0aG9kIGRlZmluZWQgb3IgYG51bGxgLlxuICAgKi9cbiAgc3RhdGljIHJlZ2lzdGVyTWFuYWdlcihzY2hlbWU6IHN0cmluZywgbWFuYWdlcjogTW9kZWxTdG9yZU1hbmFnZXIpIHtcbiAgICBhc3NlcnQoc2NoZW1lICE9IG51bGwsICgpID0+ICdzY2hlbWUgbXVzdCBub3QgYmUgdW5kZWZpbmVkIG9yIG51bGwuJyk7XG4gICAgaWYgKHNjaGVtZS5lbmRzV2l0aChVUkxfU0NIRU1FX1NVRkZJWCkpIHtcbiAgICAgIHNjaGVtZSA9IHNjaGVtZS5zbGljZSgwLCBzY2hlbWUuaW5kZXhPZihVUkxfU0NIRU1FX1NVRkZJWCkpO1xuICAgIH1cbiAgICBhc3NlcnQoc2NoZW1lLmxlbmd0aCA+IDAsICgpID0+ICdzY2hlbWUgbXVzdCBub3QgYmUgYW4gZW1wdHkgc3RyaW5nLicpO1xuICAgIGNvbnN0IHJlZ2lzdHJ5ID0gTW9kZWxTdG9yZU1hbmFnZXJSZWdpc3RyeS5nZXRJbnN0YW5jZSgpO1xuICAgIGFzc2VydChcbiAgICAgICAgcmVnaXN0cnkubWFuYWdlcnNbc2NoZW1lXSA9PSBudWxsLFxuICAgICAgICAoKSA9PiBgQSBtb2RlbCBzdG9yZSBtYW5hZ2VyIGlzIGFscmVhZHkgcmVnaXN0ZXJlZCBmb3Igc2NoZW1lICcke1xuICAgICAgICAgICAgc2NoZW1lfScuYCk7XG4gICAgcmVnaXN0cnkubWFuYWdlcnNbc2NoZW1lXSA9IG1hbmFnZXI7XG4gIH1cblxuICBzdGF0aWMgZ2V0TWFuYWdlcihzY2hlbWU6IHN0cmluZyk6IE1vZGVsU3RvcmVNYW5hZ2VyIHtcbiAgICBjb25zdCBtYW5hZ2VyID0gdGhpcy5nZXRJbnN0YW5jZSgpLm1hbmFnZXJzW3NjaGVtZV07XG4gICAgaWYgKG1hbmFnZXIgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBDYW5ub3QgZmluZCBtb2RlbCBtYW5hZ2VyIGZvciBzY2hlbWUgJyR7c2NoZW1lfSdgKTtcbiAgICB9XG4gICAgcmV0dXJuIG1hbmFnZXI7XG4gIH1cblxuICBzdGF0aWMgZ2V0U2NoZW1lcygpOiBzdHJpbmdbXSB7XG4gICAgcmV0dXJuIE9iamVjdC5rZXlzKHRoaXMuZ2V0SW5zdGFuY2UoKS5tYW5hZ2Vycyk7XG4gIH1cbn1cblxuLyoqXG4gKiBIZWxwZXIgbWV0aG9kIGZvciBwYXJzaW5nIGEgVVJMIHN0cmluZyBpbnRvIGEgc2NoZW1lIGFuZCBhIHBhdGguXG4gKlxuICogQHBhcmFtIHVybCBFLmcuLCAnbG9jYWxzdG9yYWdlOi8vbXktbW9kZWwnXG4gKiBAcmV0dXJucyBBIGRpY3Rpb25hcnkgd2l0aCB0d28gZmllbGRzOiBzY2hlbWUgYW5kIHBhdGguXG4gKiAgIFNjaGVtZTogZS5nLiwgJ2xvY2Fsc3RvcmFnZScgaW4gdGhlIGV4YW1wbGUgYWJvdmUuXG4gKiAgIFBhdGg6IGUuZy4sICdteS1tb2RlbCcgaW4gdGhlIGV4YW1wbGUgYWJvdmUuXG4gKi9cbmZ1bmN0aW9uIHBhcnNlVVJMKHVybDogc3RyaW5nKToge3NjaGVtZTogc3RyaW5nLCBwYXRoOiBzdHJpbmd9IHtcbiAgaWYgKHVybC5pbmRleE9mKFVSTF9TQ0hFTUVfU1VGRklYKSA9PT0gLTEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBUaGUgdXJsIHN0cmluZyBwcm92aWRlZCBkb2VzIG5vdCBjb250YWluIGEgc2NoZW1lLiBgICtcbiAgICAgICAgYFN1cHBvcnRlZCBzY2hlbWVzIGFyZTogYCArXG4gICAgICAgIGAke01vZGVsU3RvcmVNYW5hZ2VyUmVnaXN0cnkuZ2V0U2NoZW1lcygpLmpvaW4oJywnKX1gKTtcbiAgfVxuICByZXR1cm4ge1xuICAgIHNjaGVtZTogdXJsLnNwbGl0KFVSTF9TQ0hFTUVfU1VGRklYKVswXSxcbiAgICBwYXRoOiB1cmwuc3BsaXQoVVJMX1NDSEVNRV9TVUZGSVgpWzFdLFxuICB9O1xufVxuXG5hc3luYyBmdW5jdGlvbiBjbG9uZU1vZGVsSW50ZXJuYWwoXG4gICAgc291cmNlVVJMOiBzdHJpbmcsIGRlc3RVUkw6IHN0cmluZyxcbiAgICBkZWxldGVTb3VyY2UgPSBmYWxzZSk6IFByb21pc2U8TW9kZWxBcnRpZmFjdHNJbmZvPiB7XG4gIGFzc2VydChcbiAgICAgIHNvdXJjZVVSTCAhPT0gZGVzdFVSTCxcbiAgICAgICgpID0+IGBPbGQgcGF0aCBhbmQgbmV3IHBhdGggYXJlIHRoZSBzYW1lOiAnJHtzb3VyY2VVUkx9J2ApO1xuXG4gIGNvbnN0IGxvYWRIYW5kbGVycyA9IElPUm91dGVyUmVnaXN0cnkuZ2V0TG9hZEhhbmRsZXJzKHNvdXJjZVVSTCk7XG4gIGFzc2VydChcbiAgICAgIGxvYWRIYW5kbGVycy5sZW5ndGggPiAwLFxuICAgICAgKCkgPT4gYENvcHlpbmcgZmFpbGVkIGJlY2F1c2Ugbm8gbG9hZCBoYW5kbGVyIGlzIGZvdW5kIGZvciBzb3VyY2UgVVJMICR7XG4gICAgICAgICAgc291cmNlVVJMfS5gKTtcbiAgYXNzZXJ0KFxuICAgICAgbG9hZEhhbmRsZXJzLmxlbmd0aCA8IDIsXG4gICAgICAoKSA9PiBgQ29weWluZyBmYWlsZWQgYmVjYXVzZSBtb3JlIHRoYW4gb25lICgke2xvYWRIYW5kbGVycy5sZW5ndGh9KSBgICtcbiAgICAgICAgICBgbG9hZCBoYW5kbGVycyBmb3Igc291cmNlIFVSTCAke3NvdXJjZVVSTH0uYCk7XG4gIGNvbnN0IGxvYWRIYW5kbGVyID0gbG9hZEhhbmRsZXJzWzBdO1xuXG4gIGNvbnN0IHNhdmVIYW5kbGVycyA9IElPUm91dGVyUmVnaXN0cnkuZ2V0U2F2ZUhhbmRsZXJzKGRlc3RVUkwpO1xuICBhc3NlcnQoXG4gICAgICBzYXZlSGFuZGxlcnMubGVuZ3RoID4gMCxcbiAgICAgICgpID0+IGBDb3B5aW5nIGZhaWxlZCBiZWNhdXNlIG5vIHNhdmUgaGFuZGxlciBpcyBmb3VuZCBmb3IgZGVzdGluYXRpb24gYCArXG4gICAgICAgICAgYFVSTCAke2Rlc3RVUkx9LmApO1xuICBhc3NlcnQoXG4gICAgICBzYXZlSGFuZGxlcnMubGVuZ3RoIDwgMixcbiAgICAgICgpID0+IGBDb3B5aW5nIGZhaWxlZCBiZWNhdXNlIG1vcmUgdGhhbiBvbmUgKCR7bG9hZEhhbmRsZXJzLmxlbmd0aH0pIGAgK1xuICAgICAgICAgIGBzYXZlIGhhbmRsZXJzIGZvciBkZXN0aW5hdGlvbiBVUkwgJHtkZXN0VVJMfS5gKTtcbiAgY29uc3Qgc2F2ZUhhbmRsZXIgPSBzYXZlSGFuZGxlcnNbMF07XG5cbiAgY29uc3Qgc291cmNlU2NoZW1lID0gcGFyc2VVUkwoc291cmNlVVJMKS5zY2hlbWU7XG4gIGNvbnN0IHNvdXJjZVBhdGggPSBwYXJzZVVSTChzb3VyY2VVUkwpLnBhdGg7XG4gIGNvbnN0IHNhbWVNZWRpdW0gPSBzb3VyY2VTY2hlbWUgPT09IHBhcnNlVVJMKHNvdXJjZVVSTCkuc2NoZW1lO1xuXG4gIGNvbnN0IG1vZGVsQXJ0aWZhY3RzID0gYXdhaXQgbG9hZEhhbmRsZXIubG9hZCgpO1xuXG4gIC8vIElmIG1vdmluZyB3aXRoaW4gdGhlIHNhbWUgc3RvcmFnZSBtZWRpdW0sIHJlbW92ZSB0aGUgb2xkIG1vZGVsIGFzIHNvb24gYXNcbiAgLy8gdGhlIGxvYWRpbmcgaXMgZG9uZS4gV2l0aG91dCBkb2luZyB0aGlzLCBpdCBpcyBwb3NzaWJsZSB0aGF0IHRoZSBjb21iaW5lZFxuICAvLyBzaXplIG9mIHRoZSB0d28gbW9kZWxzIHdpbGwgY2F1c2UgdGhlIGNsb25pbmcgdG8gZmFpbC5cbiAgaWYgKGRlbGV0ZVNvdXJjZSAmJiBzYW1lTWVkaXVtKSB7XG4gICAgYXdhaXQgTW9kZWxTdG9yZU1hbmFnZXJSZWdpc3RyeS5nZXRNYW5hZ2VyKHNvdXJjZVNjaGVtZSlcbiAgICAgICAgLnJlbW92ZU1vZGVsKHNvdXJjZVBhdGgpO1xuICB9XG5cbiAgY29uc3Qgc2F2ZVJlc3VsdCA9IGF3YWl0IHNhdmVIYW5kbGVyLnNhdmUobW9kZWxBcnRpZmFjdHMpO1xuXG4gIC8vIElmIG1vdmluZyBiZXR3ZWVuIG1lZGl1bXMsIHRoZSBkZWxldGlvbiBpcyBkb25lIGFmdGVyIHRoZSBzYXZlIHN1Y2NlZWRzLlxuICAvLyBUaGlzIGd1YXJkcyBhZ2FpbnN0IHRoZSBjYXNlIGluIHdoaWNoIHNhdmluZyB0byB0aGUgZGVzdGluYXRpb24gbWVkaXVtXG4gIC8vIGZhaWxzLlxuICBpZiAoZGVsZXRlU291cmNlICYmICFzYW1lTWVkaXVtKSB7XG4gICAgYXdhaXQgTW9kZWxTdG9yZU1hbmFnZXJSZWdpc3RyeS5nZXRNYW5hZ2VyKHNvdXJjZVNjaGVtZSlcbiAgICAgICAgLnJlbW92ZU1vZGVsKHNvdXJjZVBhdGgpO1xuICB9XG5cbiAgcmV0dXJuIHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvO1xufVxuXG4vKipcbiAqIExpc3QgYWxsIG1vZGVscyBzdG9yZWQgaW4gcmVnaXN0ZXJlZCBzdG9yYWdlIG1lZGl1bXMuXG4gKlxuICogRm9yIGEgd2ViIGJyb3dzZXIgZW52aXJvbm1lbnQsIHRoZSByZWdpc3RlcmVkIG1lZGl1bXMgYXJlIExvY2FsIFN0b3JhZ2UgYW5kXG4gKiBJbmRleGVkREIuXG4gKlxuICogYGBganNcbiAqIC8vIEZpcnN0IGNyZWF0ZSBhbmQgc2F2ZSBhIG1vZGVsLlxuICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKCk7XG4gKiBtb2RlbC5hZGQodGYubGF5ZXJzLmRlbnNlKFxuICogICAgIHt1bml0czogMSwgaW5wdXRTaGFwZTogWzEwXSwgYWN0aXZhdGlvbjogJ3NpZ21vaWQnfSkpO1xuICogYXdhaXQgbW9kZWwuc2F2ZSgnbG9jYWxzdG9yYWdlOi8vZGVtby9tYW5hZ2VtZW50L21vZGVsMScpO1xuICpcbiAqIC8vIFRoZW4gbGlzdCBleGlzdGluZyBtb2RlbHMuXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShhd2FpdCB0Zi5pby5saXN0TW9kZWxzKCkpKTtcbiAqXG4gKiAvLyBEZWxldGUgdGhlIG1vZGVsLlxuICogYXdhaXQgdGYuaW8ucmVtb3ZlTW9kZWwoJ2xvY2Fsc3RvcmFnZTovL2RlbW8vbWFuYWdlbWVudC9tb2RlbDEnKTtcbiAqXG4gKiAvLyBMaXN0IG1vZGVscyBhZ2Fpbi5cbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KGF3YWl0IHRmLmlvLmxpc3RNb2RlbHMoKSkpO1xuICogYGBgXG4gKlxuICogQHJldHVybnMgQSBgUHJvbWlzZWAgb2YgYSBkaWN0aW9uYXJ5IG1hcHBpbmcgVVJMcyBvZiBleGlzdGluZyBtb2RlbHMgdG9cbiAqIHRoZWlyIG1vZGVsIGFydGlmYWN0cyBpbmZvLiBVUkxzIGluY2x1ZGUgbWVkaXVtLXNwZWNpZmljIHNjaGVtZXMsIGUuZy4sXG4gKiAgICdpbmRleGVkZGI6Ly9teS9tb2RlbC8xJy4gTW9kZWwgYXJ0aWZhY3RzIGluZm8gaW5jbHVkZSB0eXBlIG9mIHRoZVxuICogbW9kZWwncyB0b3BvbG9neSwgYnl0ZSBzaXplcyBvZiB0aGUgdG9wb2xvZ3ksIHdlaWdodHMsIGV0Yy5cbiAqXG4gKiBAZG9jIHtcbiAqICAgaGVhZGluZzogJ01vZGVscycsXG4gKiAgIHN1YmhlYWRpbmc6ICdNYW5hZ2VtZW50JyxcbiAqICAgbmFtZXNwYWNlOiAnaW8nLFxuICogICBpZ25vcmVDSTogdHJ1ZVxuICogfVxuICovXG5hc3luYyBmdW5jdGlvbiBsaXN0TW9kZWxzKCk6IFByb21pc2U8e1t1cmw6IHN0cmluZ106IE1vZGVsQXJ0aWZhY3RzSW5mb30+IHtcbiAgY29uc3Qgc2NoZW1lcyA9IE1vZGVsU3RvcmVNYW5hZ2VyUmVnaXN0cnkuZ2V0U2NoZW1lcygpO1xuICBjb25zdCBvdXQ6IHtbdXJsOiBzdHJpbmddOiBNb2RlbEFydGlmYWN0c0luZm99ID0ge307XG4gIGZvciAoY29uc3Qgc2NoZW1lIG9mIHNjaGVtZXMpIHtcbiAgICBjb25zdCBzY2hlbWVPdXQgPVxuICAgICAgICBhd2FpdCBNb2RlbFN0b3JlTWFuYWdlclJlZ2lzdHJ5LmdldE1hbmFnZXIoc2NoZW1lKS5saXN0TW9kZWxzKCk7XG4gICAgZm9yIChjb25zdCBwYXRoIGluIHNjaGVtZU91dCkge1xuICAgICAgY29uc3QgdXJsID0gc2NoZW1lICsgVVJMX1NDSEVNRV9TVUZGSVggKyBwYXRoO1xuICAgICAgb3V0W3VybF0gPSBzY2hlbWVPdXRbcGF0aF07XG4gICAgfVxuICB9XG4gIHJldHVybiBvdXQ7XG59XG5cbi8qKlxuICogUmVtb3ZlIGEgbW9kZWwgc3BlY2lmaWVkIGJ5IFVSTCBmcm9tIGEgcmVpZ3N0ZXJlZCBzdG9yYWdlIG1lZGl1bS5cbiAqXG4gKiBgYGBqc1xuICogLy8gRmlyc3QgY3JlYXRlIGFuZCBzYXZlIGEgbW9kZWwuXG4gKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoKTtcbiAqIG1vZGVsLmFkZCh0Zi5sYXllcnMuZGVuc2UoXG4gKiAgICAge3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbMTBdLCBhY3RpdmF0aW9uOiAnc2lnbW9pZCd9KSk7XG4gKiBhd2FpdCBtb2RlbC5zYXZlKCdsb2NhbHN0b3JhZ2U6Ly9kZW1vL21hbmFnZW1lbnQvbW9kZWwxJyk7XG4gKlxuICogLy8gVGhlbiBsaXN0IGV4aXN0aW5nIG1vZGVscy5cbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KGF3YWl0IHRmLmlvLmxpc3RNb2RlbHMoKSkpO1xuICpcbiAqIC8vIERlbGV0ZSB0aGUgbW9kZWwuXG4gKiBhd2FpdCB0Zi5pby5yZW1vdmVNb2RlbCgnbG9jYWxzdG9yYWdlOi8vZGVtby9tYW5hZ2VtZW50L21vZGVsMScpO1xuICpcbiAqIC8vIExpc3QgbW9kZWxzIGFnYWluLlxuICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkoYXdhaXQgdGYuaW8ubGlzdE1vZGVscygpKSk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gdXJsIEEgVVJMIHRvIGEgc3RvcmVkIG1vZGVsLCB3aXRoIGEgc2NoZW1lIHByZWZpeCwgZS5nLixcbiAqICAgJ2xvY2Fsc3RvcmFnZTovL215LW1vZGVsLTEnLCAnaW5kZXhlZGRiOi8vbXkvbW9kZWwvMicuXG4gKiBAcmV0dXJucyBNb2RlbEFydGlmYWN0c0luZm8gb2YgdGhlIGRlbGV0ZWQgbW9kZWwgKGlmIGFuZCBvbmx5IGlmIGRlbGV0aW9uXG4gKiAgIGlzIHN1Y2Nlc3NmdWwpLlxuICogQHRocm93cyBFcnJvciBpZiBkZWxldGlvbiBmYWlscywgZS5nLiwgaWYgbm8gbW9kZWwgZXhpc3RzIGF0IGBwYXRoYC5cbiAqXG4gKiBAZG9jIHtcbiAqICAgaGVhZGluZzogJ01vZGVscycsXG4gKiAgIHN1YmhlYWRpbmc6ICdNYW5hZ2VtZW50JyxcbiAqICAgbmFtZXNwYWNlOiAnaW8nLFxuICogICBpZ25vcmVDSTogdHJ1ZVxuICogfVxuICovXG5hc3luYyBmdW5jdGlvbiByZW1vdmVNb2RlbCh1cmw6IHN0cmluZyk6IFByb21pc2U8TW9kZWxBcnRpZmFjdHNJbmZvPiB7XG4gIGNvbnN0IHNjaGVtZUFuZFBhdGggPSBwYXJzZVVSTCh1cmwpO1xuICBjb25zdCBtYW5hZ2VyID0gTW9kZWxTdG9yZU1hbmFnZXJSZWdpc3RyeS5nZXRNYW5hZ2VyKHNjaGVtZUFuZFBhdGguc2NoZW1lKTtcbiAgcmV0dXJuIG1hbmFnZXIucmVtb3ZlTW9kZWwoc2NoZW1lQW5kUGF0aC5wYXRoKTtcbn1cblxuLyoqXG4gKiBDb3B5IGEgbW9kZWwgZnJvbSBvbmUgVVJMIHRvIGFub3RoZXIuXG4gKlxuICogVGhpcyBmdW5jdGlvbiBzdXBwb3J0czpcbiAqXG4gKiAxLiBDb3B5aW5nIHdpdGhpbiBhIHN0b3JhZ2UgbWVkaXVtLCBlLmcuLFxuICogICAgYHRmLmlvLmNvcHlNb2RlbCgnbG9jYWxzdG9yYWdlOi8vbW9kZWwtMScsICdsb2NhbHN0b3JhZ2U6Ly9tb2RlbC0yJylgXG4gKiAyLiBDb3B5aW5nIGJldHdlZW4gdHdvIHN0b3JhZ2UgbWVkaXVtcywgZS5nLixcbiAqICAgIGB0Zi5pby5jb3B5TW9kZWwoJ2xvY2Fsc3RvcmFnZTovL21vZGVsLTEnLCAnaW5kZXhlZGRiOi8vbW9kZWwtMScpYFxuICpcbiAqIGBgYGpzXG4gKiAvLyBGaXJzdCBjcmVhdGUgYW5kIHNhdmUgYSBtb2RlbC5cbiAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICogbW9kZWwuYWRkKHRmLmxheWVycy5kZW5zZShcbiAqICAgICB7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMF0sIGFjdGl2YXRpb246ICdzaWdtb2lkJ30pKTtcbiAqIGF3YWl0IG1vZGVsLnNhdmUoJ2xvY2Fsc3RvcmFnZTovL2RlbW8vbWFuYWdlbWVudC9tb2RlbDEnKTtcbiAqXG4gKiAvLyBUaGVuIGxpc3QgZXhpc3RpbmcgbW9kZWxzLlxuICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkoYXdhaXQgdGYuaW8ubGlzdE1vZGVscygpKSk7XG4gKlxuICogLy8gQ29weSB0aGUgbW9kZWwsIGZyb20gTG9jYWwgU3RvcmFnZSB0byBJbmRleGVkREIuXG4gKiBhd2FpdCB0Zi5pby5jb3B5TW9kZWwoXG4gKiAgICAgJ2xvY2Fsc3RvcmFnZTovL2RlbW8vbWFuYWdlbWVudC9tb2RlbDEnLFxuICogICAgICdpbmRleGVkZGI6Ly9kZW1vL21hbmFnZW1lbnQvbW9kZWwxJyk7XG4gKlxuICogLy8gTGlzdCBtb2RlbHMgYWdhaW4uXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShhd2FpdCB0Zi5pby5saXN0TW9kZWxzKCkpKTtcbiAqXG4gKiAvLyBSZW1vdmUgYm90aCBtb2RlbHMuXG4gKiBhd2FpdCB0Zi5pby5yZW1vdmVNb2RlbCgnbG9jYWxzdG9yYWdlOi8vZGVtby9tYW5hZ2VtZW50L21vZGVsMScpO1xuICogYXdhaXQgdGYuaW8ucmVtb3ZlTW9kZWwoJ2luZGV4ZWRkYjovL2RlbW8vbWFuYWdlbWVudC9tb2RlbDEnKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBzb3VyY2VVUkwgU291cmNlIFVSTCBvZiBjb3B5aW5nLlxuICogQHBhcmFtIGRlc3RVUkwgRGVzdGluYXRpb24gVVJMIG9mIGNvcHlpbmcuXG4gKiBAcmV0dXJucyBNb2RlbEFydGlmYWN0c0luZm8gb2YgdGhlIGNvcGllZCBtb2RlbCAoaWYgYW5kIG9ubHkgaWYgY29weWluZ1xuICogICBpcyBzdWNjZXNzZnVsKS5cbiAqIEB0aHJvd3MgRXJyb3IgaWYgY29weWluZyBmYWlscywgZS5nLiwgaWYgbm8gbW9kZWwgZXhpc3RzIGF0IGBzb3VyY2VVUkxgLCBvclxuICogICBpZiBgb2xkUGF0aGAgYW5kIGBuZXdQYXRoYCBhcmUgaWRlbnRpY2FsLlxuICpcbiAqIEBkb2Mge1xuICogICBoZWFkaW5nOiAnTW9kZWxzJyxcbiAqICAgc3ViaGVhZGluZzogJ01hbmFnZW1lbnQnLFxuICogICBuYW1lc3BhY2U6ICdpbycsXG4gKiAgIGlnbm9yZUNJOiB0cnVlXG4gKiB9XG4gKi9cbmFzeW5jIGZ1bmN0aW9uIGNvcHlNb2RlbChcbiAgICBzb3VyY2VVUkw6IHN0cmluZywgZGVzdFVSTDogc3RyaW5nKTogUHJvbWlzZTxNb2RlbEFydGlmYWN0c0luZm8+IHtcbiAgY29uc3QgZGVsZXRlU291cmNlID0gZmFsc2U7XG4gIHJldHVybiBjbG9uZU1vZGVsSW50ZXJuYWwoc291cmNlVVJMLCBkZXN0VVJMLCBkZWxldGVTb3VyY2UpO1xufVxuXG4vKipcbiAqIE1vdmUgYSBtb2RlbCBmcm9tIG9uZSBVUkwgdG8gYW5vdGhlci5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIHN1cHBvcnRzOlxuICpcbiAqIDEuIE1vdmluZyB3aXRoaW4gYSBzdG9yYWdlIG1lZGl1bSwgZS5nLixcbiAqICAgIGB0Zi5pby5tb3ZlTW9kZWwoJ2xvY2Fsc3RvcmFnZTovL21vZGVsLTEnLCAnbG9jYWxzdG9yYWdlOi8vbW9kZWwtMicpYFxuICogMi4gTW92aW5nIGJldHdlZW4gdHdvIHN0b3JhZ2UgbWVkaXVtcywgZS5nLixcbiAqICAgIGB0Zi5pby5tb3ZlTW9kZWwoJ2xvY2Fsc3RvcmFnZTovL21vZGVsLTEnLCAnaW5kZXhlZGRiOi8vbW9kZWwtMScpYFxuICpcbiAqIGBgYGpzXG4gKiAvLyBGaXJzdCBjcmVhdGUgYW5kIHNhdmUgYSBtb2RlbC5cbiAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICogbW9kZWwuYWRkKHRmLmxheWVycy5kZW5zZShcbiAqICAgICB7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMF0sIGFjdGl2YXRpb246ICdzaWdtb2lkJ30pKTtcbiAqIGF3YWl0IG1vZGVsLnNhdmUoJ2xvY2Fsc3RvcmFnZTovL2RlbW8vbWFuYWdlbWVudC9tb2RlbDEnKTtcbiAqXG4gKiAvLyBUaGVuIGxpc3QgZXhpc3RpbmcgbW9kZWxzLlxuICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkoYXdhaXQgdGYuaW8ubGlzdE1vZGVscygpKSk7XG4gKlxuICogLy8gTW92ZSB0aGUgbW9kZWwsIGZyb20gTG9jYWwgU3RvcmFnZSB0byBJbmRleGVkREIuXG4gKiBhd2FpdCB0Zi5pby5tb3ZlTW9kZWwoXG4gKiAgICAgJ2xvY2Fsc3RvcmFnZTovL2RlbW8vbWFuYWdlbWVudC9tb2RlbDEnLFxuICogICAgICdpbmRleGVkZGI6Ly9kZW1vL21hbmFnZW1lbnQvbW9kZWwxJyk7XG4gKlxuICogLy8gTGlzdCBtb2RlbHMgYWdhaW4uXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShhd2FpdCB0Zi5pby5saXN0TW9kZWxzKCkpKTtcbiAqXG4gKiAvLyBSZW1vdmUgdGhlIG1vdmVkIG1vZGVsLlxuICogYXdhaXQgdGYuaW8ucmVtb3ZlTW9kZWwoJ2luZGV4ZWRkYjovL2RlbW8vbWFuYWdlbWVudC9tb2RlbDEnKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBzb3VyY2VVUkwgU291cmNlIFVSTCBvZiBtb3ZpbmcuXG4gKiBAcGFyYW0gZGVzdFVSTCBEZXN0aW5hdGlvbiBVUkwgb2YgbW92aW5nLlxuICogQHJldHVybnMgTW9kZWxBcnRpZmFjdHNJbmZvIG9mIHRoZSBjb3BpZWQgbW9kZWwgKGlmIGFuZCBvbmx5IGlmIGNvcHlpbmdcbiAqICAgaXMgc3VjY2Vzc2Z1bCkuXG4gKiBAdGhyb3dzIEVycm9yIGlmIG1vdmluZyBmYWlscywgZS5nLiwgaWYgbm8gbW9kZWwgZXhpc3RzIGF0IGBzb3VyY2VVUkxgLCBvclxuICogICBpZiBgb2xkUGF0aGAgYW5kIGBuZXdQYXRoYCBhcmUgaWRlbnRpY2FsLlxuICpcbiAqIEBkb2Mge1xuICogICBoZWFkaW5nOiAnTW9kZWxzJyxcbiAqICAgc3ViaGVhZGluZzogJ01hbmFnZW1lbnQnLFxuICogICBuYW1lc3BhY2U6ICdpbycsXG4gKiAgIGlnbm9yZUNJOiB0cnVlXG4gKiB9XG4gKi9cbmFzeW5jIGZ1bmN0aW9uIG1vdmVNb2RlbChcbiAgICBzb3VyY2VVUkw6IHN0cmluZywgZGVzdFVSTDogc3RyaW5nKTogUHJvbWlzZTxNb2RlbEFydGlmYWN0c0luZm8+IHtcbiAgY29uc3QgZGVsZXRlU291cmNlID0gdHJ1ZTtcbiAgcmV0dXJuIGNsb25lTW9kZWxJbnRlcm5hbChzb3VyY2VVUkwsIGRlc3RVUkwsIGRlbGV0ZVNvdXJjZSk7XG59XG5cbmV4cG9ydCB7bW92ZU1vZGVsLCBjb3B5TW9kZWwsIHJlbW92ZU1vZGVsLCBsaXN0TW9kZWxzfTtcbiJdfQ==
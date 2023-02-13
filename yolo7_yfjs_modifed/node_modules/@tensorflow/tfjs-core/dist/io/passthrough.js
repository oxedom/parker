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
class PassthroughLoader {
    constructor(modelArtifacts) {
        this.modelArtifacts = modelArtifacts;
    }
    async load() {
        return this.modelArtifacts;
    }
}
class PassthroughSaver {
    constructor(saveHandler) {
        this.saveHandler = saveHandler;
    }
    async save(modelArtifacts) {
        return this.saveHandler(modelArtifacts);
    }
}
/**
 * Creates an IOHandler that loads model artifacts from memory.
 *
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * const model = await tf.loadLayersModel(tf.io.fromMemory(
 *     modelTopology, weightSpecs, weightData));
 * ```
 *
 * @param modelArtifacts a object containing model topology (i.e., parsed from
 *   the JSON format).
 * @param weightSpecs An array of `WeightsManifestEntry` objects describing the
 *   names, shapes, types, and quantization of the weight data.
 * @param weightData A single `ArrayBuffer` containing the weight data,
 *   concatenated in the order described by the weightSpecs.
 * @param trainingConfig Model training configuration. Optional.
 *
 * @returns A passthrough `IOHandler` that simply loads the provided data.
 */
export function fromMemory(modelArtifacts, weightSpecs, weightData, trainingConfig) {
    if (arguments.length === 1) {
        const isModelArtifacts = modelArtifacts.modelTopology != null ||
            modelArtifacts.weightSpecs != null;
        if (isModelArtifacts) {
            return new PassthroughLoader(modelArtifacts);
        }
        else {
            // Legacy support: with only modelTopology.
            // TODO(cais): Remove this deprecated API.
            console.warn('Please call tf.io.fromMemory() with only one argument. ' +
                'The argument should be of type ModelArtifacts. ' +
                'The multi-argument signature of tf.io.fromMemory() has been ' +
                'deprecated and will be removed in a future release.');
            return new PassthroughLoader({ modelTopology: modelArtifacts });
        }
    }
    else {
        // Legacy support.
        // TODO(cais): Remove this deprecated API.
        console.warn('Please call tf.io.fromMemory() with only one argument. ' +
            'The argument should be of type ModelArtifacts. ' +
            'The multi-argument signature of tf.io.fromMemory() has been ' +
            'deprecated and will be removed in a future release.');
        return new PassthroughLoader({
            modelTopology: modelArtifacts,
            weightSpecs,
            weightData,
            trainingConfig
        });
    }
}
/**
 * Creates an IOHandler that passes saved model artifacts to a callback.
 *
 * ```js
 * function handleSave(artifacts) {
 *   // ... do something with the artifacts ...
 *   return {modelArtifactsInfo: {...}, ...};
 * }
 *
 * const saveResult = model.save(tf.io.withSaveHandler(handleSave));
 * ```
 *
 * @param saveHandler A function that accepts a `ModelArtifacts` and returns a
 *     `SaveResult`.
 */
export function withSaveHandler(saveHandler) {
    return new PassthroughSaver(saveHandler);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGFzc3Rocm91Z2guanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2lvL3Bhc3N0aHJvdWdoLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQVFILE1BQU0saUJBQWlCO0lBQ3JCLFlBQTZCLGNBQStCO1FBQS9CLG1CQUFjLEdBQWQsY0FBYyxDQUFpQjtJQUFHLENBQUM7SUFFaEUsS0FBSyxDQUFDLElBQUk7UUFDUixPQUFPLElBQUksQ0FBQyxjQUFjLENBQUM7SUFDN0IsQ0FBQztDQUNGO0FBRUQsTUFBTSxnQkFBZ0I7SUFDcEIsWUFDcUIsV0FDcUM7UUFEckMsZ0JBQVcsR0FBWCxXQUFXLENBQzBCO0lBQUcsQ0FBQztJQUU5RCxLQUFLLENBQUMsSUFBSSxDQUFDLGNBQThCO1FBQ3ZDLE9BQU8sSUFBSSxDQUFDLFdBQVcsQ0FBQyxjQUFjLENBQUMsQ0FBQztJQUMxQyxDQUFDO0NBQ0Y7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FvQkc7QUFDSCxNQUFNLFVBQVUsVUFBVSxDQUN0QixjQUFpQyxFQUFFLFdBQW9DLEVBQ3ZFLFVBQXdCLEVBQUUsY0FBK0I7SUFDM0QsSUFBSSxTQUFTLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtRQUMxQixNQUFNLGdCQUFnQixHQUNqQixjQUFpQyxDQUFDLGFBQWEsSUFBSSxJQUFJO1lBQ3ZELGNBQWlDLENBQUMsV0FBVyxJQUFJLElBQUksQ0FBQztRQUMzRCxJQUFJLGdCQUFnQixFQUFFO1lBQ3BCLE9BQU8sSUFBSSxpQkFBaUIsQ0FBQyxjQUFnQyxDQUFDLENBQUM7U0FDaEU7YUFBTTtZQUNMLDJDQUEyQztZQUMzQywwQ0FBMEM7WUFDMUMsT0FBTyxDQUFDLElBQUksQ0FDUix5REFBeUQ7Z0JBQ3pELGlEQUFpRDtnQkFDakQsOERBQThEO2dCQUM5RCxxREFBcUQsQ0FBQyxDQUFDO1lBQzNELE9BQU8sSUFBSSxpQkFBaUIsQ0FBQyxFQUFDLGFBQWEsRUFBRSxjQUFvQixFQUFDLENBQUMsQ0FBQztTQUNyRTtLQUNGO1NBQU07UUFDTCxrQkFBa0I7UUFDbEIsMENBQTBDO1FBQzFDLE9BQU8sQ0FBQyxJQUFJLENBQ1IseURBQXlEO1lBQ3pELGlEQUFpRDtZQUNqRCw4REFBOEQ7WUFDOUQscURBQXFELENBQUMsQ0FBQztRQUMzRCxPQUFPLElBQUksaUJBQWlCLENBQUM7WUFDM0IsYUFBYSxFQUFFLGNBQW9CO1lBQ25DLFdBQVc7WUFDWCxVQUFVO1lBQ1YsY0FBYztTQUNmLENBQUMsQ0FBQztLQUNKO0FBQ0gsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7OztHQWNHO0FBQ0gsTUFBTSxVQUFVLGVBQWUsQ0FDM0IsV0FDdUI7SUFDekIsT0FBTyxJQUFJLGdCQUFnQixDQUFDLFdBQVcsQ0FBQyxDQUFDO0FBQzNDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogSU9IYW5kbGVycyB0aGF0IHBhc3MgdGhyb3VnaCB0aGUgaW4tbWVtb3J5IE1vZGVsQXJ0aWZhY3RzIGZvcm1hdC5cbiAqL1xuXG5pbXBvcnQge0lPSGFuZGxlciwgTW9kZWxBcnRpZmFjdHMsIFNhdmVSZXN1bHQsIFRyYWluaW5nQ29uZmlnLCBXZWlnaHRzTWFuaWZlc3RFbnRyeX0gZnJvbSAnLi90eXBlcyc7XG5cbmNsYXNzIFBhc3N0aHJvdWdoTG9hZGVyIGltcGxlbWVudHMgSU9IYW5kbGVyIHtcbiAgY29uc3RydWN0b3IocHJpdmF0ZSByZWFkb25seSBtb2RlbEFydGlmYWN0cz86IE1vZGVsQXJ0aWZhY3RzKSB7fVxuXG4gIGFzeW5jIGxvYWQoKTogUHJvbWlzZTxNb2RlbEFydGlmYWN0cz4ge1xuICAgIHJldHVybiB0aGlzLm1vZGVsQXJ0aWZhY3RzO1xuICB9XG59XG5cbmNsYXNzIFBhc3N0aHJvdWdoU2F2ZXIgaW1wbGVtZW50cyBJT0hhbmRsZXIge1xuICBjb25zdHJ1Y3RvcihcbiAgICAgIHByaXZhdGUgcmVhZG9ubHkgc2F2ZUhhbmRsZXI6XG4gICAgICAgICAgKGFydGlmYWN0czogTW9kZWxBcnRpZmFjdHMpID0+IFByb21pc2U8U2F2ZVJlc3VsdD4pIHt9XG5cbiAgYXN5bmMgc2F2ZShtb2RlbEFydGlmYWN0czogTW9kZWxBcnRpZmFjdHMpIHtcbiAgICByZXR1cm4gdGhpcy5zYXZlSGFuZGxlcihtb2RlbEFydGlmYWN0cyk7XG4gIH1cbn1cblxuLyoqXG4gKiBDcmVhdGVzIGFuIElPSGFuZGxlciB0aGF0IGxvYWRzIG1vZGVsIGFydGlmYWN0cyBmcm9tIG1lbW9yeS5cbiAqXG4gKiBXaGVuIHVzZWQgaW4gY29uanVuY3Rpb24gd2l0aCBgdGYubG9hZExheWVyc01vZGVsYCwgYW4gaW5zdGFuY2Ugb2ZcbiAqIGB0Zi5MYXllcnNNb2RlbGAgKEtlcmFzLXN0eWxlKSBjYW4gYmUgY29uc3RydWN0ZWQgZnJvbSB0aGUgbG9hZGVkIGFydGlmYWN0cy5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgbW9kZWwgPSBhd2FpdCB0Zi5sb2FkTGF5ZXJzTW9kZWwodGYuaW8uZnJvbU1lbW9yeShcbiAqICAgICBtb2RlbFRvcG9sb2d5LCB3ZWlnaHRTcGVjcywgd2VpZ2h0RGF0YSkpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIG1vZGVsQXJ0aWZhY3RzIGEgb2JqZWN0IGNvbnRhaW5pbmcgbW9kZWwgdG9wb2xvZ3kgKGkuZS4sIHBhcnNlZCBmcm9tXG4gKiAgIHRoZSBKU09OIGZvcm1hdCkuXG4gKiBAcGFyYW0gd2VpZ2h0U3BlY3MgQW4gYXJyYXkgb2YgYFdlaWdodHNNYW5pZmVzdEVudHJ5YCBvYmplY3RzIGRlc2NyaWJpbmcgdGhlXG4gKiAgIG5hbWVzLCBzaGFwZXMsIHR5cGVzLCBhbmQgcXVhbnRpemF0aW9uIG9mIHRoZSB3ZWlnaHQgZGF0YS5cbiAqIEBwYXJhbSB3ZWlnaHREYXRhIEEgc2luZ2xlIGBBcnJheUJ1ZmZlcmAgY29udGFpbmluZyB0aGUgd2VpZ2h0IGRhdGEsXG4gKiAgIGNvbmNhdGVuYXRlZCBpbiB0aGUgb3JkZXIgZGVzY3JpYmVkIGJ5IHRoZSB3ZWlnaHRTcGVjcy5cbiAqIEBwYXJhbSB0cmFpbmluZ0NvbmZpZyBNb2RlbCB0cmFpbmluZyBjb25maWd1cmF0aW9uLiBPcHRpb25hbC5cbiAqXG4gKiBAcmV0dXJucyBBIHBhc3N0aHJvdWdoIGBJT0hhbmRsZXJgIHRoYXQgc2ltcGx5IGxvYWRzIHRoZSBwcm92aWRlZCBkYXRhLlxuICovXG5leHBvcnQgZnVuY3Rpb24gZnJvbU1lbW9yeShcbiAgICBtb2RlbEFydGlmYWN0czoge318TW9kZWxBcnRpZmFjdHMsIHdlaWdodFNwZWNzPzogV2VpZ2h0c01hbmlmZXN0RW50cnlbXSxcbiAgICB3ZWlnaHREYXRhPzogQXJyYXlCdWZmZXIsIHRyYWluaW5nQ29uZmlnPzogVHJhaW5pbmdDb25maWcpOiBJT0hhbmRsZXIge1xuICBpZiAoYXJndW1lbnRzLmxlbmd0aCA9PT0gMSkge1xuICAgIGNvbnN0IGlzTW9kZWxBcnRpZmFjdHMgPVxuICAgICAgICAobW9kZWxBcnRpZmFjdHMgYXMgTW9kZWxBcnRpZmFjdHMpLm1vZGVsVG9wb2xvZ3kgIT0gbnVsbCB8fFxuICAgICAgICAobW9kZWxBcnRpZmFjdHMgYXMgTW9kZWxBcnRpZmFjdHMpLndlaWdodFNwZWNzICE9IG51bGw7XG4gICAgaWYgKGlzTW9kZWxBcnRpZmFjdHMpIHtcbiAgICAgIHJldHVybiBuZXcgUGFzc3Rocm91Z2hMb2FkZXIobW9kZWxBcnRpZmFjdHMgYXMgTW9kZWxBcnRpZmFjdHMpO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBMZWdhY3kgc3VwcG9ydDogd2l0aCBvbmx5IG1vZGVsVG9wb2xvZ3kuXG4gICAgICAvLyBUT0RPKGNhaXMpOiBSZW1vdmUgdGhpcyBkZXByZWNhdGVkIEFQSS5cbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnUGxlYXNlIGNhbGwgdGYuaW8uZnJvbU1lbW9yeSgpIHdpdGggb25seSBvbmUgYXJndW1lbnQuICcgK1xuICAgICAgICAgICdUaGUgYXJndW1lbnQgc2hvdWxkIGJlIG9mIHR5cGUgTW9kZWxBcnRpZmFjdHMuICcgK1xuICAgICAgICAgICdUaGUgbXVsdGktYXJndW1lbnQgc2lnbmF0dXJlIG9mIHRmLmlvLmZyb21NZW1vcnkoKSBoYXMgYmVlbiAnICtcbiAgICAgICAgICAnZGVwcmVjYXRlZCBhbmQgd2lsbCBiZSByZW1vdmVkIGluIGEgZnV0dXJlIHJlbGVhc2UuJyk7XG4gICAgICByZXR1cm4gbmV3IFBhc3N0aHJvdWdoTG9hZGVyKHttb2RlbFRvcG9sb2d5OiBtb2RlbEFydGlmYWN0cyBhcyB7fX0pO1xuICAgIH1cbiAgfSBlbHNlIHtcbiAgICAvLyBMZWdhY3kgc3VwcG9ydC5cbiAgICAvLyBUT0RPKGNhaXMpOiBSZW1vdmUgdGhpcyBkZXByZWNhdGVkIEFQSS5cbiAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICdQbGVhc2UgY2FsbCB0Zi5pby5mcm9tTWVtb3J5KCkgd2l0aCBvbmx5IG9uZSBhcmd1bWVudC4gJyArXG4gICAgICAgICdUaGUgYXJndW1lbnQgc2hvdWxkIGJlIG9mIHR5cGUgTW9kZWxBcnRpZmFjdHMuICcgK1xuICAgICAgICAnVGhlIG11bHRpLWFyZ3VtZW50IHNpZ25hdHVyZSBvZiB0Zi5pby5mcm9tTWVtb3J5KCkgaGFzIGJlZW4gJyArXG4gICAgICAgICdkZXByZWNhdGVkIGFuZCB3aWxsIGJlIHJlbW92ZWQgaW4gYSBmdXR1cmUgcmVsZWFzZS4nKTtcbiAgICByZXR1cm4gbmV3IFBhc3N0aHJvdWdoTG9hZGVyKHtcbiAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsQXJ0aWZhY3RzIGFzIHt9LFxuICAgICAgd2VpZ2h0U3BlY3MsXG4gICAgICB3ZWlnaHREYXRhLFxuICAgICAgdHJhaW5pbmdDb25maWdcbiAgICB9KTtcbiAgfVxufVxuXG4vKipcbiAqIENyZWF0ZXMgYW4gSU9IYW5kbGVyIHRoYXQgcGFzc2VzIHNhdmVkIG1vZGVsIGFydGlmYWN0cyB0byBhIGNhbGxiYWNrLlxuICpcbiAqIGBgYGpzXG4gKiBmdW5jdGlvbiBoYW5kbGVTYXZlKGFydGlmYWN0cykge1xuICogICAvLyAuLi4gZG8gc29tZXRoaW5nIHdpdGggdGhlIGFydGlmYWN0cyAuLi5cbiAqICAgcmV0dXJuIHttb2RlbEFydGlmYWN0c0luZm86IHsuLi59LCAuLi59O1xuICogfVxuICpcbiAqIGNvbnN0IHNhdmVSZXN1bHQgPSBtb2RlbC5zYXZlKHRmLmlvLndpdGhTYXZlSGFuZGxlcihoYW5kbGVTYXZlKSk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gc2F2ZUhhbmRsZXIgQSBmdW5jdGlvbiB0aGF0IGFjY2VwdHMgYSBgTW9kZWxBcnRpZmFjdHNgIGFuZCByZXR1cm5zIGFcbiAqICAgICBgU2F2ZVJlc3VsdGAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiB3aXRoU2F2ZUhhbmRsZXIoXG4gICAgc2F2ZUhhbmRsZXI6IChhcnRpZmFjdHM6IE1vZGVsQXJ0aWZhY3RzKSA9PlxuICAgICAgICBQcm9taXNlPFNhdmVSZXN1bHQ+KTogSU9IYW5kbGVyIHtcbiAgcmV0dXJuIG5ldyBQYXNzdGhyb3VnaFNhdmVyKHNhdmVIYW5kbGVyKTtcbn1cbiJdfQ==
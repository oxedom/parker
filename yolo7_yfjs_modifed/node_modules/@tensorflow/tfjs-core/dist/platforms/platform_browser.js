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
import '../flags';
import { env } from '../environment';
import { BrowserIndexedDB, BrowserIndexedDBManager } from '../io/indexed_db';
import { BrowserLocalStorage, BrowserLocalStorageManager } from '../io/local_storage';
import { ModelStoreManagerRegistry } from '../io/model_management';
export class PlatformBrowser {
    fetch(path, init) {
        return fetch(path, init);
    }
    now() {
        return performance.now();
    }
    encode(text, encoding) {
        if (encoding !== 'utf-8' && encoding !== 'utf8') {
            throw new Error(`Browser's encoder only supports utf-8, but got ${encoding}`);
        }
        if (this.textEncoder == null) {
            this.textEncoder = new TextEncoder();
        }
        return this.textEncoder.encode(text);
    }
    decode(bytes, encoding) {
        return new TextDecoder(encoding).decode(bytes);
    }
}
if (env().get('IS_BROWSER')) {
    env().setPlatform('browser', new PlatformBrowser());
    // Register LocalStorage IOHandler
    try {
        ModelStoreManagerRegistry.registerManager(BrowserLocalStorage.URL_SCHEME, new BrowserLocalStorageManager());
    }
    catch (err) {
    }
    // Register IndexedDB IOHandler
    try {
        ModelStoreManagerRegistry.registerManager(BrowserIndexedDB.URL_SCHEME, new BrowserIndexedDBManager());
    }
    catch (err) {
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGxhdGZvcm1fYnJvd3Nlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvcGxhdGZvcm1zL3BsYXRmb3JtX2Jyb3dzZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxVQUFVLENBQUM7QUFFbEIsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQ25DLE9BQU8sRUFBQyxnQkFBZ0IsRUFBRSx1QkFBdUIsRUFBQyxNQUFNLGtCQUFrQixDQUFDO0FBQzNFLE9BQU8sRUFBQyxtQkFBbUIsRUFBRSwwQkFBMEIsRUFBQyxNQUFNLHFCQUFxQixDQUFDO0FBQ3BGLE9BQU8sRUFBQyx5QkFBeUIsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBSWpFLE1BQU0sT0FBTyxlQUFlO0lBSzFCLEtBQUssQ0FBQyxJQUFZLEVBQUUsSUFBa0I7UUFDcEMsT0FBTyxLQUFLLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzNCLENBQUM7SUFFRCxHQUFHO1FBQ0QsT0FBTyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDM0IsQ0FBQztJQUVELE1BQU0sQ0FBQyxJQUFZLEVBQUUsUUFBZ0I7UUFDbkMsSUFBSSxRQUFRLEtBQUssT0FBTyxJQUFJLFFBQVEsS0FBSyxNQUFNLEVBQUU7WUFDL0MsTUFBTSxJQUFJLEtBQUssQ0FDWCxrREFBa0QsUUFBUSxFQUFFLENBQUMsQ0FBQztTQUNuRTtRQUNELElBQUksSUFBSSxDQUFDLFdBQVcsSUFBSSxJQUFJLEVBQUU7WUFDNUIsSUFBSSxDQUFDLFdBQVcsR0FBRyxJQUFJLFdBQVcsRUFBRSxDQUFDO1NBQ3RDO1FBQ0QsT0FBTyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUN2QyxDQUFDO0lBQ0QsTUFBTSxDQUFDLEtBQWlCLEVBQUUsUUFBZ0I7UUFDeEMsT0FBTyxJQUFJLFdBQVcsQ0FBQyxRQUFRLENBQUMsQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDakQsQ0FBQztDQUNGO0FBRUQsSUFBSSxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsWUFBWSxDQUFDLEVBQUU7SUFDM0IsR0FBRyxFQUFFLENBQUMsV0FBVyxDQUFDLFNBQVMsRUFBRSxJQUFJLGVBQWUsRUFBRSxDQUFDLENBQUM7SUFFcEQsa0NBQWtDO0lBQ2xDLElBQUk7UUFDRix5QkFBeUIsQ0FBQyxlQUFlLENBQ3JDLG1CQUFtQixDQUFDLFVBQVUsRUFBRSxJQUFJLDBCQUEwQixFQUFFLENBQUMsQ0FBQztLQUN2RTtJQUFDLE9BQU8sR0FBRyxFQUFFO0tBQ2I7SUFFRCwrQkFBK0I7SUFDL0IsSUFBSTtRQUNGLHlCQUF5QixDQUFDLGVBQWUsQ0FDckMsZ0JBQWdCLENBQUMsVUFBVSxFQUFFLElBQUksdUJBQXVCLEVBQUUsQ0FBQyxDQUFDO0tBQ2pFO0lBQUMsT0FBTyxHQUFHLEVBQUU7S0FDYjtDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgJy4uL2ZsYWdzJztcblxuaW1wb3J0IHtlbnZ9IGZyb20gJy4uL2Vudmlyb25tZW50JztcbmltcG9ydCB7QnJvd3NlckluZGV4ZWREQiwgQnJvd3NlckluZGV4ZWREQk1hbmFnZXJ9IGZyb20gJy4uL2lvL2luZGV4ZWRfZGInO1xuaW1wb3J0IHtCcm93c2VyTG9jYWxTdG9yYWdlLCBCcm93c2VyTG9jYWxTdG9yYWdlTWFuYWdlcn0gZnJvbSAnLi4vaW8vbG9jYWxfc3RvcmFnZSc7XG5pbXBvcnQge01vZGVsU3RvcmVNYW5hZ2VyUmVnaXN0cnl9IGZyb20gJy4uL2lvL21vZGVsX21hbmFnZW1lbnQnO1xuXG5pbXBvcnQge1BsYXRmb3JtfSBmcm9tICcuL3BsYXRmb3JtJztcblxuZXhwb3J0IGNsYXNzIFBsYXRmb3JtQnJvd3NlciBpbXBsZW1lbnRzIFBsYXRmb3JtIHtcbiAgLy8gQWNjb3JkaW5nIHRvIHRoZSBzcGVjLCB0aGUgYnVpbHQtaW4gZW5jb2RlciBjYW4gZG8gb25seSBVVEYtOCBlbmNvZGluZy5cbiAgLy8gaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL1RleHRFbmNvZGVyL1RleHRFbmNvZGVyXG4gIHByaXZhdGUgdGV4dEVuY29kZXI6IFRleHRFbmNvZGVyO1xuXG4gIGZldGNoKHBhdGg6IHN0cmluZywgaW5pdD86IFJlcXVlc3RJbml0KTogUHJvbWlzZTxSZXNwb25zZT4ge1xuICAgIHJldHVybiBmZXRjaChwYXRoLCBpbml0KTtcbiAgfVxuXG4gIG5vdygpOiBudW1iZXIge1xuICAgIHJldHVybiBwZXJmb3JtYW5jZS5ub3coKTtcbiAgfVxuXG4gIGVuY29kZSh0ZXh0OiBzdHJpbmcsIGVuY29kaW5nOiBzdHJpbmcpOiBVaW50OEFycmF5IHtcbiAgICBpZiAoZW5jb2RpbmcgIT09ICd1dGYtOCcgJiYgZW5jb2RpbmcgIT09ICd1dGY4Jykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBCcm93c2VyJ3MgZW5jb2RlciBvbmx5IHN1cHBvcnRzIHV0Zi04LCBidXQgZ290ICR7ZW5jb2Rpbmd9YCk7XG4gICAgfVxuICAgIGlmICh0aGlzLnRleHRFbmNvZGVyID09IG51bGwpIHtcbiAgICAgIHRoaXMudGV4dEVuY29kZXIgPSBuZXcgVGV4dEVuY29kZXIoKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMudGV4dEVuY29kZXIuZW5jb2RlKHRleHQpO1xuICB9XG4gIGRlY29kZShieXRlczogVWludDhBcnJheSwgZW5jb2Rpbmc6IHN0cmluZyk6IHN0cmluZyB7XG4gICAgcmV0dXJuIG5ldyBUZXh0RGVjb2RlcihlbmNvZGluZykuZGVjb2RlKGJ5dGVzKTtcbiAgfVxufVxuXG5pZiAoZW52KCkuZ2V0KCdJU19CUk9XU0VSJykpIHtcbiAgZW52KCkuc2V0UGxhdGZvcm0oJ2Jyb3dzZXInLCBuZXcgUGxhdGZvcm1Ccm93c2VyKCkpO1xuXG4gIC8vIFJlZ2lzdGVyIExvY2FsU3RvcmFnZSBJT0hhbmRsZXJcbiAgdHJ5IHtcbiAgICBNb2RlbFN0b3JlTWFuYWdlclJlZ2lzdHJ5LnJlZ2lzdGVyTWFuYWdlcihcbiAgICAgICAgQnJvd3NlckxvY2FsU3RvcmFnZS5VUkxfU0NIRU1FLCBuZXcgQnJvd3NlckxvY2FsU3RvcmFnZU1hbmFnZXIoKSk7XG4gIH0gY2F0Y2ggKGVycikge1xuICB9XG5cbiAgLy8gUmVnaXN0ZXIgSW5kZXhlZERCIElPSGFuZGxlclxuICB0cnkge1xuICAgIE1vZGVsU3RvcmVNYW5hZ2VyUmVnaXN0cnkucmVnaXN0ZXJNYW5hZ2VyKFxuICAgICAgICBCcm93c2VySW5kZXhlZERCLlVSTF9TQ0hFTUUsIG5ldyBCcm93c2VySW5kZXhlZERCTWFuYWdlcigpKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gIH1cbn1cbiJdfQ==
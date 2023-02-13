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
import { ENGINE } from '../../engine';
import { CropAndResize } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
/**
 * Extracts crops from the input image tensor and resizes them using bilinear
 * sampling or nearest neighbor sampling (possibly with aspect ratio change)
 * to a common output size specified by cropSize.
 *
 * @param image 4d tensor of shape `[batch,imageHeight,imageWidth, depth]`,
 *     where imageHeight and imageWidth must be positive, specifying the
 *     batch of images from which to take crops
 * @param boxes 2d float32 tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the normalized
 *     coordinates of the box in the boxInd[i]'th image in the batch
 * @param boxInd 1d int32 tensor of shape `[numBoxes]` with values in range
 *     `[0, batch)` that specifies the image that the `i`-th box refers to.
 * @param cropSize 1d int32 tensor of 2 elements `[cropHeigh, cropWidth]`
 *     specifying the size to which all crops are resized to.
 * @param method Optional string from `'bilinear' | 'nearest'`,
 *     defaults to bilinear, which specifies the sampling method for resizing
 * @param extrapolationValue A threshold for deciding when to remove boxes based
 *     on score. Defaults to 0.
 * @return A 4D tensor of the shape `[numBoxes,cropHeight,cropWidth,depth]`
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function cropAndResize_(image, boxes, boxInd, cropSize, method = 'bilinear', extrapolationValue = 0) {
    const $image = convertToTensor(image, 'image', 'cropAndResize');
    const $boxes = convertToTensor(boxes, 'boxes', 'cropAndResize', 'float32');
    const $boxInd = convertToTensor(boxInd, 'boxInd', 'cropAndResize', 'int32');
    const numBoxes = $boxes.shape[0];
    util.assert($image.rank === 4, () => 'Error in cropAndResize: image must be rank 4,' +
        `but got rank ${$image.rank}.`);
    util.assert($boxes.rank === 2 && $boxes.shape[1] === 4, () => `Error in cropAndResize: boxes must be have size [${numBoxes},4] ` +
        `but had shape ${$boxes.shape}.`);
    util.assert($boxInd.rank === 1 && $boxInd.shape[0] === numBoxes, () => `Error in cropAndResize: boxInd must be have size [${numBoxes}] ` +
        `but had shape ${$boxes.shape}.`);
    util.assert(cropSize.length === 2, () => `Error in cropAndResize: cropSize must be of length 2, but got ` +
        `length ${cropSize.length}.`);
    util.assert(cropSize[0] >= 1 && cropSize[1] >= 1, () => `cropSize must be atleast [1,1], but was ${cropSize}`);
    util.assert(method === 'bilinear' || method === 'nearest', () => `method must be bilinear or nearest, but was ${method}`);
    const inputs = { image: $image, boxes: $boxes, boxInd: $boxInd };
    const attrs = { method, extrapolationValue, cropSize };
    const res = ENGINE.runKernel(CropAndResize, inputs, attrs);
    return res;
}
export const cropAndResize = op({ cropAndResize_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY3JvcF9hbmRfcmVzaXplLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvaW1hZ2UvY3JvcF9hbmRfcmVzaXplLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLGFBQWEsRUFBMEMsTUFBTSxvQkFBb0IsQ0FBQztBQUkxRixPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFFbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXNCRztBQUNILFNBQVMsY0FBYyxDQUNuQixLQUEwQixFQUMxQixLQUEwQixFQUMxQixNQUEyQixFQUMzQixRQUEwQixFQUMxQixTQUErQixVQUFVLEVBQ3pDLGtCQUFrQixHQUFHLENBQUM7SUFFeEIsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDaEUsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsZUFBZSxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQzNFLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLGVBQWUsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUU1RSxNQUFNLFFBQVEsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRWpDLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLEdBQUcsRUFBRSxDQUFDLCtDQUErQztRQUNqRCxnQkFBZ0IsTUFBTSxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDeEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFDMUMsR0FBRyxFQUFFLENBQUMsb0RBQW9ELFFBQVEsTUFBTTtRQUNwRSxpQkFBaUIsTUFBTSxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDMUMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLFFBQVEsRUFDbkQsR0FBRyxFQUFFLENBQUMscURBQXFELFFBQVEsSUFBSTtRQUNuRSxpQkFBaUIsTUFBTSxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDMUMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDckIsR0FBRyxFQUFFLENBQUMsZ0VBQWdFO1FBQ2xFLFVBQVUsUUFBUSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUM7SUFDdEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEVBQ3BDLEdBQUcsRUFBRSxDQUFDLDJDQUEyQyxRQUFRLEVBQUUsQ0FBQyxDQUFDO0lBQ2pFLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxLQUFLLFVBQVUsSUFBSSxNQUFNLEtBQUssU0FBUyxFQUM3QyxHQUFHLEVBQUUsQ0FBQywrQ0FBK0MsTUFBTSxFQUFFLENBQUMsQ0FBQztJQUVuRSxNQUFNLE1BQU0sR0FDYyxFQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxNQUFNLEVBQUUsT0FBTyxFQUFDLENBQUM7SUFDMUUsTUFBTSxLQUFLLEdBQXVCLEVBQUMsTUFBTSxFQUFFLGtCQUFrQixFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQ3pFLE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ3hCLGFBQWEsRUFBRSxNQUE4QixFQUM3QyxLQUEyQixDQUFDLENBQUM7SUFDakMsT0FBTyxHQUFlLENBQUM7QUFDekIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBRyxFQUFFLENBQUMsRUFBQyxjQUFjLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vLi4vZW5naW5lJztcbmltcG9ydCB7Q3JvcEFuZFJlc2l6ZSwgQ3JvcEFuZFJlc2l6ZUF0dHJzLCBDcm9wQW5kUmVzaXplSW5wdXRzfSBmcm9tICcuLi8uLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uLy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjFELCBUZW5zb3IyRCwgVGVuc29yNER9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi8uLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5cbi8qKlxuICogRXh0cmFjdHMgY3JvcHMgZnJvbSB0aGUgaW5wdXQgaW1hZ2UgdGVuc29yIGFuZCByZXNpemVzIHRoZW0gdXNpbmcgYmlsaW5lYXJcbiAqIHNhbXBsaW5nIG9yIG5lYXJlc3QgbmVpZ2hib3Igc2FtcGxpbmcgKHBvc3NpYmx5IHdpdGggYXNwZWN0IHJhdGlvIGNoYW5nZSlcbiAqIHRvIGEgY29tbW9uIG91dHB1dCBzaXplIHNwZWNpZmllZCBieSBjcm9wU2l6ZS5cbiAqXG4gKiBAcGFyYW0gaW1hZ2UgNGQgdGVuc29yIG9mIHNoYXBlIGBbYmF0Y2gsaW1hZ2VIZWlnaHQsaW1hZ2VXaWR0aCwgZGVwdGhdYCxcbiAqICAgICB3aGVyZSBpbWFnZUhlaWdodCBhbmQgaW1hZ2VXaWR0aCBtdXN0IGJlIHBvc2l0aXZlLCBzcGVjaWZ5aW5nIHRoZVxuICogICAgIGJhdGNoIG9mIGltYWdlcyBmcm9tIHdoaWNoIHRvIHRha2UgY3JvcHNcbiAqIEBwYXJhbSBib3hlcyAyZCBmbG9hdDMyIHRlbnNvciBvZiBzaGFwZSBgW251bUJveGVzLCA0XWAuIEVhY2ggZW50cnkgaXNcbiAqICAgICBgW3kxLCB4MSwgeTIsIHgyXWAsIHdoZXJlIGAoeTEsIHgxKWAgYW5kIGAoeTIsIHgyKWAgYXJlIHRoZSBub3JtYWxpemVkXG4gKiAgICAgY29vcmRpbmF0ZXMgb2YgdGhlIGJveCBpbiB0aGUgYm94SW5kW2ldJ3RoIGltYWdlIGluIHRoZSBiYXRjaFxuICogQHBhcmFtIGJveEluZCAxZCBpbnQzMiB0ZW5zb3Igb2Ygc2hhcGUgYFtudW1Cb3hlc11gIHdpdGggdmFsdWVzIGluIHJhbmdlXG4gKiAgICAgYFswLCBiYXRjaClgIHRoYXQgc3BlY2lmaWVzIHRoZSBpbWFnZSB0aGF0IHRoZSBgaWAtdGggYm94IHJlZmVycyB0by5cbiAqIEBwYXJhbSBjcm9wU2l6ZSAxZCBpbnQzMiB0ZW5zb3Igb2YgMiBlbGVtZW50cyBgW2Nyb3BIZWlnaCwgY3JvcFdpZHRoXWBcbiAqICAgICBzcGVjaWZ5aW5nIHRoZSBzaXplIHRvIHdoaWNoIGFsbCBjcm9wcyBhcmUgcmVzaXplZCB0by5cbiAqIEBwYXJhbSBtZXRob2QgT3B0aW9uYWwgc3RyaW5nIGZyb20gYCdiaWxpbmVhcicgfCAnbmVhcmVzdCdgLFxuICogICAgIGRlZmF1bHRzIHRvIGJpbGluZWFyLCB3aGljaCBzcGVjaWZpZXMgdGhlIHNhbXBsaW5nIG1ldGhvZCBmb3IgcmVzaXppbmdcbiAqIEBwYXJhbSBleHRyYXBvbGF0aW9uVmFsdWUgQSB0aHJlc2hvbGQgZm9yIGRlY2lkaW5nIHdoZW4gdG8gcmVtb3ZlIGJveGVzIGJhc2VkXG4gKiAgICAgb24gc2NvcmUuIERlZmF1bHRzIHRvIDAuXG4gKiBAcmV0dXJuIEEgNEQgdGVuc29yIG9mIHRoZSBzaGFwZSBgW251bUJveGVzLGNyb3BIZWlnaHQsY3JvcFdpZHRoLGRlcHRoXWBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdJbWFnZXMnLCBuYW1lc3BhY2U6ICdpbWFnZSd9XG4gKi9cbmZ1bmN0aW9uIGNyb3BBbmRSZXNpemVfKFxuICAgIGltYWdlOiBUZW5zb3I0RHxUZW5zb3JMaWtlLFxuICAgIGJveGVzOiBUZW5zb3IyRHxUZW5zb3JMaWtlLFxuICAgIGJveEluZDogVGVuc29yMUR8VGVuc29yTGlrZSxcbiAgICBjcm9wU2l6ZTogW251bWJlciwgbnVtYmVyXSxcbiAgICBtZXRob2Q6ICdiaWxpbmVhcid8J25lYXJlc3QnID0gJ2JpbGluZWFyJyxcbiAgICBleHRyYXBvbGF0aW9uVmFsdWUgPSAwLFxuICAgICk6IFRlbnNvcjREIHtcbiAgY29uc3QgJGltYWdlID0gY29udmVydFRvVGVuc29yKGltYWdlLCAnaW1hZ2UnLCAnY3JvcEFuZFJlc2l6ZScpO1xuICBjb25zdCAkYm94ZXMgPSBjb252ZXJ0VG9UZW5zb3IoYm94ZXMsICdib3hlcycsICdjcm9wQW5kUmVzaXplJywgJ2Zsb2F0MzInKTtcbiAgY29uc3QgJGJveEluZCA9IGNvbnZlcnRUb1RlbnNvcihib3hJbmQsICdib3hJbmQnLCAnY3JvcEFuZFJlc2l6ZScsICdpbnQzMicpO1xuXG4gIGNvbnN0IG51bUJveGVzID0gJGJveGVzLnNoYXBlWzBdO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGltYWdlLnJhbmsgPT09IDQsXG4gICAgICAoKSA9PiAnRXJyb3IgaW4gY3JvcEFuZFJlc2l6ZTogaW1hZ2UgbXVzdCBiZSByYW5rIDQsJyArXG4gICAgICAgICAgYGJ1dCBnb3QgcmFuayAkeyRpbWFnZS5yYW5rfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkYm94ZXMucmFuayA9PT0gMiAmJiAkYm94ZXMuc2hhcGVbMV0gPT09IDQsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY3JvcEFuZFJlc2l6ZTogYm94ZXMgbXVzdCBiZSBoYXZlIHNpemUgWyR7bnVtQm94ZXN9LDRdIGAgK1xuICAgICAgICAgIGBidXQgaGFkIHNoYXBlICR7JGJveGVzLnNoYXBlfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkYm94SW5kLnJhbmsgPT09IDEgJiYgJGJveEluZC5zaGFwZVswXSA9PT0gbnVtQm94ZXMsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY3JvcEFuZFJlc2l6ZTogYm94SW5kIG11c3QgYmUgaGF2ZSBzaXplIFske251bUJveGVzfV0gYCArXG4gICAgICAgICAgYGJ1dCBoYWQgc2hhcGUgJHskYm94ZXMuc2hhcGV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGNyb3BTaXplLmxlbmd0aCA9PT0gMixcbiAgICAgICgpID0+IGBFcnJvciBpbiBjcm9wQW5kUmVzaXplOiBjcm9wU2l6ZSBtdXN0IGJlIG9mIGxlbmd0aCAyLCBidXQgZ290IGAgK1xuICAgICAgICAgIGBsZW5ndGggJHtjcm9wU2l6ZS5sZW5ndGh9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGNyb3BTaXplWzBdID49IDEgJiYgY3JvcFNpemVbMV0gPj0gMSxcbiAgICAgICgpID0+IGBjcm9wU2l6ZSBtdXN0IGJlIGF0bGVhc3QgWzEsMV0sIGJ1dCB3YXMgJHtjcm9wU2l6ZX1gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBtZXRob2QgPT09ICdiaWxpbmVhcicgfHwgbWV0aG9kID09PSAnbmVhcmVzdCcsXG4gICAgICAoKSA9PiBgbWV0aG9kIG11c3QgYmUgYmlsaW5lYXIgb3IgbmVhcmVzdCwgYnV0IHdhcyAke21ldGhvZH1gKTtcblxuICBjb25zdCBpbnB1dHM6XG4gICAgICBDcm9wQW5kUmVzaXplSW5wdXRzID0ge2ltYWdlOiAkaW1hZ2UsIGJveGVzOiAkYm94ZXMsIGJveEluZDogJGJveEluZH07XG4gIGNvbnN0IGF0dHJzOiBDcm9wQW5kUmVzaXplQXR0cnMgPSB7bWV0aG9kLCBleHRyYXBvbGF0aW9uVmFsdWUsIGNyb3BTaXplfTtcbiAgY29uc3QgcmVzID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIENyb3BBbmRSZXNpemUsIGlucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHt9IGFzIE5hbWVkQXR0ck1hcCk7XG4gIHJldHVybiByZXMgYXMgVGVuc29yNEQ7XG59XG5cbmV4cG9ydCBjb25zdCBjcm9wQW5kUmVzaXplID0gb3Aoe2Nyb3BBbmRSZXNpemVffSk7XG4iXX0=
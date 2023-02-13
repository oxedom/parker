/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original source: keras/callbacks.py */
import { BaseCallback } from './base_callbacks';
import { LayersModel } from './engine/training';
import { NotImplementedError } from './errors';
import { resolveScalarsInLogs } from './logs';
export class Callback extends BaseCallback {
    constructor() {
        super(...arguments);
        /** Instance of `keras.models.Model`. Reference of the model being trained. */
        this.model = null;
    }
    setModel(model) {
        if (!(model instanceof LayersModel)) {
            throw new Error('model must be a LayersModel, not some other Container');
        }
        this.model = model;
    }
}
function less(currVal, prevVal) {
    return currVal < prevVal;
}
function greater(currVal, prevVal) {
    return currVal > prevVal;
}
/**
 * A Callback that stops training when a monitored quantity has stopped
 * improving.
 */
export class EarlyStopping extends Callback {
    constructor(args) {
        super();
        if (args == null) {
            args = {};
        }
        if (args.restoreBestWeights) {
            throw new NotImplementedError('restoreBestWeights = True is not implemented in EarlyStopping yet.');
        }
        this.monitor = args.monitor || 'val_loss';
        this.minDelta = Math.abs(args.minDelta || 0);
        this.patience = args.patience || 0;
        this.verbose = args.verbose || 0;
        this.mode = args.mode || 'auto';
        this.baseline = args.baseline;
        if (['auto', 'min', 'max'].indexOf(this.mode) === -1) {
            console.warn(`EarlyStopping mode '${this.mode}' is invalid. ` +
                `Falling back to mode 'auto'.`);
            this.mode = 'auto';
        }
        if (this.mode === 'min') {
            this.monitorFunc = less;
        }
        else if (this.mode === 'max') {
            this.monitorFunc = greater;
        }
        else {
            // For mode === 'auto'.
            if (this.monitor.indexOf('acc') !== -1) {
                this.monitorFunc = greater;
            }
            else {
                this.monitorFunc = less;
            }
        }
        if (this.monitorFunc === less) {
            this.minDelta *= -1;
        }
    }
    async onTrainBegin(logs) {
        this.wait = 0;
        this.stoppedEpoch = 0;
        if (this.baseline != null) {
            this.best = this.baseline;
        }
        else {
            this.best = this.monitorFunc === less ? Infinity : -Infinity;
        }
    }
    async onEpochEnd(epoch, logs) {
        await resolveScalarsInLogs(logs);
        const current = this.getMonitorValue(logs);
        if (current == null) {
            return;
        }
        if (this.monitorFunc(current - this.minDelta, this.best)) {
            this.best = current;
            this.wait = 0;
            // TODO(cais): Logic for restoreBestWeights.
        }
        else {
            this.wait++;
            if (this.wait >= this.patience) {
                this.stoppedEpoch = epoch;
                this.model.stopTraining = true;
            }
            // TODO(cais): Logic for restoreBestWeights.
        }
    }
    async onTrainEnd(logs) {
        if (this.stoppedEpoch > 0 && this.verbose) {
            console.log(`Epoch ${this.stoppedEpoch}: early stopping.`);
        }
    }
    getMonitorValue(logs) {
        if (logs == null) {
            logs = {};
        }
        const monitorValue = logs[this.monitor];
        if (monitorValue == null) {
            console.warn(`Metric for EarlyStopping ${this.monitor} is not available. ` +
                `Available metrics are: ${Object.keys(logs)}`);
        }
        return monitorValue;
    }
}
/**
 * Factory function for a Callback that stops training when a monitored
 * quantity has stopped improving.
 *
 * Early stopping is a type of regularization, and protects model against
 * overfitting.
 *
 * The following example based on fake data illustrates how this callback
 * can be used during `tf.LayersModel.fit()`:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.dense({
 *   units: 3,
 *   activation: 'softmax',
 *   kernelInitializer: 'ones',
 *   inputShape: [2]
 * }));
 * const xs = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const ys = tf.tensor2d([[1, 0, 0], [0, 1, 0]], [2, 3]);
 * const xsVal = tf.tensor2d([4, 3, 2, 1], [2, 2]);
 * const ysVal = tf.tensor2d([[0, 0, 1], [0, 1, 0]], [2, 3]);
 * model.compile(
 *     {loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['acc']});
 *
 * // Without the EarlyStopping callback, the val_acc value would be:
 * //   0.5, 0.5, 0.5, 0.5, ...
 * // With val_acc being monitored, training should stop after the 2nd epoch.
 * const history = await model.fit(xs, ys, {
 *   epochs: 10,
 *   validationData: [xsVal, ysVal],
 *   callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})
 * });
 *
 * // Expect to see a length-2 array.
 * console.log(history.history.val_acc);
 * ```
 *
 * @doc {
 *   heading: 'Callbacks',
 *   namespace: 'callbacks'
 * }
 */
export function earlyStopping(args) {
    return new EarlyStopping(args);
}
export const callbacks = { earlyStopping };
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY2FsbGJhY2tzLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL2NhbGxiYWNrcy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVILHlDQUF5QztBQUV6QyxPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFFOUMsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQzlDLE9BQU8sRUFBQyxtQkFBbUIsRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUM3QyxPQUFPLEVBQU8sb0JBQW9CLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFbEQsTUFBTSxPQUFnQixRQUFTLFNBQVEsWUFBWTtJQUFuRDs7UUFDRSw4RUFBOEU7UUFDOUUsVUFBSyxHQUFnQixJQUFJLENBQUM7SUFRNUIsQ0FBQztJQU5DLFFBQVEsQ0FBQyxLQUFnQjtRQUN2QixJQUFJLENBQUMsQ0FBQyxLQUFLLFlBQVksV0FBVyxDQUFDLEVBQUU7WUFDbkMsTUFBTSxJQUFJLEtBQUssQ0FBQyx1REFBdUQsQ0FBQyxDQUFDO1NBQzFFO1FBQ0QsSUFBSSxDQUFDLEtBQUssR0FBRyxLQUFLLENBQUM7SUFDckIsQ0FBQztDQUNGO0FBNERELFNBQVMsSUFBSSxDQUFDLE9BQWUsRUFBRSxPQUFlO0lBQzVDLE9BQU8sT0FBTyxHQUFHLE9BQU8sQ0FBQztBQUMzQixDQUFDO0FBRUQsU0FBUyxPQUFPLENBQUMsT0FBZSxFQUFFLE9BQWU7SUFDL0MsT0FBTyxPQUFPLEdBQUcsT0FBTyxDQUFDO0FBQzNCLENBQUM7QUFFRDs7O0dBR0c7QUFDSCxNQUFNLE9BQU8sYUFBYyxTQUFRLFFBQVE7SUFjekMsWUFBWSxJQUFnQztRQUMxQyxLQUFLLEVBQUUsQ0FBQztRQUNSLElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtZQUNoQixJQUFJLEdBQUcsRUFBRSxDQUFDO1NBQ1g7UUFDRCxJQUFJLElBQUks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ZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qIE9yaWdpbmFsIHNvdXJjZToga2VyYXMvY2FsbGJhY2tzLnB5ICovXG5cbmltcG9ydCB7QmFzZUNhbGxiYWNrfSBmcm9tICcuL2Jhc2VfY2FsbGJhY2tzJztcbmltcG9ydCB7Q29udGFpbmVyfSBmcm9tICcuL2VuZ2luZS9jb250YWluZXInO1xuaW1wb3J0IHtMYXllcnNNb2RlbH0gZnJvbSAnLi9lbmdpbmUvdHJhaW5pbmcnO1xuaW1wb3J0IHtOb3RJbXBsZW1lbnRlZEVycm9yfSBmcm9tICcuL2Vycm9ycyc7XG5pbXBvcnQge0xvZ3MsIHJlc29sdmVTY2FsYXJzSW5Mb2dzfSBmcm9tICcuL2xvZ3MnO1xuXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgQ2FsbGJhY2sgZXh0ZW5kcyBCYXNlQ2FsbGJhY2sge1xuICAvKiogSW5zdGFuY2Ugb2YgYGtlcmFzLm1vZGVscy5Nb2RlbGAuIFJlZmVyZW5jZSBvZiB0aGUgbW9kZWwgYmVpbmcgdHJhaW5lZC4gKi9cbiAgbW9kZWw6IExheWVyc01vZGVsID0gbnVsbDtcblxuICBzZXRNb2RlbChtb2RlbDogQ29udGFpbmVyKTogdm9pZCB7XG4gICAgaWYgKCEobW9kZWwgaW5zdGFuY2VvZiBMYXllcnNNb2RlbCkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignbW9kZWwgbXVzdCBiZSBhIExheWVyc01vZGVsLCBub3Qgc29tZSBvdGhlciBDb250YWluZXInKTtcbiAgICB9XG4gICAgdGhpcy5tb2RlbCA9IG1vZGVsO1xuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgRWFybHlTdG9wcGluZ0NhbGxiYWNrQXJncyB7XG4gIC8qKlxuICAgKiBRdWFudGl0eSB0byBiZSBtb25pdG9yZWQuXG4gICAqXG4gICAqIERlZmF1bHRzIHRvICd2YWxfbG9zcycuXG4gICAqL1xuICBtb25pdG9yPzogc3RyaW5nO1xuXG4gIC8qKlxuICAgKiBNaW5pbXVtIGNoYW5nZSBpbiB0aGUgbW9uaXRvcmVkIHF1YW50aXR5IHRvIHF1YWxpZnkgYXMgaW1wcm92ZW1lbnQsXG4gICAqIGkuZS4sIGFuIGFic29sdXRlIGNoYW5nZSBvZiBsZXNzIHRoYW4gYG1pbkRlbHRhYCB3aWxsIGNvdW50IGFzIG5vXG4gICAqIGltcHJvdmVtZW50LlxuICAgKlxuICAgKiBEZWZhdWx0cyB0byAwLlxuICAgKi9cbiAgbWluRGVsdGE/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE51bWJlciBvZiBlcG9jaHMgd2l0aCBubyBpbXByb3ZlbWVudCBhZnRlciB3aGljaCB0cmFpbmluZyB3aWxsIGJlIHN0b3BwZWQuXG4gICAqXG4gICAqIERlZmF1bHRzIHRvIDAuXG4gICAqL1xuICBwYXRpZW5jZT86IG51bWJlcjtcblxuICAvKiogVmVyYm9zaXR5IG1vZGUuICovXG4gIHZlcmJvc2U/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE1vZGU6IG9uZSBvZiAnbWluJywgJ21heCcsIGFuZCAnYXV0bycuXG4gICAqIC0gSW4gJ21pbicgbW9kZSwgdHJhaW5pbmcgd2lsbCBiZSBzdG9wcGVkIHdoZW4gdGhlIHF1YW50aXR5IG1vbml0b3JlZCBoYXNcbiAgICogICBzdG9wcGVkIGRlY3JlYXNpbmcuXG4gICAqIC0gSW4gJ21heCcgbW9kZSwgdHJhaW5pbmcgd2lsbCBiZSBzdG9wcGVkIHdoZW4gdGhlIHF1YW50aXR5IG1vbml0b3JlZCBoYXNcbiAgICogICBzdG9wcGVkIGluY3JlYXNpbmcuXG4gICAqIC0gSW4gJ2F1dG8nIG1vZGUsIHRoZSBkaXJlY3Rpb24gaXMgaW5mZXJyZWQgYXV0b21hdGljYWxseSBmcm9tIHRoZSBuYW1lIG9mXG4gICAqICAgdGhlIG1vbml0b3JlZCBxdWFudGl0eS5cbiAgICpcbiAgICogRGVmYXVsdHMgdG8gJ2F1dG8nLlxuICAgKi9cbiAgbW9kZT86ICdhdXRvJ3wnbWluJ3wnbWF4JztcblxuICAvKipcbiAgICogQmFzZWxpbmUgdmFsdWUgb2YgdGhlIG1vbml0b3JlZCBxdWFudGl0eS5cbiAgICpcbiAgICogSWYgc3BlY2lmaWVkLCB0cmFpbmluZyB3aWxsIGJlIHN0b3BwZWQgaWYgdGhlIG1vZGVsIGRvZXNuJ3Qgc2hvd1xuICAgKiBpbXByb3ZlbWVudCBvdmVyIHRoZSBiYXNlbGluZS5cbiAgICovXG4gIGJhc2VsaW5lPzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBXaGV0aGVyIHRvIHJlc3RvcmUgbW9kZWwgd2VpZ2h0cyBmcm9tIHRoZSBlcG9jaCB3aXRoIHRoZSBiZXN0IHZhbHVlXG4gICAqIG9mIHRoZSBtb25pdG9yZWQgcXVhbnRpdHkuIElmIGBGYWxzZWAsIHRoZSBtb2RlbCB3ZWlnaHRzIG9idGFpbmVkIGF0IHRoZVxuICAgKiBhdCB0aGUgbGFzdCBzdGVwIG9mIHRyYWluaW5nIGFyZSB1c2VkLlxuICAgKlxuICAgKiAqKmBUcnVlYCBpcyBub3Qgc3VwcG9ydGVkIHlldC4qKlxuICAgKi9cbiAgcmVzdG9yZUJlc3RXZWlnaHRzPzogYm9vbGVhbjtcbn1cblxuZnVuY3Rpb24gbGVzcyhjdXJyVmFsOiBudW1iZXIsIHByZXZWYWw6IG51bWJlcikge1xuICByZXR1cm4gY3VyclZhbCA8IHByZXZWYWw7XG59XG5cbmZ1bmN0aW9uIGdyZWF0ZXIoY3VyclZhbDogbnVtYmVyLCBwcmV2VmFsOiBudW1iZXIpIHtcbiAgcmV0dXJuIGN1cnJWYWwgPiBwcmV2VmFsO1xufVxuXG4vKipcbiAqIEEgQ2FsbGJhY2sgdGhhdCBzdG9wcyB0cmFpbmluZyB3aGVuIGEgbW9uaXRvcmVkIHF1YW50aXR5IGhhcyBzdG9wcGVkXG4gKiBpbXByb3ZpbmcuXG4gKi9cbmV4cG9ydCBjbGFzcyBFYXJseVN0b3BwaW5nIGV4dGVuZHMgQ2FsbGJhY2sge1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgbW9uaXRvcjogc3RyaW5nO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgbWluRGVsdGE6IG51bWJlcjtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHBhdGllbmNlOiBudW1iZXI7XG4gIHByb3RlY3RlZCByZWFkb25seSBiYXNlbGluZTogbnVtYmVyO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgdmVyYm9zZTogbnVtYmVyO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgbW9kZTogJ2F1dG8nfCdtaW4nfCdtYXgnO1xuXG4gIHByb3RlY3RlZCBtb25pdG9yRnVuYzogKGN1cnJWYWw6IG51bWJlciwgcHJldlZhbDogbnVtYmVyKSA9PiBib29sZWFuO1xuXG4gIHByaXZhdGUgd2FpdDogbnVtYmVyO1xuICBwcml2YXRlIHN0b3BwZWRFcG9jaDogbnVtYmVyO1xuICBwcml2YXRlIGJlc3Q6IG51bWJlcjtcblxuICBjb25zdHJ1Y3RvcihhcmdzPzogRWFybHlTdG9wcGluZ0NhbGxiYWNrQXJncykge1xuICAgIHN1cGVyKCk7XG4gICAgaWYgKGFyZ3MgPT0gbnVsbCkge1xuICAgICAgYXJncyA9IHt9O1xuICAgIH1cbiAgICBpZiAoYXJncy5yZXN0b3JlQmVzdFdlaWdodHMpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICdyZXN0b3JlQmVzdFdlaWdodHMgPSBUcnVlIGlzIG5vdCBpbXBsZW1lbnRlZCBpbiBFYXJseVN0b3BwaW5nIHlldC4nKTtcbiAgICB9XG5cbiAgICB0aGlzLm1vbml0b3IgPSBhcmdzLm1vbml0b3IgfHwgJ3ZhbF9sb3NzJztcbiAgICB0aGlzLm1pbkRlbHRhID0gTWF0aC5hYnMoYXJncy5taW5EZWx0YSB8fCAwKTtcbiAgICB0aGlzLnBhdGllbmNlID0gYXJncy5wYXRpZW5jZSB8fCAwO1xuICAgIHRoaXMudmVyYm9zZSA9IGFyZ3MudmVyYm9zZSB8fCAwO1xuICAgIHRoaXMubW9kZSA9IGFyZ3MubW9kZSB8fCAnYXV0byc7XG4gICAgdGhpcy5iYXNlbGluZSA9IGFyZ3MuYmFzZWxpbmU7XG5cbiAgICBpZiAoWydhdXRvJywgJ21pbicsICdtYXgnXS5pbmRleE9mKHRoaXMubW9kZSkgPT09IC0xKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYEVhcmx5U3RvcHBpbmcgbW9kZSAnJHt0aGlzLm1vZGV9JyBpcyBpbnZhbGlkLiBgICtcbiAgICAgICAgICBgRmFsbGluZyBiYWNrIHRvIG1vZGUgJ2F1dG8nLmApO1xuICAgICAgdGhpcy5tb2RlID0gJ2F1dG8nO1xuICAgIH1cblxuICAgIGlmICh0aGlzLm1vZGUgPT09ICdtaW4nKSB7XG4gICAgICB0aGlzLm1vbml0b3JGdW5jID0gbGVzcztcbiAgICB9IGVsc2UgaWYgKHRoaXMubW9kZSA9PT0gJ21heCcpIHtcbiAgICAgIHRoaXMubW9uaXRvckZ1bmMgPSBncmVhdGVyO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBGb3IgbW9kZSA9PT0gJ2F1dG8nLlxuICAgICAgaWYgKHRoaXMubW9uaXRvci5pbmRleE9mKCdhY2MnKSAhPT0gLTEpIHtcbiAgICAgICAgdGhpcy5tb25pdG9yRnVuYyA9IGdyZWF0ZXI7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aGlzLm1vbml0b3JGdW5jID0gbGVzcztcbiAgICAgIH1cbiAgICB9XG5cbiAgICBpZiAodGhpcy5tb25pdG9yRnVuYyA9PT0gbGVzcykge1xuICAgICAgdGhpcy5taW5EZWx0YSAqPSAtMTtcbiAgICB9XG4gIH1cblxuICBhc3luYyBvblRyYWluQmVnaW4obG9ncz86IExvZ3MpIHtcbiAgICB0aGlzLndhaXQgPSAwO1xuICAgIHRoaXMuc3RvcHBlZEVwb2NoID0gMDtcbiAgICBpZiAodGhpcy5iYXNlbGluZSAhPSBudWxsKSB7XG4gICAgICB0aGlzLmJlc3QgPSB0aGlzLmJhc2VsaW5lO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmJlc3QgPSB0aGlzLm1vbml0b3JGdW5jID09PSBsZXNzID8gSW5maW5pdHkgOiAtSW5maW5pdHk7XG4gICAgfVxuICB9XG5cbiAgYXN5bmMgb25FcG9jaEVuZChlcG9jaDogbnVtYmVyLCBsb2dzPzogTG9ncykge1xuICAgIGF3YWl0IHJlc29sdmVTY2FsYXJzSW5Mb2dzKGxvZ3MpO1xuICAgIGNvbnN0IGN1cnJlbnQgPSB0aGlzLmdldE1vbml0b3JWYWx1ZShsb2dzKTtcbiAgICBpZiAoY3VycmVudCA9PSBudWxsKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgaWYgKHRoaXMubW9uaXRvckZ1bmMoY3VycmVudCAtIHRoaXMubWluRGVsdGEsIHRoaXMuYmVzdCkpIHtcbiAgICAgIHRoaXMuYmVzdCA9IGN1cnJlbnQ7XG4gICAgICB0aGlzLndhaXQgPSAwO1xuICAgICAgLy8gVE9ETyhjYWlzKTogTG9naWMgZm9yIHJlc3RvcmVCZXN0V2VpZ2h0cy5cbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy53YWl0Kys7XG4gICAgICBpZiAodGhpcy53YWl0ID49IHRoaXMucGF0aWVuY2UpIHtcbiAgICAgICAgdGhpcy5zdG9wcGVkRXBvY2ggPSBlcG9jaDtcbiAgICAgICAgdGhpcy5tb2RlbC5zdG9wVHJhaW5pbmcgPSB0cnVlO1xuICAgICAgfVxuICAgICAgLy8gVE9ETyhjYWlzKTogTG9naWMgZm9yIHJlc3RvcmVCZXN0V2VpZ2h0cy5cbiAgICB9XG4gIH1cblxuICBhc3luYyBvblRyYWluRW5kKGxvZ3M/OiBMb2dzKSB7XG4gICAgaWYgKHRoaXMuc3RvcHBlZEVwb2NoID4gMCAmJiB0aGlzLnZlcmJvc2UpIHtcbiAgICAgIGNvbnNvbGUubG9nKGBFcG9jaCAke3RoaXMuc3RvcHBlZEVwb2NofTogZWFybHkgc3RvcHBpbmcuYCk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBnZXRNb25pdG9yVmFsdWUobG9nczogTG9ncykge1xuICAgIGlmIChsb2dzID09IG51bGwpIHtcbiAgICAgIGxvZ3MgPSB7fTtcbiAgICB9XG4gICAgY29uc3QgbW9uaXRvclZhbHVlID0gbG9nc1t0aGlzLm1vbml0b3JdO1xuICAgIGlmIChtb25pdG9yVmFsdWUgPT0gbnVsbCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBNZXRyaWMgZm9yIEVhcmx5U3RvcHBpbmcgJHt0aGlzLm1vbml0b3J9IGlzIG5vdCBhdmFpbGFibGUuIGAgK1xuICAgICAgICAgIGBBdmFpbGFibGUgbWV0cmljcyBhcmU6ICR7T2JqZWN0LmtleXMobG9ncyl9YCk7XG4gICAgfVxuICAgIHJldHVybiBtb25pdG9yVmFsdWU7XG4gIH1cbn1cblxuLyoqXG4gKiBGYWN0b3J5IGZ1bmN0aW9uIGZvciBhIENhbGxiYWNrIHRoYXQgc3RvcHMgdHJhaW5pbmcgd2hlbiBhIG1vbml0b3JlZFxuICogcXVhbnRpdHkgaGFzIHN0b3BwZWQgaW1wcm92aW5nLlxuICpcbiAqIEVhcmx5IHN0b3BwaW5nIGlzIGEgdHlwZSBvZiByZWd1bGFyaXphdGlvbiwgYW5kIHByb3RlY3RzIG1vZGVsIGFnYWluc3RcbiAqIG92ZXJmaXR0aW5nLlxuICpcbiAqIFRoZSBmb2xsb3dpbmcgZXhhbXBsZSBiYXNlZCBvbiBmYWtlIGRhdGEgaWxsdXN0cmF0ZXMgaG93IHRoaXMgY2FsbGJhY2tcbiAqIGNhbiBiZSB1c2VkIGR1cmluZyBgdGYuTGF5ZXJzTW9kZWwuZml0KClgOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoKTtcbiAqIG1vZGVsLmFkZCh0Zi5sYXllcnMuZGVuc2Uoe1xuICogICB1bml0czogMyxcbiAqICAgYWN0aXZhdGlvbjogJ3NvZnRtYXgnLFxuICogICBrZXJuZWxJbml0aWFsaXplcjogJ29uZXMnLFxuICogICBpbnB1dFNoYXBlOiBbMl1cbiAqIH0pKTtcbiAqIGNvbnN0IHhzID0gdGYudGVuc29yMmQoWzEsIDIsIDMsIDRdLCBbMiwgMl0pO1xuICogY29uc3QgeXMgPSB0Zi50ZW5zb3IyZChbWzEsIDAsIDBdLCBbMCwgMSwgMF1dLCBbMiwgM10pO1xuICogY29uc3QgeHNWYWwgPSB0Zi50ZW5zb3IyZChbNCwgMywgMiwgMV0sIFsyLCAyXSk7XG4gKiBjb25zdCB5c1ZhbCA9IHRmLnRlbnNvcjJkKFtbMCwgMCwgMV0sIFswLCAxLCAwXV0sIFsyLCAzXSk7XG4gKiBtb2RlbC5jb21waWxlKFxuICogICAgIHtsb3NzOiAnY2F0ZWdvcmljYWxDcm9zc2VudHJvcHknLCBvcHRpbWl6ZXI6ICdzZ2QnLCBtZXRyaWNzOiBbJ2FjYyddfSk7XG4gKlxuICogLy8gV2l0aG91dCB0aGUgRWFybHlTdG9wcGluZyBjYWxsYmFjaywgdGhlIHZhbF9hY2MgdmFsdWUgd291bGQgYmU6XG4gKiAvLyAgIDAuNSwgMC41LCAwLjUsIDAuNSwgLi4uXG4gKiAvLyBXaXRoIHZhbF9hY2MgYmVpbmcgbW9uaXRvcmVkLCB0cmFpbmluZyBzaG91bGQgc3RvcCBhZnRlciB0aGUgMm5kIGVwb2NoLlxuICogY29uc3QgaGlzdG9yeSA9IGF3YWl0IG1vZGVsLmZpdCh4cywgeXMsIHtcbiAqICAgZXBvY2hzOiAxMCxcbiAqICAgdmFsaWRhdGlvbkRhdGE6IFt4c1ZhbCwgeXNWYWxdLFxuICogICBjYWxsYmFja3M6IHRmLmNhbGxiYWNrcy5lYXJseVN0b3BwaW5nKHttb25pdG9yOiAndmFsX2FjYyd9KVxuICogfSk7XG4gKlxuICogLy8gRXhwZWN0IHRvIHNlZSBhIGxlbmd0aC0yIGFycmF5LlxuICogY29uc29sZS5sb2coaGlzdG9yeS5oaXN0b3J5LnZhbF9hY2MpO1xuICogYGBgXG4gKlxuICogQGRvYyB7XG4gKiAgIGhlYWRpbmc6ICdDYWxsYmFja3MnLFxuICogICBuYW1lc3BhY2U6ICdjYWxsYmFja3MnXG4gKiB9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBlYXJseVN0b3BwaW5nKGFyZ3M/OiBFYXJseVN0b3BwaW5nQ2FsbGJhY2tBcmdzKSB7XG4gIHJldHVybiBuZXcgRWFybHlTdG9wcGluZyhhcmdzKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNhbGxiYWNrcyA9IHtlYXJseVN0b3BwaW5nfTtcbiJdfQ==
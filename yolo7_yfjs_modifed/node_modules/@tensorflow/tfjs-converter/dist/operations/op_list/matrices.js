/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
export const json = [
    {
        'tfOpName': '_FusedMatMul',
        'category': 'matrices',
        'inputs': [
            {
                'start': 0,
                'name': 'a',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'b',
                'type': 'tensor'
            },
            {
                'start': 2,
                'end': 0,
                'name': 'args',
                'type': 'tensors'
            }
        ],
        'attrs': [
            {
                'tfName': 'num_args',
                'name': 'numArgs',
                'type': 'number'
            },
            {
                'tfName': 'fused_ops',
                'name': 'fusedOps',
                'type': 'string[]',
                'defaultValue': []
            },
            {
                'tfName': 'epsilon',
                'name': 'epsilon',
                'type': 'number',
                'defaultValue': 0.0001
            },
            {
                'tfName': 'transpose_a',
                'name': 'transposeA',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'transpose_b',
                'name': 'transposeB',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'MatMul',
        'category': 'matrices',
        'inputs': [
            {
                'start': 0,
                'name': 'a',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'b',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'transpose_a',
                'name': 'transposeA',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'transpose_b',
                'name': 'transposeB',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'BatchMatMul',
        'category': 'matrices',
        'inputs': [
            {
                'start': 0,
                'name': 'a',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'b',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'adj_x',
                'name': 'transposeA',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'adj_y',
                'name': 'transposeB',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'BatchMatMulV2',
        'category': 'matrices',
        'inputs': [
            {
                'start': 0,
                'name': 'a',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'b',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'adj_x',
                'name': 'transposeA',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'adj_y',
                'name': 'transposeB',
                'type': 'bool',
                'defaultValue': false
            },
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'Transpose',
        'category': 'matrices',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'perm',
                'type': 'number[]'
            }
        ],
        'attrs': [
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'Einsum',
        'category': 'matrices',
        'inputs': [
            {
                'start': 0,
                'end': 0,
                'name': 'tensors',
                'type': 'tensors'
            }
        ],
        'attrs': [
            {
                'tfName': 'equation',
                'name': 'equation',
                'type': 'string'
            },
            {
                'tfName': 'N',
                'name': 'n',
                'type': 'number',
                'defaultValue': 2
            },
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype'
            }
        ]
    }
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF0cmljZXMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvbnZlcnRlci9zcmMvb3BlcmF0aW9ucy9vcF9saXN0L21hdHJpY2VzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUNBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUlILE1BQU0sQ0FBQyxNQUFNLElBQUksR0FBZTtJQUM5QjtRQUNFLFVBQVUsRUFBRSxjQUFjO1FBQzFCLFVBQVUsRUFBRSxVQUFVO1FBQ3RCLFFBQVEsRUFBRTtZQUNSO2dCQUNFLE9BQU8sRUFBRSxDQUFDO2dCQUNWLE1BQU0sRUFBRSxHQUFHO2dCQUNYLE1BQU0sRUFBRSxRQUFRO2FBQ2pCO1lBQ0Q7Z0JBQ0UsT0FBTyxFQUFFLENBQUM7Z0JBQ1YsTUFBTSxFQUFFLEdBQUc7Z0JBQ1gsTUFBTSxFQUFFLFFBQVE7YUFDakI7WUFDRDtnQkFDRSxPQUFPLEVBQUUsQ0FBQztnQkFDVixLQUFLLEVBQUUsQ0FBQztnQkFDUixNQUFNLEVBQUUsTUFBTTtnQkFDZCxNQUFNLEVBQUUsU0FBUzthQUNsQjtTQUNGO1FBQ0QsT0FBTyxFQUFFO1lBQ1A7Z0JBQ0UsUUFBUSxFQUFFLFVBQVU7Z0JBQ3BCLE1BQU0sRUFBRSxTQUFTO2dCQUNqQixNQUFNLEVBQUUsUUFBUTthQUNqQjtZQUNEO2dCQUNFLFFBQVEsRUFBRSxXQUFXO2dCQUNyQixNQUFNLEVBQUUsVUFBVTtnQkFDbEIsTUFBTSxFQUFFLFVBQVU7Z0JBQ2xCLGNBQWMsRUFBRSxFQUFFO2FBQ25CO1lBQ0Q7Z0JBQ0UsUUFBUSxFQUFFLFNBQVM7Z0JBQ25CLE1BQU0sRUFBRSxTQUFTO2dCQUNqQixNQUFNLEVBQUUsUUFBUTtnQkFDaEIsY0FBYyxFQUFFLE1BQU07YUFDdkI7WUFDRDtnQkFDRSxRQUFRLEVBQUUsYUFBYTtnQkFDdkIsTUFBTSxFQUFFLFlBQVk7Z0JBQ3BCLE1BQU0sRUFBRSxNQUFNO2dCQUNkLGNBQWMsRUFBRSxLQUFL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O1FBQ0UsVUFBVSxFQUFFLFdBQVc7UUFDdkIsVUFBVSxFQUFFLFVBQVU7UUFDdEIsUUFBUSxFQUFFO1lBQ1I7Z0JBQ0UsT0FBTyxFQUFFLENBQUM7Z0JBQ1YsTUFBTSxFQUFFLEdBQUc7Z0JBQ1gsTUFBTSxFQUFFLFFBQVE7YUFDakI7WUFDRDtnQkFDRSxPQUFPLEVBQUUsQ0FBQztnQkFDVixNQUFNLEVBQUUsTUFBTTtnQkFDZCxNQUFNLEVBQUUsVUFBVTthQUNuQjtTQUNGO1FBQ0QsT0FBTyxFQUFFO1lBQ1A7Z0JBQ0UsUUFBUSxFQUFFLEdBQUc7Z0JBQ2IsTUFBTSxFQUFFLE9BQU87Z0JBQ2YsTUFBTSxFQUFFLE9BQU87Z0JBQ2YsY0FBYyxFQUFFLElBQUk7YUFDckI7U0FDRjtLQUNGO0lBQ0Q7UUFDRSxVQUFVLEVBQUUsUUFBUTtRQUNwQixVQUFVLEVBQUUsVUFBVTtRQUN0QixRQUFRLEVBQUU7WUFDUjtnQkFDRSxPQUFPLEVBQUUsQ0FBQztnQkFDVixLQUFLLEVBQUUsQ0FBQztnQkFDUixNQUFNLEVBQUUsU0FBUztnQkFDakIsTUFBTSxFQUFFLFNBQVM7YUFDbEI7U0FDRjtRQUNELE9BQU8sRUFBRTtZQUNQO2dCQUNFLFFBQVEsRUFBRSxVQUFVO2dCQUNwQixNQUFNLEVBQUUsVUFBVTtnQkFDbEIsTUFBTSxFQUFFLFFBQVE7YUFDakI7WUFDRDtnQkFDRSxRQUFRLEVBQUUsR0FBRztnQkFDYixNQUFNLEVBQUUsR0FBRztnQkFDWCxNQUFNLEVBQUUsUUFBUTtnQkFDaEIsY0FBYyxFQUFFLENBQUM7YUFDbEI7WUFDRDtnQkFDRSxRQUFRLEVBQUUsR0FBRztnQkFDYixNQUFNLEVBQUUsT0FBTztnQkFDZixNQUFNLEVBQUUsT0FBTzthQUNoQjtTQUNGO0tBQ0Y7Q0FDRixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiXG4vKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7T3BNYXBwZXJ9IGZyb20gJy4uL3R5cGVzJztcblxuZXhwb3J0IGNvbnN0IGpzb246IE9wTWFwcGVyW10gPSBbXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnX0Z1c2VkTWF0TXVsJyxcbiAgICAnY2F0ZWdvcnknOiAnbWF0cmljZXMnLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ2EnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAxLFxuICAgICAgICAnbmFtZSc6ICdiJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMixcbiAgICAgICAgJ2VuZCc6IDAsXG4gICAgICAgICduYW1lJzogJ2FyZ3MnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3JzJ1xuICAgICAgfVxuICAgIF0sXG4gICAgJ2F0dHJzJzogW1xuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ251bV9hcmdzJyxcbiAgICAgICAgJ25hbWUnOiAnbnVtQXJncycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZnVzZWRfb3BzJyxcbiAgICAgICAgJ25hbWUnOiAnZnVzZWRPcHMnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmdbXScsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBbXVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdlcHNpbG9uJyxcbiAgICAgICAgJ25hbWUnOiAnZXBzaWxvbicsXG4gICAgICAgICd0eXBlJzogJ251bWJlcicsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiAwLjAwMDFcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAndHJhbnNwb3NlX2EnLFxuICAgICAgICAnbmFtZSc6ICd0cmFuc3Bvc2VBJyxcbiAgICAgICAgJ3R5cGUnOiAnYm9vbCcsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBmYWxzZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICd0cmFuc3Bvc2VfYicsXG4gICAgICAgICduYW1lJzogJ3RyYW5zcG9zZUInLFxuICAgICAgICAndHlwZSc6ICdib29sJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IGZhbHNlXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ1QnLFxuICAgICAgICAnbmFtZSc6ICdkdHlwZScsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJyxcbiAgICAgICAgJ25vdFN1cHBvcnRlZCc6IHRydWVcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnTWF0TXVsJyxcbiAgICAnY2F0ZWdvcnknOiAnbWF0cmljZXMnLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ2EnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAxLFxuICAgICAgICAnbmFtZSc6ICdiJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfVxuICAgIF0sXG4gICAgJ2F0dHJzJzogW1xuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3RyYW5zcG9zZV9hJyxcbiAgICAgICAgJ25hbWUnOiAndHJhbnNwb3NlQScsXG4gICAgICAgICd0eXBlJzogJ2Jvb2wnLFxuICAgICAgICAnZGVmYXVsdFZhbHVlJzogZmFsc2VcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAndHJhbnNwb3NlX2InLFxuICAgICAgICAnbmFtZSc6ICd0cmFuc3Bvc2VCJyxcbiAgICAgICAgJ3R5cGUnOiAnYm9vbCcsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBmYWxzZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUJyxcbiAgICAgICAgJ25hbWUnOiAnZHR5cGUnLFxuICAgICAgICAndHlwZSc6ICdkdHlwZScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9XG4gICAgXVxuICB9LFxuICB7XG4gICAgJ3RmT3BOYW1lJzogJ0JhdGNoTWF0TXVsJyxcbiAgICAnY2F0ZWdvcnknOiAnbWF0cmljZXMnLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ2EnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAxLFxuICAgICAgICAnbmFtZSc6ICdiJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfVxuICAgIF0sXG4gICAgJ2F0dHJzJzogW1xuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2Fkal94JyxcbiAgICAgICAgJ25hbWUnOiAndHJhbnNwb3NlQScsXG4gICAgICAgICd0eXBlJzogJ2Jvb2wnLFxuICAgICAgICAnZGVmYXVsdFZhbHVlJzogZmFsc2VcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnYWRqX3knLFxuICAgICAgICAnbmFtZSc6ICd0cmFuc3Bvc2VCJyxcbiAgICAgICAgJ3R5cGUnOiAnYm9vbCcsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBmYWxzZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUJyxcbiAgICAgICAgJ25hbWUnOiAnZHR5cGUnLFxuICAgICAgICAndHlwZSc6ICdkdHlwZScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9XG4gICAgXVxuICB9LFxuICB7XG4gICAgJ3RmT3BOYW1lJzogJ0JhdGNoTWF0TXVsVjInLFxuICAgICdjYXRlZ29yeSc6ICdtYXRyaWNlcycsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAnYScsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2InLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnYWRqX3gnLFxuICAgICAgICAnbmFtZSc6ICd0cmFuc3Bvc2VBJyxcbiAgICAgICAgJ3R5cGUnOiAnYm9vbCcsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBmYWxzZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdhZGpfeScsXG4gICAgICAgICduYW1lJzogJ3RyYW5zcG9zZUInLFxuICAgICAgICAndHlwZSc6ICdib29sJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IGZhbHNlXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ1QnLFxuICAgICAgICAnbmFtZSc6ICdkdHlwZScsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJyxcbiAgICAgICAgJ25vdFN1cHBvcnRlZCc6IHRydWVcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnVHJhbnNwb3NlJyxcbiAgICAnY2F0ZWdvcnknOiAnbWF0cmljZXMnLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ3gnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAxLFxuICAgICAgICAnbmFtZSc6ICdwZXJtJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVCcsXG4gICAgICAgICduYW1lJzogJ2R0eXBlJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdFaW5zdW0nLFxuICAgICdjYXRlZ29yeSc6ICdtYXRyaWNlcycsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ2VuZCc6IDAsXG4gICAgICAgICduYW1lJzogJ3RlbnNvcnMnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3JzJ1xuICAgICAgfVxuICAgIF0sXG4gICAgJ2F0dHJzJzogW1xuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2VxdWF0aW9uJyxcbiAgICAgICAgJ25hbWUnOiAnZXF1YXRpb24nLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ04nLFxuICAgICAgICAnbmFtZSc6ICduJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IDJcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVCcsXG4gICAgICAgICduYW1lJzogJ2R0eXBlJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnXG4gICAgICB9XG4gICAgXVxuICB9XG5dO1xuIl19
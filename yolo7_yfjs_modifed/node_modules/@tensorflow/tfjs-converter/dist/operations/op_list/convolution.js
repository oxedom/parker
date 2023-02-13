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
        'tfOpName': 'AvgPool',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'notSupported': true
            },
            {
                'tfName': 'ksize',
                'name': 'kernelSize',
                'type': 'number[]'
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
        'tfOpName': 'MaxPool',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'notSupported': true
            },
            {
                'tfName': 'ksize',
                'name': 'kernelSize',
                'type': 'number[]'
            },
            {
                'tfName': 'explicit_paddings',
                'name': 'explicitPaddings',
                'type': 'number[]',
                'defaultValue': [],
                'notSupported': true
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
        'tfOpName': 'MaxPoolWithArgmax',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'ksize',
                'name': 'kernelSize',
                'type': 'number[]'
            },
            {
                'tfName': 'include_batch_in_index',
                'name': 'includeBatchInIndex',
                'type': 'bool'
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
        'tfOpName': 'AvgPool3D',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'notSupported': true
            },
            {
                'tfName': 'ksize',
                'name': 'kernelSize',
                'type': 'number[]'
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
        'tfOpName': 'MaxPool3D',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'notSupported': true
            },
            {
                'tfName': 'ksize',
                'name': 'kernelSize',
                'type': 'number[]'
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
        'tfOpName': 'Conv1D',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'stride',
                'name': 'stride',
                'type': 'number'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'defaultValue': 'NWC'
            },
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'dilation',
                'name': 'dilation',
                'type': 'number',
                'defaultValue': 1
            }
        ]
    },
    {
        'tfOpName': 'Conv2D',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'useCudnnOnGpu',
                'name': 'useCudnnOnGpu',
                'type': 'bool'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'defaultValue': 'NHWC'
            },
            {
                'tfName': 'explicit_paddings',
                'name': 'explicitPaddings',
                'type': 'number[]',
                'defaultValue': []
            },
            {
                'tfName': 'dilations',
                'name': 'dilations',
                'type': 'number[]'
            }
        ]
    },
    {
        'tfOpName': '_FusedConv2D',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
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
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'explicit_paddings',
                'name': 'explicitPaddings',
                'type': 'number[]',
                'defaultValue': []
            },
            {
                'tfName': 'use_cudnn_on_gpu',
                'name': 'useCudnnOnGpu',
                'type': 'bool',
                'defaultValue': true
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'defaultValue': 'NHWC'
            },
            {
                'tfName': 'dilations',
                'name': 'dilations',
                'type': 'number[]',
                'defaultValue': [
                    1,
                    1,
                    1,
                    1
                ]
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
                'tfName': 'leakyrelu_alpha',
                'name': 'leakyreluAlpha',
                'type': 'number'
            }
        ]
    },
    {
        'tfOpName': 'Conv2DBackpropInput',
        'category': 'convolution',
        'inputs': [
            {
                'start': 2,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
                'type': 'tensor'
            },
            {
                'start': 0,
                'name': 'outputShape',
                'type': 'number[]'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'notSupported': true
            },
            {
                'tfName': 'explicit_paddings',
                'name': 'explicitPaddings',
                'type': 'number[]',
                'defaultValue': []
            },
            {
                'tfName': 'dilations',
                'name': 'dilations',
                'type': 'number[]',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'DepthwiseConv2d',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'input',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'defaultValue': 'NHWC'
            },
            {
                'tfName': 'explicit_paddings',
                'name': 'explicitPaddings',
                'type': 'number[]',
                'defaultValue': []
            },
            {
                'tfName': 'dilations',
                'name': 'dilations',
                'type': 'number[]'
            }
        ]
    },
    {
        'tfOpName': 'DepthwiseConv2dNative',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'input',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'defaultValue': 'NHWC'
            },
            {
                'tfName': 'explicit_paddings',
                'name': 'explicitPaddings',
                'type': 'number[]',
                'defaultValue': []
            },
            {
                'tfName': 'dilations',
                'name': 'dilations',
                'type': 'number[]'
            }
        ]
    },
    {
        'tfOpName': 'FusedDepthwiseConv2dNative',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
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
                'tfName': 'T',
                'name': 'dtype',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'defaultValue': 'NHWC'
            },
            {
                'tfName': 'dilations',
                'name': 'dilations',
                'type': 'number[]',
                'defaultValue': [
                    1,
                    1,
                    1,
                    1
                ]
            },
            {
                'tfName': 'fused_ops',
                'name': 'fusedOps',
                'type': 'string[]',
                'defaultValue': []
            },
            {
                'tfName': 'explicit_paddings',
                'name': 'explicitPaddings',
                'type': 'number[]',
                'defaultValue': []
            }
        ]
    },
    {
        'tfOpName': 'Conv3D',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            },
            {
                'tfName': 'data_format',
                'name': 'dataFormat',
                'type': 'string',
                'defaultValue': 'NHWC'
            },
            {
                'tfName': 'dilations',
                'name': 'dilations',
                'type': 'number[]'
            }
        ]
    },
    {
        'tfOpName': 'Dilation2D',
        'category': 'convolution',
        'inputs': [
            {
                'start': 0,
                'name': 'x',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'filter',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'strides',
                'name': 'strides',
                'type': 'number[]'
            },
            {
                'tfName': 'rates',
                'name': 'dilations',
                'type': 'number[]'
            },
            {
                'tfName': 'padding',
                'name': 'pad',
                'type': 'string'
            }
        ]
    }
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udm9sdXRpb24uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvbnZlcnRlci9zcmMvb3BlcmF0aW9ucy9vcF9saXN0L2NvbnZvbHV0aW9uLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUNBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUlILE1BQU0sQ0FBQyxNQUFNLElBQUksR0FBZTtJQUM5QjtRQUNFLFVBQVUsRUFBRSxTQUFTO1FBQ3JCLFVBQVUsRUFBRSxhQUFhO1FBQ3pCLFFBQVEsRUFBRTtZQUNSO2dCQUNFLE9BQU8sRUFBRSxDQUFDO2dCQUNWLE1BQU0sRUFBRSxHQUFHO2dCQUNYLE1BQU0sRUFBRSxRQUFRO2FBQ2pCO1NBQ0Y7UUFDRCxPQUFPLEVBQUU7WUFDUDtnQkFDRSxRQUFRLEVBQUUsU0FBUztnQkFDbkIsTUFBTSxFQUFFLFNBQVM7Z0JBQ2pCLE1BQU0sRUFBRSxVQUFVO2FBQ25CO1lBQ0Q7Z0JBQ0UsUUFBUSxFQUFFLFNBQVM7Z0JBQ25CLE1BQU0sRUFBRSxLQUFLO2dCQUNiLE1BQU0sRUFBRSxRQUFRO2FBQ2pCO1lBQ0Q7Z0JBQ0UsUUFBUSxFQUFFLGFBQWE7Z0JBQ3ZCLE1BQU0sRUFBRSxZQUFZO2dCQUNwQixNQUFNLEVBQUUsUUFBUTtnQkFDaEIsY0FBYyxFQUFFLElBQUk7YUFDckI7WUFDRDtnQkFDRSxRQUFRLEVBQUUsT0FBTztnQkFDakIsTUFBTSxFQUFFLFlBQVk7Z0JBQ3BCLE1BQU0sRUFBRSxVQUFVO2FBQ25CO1lBQ0Q7Z0JBQ0UsUUFBUSxFQUFFLEdBQUc7Z0JBQ2IsTUFBTSxFQUFFLE9BQU87Z0JBQ2YsTUFBTSxFQUFFLE9BQU87Z0JBQ2YsY0FBYyxFQUFFLElBQUk7YUFDckI7U0FDRjtLQUNGO0lBQ0Q7UUFDRSxVQUFVLEVBQUUsU0FBUztRQUNyQixVQUFVLEVBQUUsYUFBYTtRQUN6QixRQUFRLEVBQUU7WUFDUjtnQkFDRSxPQUFPLEVBQUUsQ0FBQztnQkFDVixN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QUFFLFVBQVU7YUFDbkI7U0FDRjtLQUNGO0lBQ0Q7UUFDRSxVQUFVLEVBQUUsWUFBWTtRQUN4QixVQUFVLEVBQUUsYUFBYTtRQUN6QixRQUFRLEVBQUU7WUFDUjtnQkFDRSxPQUFPLEVBQUUsQ0FBQztnQkFDVixNQUFNLEVBQUUsR0FBRztnQkFDWCxNQUFNLEVBQUUsUUFBUTthQUNqQjtZQUNEO2dCQUNFLE9BQU8sRUFBRSxDQUFDO2dCQUNWLE1BQU0sRUFBRSxRQUFRO2dCQUNoQixNQUFNLEVBQUUsUUFBUTthQUNqQjtTQUNGO1FBQ0QsT0FBTyxFQUFFO1lBQ1A7Z0JBQ0UsUUFBUSxFQUFFLFNBQVM7Z0JBQ25CLE1BQU0sRUFBRSxTQUFTO2dCQUNqQixNQUFNLEVBQUUsVUFBVTthQUNuQjtZQUNEO2dCQUNFLFFBQVEsRUFBRSxPQUFPO2dCQUNqQixNQUFNLEVBQUUsV0FBVztnQkFDbkIsTUFBTSxFQUFFLFVBQVU7YUFDbkI7WUFDRDtnQkFDRSxRQUFRLEVBQUUsU0FBUztnQkFDbkIsTUFBTSxFQUFFLEtBQUs7Z0JBQ2IsTUFBTSxFQUFFLFFBQVE7YUFDakI7U0FDRjtLQUNGO0NBQ0YsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIlxuLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge09wTWFwcGVyfSBmcm9tICcuLi90eXBlcyc7XG5cbmV4cG9ydCBjb25zdCBqc29uOiBPcE1hcHBlcltdID0gW1xuICB7XG4gICAgJ3RmT3BOYW1lJzogJ0F2Z1Bvb2wnLFxuICAgICdjYXRlZ29yeSc6ICdjb252b2x1dGlvbicsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAneCcsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ25hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdwYWRkaW5nJyxcbiAgICAgICAgJ25hbWUnOiAncGFkJyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkYXRhX2Zvcm1hdCcsXG4gICAgICAgICduYW1lJzogJ2RhdGFGb3JtYXQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdrc2l6ZScsXG4gICAgICAgICduYW1lJzogJ2tlcm5lbFNpemUnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVCcsXG4gICAgICAgICduYW1lJzogJ2R0eXBlJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdNYXhQb29sJyxcbiAgICAnY2F0ZWdvcnknOiAnY29udm9sdXRpb24nLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ3gnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICduYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAncGFkZGluZycsXG4gICAgICAgICduYW1lJzogJ3BhZCcsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZydcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZGF0YV9mb3JtYXQnLFxuICAgICAgICAnbmFtZSc6ICdkYXRhRm9ybWF0JyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJyxcbiAgICAgICAgJ25vdFN1cHBvcnRlZCc6IHRydWVcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAna3NpemUnLFxuICAgICAgICAnbmFtZSc6ICdrZXJuZWxTaXplJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2V4cGxpY2l0X3BhZGRpbmdzJyxcbiAgICAgICAgJ25hbWUnOiAnZXhwbGljaXRQYWRkaW5ncycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IFtdLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUJyxcbiAgICAgICAgJ25hbWUnOiAnZHR5cGUnLFxuICAgICAgICAndHlwZSc6ICdkdHlwZScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9XG4gICAgXVxuICB9LFxuICB7XG4gICAgJ3RmT3BOYW1lJzogJ01heFBvb2xXaXRoQXJnbWF4JyxcbiAgICAnY2F0ZWdvcnknOiAnY29udm9sdXRpb24nLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ3gnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICduYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAncGFkZGluZycsXG4gICAgICAgICduYW1lJzogJ3BhZCcsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZydcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAna3NpemUnLFxuICAgICAgICAnbmFtZSc6ICdrZXJuZWxTaXplJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2luY2x1ZGVfYmF0Y2hfaW5faW5kZXgnLFxuICAgICAgICAnbmFtZSc6ICdpbmNsdWRlQmF0Y2hJbkluZGV4JyxcbiAgICAgICAgJ3R5cGUnOiAnYm9vbCdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVCcsXG4gICAgICAgICduYW1lJzogJ2R0eXBlJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdBdmdQb29sM0QnLFxuICAgICdjYXRlZ29yeSc6ICdjb252b2x1dGlvbicsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAneCcsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ25hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdwYWRkaW5nJyxcbiAgICAgICAgJ25hbWUnOiAncGFkJyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkYXRhX2Zvcm1hdCcsXG4gICAgICAgICduYW1lJzogJ2RhdGFGb3JtYXQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdrc2l6ZScsXG4gICAgICAgICduYW1lJzogJ2tlcm5lbFNpemUnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVCcsXG4gICAgICAgICduYW1lJzogJ2R0eXBlJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdNYXhQb29sM0QnLFxuICAgICdjYXRlZ29yeSc6ICdjb252b2x1dGlvbicsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAneCcsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ25hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdwYWRkaW5nJyxcbiAgICAgICAgJ25hbWUnOiAncGFkJyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkYXRhX2Zvcm1hdCcsXG4gICAgICAgICduYW1lJzogJ2RhdGFGb3JtYXQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdrc2l6ZScsXG4gICAgICAgICduYW1lJzogJ2tlcm5lbFNpemUnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVCcsXG4gICAgICAgICduYW1lJzogJ2R0eXBlJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdDb252MUQnLFxuICAgICdjYXRlZ29yeSc6ICdjb252b2x1dGlvbicsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAneCcsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2ZpbHRlcicsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdzdHJpZGUnLFxuICAgICAgICAnbmFtZSc6ICdzdHJpZGUnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXInXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3BhZGRpbmcnLFxuICAgICAgICAnbmFtZSc6ICdwYWQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2RhdGFfZm9ybWF0JyxcbiAgICAgICAgJ25hbWUnOiAnZGF0YUZvcm1hdCcsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZycsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiAnTldDJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUJyxcbiAgICAgICAgJ25hbWUnOiAnZHR5cGUnLFxuICAgICAgICAndHlwZSc6ICdkdHlwZScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2RpbGF0aW9uJyxcbiAgICAgICAgJ25hbWUnOiAnZGlsYXRpb24nLFxuICAgICAgICAndHlwZSc6ICdudW1iZXInLFxuICAgICAgICAnZGVmYXVsdFZhbHVlJzogMVxuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdDb252MkQnLFxuICAgICdjYXRlZ29yeSc6ICdjb252b2x1dGlvbicsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAneCcsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2ZpbHRlcicsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUJyxcbiAgICAgICAgJ25hbWUnOiAnZHR5cGUnLFxuICAgICAgICAndHlwZSc6ICdkdHlwZScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAnbmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3BhZGRpbmcnLFxuICAgICAgICAnbmFtZSc6ICdwYWQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3VzZUN1ZG5uT25HcHUnLFxuICAgICAgICAnbmFtZSc6ICd1c2VDdWRubk9uR3B1JyxcbiAgICAgICAgJ3R5cGUnOiAnYm9vbCdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZGF0YV9mb3JtYXQnLFxuICAgICAgICAnbmFtZSc6ICdkYXRhRm9ybWF0JyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6ICdOSFdDJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdleHBsaWNpdF9wYWRkaW5ncycsXG4gICAgICAgICduYW1lJzogJ2V4cGxpY2l0UGFkZGluZ3MnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXScsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBbXVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAnbmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnX0Z1c2VkQ29udjJEJyxcbiAgICAnY2F0ZWdvcnknOiAnY29udm9sdXRpb24nLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ3gnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAxLFxuICAgICAgICAnbmFtZSc6ICdmaWx0ZXInLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAyLFxuICAgICAgICAnZW5kJzogMCxcbiAgICAgICAgJ25hbWUnOiAnYXJncycsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcnMnXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnbnVtX2FyZ3MnLFxuICAgICAgICAnbmFtZSc6ICdudW1BcmdzJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUJyxcbiAgICAgICAgJ25hbWUnOiAnZHR5cGUnLFxuICAgICAgICAndHlwZSc6ICdkdHlwZScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAnbmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3BhZGRpbmcnLFxuICAgICAgICAnbmFtZSc6ICdwYWQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2V4cGxpY2l0X3BhZGRpbmdzJyxcbiAgICAgICAgJ25hbWUnOiAnZXhwbGljaXRQYWRkaW5ncycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IFtdXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3VzZV9jdWRubl9vbl9ncHUnLFxuICAgICAgICAnbmFtZSc6ICd1c2VDdWRubk9uR3B1JyxcbiAgICAgICAgJ3R5cGUnOiAnYm9vbCcsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiB0cnVlXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2RhdGFfZm9ybWF0JyxcbiAgICAgICAgJ25hbWUnOiAnZGF0YUZvcm1hdCcsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZycsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiAnTkhXQydcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZGlsYXRpb25zJyxcbiAgICAgICAgJ25hbWUnOiAnZGlsYXRpb25zJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nLFxuICAgICAgICAnZGVmYXVsdFZhbHVlJzogW1xuICAgICAgICAgIDEsXG4gICAgICAgICAgMSxcbiAgICAgICAgICAxLFxuICAgICAgICAgIDFcbiAgICAgICAgXVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdmdXNlZF9vcHMnLFxuICAgICAgICAnbmFtZSc6ICdmdXNlZE9wcycsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZ1tdJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IFtdXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2Vwc2lsb24nLFxuICAgICAgICAnbmFtZSc6ICdlcHNpbG9uJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IDAuMDAwMVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdsZWFreXJlbHVfYWxwaGEnLFxuICAgICAgICAnbmFtZSc6ICdsZWFreXJlbHVBbHBoYScsXG4gICAgICAgICd0eXBlJzogJ251bWJlcidcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnQ29udjJEQmFja3Byb3BJbnB1dCcsXG4gICAgJ2NhdGVnb3J5JzogJ2NvbnZvbHV0aW9uJyxcbiAgICAnaW5wdXRzJzogW1xuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAyLFxuICAgICAgICAnbmFtZSc6ICd4JyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMSxcbiAgICAgICAgJ25hbWUnOiAnZmlsdGVyJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAnb3V0cHV0U2hhcGUnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ25hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdwYWRkaW5nJyxcbiAgICAgICAgJ25hbWUnOiAncGFkJyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkYXRhX2Zvcm1hdCcsXG4gICAgICAgICduYW1lJzogJ2RhdGFGb3JtYXQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdleHBsaWNpdF9wYWRkaW5ncycsXG4gICAgICAgICduYW1lJzogJ2V4cGxpY2l0UGFkZGluZ3MnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXScsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBbXVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAnbmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9XG4gICAgXVxuICB9LFxuICB7XG4gICAgJ3RmT3BOYW1lJzogJ0RlcHRod2lzZUNvbnYyZCcsXG4gICAgJ2NhdGVnb3J5JzogJ2NvbnZvbHV0aW9uJyxcbiAgICAnaW5wdXRzJzogW1xuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAwLFxuICAgICAgICAnbmFtZSc6ICdpbnB1dCcsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2ZpbHRlcicsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ25hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdwYWRkaW5nJyxcbiAgICAgICAgJ25hbWUnOiAncGFkJyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkYXRhX2Zvcm1hdCcsXG4gICAgICAgICduYW1lJzogJ2RhdGFGb3JtYXQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnLFxuICAgICAgICAnZGVmYXVsdFZhbHVlJzogJ05IV0MnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2V4cGxpY2l0X3BhZGRpbmdzJyxcbiAgICAgICAgJ25hbWUnOiAnZXhwbGljaXRQYWRkaW5ncycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6IFtdXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2RpbGF0aW9ucycsXG4gICAgICAgICduYW1lJzogJ2RpbGF0aW9ucycsXG4gICAgICAgICd0eXBlJzogJ251bWJlcltdJ1xuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdEZXB0aHdpc2VDb252MmROYXRpdmUnLFxuICAgICdjYXRlZ29yeSc6ICdjb252b2x1dGlvbicsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAnaW5wdXQnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAxLFxuICAgICAgICAnbmFtZSc6ICdmaWx0ZXInLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICduYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAncGFkZGluZycsXG4gICAgICAgICduYW1lJzogJ3BhZCcsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZydcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZGF0YV9mb3JtYXQnLFxuICAgICAgICAnbmFtZSc6ICdkYXRhRm9ybWF0JyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6ICdOSFdDJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdleHBsaWNpdF9wYWRkaW5ncycsXG4gICAgICAgICduYW1lJzogJ2V4cGxpY2l0UGFkZGluZ3MnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXScsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBbXVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAnbmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnRnVzZWREZXB0aHdpc2VDb252MmROYXRpdmUnLFxuICAgICdjYXRlZ29yeSc6ICdjb252b2x1dGlvbicsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAneCcsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2ZpbHRlcicsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDIsXG4gICAgICAgICdlbmQnOiAwLFxuICAgICAgICAnbmFtZSc6ICdhcmdzJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29ycydcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdudW1fYXJncycsXG4gICAgICAgICduYW1lJzogJ251bUFyZ3MnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXInXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ1QnLFxuICAgICAgICAnbmFtZSc6ICdkdHlwZScsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJyxcbiAgICAgICAgJ25vdFN1cHBvcnRlZCc6IHRydWVcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICduYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAncGFkZGluZycsXG4gICAgICAgICduYW1lJzogJ3BhZCcsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZydcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZGF0YV9mb3JtYXQnLFxuICAgICAgICAnbmFtZSc6ICdkYXRhRm9ybWF0JyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6ICdOSFdDJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAnbmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXScsXG4gICAgICAgICdkZWZhdWx0VmFsdWUnOiBbXG4gICAgICAgICAgMSxcbiAgICAgICAgICAxLFxuICAgICAgICAgIDEsXG4gICAgICAgICAgMVxuICAgICAgICBdXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2Z1c2VkX29wcycsXG4gICAgICAgICduYW1lJzogJ2Z1c2VkT3BzJyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nW10nLFxuICAgICAgICAnZGVmYXVsdFZhbHVlJzogW11cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZXhwbGljaXRfcGFkZGluZ3MnLFxuICAgICAgICAnbmFtZSc6ICdleHBsaWNpdFBhZGRpbmdzJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nLFxuICAgICAgICAnZGVmYXVsdFZhbHVlJzogW11cbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnQ29udjNEJyxcbiAgICAnY2F0ZWdvcnknOiAnY29udm9sdXRpb24nLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ3gnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAxLFxuICAgICAgICAnbmFtZSc6ICdmaWx0ZXInLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnc3RyaWRlcycsXG4gICAgICAgICduYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAncGFkZGluZycsXG4gICAgICAgICduYW1lJzogJ3BhZCcsXG4gICAgICAgICd0eXBlJzogJ3N0cmluZydcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnZGF0YV9mb3JtYXQnLFxuICAgICAgICAnbmFtZSc6ICdkYXRhRm9ybWF0JyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJyxcbiAgICAgICAgJ2RlZmF1bHRWYWx1ZSc6ICdOSFdDJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAnbmFtZSc6ICdkaWxhdGlvbnMnLFxuICAgICAgICAndHlwZSc6ICdudW1iZXJbXSdcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnRGlsYXRpb24yRCcsXG4gICAgJ2NhdGVnb3J5JzogJ2NvbnZvbHV0aW9uJyxcbiAgICAnaW5wdXRzJzogW1xuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAwLFxuICAgICAgICAnbmFtZSc6ICd4JyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMSxcbiAgICAgICAgJ25hbWUnOiAnZmlsdGVyJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfVxuICAgIF0sXG4gICAgJ2F0dHJzJzogW1xuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3N0cmlkZXMnLFxuICAgICAgICAnbmFtZSc6ICdzdHJpZGVzJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3JhdGVzJyxcbiAgICAgICAgJ25hbWUnOiAnZGlsYXRpb25zJyxcbiAgICAgICAgJ3R5cGUnOiAnbnVtYmVyW10nXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3BhZGRpbmcnLFxuICAgICAgICAnbmFtZSc6ICdwYWQnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnXG4gICAgICB9XG4gICAgXVxuICB9XG5dO1xuIl19
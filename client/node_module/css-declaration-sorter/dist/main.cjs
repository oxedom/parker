'use strict';

const stableSort = require('./stable-sort.cjs');

Object.defineProperty(exports, '__esModule', { value: true });

const shorthandData = {
  'animation': [
    'animation-name',
    'animation-duration',
    'animation-timing-function',
    'animation-delay',
    'animation-iteration-count',
    'animation-direction',
    'animation-fill-mode',
    'animation-play-state',
  ],
  'background': [
    'background-image',
    'background-size',
    'background-position',
    'background-repeat',
    'background-origin',
    'background-clip',
    'background-attachment',
    'background-color',
  ],
  'columns': [
    'column-width',
    'column-count',
  ],
  'column-rule': [
    'column-rule-width',
    'column-rule-style',
    'column-rule-color',
  ],
  'flex': [
    'flex-grow',
    'flex-shrink',
    'flex-basis',
  ],
  'flex-flow': [
    'flex-direction',
    'flex-wrap',
  ],
  'font': [
    'font-style',
    'font-variant',
    'font-weight',
    'font-stretch',
    'font-size',
    'font-family',
    'line-height',
  ],
  'grid': [
    'grid-template-rows',
    'grid-template-columns',
    'grid-template-areas',
    'grid-auto-rows',
    'grid-auto-columns',
    'grid-auto-flow',
    'column-gap',
    'row-gap',
  ],
  'grid-area': [
    'grid-row-start',
    'grid-column-start',
    'grid-row-end',
    'grid-column-end',
  ],
  'grid-column': [
    'grid-column-start',
    'grid-column-end',
  ],
  'grid-row': [
    'grid-row-start',
    'grid-row-end',
  ],
  'grid-template': [
    'grid-template-columns',
    'grid-template-rows',
    'grid-template-areas',
  ],
  'list-style': [
    'list-style-type',
    'list-style-position',
    'list-style-image',
  ],
  'padding': [
    'padding-block',
    'padding-block-start',
    'padding-block-end',
    'padding-inline',
    'padding-inline-start',
    'padding-inline-end',
    'padding-top',
    'padding-right',
    'padding-bottom',
    'padding-left',
  ],
  'padding-block': [
    'padding-block-start',
    'padding-block-end',
    'padding-top',
    'padding-right',
    'padding-bottom',
    'padding-left',
  ],
  'padding-block-start': [
    'padding-top',
    'padding-right',
    'padding-left',
  ],
  'padding-block-end': [
    'padding-right',
    'padding-bottom',
    'padding-left',
  ],
  'padding-inline': [
    'padding-inline-start',
    'padding-inline-end',
    'padding-top',
    'padding-right',
    'padding-bottom',
    'padding-left',
  ],
  'padding-inline-start': [
    'padding-top',
    'padding-right',
    'padding-left',
  ],
  'padding-inline-end': [
    'padding-right',
    'padding-bottom',
    'padding-left',
  ],
  'margin': [
    'margin-block',
    'margin-block-start',
    'margin-block-end',
    'margin-inline',
    'margin-inline-start',
    'margin-inline-end',
    'margin-top',
    'margin-right',
    'margin-bottom',
    'margin-left',
  ],
  'margin-block': [
    'margin-block-start',
    'margin-block-end',
    'margin-top',
    'margin-right',
    'margin-bottom',
    'margin-left',
  ],
  'margin-inline': [
    'margin-inline-start',
    'margin-inline-end',
    'margin-top',
    'margin-right',
    'margin-bottom',
    'margin-left',
  ],
  'margin-inline-start': [
    'margin-top',
    'margin-right',
    'margin-bottom',
    'margin-left',
  ],
  'margin-inline-end': [
    'margin-top',
    'margin-right',
    'margin-bottom',
    'margin-left',
  ],
  'border': [
    'border-top',
    'border-right',
    'border-bottom',
    'border-left',
    'border-width',
    'border-style',
    'border-color',
    'border-top-width',
    'border-right-width',
    'border-bottom-width',
    'border-left-width',
    'border-inline-start-width',
    'border-inline-end-width',
    'border-block-start-width',
    'border-block-end-width',
    'border-top-style',
    'border-right-style',
    'border-bottom-style',
    'border-left-style',
    'border-inline-start-style',
    'border-inline-end-style',
    'border-block-start-style',
    'border-block-end-style',
    'border-top-color',
    'border-right-color',
    'border-bottom-color',
    'border-left-color',
    'border-inline-start-color',
    'border-inline-end-color',
    'border-block-start-color',
    'border-block-end-color',
    'border-block',
    'border-block-start',
    'border-block-end',
    'border-block-width',
    'border-block-style',
    'border-block-color',
    'border-inline',
    'border-inline-start',
    'border-inline-end',
    'border-inline-width',
    'border-inline-style',
    'border-inline-color',
  ],
  'border-top': [
    'border-width',
    'border-style',
    'border-color',
    'border-top-width',
    'border-top-style',
    'border-top-color',
  ],
  'border-right': [
    'border-width',
    'border-style',
    'border-color',
    'border-right-width',
    'border-right-style',
    'border-right-color',
  ],
  'border-bottom': [
    'border-width',
    'border-style',
    'border-color',
    'border-bottom-width',
    'border-bottom-style',
    'border-bottom-color',
  ],
  'border-left': [
    'border-width',
    'border-style',
    'border-color',
    'border-left-width',
    'border-left-style',
    'border-left-color',
  ],
  'border-color': [
    'border-top-color',
    'border-bottom-color',
    'border-left-color',
    'border-right-color',
    'border-inline-start-color',
    'border-inline-end-color',
    'border-block-start-color',
    'border-block-end-color',
  ],
  'border-width': [
    'border-top-width',
    'border-bottom-width',
    'border-left-width',
    'border-right-width',
    'border-inline-start-width',
    'border-inline-end-width',
    'border-block-start-width',
    'border-block-end-width',
  ],
  'border-style': [
    'border-top-style',
    'border-bottom-style',
    'border-left-style',
    'border-right-style',
    'border-inline-start-style',
    'border-inline-end-style',
    'border-block-start-style',
    'border-block-end-style',
  ],
  'border-radius': [
    'border-top-right-radius',
    'border-top-left-radius',
    'border-bottom-right-radius',
    'border-bottom-left-radius',
  ],
  'border-block': [
    'border-block-start',
    'border-block-end',
    'border-block-width',
    'border-width',
    'border-block-style',
    'border-style',
    'border-block-color',
    'border-color',
  ],
  'border-block-start': [
    'border-block-start-width',
    'border-width',
    'border-block-start-style',
    'border-style',
    'border-block-start-color',
    'border-color',
  ],
  'border-block-end': [
    'border-block-end-width',
    'border-width',
    'border-block-end-style',
    'border-style',
    'border-block-end-color',
    'border-color',
  ],
  'border-inline': [
    'border-inline-start',
    'border-inline-end',
    'border-inline-width',
    'border-width',
    'border-inline-style',
    'border-style',
    'border-inline-color',
    'border-color',
  ],
  'border-inline-start': [
    'border-inline-start-width',
    'border-width',
    'border-inline-start-style',
    'border-style',
    'border-inline-start-color',
    'border-color',
  ],
  'border-inline-end': [
    'border-inline-end-width',
    'border-width',
    'border-inline-end-style',
    'border-style',
    'border-inline-end-color',
    'border-color',
  ],
  'border-image': [
    'border-image-source',
    'border-image-slice',
    'border-image-width',
    'border-image-outset',
    'border-image-repeat',
  ],
  'mask': [
    'mask-image',
    'mask-mode',
    'mask-position',
    'mask-size',
    'mask-repeat',
    'mask-origin',
    'mask-clip',
    'mask-composite',
  ],
  'inline-size': [
    'width',
    'height',
  ],
  'block-size': [
    'width',
    'height',
  ],
  'max-inline-size': [
    'max-width',
    'max-height',
  ],
  'max-block-size': [
    'max-width',
    'max-height',
  ],
  'inset': [
    'inset-block',
    'inset-block-start',
    'inset-block-end',
    'inset-inline',
    'inset-inline-start',
    'inset-inline-end',
    'top',
    'right',
    'bottom',
    'left',
  ],
  'inset-block': [
    'inset-block-start',
    'inset-block-end',
    'top',
    'right',
    'bottom',
    'left',
  ],
  'inset-inline': [
    'inset-inline-start',
    'inset-inline-end',
    'top',
    'right',
    'bottom',
    'left',
  ],
  'outline': [
    'outline-color',
    'outline-style',
    'outline-width',
  ],
  'overflow': [
    'overflow-x',
    'overflow-y',
  ],
  'place-content': [
    'align-content',
    'justify-content',
  ],
  'place-items': [
    'align-items',
    'justify-items',
  ],
  'place-self': [
    'align-self',
    'justify-self',
  ],
  'text-decoration': [
    'text-decoration-color',
    'text-decoration-style',
    'text-decoration-line',
  ],
  'transition': [
    'transition-delay',
    'transition-duration',
    'transition-property',
    'transition-timing-function',
  ],
  'text-emphasis': [
    'text-emphasis-style',
    'text-emphasis-color',
  ],
};

function __variableDynamicImportRuntime0__(path) {
  switch (path) {
    case '../orders/alphabetical.mjs': return Promise.resolve().then(function () { return alphabetical; });
    case '../orders/concentric-css.mjs': return Promise.resolve().then(function () { return concentricCss; });
    case '../orders/smacss.mjs': return Promise.resolve().then(function () { return smacss; });
    default: return new Promise(function(resolve, reject) {
      (typeof queueMicrotask === 'function' ? queueMicrotask : setTimeout)(
        reject.bind(null, new Error("Unknown variable dynamic import: " + path))
      );
    })
   }
 }

const builtInOrders = [
  'alphabetical',
  'concentric-css',
  'smacss',
];

const cssDeclarationSorter = ({ order = 'alphabetical', keepOverrides = false } = {}) => ({
  postcssPlugin: 'css-declaration-sorter',
  OnceExit (css) {
    let withKeepOverrides = comparator => comparator;
    if (keepOverrides) {
      withKeepOverrides = withOverridesComparator(shorthandData);
    }

    if (typeof order === 'function') {
      return processCss({ css, comparator: withKeepOverrides(order) });
    }

    if (!builtInOrders.includes(order))
      return Promise.reject(
        Error([
          `Invalid built-in order '${order}' provided.`,
          `Available built-in orders are: ${builtInOrders}`,
        ].join('\n'))
      );

    return __variableDynamicImportRuntime0__(`../orders/${order}.mjs`)
      .then(({ properties }) => processCss({
        css,
        comparator: withKeepOverrides(orderComparator(properties)),
      }));
  },
});

cssDeclarationSorter.postcss = true;

function processCss ({ css, comparator }) {
  const comments = [];
  const rulesCache = [];

  css.walk(node => {
    const nodes = node.nodes;
    const type = node.type;

    if (type === 'comment') {
      // Don't do anything to root comments or the last newline comment
      const isNewlineNode = node.raws.before && node.raws.before.includes('\n');
      const lastNewlineNode = isNewlineNode && !node.next();
      const onlyNode = !node.prev() && !node.next() || !node.parent;

      if (lastNewlineNode || onlyNode || node.parent.type === 'root') {
        return;
      }

      if (isNewlineNode) {
        const pairedNode = node.next() || node.prev();
        if (pairedNode) {
          comments.unshift({
            'comment': node,
            'pairedNode': pairedNode,
            'insertPosition': node.next() ? 'Before' : 'After',
          });
          node.remove();
        }
      } else {
        const pairedNode = node.prev() || node.next();
        if (pairedNode) {
          comments.push({
            'comment': node,
            'pairedNode': pairedNode,
            'insertPosition': 'After',
          });
          node.remove();
        }
      }
      return;
    }

    // Add rule-like nodes to a cache so that we can remove all
    // comment nodes before we start sorting.
    const isRule = type === 'rule' || type === 'atrule';
    if (isRule && nodes && nodes.length > 1) {
      rulesCache.push(nodes);
    }
  });

  // Perform a sort once all comment nodes are removed
  rulesCache.forEach(nodes => {
    sortCssDeclarations({ nodes, comparator });
  });

  // Add comments back to the nodes they are paired with
  comments.forEach(node => {
    const pairedNode = node.pairedNode;
    node.comment.remove();
    pairedNode.parent && pairedNode.parent['insert' + node.insertPosition](pairedNode, node.comment);
  });
}

function sortCssDeclarations ({ nodes, comparator }) {
  stableSort(nodes,(a, b) => {
    if (a.type === 'decl' && b.type === 'decl') {
      return comparator(a.prop, b.prop);
    } else {
      return compareDifferentType(a, b);
    }
  });
}

function withOverridesComparator (shorthandData) {
  return function (comparator) {
    return function (a, b) {
      a = removeVendorPrefix(a);
      b = removeVendorPrefix(b);

      if (shorthandData[a] && shorthandData[a].includes(b)) return 0;
      if (shorthandData[b] && shorthandData[b].includes(a)) return 0;

      return comparator(a, b);
    };
  };
}

function orderComparator (order) {
  return function (a, b) {
    return order.indexOf(a) - order.indexOf(b);
  };
}

function compareDifferentType (a, b) {
  if (b.type === 'atrule') {
    return 0;
  }

  return a.type === 'decl' ? -1 : b.type === 'decl' ? 1 : 0;
}

function removeVendorPrefix (property) {
  return property.replace(/^-\w+-/, '');
}

const properties$2 = [
  "all",
  "-webkit-line-clamp",
  "-webkit-text-fill-color",
  "-webkit-text-stroke",
  "-webkit-text-stroke-color",
  "-webkit-text-stroke-width",
  "accent-color",
  "align-content",
  "align-items",
  "align-self",
  "animation",
  "animation-delay",
  "animation-direction",
  "animation-duration",
  "animation-fill-mode",
  "animation-iteration-count",
  "animation-name",
  "animation-play-state",
  "animation-timing-function",
  "appearance",
  "ascent-override",
  "aspect-ratio",
  "backdrop-filter",
  "backface-visibility",
  "background",
  "background-attachment",
  "background-blend-mode",
  "background-clip",
  "background-color",
  "background-image",
  "background-origin",
  "background-position",
  "background-position-x",
  "background-position-y",
  "background-repeat",
  "background-size",
  "block-size",
  "border",
  "border-block",
  "border-block-color",
  "border-block-end",
  "border-block-end-color",
  "border-block-end-style",
  "border-block-end-width",
  "border-block-start",
  "border-block-start-color",
  "border-block-start-style",
  "border-block-start-width",
  "border-block-style",
  "border-block-width",
  "border-bottom",
  "border-bottom-color",
  "border-bottom-left-radius",
  "border-bottom-right-radius",
  "border-bottom-style",
  "border-bottom-width",
  "border-collapse",
  "border-color",
  "border-end-end-radius",
  "border-end-start-radius",
  "border-image",
  "border-image-outset",
  "border-image-repeat",
  "border-image-slice",
  "border-image-source",
  "border-image-width",
  "border-inline",
  "border-inline-color",
  "border-inline-end",
  "border-inline-end-color",
  "border-inline-end-style",
  "border-inline-end-width",
  "border-inline-start",
  "border-inline-start-color",
  "border-inline-start-style",
  "border-inline-start-width",
  "border-inline-style",
  "border-inline-width",
  "border-left",
  "border-left-color",
  "border-left-style",
  "border-left-width",
  "border-radius",
  "border-right",
  "border-right-color",
  "border-right-style",
  "border-right-width",
  "border-spacing",
  "border-start-end-radius",
  "border-start-start-radius",
  "border-style",
  "border-top",
  "border-top-color",
  "border-top-left-radius",
  "border-top-right-radius",
  "border-top-style",
  "border-top-width",
  "border-width",
  "bottom",
  "box-decoration-break",
  "box-shadow",
  "box-sizing",
  "break-after",
  "break-before",
  "break-inside",
  "caption-side",
  "caret-color",
  "clear",
  "clip-path",
  "color",
  "color-scheme",
  "column-count",
  "column-fill",
  "column-gap",
  "column-rule",
  "column-rule-color",
  "column-rule-style",
  "column-rule-width",
  "column-span",
  "column-width",
  "columns",
  "contain",
  "content",
  "content-visibility",
  "counter-increment",
  "counter-reset",
  "counter-set",
  "cursor",
  "descent-override",
  "direction",
  "display",
  "empty-cells",
  "filter",
  "flex",
  "flex-basis",
  "flex-direction",
  "flex-flow",
  "flex-grow",
  "flex-shrink",
  "flex-wrap",
  "float",
  "font",
  "font-display",
  "font-family",
  "font-kerning",
  "font-language-override",
  "font-optical-sizing",
  "font-size",
  "font-size-adjust",
  "font-stretch",
  "font-style",
  "font-synthesis",
  "font-variant",
  "font-variant-alternates",
  "font-variant-caps",
  "font-variant-east-asian",
  "font-variant-ligatures",
  "font-variant-numeric",
  "font-variant-position",
  "font-variation-settings",
  "font-weight",
  "forced-color-adjust",
  "gap",
  "grid",
  "grid-area",
  "grid-auto-columns",
  "grid-auto-flow",
  "grid-auto-rows",
  "grid-column",
  "grid-column-end",
  "grid-column-start",
  "grid-row",
  "grid-row-end",
  "grid-row-start",
  "grid-template",
  "grid-template-areas",
  "grid-template-columns",
  "grid-template-rows",
  "hanging-punctuation",
  "height",
  "hyphenate-character",
  "hyphens",
  "image-orientation",
  "image-rendering",
  "inline-size",
  "inset",
  "inset-block",
  "inset-block-end",
  "inset-block-start",
  "inset-inline",
  "inset-inline-end",
  "inset-inline-start",
  "isolation",
  "justify-content",
  "justify-items",
  "justify-self",
  "left",
  "letter-spacing",
  "line-break",
  "line-gap-override",
  "line-height",
  "list-style",
  "list-style-image",
  "list-style-position",
  "list-style-type",
  "margin",
  "margin-block",
  "margin-block-end",
  "margin-block-start",
  "margin-bottom",
  "margin-inline",
  "margin-inline-end",
  "margin-inline-start",
  "margin-left",
  "margin-right",
  "margin-top",
  "mask",
  "mask-border",
  "mask-border-outset",
  "mask-border-repeat",
  "mask-border-slice",
  "mask-border-source",
  "mask-border-width",
  "mask-clip",
  "mask-composite",
  "mask-image",
  "mask-mode",
  "mask-origin",
  "mask-position",
  "mask-repeat",
  "mask-size",
  "mask-type",
  "max-block-size",
  "max-height",
  "max-inline-size",
  "max-width",
  "min-block-size",
  "min-height",
  "min-inline-size",
  "min-width",
  "mix-blend-mode",
  "object-fit",
  "object-position",
  "offset",
  "offset-anchor",
  "offset-distance",
  "offset-path",
  "offset-rotate",
  "opacity",
  "order",
  "orphans",
  "outline",
  "outline-color",
  "outline-offset",
  "outline-style",
  "outline-width",
  "overflow",
  "overflow-anchor",
  "overflow-block",
  "overflow-inline",
  "overflow-wrap",
  "overflow-x",
  "overflow-y",
  "overscroll-behavior",
  "overscroll-behavior-block",
  "overscroll-behavior-inline",
  "overscroll-behavior-x",
  "overscroll-behavior-y",
  "padding",
  "padding-block",
  "padding-block-end",
  "padding-block-start",
  "padding-bottom",
  "padding-inline",
  "padding-inline-end",
  "padding-inline-start",
  "padding-left",
  "padding-right",
  "padding-top",
  "page",
  "page-break-after",
  "page-break-before",
  "page-break-inside",
  "paint-order",
  "perspective",
  "perspective-origin",
  "place-content",
  "place-items",
  "place-self",
  "pointer-events",
  "position",
  "print-color-adjust",
  "quotes",
  "resize",
  "right",
  "rotate",
  "row-gap",
  "ruby-position",
  "scale",
  "scroll-behavior",
  "scroll-margin",
  "scroll-margin-block",
  "scroll-margin-block-end",
  "scroll-margin-block-start",
  "scroll-margin-bottom",
  "scroll-margin-inline",
  "scroll-margin-inline-end",
  "scroll-margin-inline-start",
  "scroll-margin-left",
  "scroll-margin-right",
  "scroll-margin-top",
  "scroll-padding",
  "scroll-padding-block",
  "scroll-padding-block-end",
  "scroll-padding-block-start",
  "scroll-padding-bottom",
  "scroll-padding-inline",
  "scroll-padding-inline-end",
  "scroll-padding-inline-start",
  "scroll-padding-left",
  "scroll-padding-right",
  "scroll-padding-top",
  "scroll-snap-align",
  "scroll-snap-stop",
  "scroll-snap-type",
  "scrollbar-color",
  "scrollbar-gutter",
  "scrollbar-width",
  "shape-image-threshold",
  "shape-margin",
  "shape-outside",
  "size-adjust",
  "src",
  "tab-size",
  "table-layout",
  "text-align",
  "text-align-last",
  "text-combine-upright",
  "text-decoration",
  "text-decoration-color",
  "text-decoration-line",
  "text-decoration-skip-ink",
  "text-decoration-style",
  "text-decoration-thickness",
  "text-emphasis",
  "text-emphasis-color",
  "text-emphasis-position",
  "text-emphasis-style",
  "text-indent",
  "text-justify",
  "text-orientation",
  "text-overflow",
  "text-rendering",
  "text-shadow",
  "text-transform",
  "text-underline-offset",
  "text-underline-position",
  "top",
  "touch-action",
  "transform",
  "transform-box",
  "transform-origin",
  "transform-style",
  "transition",
  "transition-delay",
  "transition-duration",
  "transition-property",
  "transition-timing-function",
  "translate",
  "unicode-bidi",
  "unicode-range",
  "user-select",
  "vertical-align",
  "visibility",
  "white-space",
  "widows",
  "width",
  "will-change",
  "word-break",
  "word-spacing",
  "writing-mode",
  "z-index"
];

var alphabetical = /*#__PURE__*/Object.freeze({
  __proto__: null,
  properties: properties$2
});

const properties$1 = [
  "all",
  "display",
  "position",
  "top",
  "right",
  "bottom",
  "left",
  "offset",
  "offset-anchor",
  "offset-distance",
  "offset-path",
  "offset-rotate",
  "grid",
  "grid-template-rows",
  "grid-template-columns",
  "grid-template-areas",
  "grid-auto-rows",
  "grid-auto-columns",
  "grid-auto-flow",
  "column-gap",
  "row-gap",
  "grid-area",
  "grid-row",
  "grid-row-start",
  "grid-row-end",
  "grid-column",
  "grid-column-start",
  "grid-column-end",
  "grid-template",
  "flex",
  "flex-grow",
  "flex-shrink",
  "flex-basis",
  "flex-direction",
  "flex-flow",
  "flex-wrap",
  "box-decoration-break",
  "place-content",
  "align-content",
  "justify-content",
  "place-items",
  "align-items",
  "justify-items",
  "place-self",
  "align-self",
  "justify-self",
  "vertical-align",
  "order",
  "float",
  "clear",
  "shape-margin",
  "shape-outside",
  "shape-image-threshold",
  "orphans",
  "gap",
  "columns",
  "column-fill",
  "column-rule",
  "column-rule-width",
  "column-rule-style",
  "column-rule-color",
  "column-width",
  "column-span",
  "column-count",
  "break-before",
  "break-after",
  "break-inside",
  "page",
  "page-break-before",
  "page-break-after",
  "page-break-inside",
  "transform",
  "transform-box",
  "transform-origin",
  "transform-style",
  "translate",
  "rotate",
  "scale",

  "perspective",
  "perspective-origin",
  "appearance",
  "visibility",
  "content-visibility",
  "opacity",
  "z-index",
  "paint-order",
  "mix-blend-mode",
  "backface-visibility",
  "backdrop-filter",
  "clip-path",
  "mask",
  "mask-border",
  "mask-border-outset",
  "mask-border-repeat",
  "mask-border-slice",
  "mask-border-source",
  "mask-border-width",
  "mask-image",
  "mask-mode",
  "mask-position",
  "mask-size",
  "mask-repeat",
  "mask-origin",
  "mask-clip",
  "mask-composite",
  "mask-type",
  "filter",
  "animation",
  "animation-duration",
  "animation-timing-function",
  "animation-delay",
  "animation-iteration-count",
  "animation-direction",
  "animation-fill-mode",
  "animation-play-state",
  "animation-name",
  "transition",
  "transition-delay",
  "transition-duration",
  "transition-property",
  "transition-timing-function",
  "will-change",
  "counter-increment",
  "counter-reset",
  "counter-set",
  "cursor",

  "box-sizing",
  "contain",
  "margin",
  "margin-top",
  "margin-right",
  "margin-bottom",
  "margin-left",
  "margin-inline",
  "margin-inline-start",
  "margin-inline-end",
  "margin-block",
  "margin-block-start",
  "margin-block-end",
  "inset",
  "inset-block",
  "inset-block-end",
  "inset-block-start",
  "inset-inline",
  "inset-inline-end",
  "inset-inline-start",
  "outline",
  "outline-color",
  "outline-style",
  "outline-width",
  "outline-offset",
  "box-shadow",
  "border",
  "border-top",
  "border-right",
  "border-bottom",
  "border-left",
  "border-width",
  "border-top-width",
  "border-right-width",
  "border-bottom-width",
  "border-left-width",
  "border-style",
  "border-top-style",
  "border-right-style",
  "border-bottom-style",
  "border-left-style",
  "border-color",
  "border-top-color",
  "border-right-color",
  "border-bottom-color",
  "border-left-color",
  "border-radius",
  "border-top-right-radius",
  "border-top-left-radius",
  "border-bottom-right-radius",
  "border-bottom-left-radius",
  "border-inline",
  "border-inline-width",
  "border-inline-style",
  "border-inline-color",
  "border-inline-start",
  "border-inline-start-width",
  "border-inline-start-style",
  "border-inline-start-color",
  "border-inline-end",
  "border-inline-end-width",
  "border-inline-end-style",
  "border-inline-end-color",
  "border-block",
  "border-block-width",
  "border-block-style",
  "border-block-color",
  "border-block-start",
  "border-block-start-width",
  "border-block-start-style",
  "border-block-start-color",
  "border-block-end",
  "border-block-end-width",
  "border-block-end-style",
  "border-block-end-color",
  "border-image",
  "border-image-source",
  "border-image-slice",
  "border-image-width",
  "border-image-outset",
  "border-image-repeat",
  "border-collapse",
  "border-spacing",
  "border-start-start-radius",
  "border-start-end-radius",
  "border-end-start-radius",
  "border-end-end-radius",
  "background",
  "background-image",
  "background-position",
  "background-size",
  "background-repeat",
  "background-origin",
  "background-clip",
  "background-attachment",
  "background-color",
  "background-blend-mode",
  "background-position-x",
  "background-position-y",
  "isolation",
  "padding",
  "padding-top",
  "padding-right",
  "padding-bottom",
  "padding-left",
  "padding-inline",
  "padding-inline-start",
  "padding-inline-end",
  "padding-block",
  "padding-block-start",
  "padding-block-end",
  "image-orientation",
  "image-rendering",

  "aspect-ratio",
  "width",
  "min-width",
  "max-width",
  "height",
  "min-height",
  "max-height",
  "-webkit-line-clamp",
  "-webkit-text-fill-color",
  "-webkit-text-stroke",
  "-webkit-text-stroke-color",
  "-webkit-text-stroke-width",
  "inline-size",
  "min-inline-size",
  "max-inline-size",
  "block-size",
  "min-block-size",
  "max-block-size",
  "table-layout",
  "caption-side",
  "empty-cells",
  "overflow",
  "overflow-anchor",
  "overflow-block",
  "overflow-inline",
  "overflow-x",
  "overflow-y",
  "overscroll-behavior",
  "overscroll-behavior-block",
  "overscroll-behavior-inline",
  "overscroll-behavior-x",
  "overscroll-behavior-y",
  "resize",
  "object-fit",
  "object-position",
  "scroll-behavior",
  "scroll-margin",
  "scroll-margin-block",
  "scroll-margin-block-end",
  "scroll-margin-block-start",
  "scroll-margin-bottom",
  "scroll-margin-inline",
  "scroll-margin-inline-end",
  "scroll-margin-inline-start",
  "scroll-margin-left",
  "scroll-margin-right",
  "scroll-margin-top",
  "scroll-padding",
  "scroll-padding-block",
  "scroll-padding-block-end",
  "scroll-padding-block-start",
  "scroll-padding-bottom",
  "scroll-padding-inline",
  "scroll-padding-inline-end",
  "scroll-padding-inline-start",
  "scroll-padding-left",
  "scroll-padding-right",
  "scroll-padding-top",
  "scroll-snap-align",
  "scroll-snap-stop",
  "scroll-snap-type",
  "scrollbar-color",
  "scrollbar-gutter",
  "scrollbar-width",
  "touch-action",
  "pointer-events",

  "content",
  "quotes",
  "hanging-punctuation",
  "color",
  "accent-color",
  "print-color-adjust",
  "forced-color-adjust",
  "color-scheme",
  "caret-color",
  "font",
  "font-style",
  "font-variant",
  "font-weight",
  "font-stretch",
  "font-size",
  "size-adjust",
  "line-height",
  "src",
  "font-family",
  "font-display",
  "font-kerning",
  "font-language-override",
  "font-optical-sizing",
  "font-size-adjust",
  "font-synthesis",
  "font-variant-alternates",
  "font-variant-caps",
  "font-variant-east-asian",
  "font-variant-ligatures",
  "font-variant-numeric",
  "font-variant-position",
  "font-variation-settings",
  "ascent-override",
  "descent-override",
  "line-gap-override",
  "hyphens",
  "hyphenate-character",
  "letter-spacing",
  "line-break",
  "list-style",
  "list-style-type",
  "list-style-image",
  "list-style-position",
  "writing-mode",
  "direction",
  "unicode-bidi",
  "unicode-range",
  "user-select",
  "ruby-position",
  "text-combine-upright",
  "text-align",
  "text-align-last",
  "text-decoration",
  "text-decoration-line",
  "text-decoration-style",
  "text-decoration-color",
  "text-decoration-thickness",
  "text-decoration-skip-ink",
  "text-emphasis",
  "text-emphasis-style",
  "text-emphasis-color",
  "text-emphasis-position",
  "text-indent",
  "text-justify",
  "text-underline-position",
  "text-underline-offset",
  "text-orientation",
  "text-overflow",
  "text-rendering",
  "text-shadow",
  "text-transform",
  "white-space",
  "word-break",
  "word-spacing",
  "overflow-wrap",
  "tab-size",
  "widows"
];

var concentricCss = /*#__PURE__*/Object.freeze({
  __proto__: null,
  properties: properties$1
});

const properties = [
  "all",
  "box-sizing",
  "contain",
  "display",
  "appearance",
  "visibility",
  "content-visibility",
  "z-index",
  "paint-order",
  "position",
  "top",
  "right",
  "bottom",
  "left",
  "offset",
  "offset-anchor",
  "offset-distance",
  "offset-path",
  "offset-rotate",


  "grid",
  "grid-template-rows",
  "grid-template-columns",
  "grid-template-areas",
  "grid-auto-rows",
  "grid-auto-columns",
  "grid-auto-flow",
  "column-gap",
  "row-gap",
  "grid-area",
  "grid-row",
  "grid-row-start",
  "grid-row-end",
  "grid-column",
  "grid-column-start",
  "grid-column-end",
  "grid-template",
  "flex",
  "flex-grow",
  "flex-shrink",
  "flex-basis",
  "flex-direction",
  "flex-flow",
  "flex-wrap",
  "box-decoration-break",
  "place-content",
  "place-items",
  "place-self",
  "align-content",
  "align-items",
  "align-self",
  "justify-content",
  "justify-items",
  "justify-self",
  "order",
  "aspect-ratio",
  "width",
  "min-width",
  "max-width",
  "height",
  "min-height",
  "max-height",
  "-webkit-line-clamp",
  "-webkit-text-fill-color",
  "-webkit-text-stroke",
  "-webkit-text-stroke-color",
  "-webkit-text-stroke-width",
  "inline-size",
  "min-inline-size",
  "max-inline-size",
  "block-size",
  "min-block-size",
  "max-block-size",
  "margin",
  "margin-top",
  "margin-right",
  "margin-bottom",
  "margin-left",
  "margin-inline",
  "margin-inline-start",
  "margin-inline-end",
  "margin-block",
  "margin-block-start",
  "margin-block-end",
  "inset",
  "inset-block",
  "inset-block-end",
  "inset-block-start",
  "inset-inline",
  "inset-inline-end",
  "inset-inline-start",
  "padding",
  "padding-top",
  "padding-right",
  "padding-bottom",
  "padding-left",
  "padding-inline",
  "padding-inline-start",
  "padding-inline-end",
  "padding-block",
  "padding-block-start",
  "padding-block-end",
  "float",
  "clear",
  "overflow",
  "overflow-anchor",
  "overflow-block",
  "overflow-inline",
  "overflow-x",
  "overflow-y",
  "overscroll-behavior",
  "overscroll-behavior-block",
  "overscroll-behavior-inline",
  "overscroll-behavior-x",
  "overscroll-behavior-y",
  "orphans",
  "gap",
  "columns",
  "column-fill",
  "column-rule",
  "column-rule-color",
  "column-rule-style",
  "column-rule-width",
  "column-span",
  "column-count",
  "column-width",
  "object-fit",
  "object-position",
  "transform",
  "transform-box",
  "transform-origin",
  "transform-style",
  "translate",
  "rotate",
  "scale",

  "border",
  "border-top",
  "border-right",
  "border-bottom",
  "border-left",
  "border-width",
  "border-top-width",
  "border-right-width",
  "border-bottom-width",
  "border-left-width",
  "border-style",
  "border-top-style",
  "border-right-style",
  "border-bottom-style",
  "border-left-style",
  "border-radius",
  "border-top-right-radius",
  "border-top-left-radius",
  "border-bottom-right-radius",
  "border-bottom-left-radius",
  "border-inline",
  "border-inline-color",
  "border-inline-style",
  "border-inline-width",
  "border-inline-start",
  "border-inline-start-color",
  "border-inline-start-style",
  "border-inline-start-width",
  "border-inline-end",
  "border-inline-end-color",
  "border-inline-end-style",
  "border-inline-end-width",
  "border-block",
  "border-block-color",
  "border-block-style",
  "border-block-width",
  "border-block-start",
  "border-block-start-color",
  "border-block-start-style",
  "border-block-start-width",
  "border-block-end",
  "border-block-end-color",
  "border-block-end-style",
  "border-block-end-width",
  "border-color",
  "border-image",
  "border-image-outset",
  "border-image-repeat",
  "border-image-slice",
  "border-image-source",
  "border-image-width",
  "border-top-color",
  "border-right-color",
  "border-bottom-color",
  "border-left-color",
  "border-collapse",
  "border-spacing",
  "border-start-start-radius",
  "border-start-end-radius",
  "border-end-start-radius",
  "border-end-end-radius",
  "outline",
  "outline-color",
  "outline-style",
  "outline-width",
  "outline-offset",

  "backdrop-filter",
  "backface-visibility",
  "background",
  "background-image",
  "background-position",
  "background-size",
  "background-repeat",
  "background-origin",
  "background-clip",
  "background-attachment",
  "background-color",
  "background-blend-mode",
  "background-position-x",
  "background-position-y",
  "box-shadow",
  "isolation",

  "content",
  "quotes",
  "hanging-punctuation",
  "color",
  "accent-color",
  "print-color-adjust",
  "forced-color-adjust",
  "color-scheme",
  "caret-color",
  "font",
  "font-style",
  "font-variant",
  "font-weight",
  "src",
  "font-stretch",
  "font-size",
  "size-adjust",
  "line-height",
  "font-family",
  "font-display",
  "font-kerning",
  "font-language-override",
  "font-optical-sizing",
  "font-size-adjust",
  "font-synthesis",
  "font-variant-alternates",
  "font-variant-caps",
  "font-variant-east-asian",
  "font-variant-ligatures",
  "font-variant-numeric",
  "font-variant-position",
  "font-variation-settings",
  "ascent-override",
  "descent-override",
  "line-gap-override",
  "hyphens",
  "hyphenate-character",
  "letter-spacing",
  "line-break",
  "list-style",
  "list-style-image",
  "list-style-position",
  "list-style-type",
  "direction",
  "text-align",
  "text-align-last",
  "text-decoration",
  "text-decoration-line",
  "text-decoration-style",
  "text-decoration-color",
  "text-decoration-thickness",
  "text-decoration-skip-ink",
  "text-emphasis",
  "text-emphasis-style",
  "text-emphasis-color",
  "text-emphasis-position",
  "text-indent",
  "text-justify",
  "text-underline-position",
  "text-underline-offset",
  "text-orientation",
  "text-overflow",
  "text-rendering",
  "text-shadow",
  "text-transform",
  "vertical-align",
  "white-space",
  "word-break",
  "word-spacing",
  "overflow-wrap",

  "animation",
  "animation-duration",
  "animation-timing-function",
  "animation-delay",
  "animation-iteration-count",
  "animation-direction",
  "animation-fill-mode",
  "animation-play-state",
  "animation-name",
  "mix-blend-mode",
  "break-before",
  "break-after",
  "break-inside",
  "page",
  "page-break-before",
  "page-break-after",
  "page-break-inside",
  "caption-side",
  "clip-path",
  "counter-increment",
  "counter-reset",
  "counter-set",
  "cursor",
  "empty-cells",
  "filter",
  "image-orientation",
  "image-rendering",
  "mask",
  "mask-border",
  "mask-border-outset",
  "mask-border-repeat",
  "mask-border-slice",
  "mask-border-source",
  "mask-border-width",
  "mask-clip",
  "mask-composite",
  "mask-image",
  "mask-mode",
  "mask-origin",
  "mask-position",
  "mask-repeat",
  "mask-size",
  "mask-type",
  "opacity",
  "perspective",
  "perspective-origin",
  "pointer-events",
  "resize",
  "scroll-behavior",
  "scroll-margin",
  "scroll-margin-block",
  "scroll-margin-block-end",
  "scroll-margin-block-start",
  "scroll-margin-bottom",
  "scroll-margin-inline",
  "scroll-margin-inline-end",
  "scroll-margin-inline-start",
  "scroll-margin-left",
  "scroll-margin-right",
  "scroll-margin-top",
  "scroll-padding",
  "scroll-padding-block",
  "scroll-padding-block-end",
  "scroll-padding-block-start",
  "scroll-padding-bottom",
  "scroll-padding-inline",
  "scroll-padding-inline-end",
  "scroll-padding-inline-start",
  "scroll-padding-left",
  "scroll-padding-right",
  "scroll-padding-top",
  "scroll-snap-align",
  "scroll-snap-stop",
  "scroll-snap-type",
  "scrollbar-color",
  "scrollbar-gutter",
  "scrollbar-width",
  "shape-image-threshold",
  "shape-margin",
  "shape-outside",
  "tab-size",
  "table-layout",
  "ruby-position",
  "text-combine-upright",
  "touch-action",
  "transition",
  "transition-delay",
  "transition-duration",
  "transition-property",
  "transition-timing-function",
  "will-change",
  "unicode-bidi",
  "unicode-range",
  "user-select",
  "widows",
  "writing-mode"
];

var smacss = /*#__PURE__*/Object.freeze({
  __proto__: null,
  properties: properties
});

exports.cssDeclarationSorter = cssDeclarationSorter;
exports["default"] = cssDeclarationSorter;

module.exports = cssDeclarationSorter;

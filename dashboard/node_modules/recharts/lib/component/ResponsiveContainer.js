"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useResponsiveContainerContext = exports.ResponsiveContainer = void 0;
var _clsx = require("clsx");
var _react = _interopRequireWildcard(require("react"));
var React = _react;
var _throttle = _interopRequireDefault(require("es-toolkit/compat/throttle"));
var _DataUtils = require("../util/DataUtils");
var _LogUtils = require("../util/LogUtils");
var _responsiveContainerUtils = require("./responsiveContainerUtils");
var _isWellBehavedNumber = require("../util/isWellBehavedNumber");
function _interopRequireDefault(e) { return e && e.__esModule ? e : { default: e }; }
function _interopRequireWildcard(e, t) { if ("function" == typeof WeakMap) var r = new WeakMap(), n = new WeakMap(); return (_interopRequireWildcard = function _interopRequireWildcard(e, t) { if (!t && e && e.__esModule) return e; var o, i, f = { __proto__: null, default: e }; if (null === e || "object" != typeof e && "function" != typeof e) return f; if (o = t ? n : r) { if (o.has(e)) return o.get(e); o.set(e, f); } for (var _t in e) "default" !== _t && {}.hasOwnProperty.call(e, _t) && ((i = (o = Object.defineProperty) && Object.getOwnPropertyDescriptor(e, _t)) && (i.get || i.set) ? o(f, _t, i) : f[_t] = e[_t]); return f; })(e, t); }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
var ResponsiveContainerContext = /*#__PURE__*/(0, _react.createContext)(_responsiveContainerUtils.defaultResponsiveContainerProps.initialDimension);
function isAcceptableSize(size) {
  return (0, _isWellBehavedNumber.isPositiveNumber)(size.width) && (0, _isWellBehavedNumber.isPositiveNumber)(size.height);
}
function ResponsiveContainerContextProvider(_ref) {
  var {
    children,
    width,
    height
  } = _ref;
  var size = (0, _react.useMemo)(() => ({
    width,
    height
  }), [width, height]);
  if (!isAcceptableSize(size)) {
    /*
     * Don't render the container if width or height is non-positive because
     * in that case the chart will not be rendered properly anyway.
     * We will instead wait for the next resize event to provide the correct dimensions.
     */
    return null;
  }
  return /*#__PURE__*/React.createElement(ResponsiveContainerContext.Provider, {
    value: size
  }, children);
}
var useResponsiveContainerContext = () => (0, _react.useContext)(ResponsiveContainerContext);
exports.useResponsiveContainerContext = useResponsiveContainerContext;
var SizeDetectorContainer = /*#__PURE__*/(0, _react.forwardRef)((_ref2, ref) => {
  var {
    aspect,
    initialDimension = _responsiveContainerUtils.defaultResponsiveContainerProps.initialDimension,
    width,
    height,
    /*
     * default min-width to 0 if not specified - 'auto' causes issues with flexbox
     * https://github.com/recharts/recharts/issues/172
     */
    minWidth = _responsiveContainerUtils.defaultResponsiveContainerProps.minWidth,
    minHeight,
    maxHeight,
    children,
    debounce = _responsiveContainerUtils.defaultResponsiveContainerProps.debounce,
    id,
    className,
    onResize,
    style = {}
  } = _ref2;
  var containerRef = (0, _react.useRef)(null);
  /*
   * We are using a ref to avoid re-creating the ResizeObserver when the onResize function changes.
   * The ref is updated on every render, so the latest onResize function is always available in the effect.
   */
  var onResizeRef = (0, _react.useRef)();
  onResizeRef.current = onResize;
  (0, _react.useImperativeHandle)(ref, () => containerRef.current);
  var [sizes, setSizes] = (0, _react.useState)({
    containerWidth: initialDimension.width,
    containerHeight: initialDimension.height
  });
  var setContainerSize = (0, _react.useCallback)((newWidth, newHeight) => {
    setSizes(prevState => {
      var roundedWidth = Math.round(newWidth);
      var roundedHeight = Math.round(newHeight);
      if (prevState.containerWidth === roundedWidth && prevState.containerHeight === roundedHeight) {
        return prevState;
      }
      return {
        containerWidth: roundedWidth,
        containerHeight: roundedHeight
      };
    });
  }, []);
  (0, _react.useEffect)(() => {
    if (containerRef.current == null || typeof ResizeObserver === 'undefined') {
      return _DataUtils.noop;
    }
    var callback = entries => {
      var _onResizeRef$current;
      var entry = entries[0];
      if (entry == null) {
        return;
      }
      var {
        width: containerWidth,
        height: containerHeight
      } = entry.contentRect;
      setContainerSize(containerWidth, containerHeight);
      (_onResizeRef$current = onResizeRef.current) === null || _onResizeRef$current === void 0 || _onResizeRef$current.call(onResizeRef, containerWidth, containerHeight);
    };
    if (debounce > 0) {
      callback = (0, _throttle.default)(callback, debounce, {
        trailing: true,
        leading: false
      });
    }
    var observer = new ResizeObserver(callback);
    var {
      width: containerWidth,
      height: containerHeight
    } = containerRef.current.getBoundingClientRect();
    setContainerSize(containerWidth, containerHeight);
    observer.observe(containerRef.current);
    return () => {
      observer.disconnect();
    };
  }, [setContainerSize, debounce]);
  var {
    containerWidth,
    containerHeight
  } = sizes;
  (0, _LogUtils.warn)(!aspect || aspect > 0, 'The aspect(%s) must be greater than zero.', aspect);
  var {
    calculatedWidth,
    calculatedHeight
  } = (0, _responsiveContainerUtils.calculateChartDimensions)(containerWidth, containerHeight, {
    width,
    height,
    aspect,
    maxHeight
  });
  (0, _LogUtils.warn)(calculatedWidth != null && calculatedWidth > 0 || calculatedHeight != null && calculatedHeight > 0, "The width(%s) and height(%s) of chart should be greater than 0,\n       please check the style of container, or the props width(%s) and height(%s),\n       or add a minWidth(%s) or minHeight(%s) or use aspect(%s) to control the\n       height and width.", calculatedWidth, calculatedHeight, width, height, minWidth, minHeight, aspect);
  return /*#__PURE__*/React.createElement("div", {
    id: id ? "".concat(id) : undefined,
    className: (0, _clsx.clsx)('recharts-responsive-container', className),
    style: _objectSpread(_objectSpread({}, style), {}, {
      width,
      height,
      minWidth,
      minHeight,
      maxHeight
    }),
    ref: containerRef
  }, /*#__PURE__*/React.createElement("div", {
    style: (0, _responsiveContainerUtils.getInnerDivStyle)({
      width,
      height
    })
  }, /*#__PURE__*/React.createElement(ResponsiveContainerContextProvider, {
    width: calculatedWidth,
    height: calculatedHeight
  }, children)));
});

/**
 * The `ResponsiveContainer` component is a container that adjusts its width and height based on the size of its parent element.
 * It is used to create responsive charts that adapt to different screen sizes.
 *
 * This component uses the {@link https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver ResizeObserver} API to monitor changes to the size of its parent element.
 * If you need to support older browsers that do not support this API, you may need to include a polyfill.
 *
 * @see {@link https://recharts.github.io/en-US/guide/sizes/ Chart size guide}
 *
 * @provides ResponsiveContainerContext
 */
var ResponsiveContainer = exports.ResponsiveContainer = /*#__PURE__*/(0, _react.forwardRef)((props, ref) => {
  var responsiveContainerContext = useResponsiveContainerContext();
  if ((0, _isWellBehavedNumber.isPositiveNumber)(responsiveContainerContext.width) && (0, _isWellBehavedNumber.isPositiveNumber)(responsiveContainerContext.height)) {
    /*
     * If we detect that we are already inside another ResponsiveContainer,
     * we do not attempt to add another layer of responsiveness.
     */
    return props.children;
  }
  var {
    width,
    height
  } = (0, _responsiveContainerUtils.getDefaultWidthAndHeight)({
    width: props.width,
    height: props.height,
    aspect: props.aspect
  });

  /*
   * Let's try to get the calculated dimensions without having the div container set up.
   * Sometimes this does produce fixed, positive dimensions. If so, we can skip rendering the div and monitoring its size.
   */
  var {
    calculatedWidth,
    calculatedHeight
  } = (0, _responsiveContainerUtils.calculateChartDimensions)(undefined, undefined, {
    width,
    height,
    aspect: props.aspect,
    maxHeight: props.maxHeight
  });
  if ((0, _DataUtils.isNumber)(calculatedWidth) && (0, _DataUtils.isNumber)(calculatedHeight)) {
    /*
     * If it just so happens that the combination of width, height, and aspect ratio
     * results in fixed dimensions, then we don't need to monitor the container's size.
     * We can just provide these fixed dimensions to the context.
     *
     * Note that here we are not checking for positive numbers;
     * if the user provides a zero or negative width/height, we will just pass that along
     * as whatever size we detect won't be helping anyway.
     */
    return /*#__PURE__*/React.createElement(ResponsiveContainerContextProvider, {
      width: calculatedWidth,
      height: calculatedHeight
    }, props.children);
  }
  /*
   * Static analysis did not produce fixed dimensions,
   * so we need to render a special div and monitor its size.
   */
  return /*#__PURE__*/React.createElement(SizeDetectorContainer, _extends({}, props, {
    width: width,
    height: height,
    ref: ref
  }));
});
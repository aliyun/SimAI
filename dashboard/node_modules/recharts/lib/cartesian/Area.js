"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.Area = void 0;
exports.computeArea = computeArea;
exports.getBaseValue = exports.defaultAreaProps = void 0;
var _react = _interopRequireWildcard(require("react"));
var React = _react;
var _clsx = require("clsx");
var _Curve = require("../shape/Curve");
var _Layer = require("../container/Layer");
var _LabelList = require("../component/LabelList");
var _Dots = require("../component/Dots");
var _Global = require("../util/Global");
var _DataUtils = require("../util/DataUtils");
var _ChartUtils = require("../util/ChartUtils");
var _ReactUtils = require("../util/ReactUtils");
var _ActivePoints = require("../component/ActivePoints");
var _SetTooltipEntrySettings = require("../state/SetTooltipEntrySettings");
var _GraphicalItemClipPath = require("./GraphicalItemClipPath");
var _areaSelectors = require("../state/selectors/areaSelectors");
var _PanoramaContext = require("../context/PanoramaContext");
var _chartLayoutContext = require("../context/chartLayoutContext");
var _selectors = require("../state/selectors/selectors");
var _SetLegendPayload = require("../state/SetLegendPayload");
var _hooks = require("../state/hooks");
var _useAnimationId = require("../util/useAnimationId");
var _resolveDefaultProps2 = require("../util/resolveDefaultProps");
var _isWellBehavedNumber = require("../util/isWellBehavedNumber");
var _hooks2 = require("../hooks");
var _RegisterGraphicalItemId = require("../context/RegisterGraphicalItemId");
var _SetGraphicalItem = require("../state/SetGraphicalItem");
var _svgPropertiesNoEvents = require("../util/svgPropertiesNoEvents");
var _JavascriptAnimate = require("../animation/JavascriptAnimate");
var _getRadiusAndStrokeWidthFromDot = require("../util/getRadiusAndStrokeWidthFromDot");
var _svgPropertiesAndEvents = require("../util/svgPropertiesAndEvents");
var _ZIndexLayer = require("../zIndex/ZIndexLayer");
var _DefaultZIndexes = require("../zIndex/DefaultZIndexes");
var _propsAreEqual = require("../util/propsAreEqual");
var _excluded = ["id"],
  _excluded2 = ["activeDot", "animationBegin", "animationDuration", "animationEasing", "connectNulls", "dot", "fill", "fillOpacity", "hide", "isAnimationActive", "legendType", "stroke", "xAxisId", "yAxisId"];
function _interopRequireWildcard(e, t) { if ("function" == typeof WeakMap) var r = new WeakMap(), n = new WeakMap(); return (_interopRequireWildcard = function _interopRequireWildcard(e, t) { if (!t && e && e.__esModule) return e; var o, i, f = { __proto__: null, default: e }; if (null === e || "object" != typeof e && "function" != typeof e) return f; if (o = t ? n : r) { if (o.has(e)) return o.get(e); o.set(e, f); } for (var _t in e) "default" !== _t && {}.hasOwnProperty.call(e, _t) && ((i = (o = Object.defineProperty) && Object.getOwnPropertyDescriptor(e, _t)) && (i.get || i.set) ? o(f, _t, i) : f[_t] = e[_t]); return f; })(e, t); }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
/**
 * @inline
 */

/**
 * Our base value array has payload in it, and we expose it externally too.
 */

/**
 * Internal props, combination of external props + defaultProps + private Recharts state
 */

/**
 * External props, intended for end users to fill in
 */

/**
 * Because of naming conflict, we are forced to ignore certain (valid) SVG attributes.
 */

function getLegendItemColor(stroke, fill) {
  return stroke && stroke !== 'none' ? stroke : fill;
}
var computeLegendPayloadFromAreaData = props => {
  var {
    dataKey,
    name,
    stroke,
    fill,
    legendType,
    hide
  } = props;
  return [{
    inactive: hide,
    dataKey,
    type: legendType,
    color: getLegendItemColor(stroke, fill),
    value: (0, _ChartUtils.getTooltipNameProp)(name, dataKey),
    payload: props
  }];
};
var SetAreaTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
    data,
    stroke,
    strokeWidth,
    fill,
    name,
    hide,
    unit,
    tooltipType,
    id
  } = _ref;
  var tooltipEntrySettings = {
    dataDefinedOnItem: data,
    getPosition: _DataUtils.noop,
    settings: {
      stroke,
      strokeWidth,
      fill,
      dataKey,
      nameKey: undefined,
      name: (0, _ChartUtils.getTooltipNameProp)(name, dataKey),
      hide,
      type: tooltipType,
      color: getLegendItemColor(stroke, fill),
      unit,
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(_SetTooltipEntrySettings.SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});
function AreaDotsWrapper(_ref2) {
  var {
    clipPathId,
    points,
    props
  } = _ref2;
  var {
    needClip,
    dot,
    dataKey
  } = props;
  var areaProps = (0, _svgPropertiesNoEvents.svgPropertiesNoEvents)(props);
  return /*#__PURE__*/React.createElement(_Dots.Dots, {
    points: points,
    dot: dot,
    className: "recharts-area-dots",
    dotClassName: "recharts-area-dot",
    dataKey: dataKey,
    baseProps: areaProps,
    needClip: needClip,
    clipPathId: clipPathId
  });
}
function AreaLabelListProvider(_ref3) {
  var {
    showLabels,
    children,
    points
  } = _ref3;
  var labelListEntries = points.map(point => {
    var _point$x, _point$y;
    var viewBox = {
      x: (_point$x = point.x) !== null && _point$x !== void 0 ? _point$x : 0,
      y: (_point$y = point.y) !== null && _point$y !== void 0 ? _point$y : 0,
      width: 0,
      lowerWidth: 0,
      upperWidth: 0,
      height: 0
    };
    return _objectSpread(_objectSpread({}, viewBox), {}, {
      value: point.value,
      payload: point.payload,
      parentViewBox: undefined,
      viewBox,
      fill: undefined
    });
  });
  return /*#__PURE__*/React.createElement(_LabelList.CartesianLabelListContextProvider, {
    value: showLabels ? labelListEntries : undefined
  }, children);
}
function StaticArea(_ref4) {
  var {
    points,
    baseLine,
    needClip,
    clipPathId,
    props
  } = _ref4;
  var {
    layout,
    type,
    stroke,
    connectNulls,
    isRange
  } = props;
  var {
      id
    } = props,
    propsWithoutId = _objectWithoutProperties(props, _excluded);
  var allOtherProps = (0, _svgPropertiesNoEvents.svgPropertiesNoEvents)(propsWithoutId);
  var propsWithEvents = (0, _svgPropertiesAndEvents.svgPropertiesAndEvents)(propsWithoutId);
  return /*#__PURE__*/React.createElement(React.Fragment, null, (points === null || points === void 0 ? void 0 : points.length) > 1 && /*#__PURE__*/React.createElement(_Layer.Layer, {
    clipPath: needClip ? "url(#clipPath-".concat(clipPathId, ")") : undefined
  }, /*#__PURE__*/React.createElement(_Curve.Curve, _extends({}, propsWithEvents, {
    id: id,
    points: points,
    connectNulls: connectNulls,
    type: type,
    baseLine: baseLine,
    layout: layout,
    stroke: "none",
    className: "recharts-area-area"
  })), stroke !== 'none' && /*#__PURE__*/React.createElement(_Curve.Curve, _extends({}, allOtherProps, {
    className: "recharts-area-curve",
    layout: layout,
    type: type,
    connectNulls: connectNulls,
    fill: "none",
    points: points
  })), stroke !== 'none' && isRange && /*#__PURE__*/React.createElement(_Curve.Curve, _extends({}, allOtherProps, {
    className: "recharts-area-curve",
    layout: layout,
    type: type,
    connectNulls: connectNulls,
    fill: "none",
    points: baseLine
  }))), /*#__PURE__*/React.createElement(AreaDotsWrapper, {
    points: points,
    props: propsWithoutId,
    clipPathId: clipPathId
  }));
}
function VerticalRect(_ref5) {
  var _points$, _points;
  var {
    alpha,
    baseLine,
    points,
    strokeWidth
  } = _ref5;
  var startY = (_points$ = points[0]) === null || _points$ === void 0 ? void 0 : _points$.y;
  var endY = (_points = points[points.length - 1]) === null || _points === void 0 ? void 0 : _points.y;
  if (!(0, _isWellBehavedNumber.isWellBehavedNumber)(startY) || !(0, _isWellBehavedNumber.isWellBehavedNumber)(endY)) {
    return null;
  }
  var height = alpha * Math.abs(startY - endY);
  var maxX = Math.max(...points.map(entry => entry.x || 0));
  if ((0, _DataUtils.isNumber)(baseLine)) {
    maxX = Math.max(baseLine, maxX);
  } else if (baseLine && Array.isArray(baseLine) && baseLine.length) {
    maxX = Math.max(...baseLine.map(entry => entry.x || 0), maxX);
  }
  if ((0, _DataUtils.isNumber)(maxX)) {
    return /*#__PURE__*/React.createElement("rect", {
      x: 0,
      y: startY < endY ? startY : startY - height,
      width: maxX + (strokeWidth ? parseInt("".concat(strokeWidth), 10) : 1),
      height: Math.floor(height)
    });
  }
  return null;
}
function HorizontalRect(_ref6) {
  var _points$2, _points2;
  var {
    alpha,
    baseLine,
    points,
    strokeWidth
  } = _ref6;
  var startX = (_points$2 = points[0]) === null || _points$2 === void 0 ? void 0 : _points$2.x;
  var endX = (_points2 = points[points.length - 1]) === null || _points2 === void 0 ? void 0 : _points2.x;
  if (!(0, _isWellBehavedNumber.isWellBehavedNumber)(startX) || !(0, _isWellBehavedNumber.isWellBehavedNumber)(endX)) {
    return null;
  }
  var width = alpha * Math.abs(startX - endX);
  var maxY = Math.max(...points.map(entry => entry.y || 0));
  if ((0, _DataUtils.isNumber)(baseLine)) {
    maxY = Math.max(baseLine, maxY);
  } else if (baseLine && Array.isArray(baseLine) && baseLine.length) {
    maxY = Math.max(...baseLine.map(entry => entry.y || 0), maxY);
  }
  if ((0, _DataUtils.isNumber)(maxY)) {
    return /*#__PURE__*/React.createElement("rect", {
      x: startX < endX ? startX : startX - width,
      y: 0,
      width: width,
      height: Math.floor(maxY + (strokeWidth ? parseInt("".concat(strokeWidth), 10) : 1))
    });
  }
  return null;
}
function ClipRect(_ref7) {
  var {
    alpha,
    layout,
    points,
    baseLine,
    strokeWidth
  } = _ref7;
  if (layout === 'vertical') {
    return /*#__PURE__*/React.createElement(VerticalRect, {
      alpha: alpha,
      points: points,
      baseLine: baseLine,
      strokeWidth: strokeWidth
    });
  }
  return /*#__PURE__*/React.createElement(HorizontalRect, {
    alpha: alpha,
    points: points,
    baseLine: baseLine,
    strokeWidth: strokeWidth
  });
}
function AreaWithAnimation(_ref8) {
  var {
    needClip,
    clipPathId,
    props,
    previousPointsRef,
    previousBaselineRef
  } = _ref8;
  var {
    points,
    baseLine,
    isAnimationActive,
    animationBegin,
    animationDuration,
    animationEasing,
    onAnimationStart,
    onAnimationEnd
  } = props;
  var animationInput = (0, _react.useMemo)(() => ({
    points,
    baseLine
  }), [points, baseLine]);
  var animationId = (0, _useAnimationId.useAnimationId)(animationInput, 'recharts-area-');
  var layout = (0, _chartLayoutContext.useCartesianChartLayout)();
  var [isAnimating, setIsAnimating] = (0, _react.useState)(false);
  var showLabels = !isAnimating;
  var handleAnimationEnd = (0, _react.useCallback)(() => {
    if (typeof onAnimationEnd === 'function') {
      onAnimationEnd();
    }
    setIsAnimating(false);
  }, [onAnimationEnd]);
  var handleAnimationStart = (0, _react.useCallback)(() => {
    if (typeof onAnimationStart === 'function') {
      onAnimationStart();
    }
    setIsAnimating(true);
  }, [onAnimationStart]);
  if (layout == null) {
    return null;
  }
  var prevPoints = previousPointsRef.current;
  var prevBaseLine = previousBaselineRef.current;
  return /*#__PURE__*/React.createElement(AreaLabelListProvider, {
    showLabels: showLabels,
    points: points
  }, props.children, /*#__PURE__*/React.createElement(_JavascriptAnimate.JavascriptAnimate, {
    animationId: animationId,
    begin: animationBegin,
    duration: animationDuration,
    isActive: isAnimationActive,
    easing: animationEasing,
    onAnimationEnd: handleAnimationEnd,
    onAnimationStart: handleAnimationStart,
    key: animationId
  }, t => {
    if (prevPoints) {
      var prevPointsDiffFactor = prevPoints.length / points.length;
      var stepPoints =
      /*
       * Here it is important that at the very end of the animation, on the last frame,
       * we render the original points without any interpolation.
       * This is needed because the code above is checking for reference equality to decide if the animation should run
       * and if we create a new array instance (even if the numbers were the same)
       * then we would break animations.
       */
      t === 1 ? points : points.map((entry, index) => {
        var prevPointIndex = Math.floor(index * prevPointsDiffFactor);
        if (prevPoints[prevPointIndex]) {
          var prev = prevPoints[prevPointIndex];
          return _objectSpread(_objectSpread({}, entry), {}, {
            x: (0, _DataUtils.interpolate)(prev.x, entry.x, t),
            y: (0, _DataUtils.interpolate)(prev.y, entry.y, t)
          });
        }
        return entry;
      });
      var stepBaseLine;
      if ((0, _DataUtils.isNumber)(baseLine)) {
        stepBaseLine = (0, _DataUtils.interpolate)(prevBaseLine, baseLine, t);
      } else if ((0, _DataUtils.isNullish)(baseLine) || (0, _DataUtils.isNan)(baseLine)) {
        stepBaseLine = (0, _DataUtils.interpolate)(prevBaseLine, 0, t);
      } else {
        stepBaseLine = baseLine.map((entry, index) => {
          var prevPointIndex = Math.floor(index * prevPointsDiffFactor);
          if (Array.isArray(prevBaseLine) && prevBaseLine[prevPointIndex]) {
            var prev = prevBaseLine[prevPointIndex];
            return _objectSpread(_objectSpread({}, entry), {}, {
              x: (0, _DataUtils.interpolate)(prev.x, entry.x, t),
              y: (0, _DataUtils.interpolate)(prev.y, entry.y, t)
            });
          }
          return entry;
        });
      }
      if (t > 0) {
        /*
         * We need to keep the refs in the parent component because we need to remember the last shape of the animation
         * even if AreaWithAnimation is unmounted as that happens when changing props.
         *
         * And we need to update the refs here because here is where the interpolation is computed.
         * Eslint doesn't like changing function arguments, but we need it so here is an eslint-disable.
         */
        // eslint-disable-next-line no-param-reassign
        previousPointsRef.current = stepPoints;
        // eslint-disable-next-line no-param-reassign
        previousBaselineRef.current = stepBaseLine;
      }
      return /*#__PURE__*/React.createElement(StaticArea, {
        points: stepPoints,
        baseLine: stepBaseLine,
        needClip: needClip,
        clipPathId: clipPathId,
        props: props
      });
    }
    if (t > 0) {
      // eslint-disable-next-line no-param-reassign
      previousPointsRef.current = points;
      // eslint-disable-next-line no-param-reassign
      previousBaselineRef.current = baseLine;
    }
    return /*#__PURE__*/React.createElement(_Layer.Layer, null, isAnimationActive && /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("clipPath", {
      id: "animationClipPath-".concat(clipPathId)
    }, /*#__PURE__*/React.createElement(ClipRect, {
      alpha: t,
      points: points,
      baseLine: baseLine,
      layout: layout,
      strokeWidth: props.strokeWidth
    }))), /*#__PURE__*/React.createElement(_Layer.Layer, {
      clipPath: "url(#animationClipPath-".concat(clipPathId, ")")
    }, /*#__PURE__*/React.createElement(StaticArea, {
      points: points,
      baseLine: baseLine,
      needClip: needClip,
      clipPathId: clipPathId,
      props: props
    })));
  }), /*#__PURE__*/React.createElement(_LabelList.LabelListFromLabelProp, {
    label: props.label
  }));
}

/*
 * This components decides if the area should be animated or not.
 * It also holds the state of the animation.
 */
function RenderArea(_ref9) {
  var {
    needClip,
    clipPathId,
    props
  } = _ref9;
  /*
   * These two must be refs, not state!
   * Because we want to store the most recent shape of the animation in case we have to interrupt the animation;
   * that happens when user initiates another animation before the current one finishes.
   *
   * If this was a useState, then every step in the animation would trigger a re-render.
   * So, useRef it is.
   */
  var previousPointsRef = (0, _react.useRef)(null);
  var previousBaselineRef = (0, _react.useRef)();
  return /*#__PURE__*/React.createElement(AreaWithAnimation, {
    needClip: needClip,
    clipPathId: clipPathId,
    props: props,
    previousPointsRef: previousPointsRef,
    previousBaselineRef: previousBaselineRef
  });
}
class AreaWithState extends _react.PureComponent {
  render() {
    var {
      hide,
      dot,
      points,
      className,
      top,
      left,
      needClip,
      xAxisId,
      yAxisId,
      width,
      height,
      id,
      baseLine,
      zIndex
    } = this.props;
    if (hide) {
      return null;
    }
    var layerClass = (0, _clsx.clsx)('recharts-area', className);
    var clipPathId = id;
    var {
      r,
      strokeWidth
    } = (0, _getRadiusAndStrokeWidthFromDot.getRadiusAndStrokeWidthFromDot)(dot);
    var clipDot = (0, _ReactUtils.isClipDot)(dot);
    var dotSize = r * 2 + strokeWidth;
    var activePointsClipPath = needClip ? "url(#clipPath-".concat(clipDot ? '' : 'dots-').concat(clipPathId, ")") : undefined;
    return /*#__PURE__*/React.createElement(_ZIndexLayer.ZIndexLayer, {
      zIndex: zIndex
    }, /*#__PURE__*/React.createElement(_Layer.Layer, {
      className: layerClass
    }, needClip && /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement(_GraphicalItemClipPath.GraphicalItemClipPath, {
      clipPathId: clipPathId,
      xAxisId: xAxisId,
      yAxisId: yAxisId
    }), !clipDot && /*#__PURE__*/React.createElement("clipPath", {
      id: "clipPath-dots-".concat(clipPathId)
    }, /*#__PURE__*/React.createElement("rect", {
      x: left - dotSize / 2,
      y: top - dotSize / 2,
      width: width + dotSize,
      height: height + dotSize
    }))), /*#__PURE__*/React.createElement(RenderArea, {
      needClip: needClip,
      clipPathId: clipPathId,
      props: this.props
    })), /*#__PURE__*/React.createElement(_ActivePoints.ActivePoints, {
      points: points,
      mainColor: getLegendItemColor(this.props.stroke, this.props.fill),
      itemDataKey: this.props.dataKey,
      activeDot: this.props.activeDot,
      clipPath: activePointsClipPath
    }), this.props.isRange && Array.isArray(baseLine) && /*#__PURE__*/React.createElement(_ActivePoints.ActivePoints, {
      points: baseLine,
      mainColor: getLegendItemColor(this.props.stroke, this.props.fill),
      itemDataKey: this.props.dataKey,
      activeDot: this.props.activeDot,
      clipPath: activePointsClipPath
    }));
  }
}
var defaultAreaProps = exports.defaultAreaProps = {
  activeDot: true,
  animationBegin: 0,
  animationDuration: 1500,
  animationEasing: 'ease',
  connectNulls: false,
  dot: false,
  fill: '#3182bd',
  fillOpacity: 0.6,
  hide: false,
  isAnimationActive: 'auto',
  legendType: 'line',
  stroke: '#3182bd',
  strokeWidth: 1,
  type: 'linear',
  label: false,
  xAxisId: 0,
  yAxisId: 0,
  zIndex: _DefaultZIndexes.DefaultZIndexes.area
};
function AreaImpl(props) {
  var _useAppSelector;
  var _resolveDefaultProps = (0, _resolveDefaultProps2.resolveDefaultProps)(props, defaultAreaProps),
    {
      activeDot,
      animationBegin,
      animationDuration,
      animationEasing,
      connectNulls,
      dot,
      fill,
      fillOpacity,
      hide,
      isAnimationActive,
      legendType,
      stroke,
      xAxisId,
      yAxisId
    } = _resolveDefaultProps,
    everythingElse = _objectWithoutProperties(_resolveDefaultProps, _excluded2);
  var layout = (0, _chartLayoutContext.useChartLayout)();
  var chartName = (0, _selectors.useChartName)();
  var {
    needClip
  } = (0, _GraphicalItemClipPath.useNeedsClip)(xAxisId, yAxisId);
  var isPanorama = (0, _PanoramaContext.useIsPanorama)();
  var {
    points,
    isRange,
    baseLine
  } = (_useAppSelector = (0, _hooks.useAppSelector)(state => (0, _areaSelectors.selectArea)(state, props.id, isPanorama))) !== null && _useAppSelector !== void 0 ? _useAppSelector : {};
  var plotArea = (0, _hooks2.usePlotArea)();
  if (layout !== 'horizontal' && layout !== 'vertical' || plotArea == null) {
    // Can't render Area in an unsupported layout
    return null;
  }
  if (chartName !== 'AreaChart' && chartName !== 'ComposedChart') {
    // There is nothing stopping us from rendering Area in other charts, except for historical reasons. Do we want to allow that?
    return null;
  }
  var {
    height,
    width,
    x: left,
    y: top
  } = plotArea;
  if (!points || !points.length) {
    return null;
  }
  return /*#__PURE__*/React.createElement(AreaWithState, _extends({}, everythingElse, {
    activeDot: activeDot,
    animationBegin: animationBegin,
    animationDuration: animationDuration,
    animationEasing: animationEasing,
    baseLine: baseLine,
    connectNulls: connectNulls,
    dot: dot,
    fill: fill,
    fillOpacity: fillOpacity,
    height: height,
    hide: hide,
    layout: layout,
    isAnimationActive: isAnimationActive === 'auto' ? !_Global.Global.isSsr : isAnimationActive,
    isRange: isRange,
    legendType: legendType,
    needClip: needClip,
    points: points,
    stroke: stroke,
    width: width,
    left: left,
    top: top,
    xAxisId: xAxisId,
    yAxisId: yAxisId
  }));
}
var getBaseValue = (layout, chartBaseValue, itemBaseValue, xAxis, yAxis) => {
  // The baseValue can be defined both on the AreaChart, and on the Area.
  // The value for the item takes precedence.
  var baseValue = itemBaseValue !== null && itemBaseValue !== void 0 ? itemBaseValue : chartBaseValue;
  if ((0, _DataUtils.isNumber)(baseValue)) {
    return baseValue;
  }
  var numericAxis = layout === 'horizontal' ? yAxis : xAxis;
  // @ts-expect-error d3scale .domain() returns unknown, Math.max expects number
  var domain = numericAxis.scale.domain();
  if (numericAxis.type === 'number') {
    var domainMax = Math.max(domain[0], domain[1]);
    var domainMin = Math.min(domain[0], domain[1]);
    if (baseValue === 'dataMin') {
      return domainMin;
    }
    if (baseValue === 'dataMax') {
      return domainMax;
    }
    return domainMax < 0 ? domainMax : Math.max(Math.min(domain[0], domain[1]), 0);
  }
  if (baseValue === 'dataMin') {
    return domain[0];
  }
  if (baseValue === 'dataMax') {
    return domain[1];
  }
  return domain[0];
};
exports.getBaseValue = getBaseValue;
function computeArea(_ref0) {
  var {
    areaSettings: {
      connectNulls,
      baseValue: itemBaseValue,
      dataKey
    },
    stackedData,
    layout,
    chartBaseValue,
    xAxis,
    yAxis,
    displayedData,
    dataStartIndex,
    xAxisTicks,
    yAxisTicks,
    bandSize
  } = _ref0;
  var hasStack = stackedData && stackedData.length;
  var baseValue = getBaseValue(layout, chartBaseValue, itemBaseValue, xAxis, yAxis);
  var isHorizontalLayout = layout === 'horizontal';
  var isRange = false;
  var points = displayedData.map((entry, index) => {
    var _valueAsArray$, _valueAsArray, _xAxis$scale$map;
    var valueAsArray;
    if (hasStack) {
      valueAsArray = stackedData[dataStartIndex + index];
    } else {
      var rawValue = (0, _ChartUtils.getValueByDataKey)(entry, dataKey);
      if (!Array.isArray(rawValue)) {
        // @ts-expect-error getValueByDataKey is not checking its return value
        valueAsArray = [baseValue, rawValue];
      } else {
        // @ts-expect-error getValueByDataKey is not checking its return value
        valueAsArray = rawValue;
        isRange = true;
      }
    }
    var value1 = (_valueAsArray$ = (_valueAsArray = valueAsArray) === null || _valueAsArray === void 0 ? void 0 : _valueAsArray[1]) !== null && _valueAsArray$ !== void 0 ? _valueAsArray$ : null;
    var isBreakPoint = value1 == null || hasStack && !connectNulls && (0, _ChartUtils.getValueByDataKey)(entry, dataKey) == null;
    if (isHorizontalLayout) {
      var _yAxis$scale$map;
      return {
        x: (0, _ChartUtils.getCateCoordinateOfLine)({
          axis: xAxis,
          ticks: xAxisTicks,
          bandSize,
          entry,
          index
        }),
        y: isBreakPoint ? null : (_yAxis$scale$map = yAxis.scale.map(value1)) !== null && _yAxis$scale$map !== void 0 ? _yAxis$scale$map : null,
        value: valueAsArray,
        payload: entry
      };
    }
    return {
      x: isBreakPoint ? null : (_xAxis$scale$map = xAxis.scale.map(value1)) !== null && _xAxis$scale$map !== void 0 ? _xAxis$scale$map : null,
      y: (0, _ChartUtils.getCateCoordinateOfLine)({
        axis: yAxis,
        ticks: yAxisTicks,
        bandSize,
        entry,
        index
      }),
      value: valueAsArray,
      payload: entry
    };
  });
  var baseLine;
  if (hasStack || isRange) {
    baseLine = points.map(entry => {
      var _xAxis$scale$map2;
      var x = Array.isArray(entry.value) ? entry.value[0] : null;
      if (isHorizontalLayout) {
        var _yAxis$scale$map2;
        return {
          x: entry.x,
          y: x != null && entry.y != null ? (_yAxis$scale$map2 = yAxis.scale.map(x)) !== null && _yAxis$scale$map2 !== void 0 ? _yAxis$scale$map2 : null : null,
          payload: entry.payload
        };
      }
      return {
        x: x != null ? (_xAxis$scale$map2 = xAxis.scale.map(x)) !== null && _xAxis$scale$map2 !== void 0 ? _xAxis$scale$map2 : null : null,
        y: entry.y,
        payload: entry.payload
      };
    });
  } else {
    baseLine = isHorizontalLayout ? yAxis.scale.map(baseValue) : xAxis.scale.map(baseValue);
  }
  return {
    points,
    baseLine: baseLine !== null && baseLine !== void 0 ? baseLine : 0,
    isRange
  };
}
function AreaFn(outsideProps) {
  var props = (0, _resolveDefaultProps2.resolveDefaultProps)(outsideProps, defaultAreaProps);
  var isPanorama = (0, _PanoramaContext.useIsPanorama)();
  // Report all props to Redux store first, before calling any hooks, to avoid circular dependencies.
  return /*#__PURE__*/React.createElement(_RegisterGraphicalItemId.RegisterGraphicalItemId, {
    id: props.id,
    type: "area"
  }, id => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(_SetLegendPayload.SetLegendPayload, {
    legendPayload: computeLegendPayloadFromAreaData(props)
  }), /*#__PURE__*/React.createElement(SetAreaTooltipEntrySettings, {
    dataKey: props.dataKey,
    data: props.data,
    stroke: props.stroke,
    strokeWidth: props.strokeWidth,
    fill: props.fill,
    name: props.name,
    hide: props.hide,
    unit: props.unit,
    tooltipType: props.tooltipType,
    id: id
  }), /*#__PURE__*/React.createElement(_SetGraphicalItem.SetCartesianGraphicalItem, {
    type: "area",
    id: id,
    data: props.data,
    dataKey: props.dataKey,
    xAxisId: props.xAxisId,
    yAxisId: props.yAxisId,
    zAxisId: 0,
    stackId: (0, _ChartUtils.getNormalizedStackId)(props.stackId),
    hide: props.hide,
    barSize: undefined,
    baseValue: props.baseValue,
    isPanorama: isPanorama,
    connectNulls: props.connectNulls
  }), /*#__PURE__*/React.createElement(AreaImpl, _extends({}, props, {
    id: id
  }))));
}

/**
 * @provides LabelListContext
 * @consumes CartesianChartContext
 */
var Area = exports.Area = /*#__PURE__*/React.memo(AreaFn, _propsAreEqual.propsAreEqual);
Area.displayName = 'Area';
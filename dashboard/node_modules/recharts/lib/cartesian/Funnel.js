"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.Funnel = Funnel;
exports.computeFunnelTrapezoids = computeFunnelTrapezoids;
exports.defaultFunnelProps = void 0;
var _react = _interopRequireWildcard(require("react"));
var React = _react;
var _omit = _interopRequireDefault(require("es-toolkit/compat/omit"));
var _clsx = require("clsx");
var _selectors = require("../state/selectors/selectors");
var _hooks = require("../state/hooks");
var _Layer = require("../container/Layer");
var _LabelList = require("../component/LabelList");
var _DataUtils = require("../util/DataUtils");
var _ChartUtils = require("../util/ChartUtils");
var _types = require("../util/types");
var _FunnelUtils = require("../util/FunnelUtils");
var _tooltipContext = require("../context/tooltipContext");
var _SetTooltipEntrySettings = require("../state/SetTooltipEntrySettings");
var _funnelSelectors = require("../state/selectors/funnelSelectors");
var _ReactUtils = require("../util/ReactUtils");
var _Cell = require("../component/Cell");
var _resolveDefaultProps2 = require("../util/resolveDefaultProps");
var _hooks2 = require("../hooks");
var _svgPropertiesNoEvents = require("../util/svgPropertiesNoEvents");
var _JavascriptAnimate = require("../animation/JavascriptAnimate");
var _useAnimationId = require("../util/useAnimationId");
var _RegisterGraphicalItemId = require("../context/RegisterGraphicalItemId");
var _excluded = ["onMouseEnter", "onClick", "onMouseLeave", "shape", "activeShape"],
  _excluded2 = ["id"],
  _excluded3 = ["stroke", "fill", "legendType", "hide", "isAnimationActive", "animationBegin", "animationDuration", "animationEasing", "nameKey", "lastShapeType", "id"],
  _excluded4 = ["id"];
function _interopRequireDefault(e) { return e && e.__esModule ? e : { default: e }; }
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
 * Internal props, combination of external props + defaultProps + private Recharts state
 */

/**
 * External props, intended for end users to fill in
 */

var SetFunnelTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
    nameKey,
    stroke,
    strokeWidth,
    fill,
    name,
    hide,
    tooltipType,
    data,
    trapezoids,
    id
  } = _ref;
  var tooltipEntrySettings = {
    dataDefinedOnItem: data,
    getPosition: index => {
      var _trapezoids$Number;
      return (_trapezoids$Number = trapezoids[Number(index)]) === null || _trapezoids$Number === void 0 ? void 0 : _trapezoids$Number.tooltipPosition;
    },
    settings: {
      stroke,
      strokeWidth,
      fill,
      dataKey,
      name,
      nameKey,
      hide,
      type: tooltipType,
      color: fill,
      unit: '',
      // Funnel does not have unit, why?
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(_SetTooltipEntrySettings.SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});
function FunnelLabelListProvider(_ref2) {
  var {
    showLabels,
    trapezoids,
    children
  } = _ref2;
  var labelListEntries = (0, _react.useMemo)(() => {
    if (!showLabels) {
      return undefined;
    }
    return trapezoids === null || trapezoids === void 0 ? void 0 : trapezoids.map(entry => {
      var viewBox = entry.labelViewBox;
      return _objectSpread(_objectSpread({}, viewBox), {}, {
        value: entry.name,
        payload: entry.payload,
        parentViewBox: entry.parentViewBox,
        viewBox,
        fill: entry.fill
      });
    });
  }, [showLabels, trapezoids]);
  return /*#__PURE__*/React.createElement(_LabelList.CartesianLabelListContextProvider, {
    value: labelListEntries
  }, children);
}
function FunnelTrapezoids(props) {
  var {
    trapezoids,
    allOtherFunnelProps
  } = props;
  var activeItemIndex = (0, _hooks.useAppSelector)(state => (0, _selectors.selectActiveIndex)(state, 'item', state.tooltip.settings.trigger, undefined));
  var {
      onMouseEnter: onMouseEnterFromProps,
      onClick: onItemClickFromProps,
      onMouseLeave: onMouseLeaveFromProps,
      shape,
      activeShape
    } = allOtherFunnelProps,
    restOfAllOtherProps = _objectWithoutProperties(allOtherFunnelProps, _excluded);
  var onMouseEnterFromContext = (0, _tooltipContext.useMouseEnterItemDispatch)(onMouseEnterFromProps, allOtherFunnelProps.dataKey, allOtherFunnelProps.id);
  var onMouseLeaveFromContext = (0, _tooltipContext.useMouseLeaveItemDispatch)(onMouseLeaveFromProps);
  var onClickFromContext = (0, _tooltipContext.useMouseClickItemDispatch)(onItemClickFromProps, allOtherFunnelProps.dataKey, allOtherFunnelProps.id);
  return /*#__PURE__*/React.createElement(React.Fragment, null, trapezoids.map((entry, i) => {
    var isActiveIndex = Boolean(activeShape) && activeItemIndex === String(i);
    var trapezoidOptions = isActiveIndex ? activeShape : shape;
    var _entry$option$isActiv = _objectSpread(_objectSpread({}, entry), {}, {
        option: trapezoidOptions,
        isActive: isActiveIndex,
        stroke: entry.stroke
      }),
      {
        id
      } = _entry$option$isActiv,
      trapezoidProps = _objectWithoutProperties(_entry$option$isActiv, _excluded2);
    return /*#__PURE__*/React.createElement(_Layer.Layer, _extends({
      key: "trapezoid-".concat(entry === null || entry === void 0 ? void 0 : entry.x, "-").concat(entry === null || entry === void 0 ? void 0 : entry.y, "-").concat(entry === null || entry === void 0 ? void 0 : entry.name, "-").concat(entry === null || entry === void 0 ? void 0 : entry.value),
      className: "recharts-funnel-trapezoid"
    }, (0, _types.adaptEventsOfChild)(restOfAllOtherProps, entry, i), {
      // @ts-expect-error the types need a bit of attention
      onMouseEnter: onMouseEnterFromContext(entry, i)
      // @ts-expect-error the types need a bit of attention
      ,
      onMouseLeave: onMouseLeaveFromContext(entry, i)
      // @ts-expect-error the types need a bit of attention
      ,
      onClick: onClickFromContext(entry, i)
    }), /*#__PURE__*/React.createElement(_FunnelUtils.FunnelTrapezoid, trapezoidProps));
  }));
}
function TrapezoidsWithAnimation(_ref3) {
  var {
    previousTrapezoidsRef,
    props
  } = _ref3;
  var {
    trapezoids,
    isAnimationActive,
    animationBegin,
    animationDuration,
    animationEasing,
    onAnimationEnd,
    onAnimationStart
  } = props;
  var prevTrapezoids = previousTrapezoidsRef.current;
  var [isAnimating, setIsAnimating] = (0, _react.useState)(false);
  var showLabels = !isAnimating;
  var animationId = (0, _useAnimationId.useAnimationId)(trapezoids, 'recharts-funnel-');
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
  return /*#__PURE__*/React.createElement(FunnelLabelListProvider, {
    showLabels: showLabels,
    trapezoids: trapezoids
  }, /*#__PURE__*/React.createElement(_JavascriptAnimate.JavascriptAnimate, {
    animationId: animationId,
    begin: animationBegin,
    duration: animationDuration,
    isActive: isAnimationActive,
    easing: animationEasing,
    key: animationId,
    onAnimationStart: handleAnimationStart,
    onAnimationEnd: handleAnimationEnd
  }, t => {
    var stepData = t === 1 ? trapezoids : trapezoids.map((entry, index) => {
      var prev = prevTrapezoids && prevTrapezoids[index];
      if (prev) {
        return _objectSpread(_objectSpread({}, entry), {}, {
          x: (0, _DataUtils.interpolate)(prev.x, entry.x, t),
          y: (0, _DataUtils.interpolate)(prev.y, entry.y, t),
          upperWidth: (0, _DataUtils.interpolate)(prev.upperWidth, entry.upperWidth, t),
          lowerWidth: (0, _DataUtils.interpolate)(prev.lowerWidth, entry.lowerWidth, t),
          height: (0, _DataUtils.interpolate)(prev.height, entry.height, t)
        });
      }
      return _objectSpread(_objectSpread({}, entry), {}, {
        x: (0, _DataUtils.interpolate)(entry.x + entry.upperWidth / 2, entry.x, t),
        y: (0, _DataUtils.interpolate)(entry.y + entry.height / 2, entry.y, t),
        upperWidth: (0, _DataUtils.interpolate)(0, entry.upperWidth, t),
        lowerWidth: (0, _DataUtils.interpolate)(0, entry.lowerWidth, t),
        height: (0, _DataUtils.interpolate)(0, entry.height, t)
      });
    });
    if (t > 0) {
      // eslint-disable-next-line no-param-reassign
      previousTrapezoidsRef.current = stepData;
    }
    return /*#__PURE__*/React.createElement(_Layer.Layer, null, /*#__PURE__*/React.createElement(FunnelTrapezoids, {
      trapezoids: stepData,
      allOtherFunnelProps: props
    }));
  }), /*#__PURE__*/React.createElement(_LabelList.LabelListFromLabelProp, {
    label: props.label
  }), props.children);
}
function RenderTrapezoids(props) {
  var previousTrapezoidsRef = (0, _react.useRef)(undefined);
  return /*#__PURE__*/React.createElement(TrapezoidsWithAnimation, {
    props: props,
    previousTrapezoidsRef: previousTrapezoidsRef
  });
}
var getRealWidthHeight = (customWidth, offset) => {
  var {
    width,
    height,
    left,
    top
  } = offset;
  var realWidth = (0, _DataUtils.getPercentValue)(customWidth, width, width);
  return {
    realWidth,
    realHeight: height,
    offsetX: left,
    offsetY: top
  };
};
var defaultFunnelProps = exports.defaultFunnelProps = {
  animationBegin: 400,
  animationDuration: 1500,
  animationEasing: 'ease',
  fill: '#808080',
  hide: false,
  isAnimationActive: 'auto',
  lastShapeType: 'triangle',
  legendType: 'rect',
  nameKey: 'name',
  reversed: false,
  stroke: '#fff'
};
function FunnelImpl(props) {
  var plotArea = (0, _hooks2.usePlotArea)();
  var {
      stroke,
      fill,
      legendType,
      hide,
      isAnimationActive,
      animationBegin,
      animationDuration,
      animationEasing,
      nameKey,
      lastShapeType,
      id
    } = props,
    everythingElse = _objectWithoutProperties(props, _excluded3);
  var presentationProps = (0, _svgPropertiesNoEvents.svgPropertiesNoEvents)(props);
  var cells = (0, _ReactUtils.findAllByType)(props.children, _Cell.Cell);
  var funnelSettings = (0, _react.useMemo)(() => ({
    dataKey: props.dataKey,
    nameKey,
    data: props.data,
    tooltipType: props.tooltipType,
    lastShapeType,
    reversed: props.reversed,
    customWidth: props.width,
    cells,
    presentationProps,
    id
  }), [props.dataKey, nameKey, props.data, props.tooltipType, lastShapeType, props.reversed, props.width, cells, presentationProps, id]);
  var trapezoids = (0, _hooks.useAppSelector)(state => (0, _funnelSelectors.selectFunnelTrapezoids)(state, funnelSettings));
  if (hide || !trapezoids || !trapezoids.length || !plotArea) {
    return null;
  }
  var {
    height,
    width
  } = plotArea;
  var layerClass = (0, _clsx.clsx)('recharts-trapezoids', props.className);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(SetFunnelTooltipEntrySettings, {
    dataKey: props.dataKey,
    nameKey: props.nameKey,
    stroke: props.stroke,
    strokeWidth: props.strokeWidth,
    fill: props.fill,
    name: props.name,
    hide: props.hide,
    tooltipType: props.tooltipType,
    data: props.data,
    trapezoids: trapezoids,
    id: id
  }), /*#__PURE__*/React.createElement(_Layer.Layer, {
    className: layerClass
  }, /*#__PURE__*/React.createElement(RenderTrapezoids, _extends({}, everythingElse, {
    id: id,
    stroke: stroke,
    fill: fill,
    nameKey: nameKey,
    lastShapeType: lastShapeType,
    animationBegin: animationBegin,
    animationDuration: animationDuration,
    animationEasing: animationEasing,
    isAnimationActive: isAnimationActive,
    hide: hide,
    legendType: legendType,
    height: height,
    width: width,
    trapezoids: trapezoids
  }))));
}
function computeFunnelTrapezoids(_ref4) {
  var {
    dataKey,
    nameKey,
    displayedData,
    tooltipType,
    lastShapeType,
    reversed,
    offset,
    customWidth,
    graphicalItemId
  } = _ref4;
  var {
    realHeight,
    realWidth,
    offsetX,
    offsetY
  } = getRealWidthHeight(customWidth, offset);
  var values = displayedData.map(entry => {
    var val = (0, _ChartUtils.getValueByDataKey)(entry, dataKey, 0);
    return typeof val === 'number' ? val : 0;
  });
  var maxValue = Math.max.apply(null, values);
  var len = displayedData.length;
  var rowHeight = realHeight / len;
  var parentViewBox = {
    x: offset.left,
    y: offset.top,
    width: offset.width,
    height: offset.height
  };
  var trapezoids = displayedData.map((entry, i) => {
    // @ts-expect-error getValueByDataKey does not validate the output type
    var rawVal = (0, _ChartUtils.getValueByDataKey)(entry, dataKey, 0);
    var name = String((0, _ChartUtils.getValueByDataKey)(entry, nameKey, i));
    var val = rawVal;
    var nextVal;
    if (i !== len - 1) {
      var nextDataValue = (0, _ChartUtils.getValueByDataKey)(displayedData[i + 1], dataKey, 0);
      if (typeof nextDataValue === 'number') {
        nextVal = nextDataValue;
      } else if (Array.isArray(nextDataValue)) {
        var [first, second] = nextDataValue;
        if (typeof first === 'number') {
          val = first;
        }
        if (typeof second === 'number') {
          nextVal = second;
        }
      }
    } else if (rawVal instanceof Array && rawVal.length === 2) {
      var [_first, _second] = rawVal;
      if (typeof _first === 'number') {
        val = _first;
      }
      if (typeof _second === 'number') {
        nextVal = _second;
      }
    } else if (lastShapeType === 'rectangle') {
      nextVal = val;
    } else {
      nextVal = 0;
    }

    // @ts-expect-error this is a problem if we have ranged values because `val` can be an array
    var x = (maxValue - val) * realWidth / (2 * maxValue) + offsetX;
    var y = rowHeight * i + offsetY;
    // @ts-expect-error getValueByDataKey does not validate the output type
    var upperWidth = val / maxValue * realWidth;
    // @ts-expect-error nextVal could be an array
    var lowerWidth = nextVal / maxValue * realWidth;
    var tooltipPayload = [{
      name,
      value: val,
      payload: entry,
      dataKey,
      type: tooltipType,
      graphicalItemId
    }];
    var tooltipPosition = {
      x: x + upperWidth / 2,
      y: y + rowHeight / 2
    };
    var trapezoidViewBox = {
      x,
      y,
      upperWidth,
      lowerWidth,
      width: Math.max(upperWidth, lowerWidth),
      height: rowHeight
    };
    return _objectSpread(_objectSpread(_objectSpread({}, trapezoidViewBox), {}, {
      name,
      val,
      tooltipPayload,
      tooltipPosition
    }, entry != null && typeof entry === 'object' ? (0, _omit.default)(entry, ['width']) : {}), {}, {
      payload: entry,
      parentViewBox,
      labelViewBox: trapezoidViewBox
    });
  });
  if (reversed) {
    trapezoids = trapezoids.map((entry, index) => {
      var reversedViewBox = {
        x: entry.x - (entry.lowerWidth - entry.upperWidth) / 2,
        y: entry.y - index * rowHeight + (len - 1 - index) * rowHeight,
        upperWidth: entry.lowerWidth,
        lowerWidth: entry.upperWidth,
        width: Math.max(entry.lowerWidth, entry.upperWidth),
        height: rowHeight
      };
      return _objectSpread(_objectSpread(_objectSpread({}, entry), reversedViewBox), {}, {
        tooltipPosition: _objectSpread(_objectSpread({}, entry.tooltipPosition), {}, {
          y: entry.y - index * rowHeight + (len - 1 - index) * rowHeight + rowHeight / 2
        }),
        labelViewBox: reversedViewBox
      });
    });
  }
  return trapezoids;
}

/**
 * @consumes CartesianViewBoxContext
 * @provides LabelListContext
 * @provides CellReader
 */
function Funnel(outsideProps) {
  var _resolveDefaultProps = (0, _resolveDefaultProps2.resolveDefaultProps)(outsideProps, defaultFunnelProps),
    {
      id: externalId
    } = _resolveDefaultProps,
    props = _objectWithoutProperties(_resolveDefaultProps, _excluded4);
  return /*#__PURE__*/React.createElement(_RegisterGraphicalItemId.RegisterGraphicalItemId, {
    id: externalId,
    type: "funnel"
  }, id => /*#__PURE__*/React.createElement(FunnelImpl, _extends({}, props, {
    id: id
  })));
}
Funnel.displayName = 'Funnel';
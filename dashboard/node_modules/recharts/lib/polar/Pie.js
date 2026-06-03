"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.Pie = Pie;
exports.computePieSectors = computePieSectors;
exports.defaultPieProps = void 0;
var _react = _interopRequireWildcard(require("react"));
var React = _react;
var _get = _interopRequireDefault(require("es-toolkit/compat/get"));
var _clsx = require("clsx");
var _pieSelectors = require("../state/selectors/pieSelectors");
var _hooks = require("../state/hooks");
var _Layer = require("../container/Layer");
var _Curve = require("../shape/Curve");
var _Text = require("../component/Text");
var _Cell = require("../component/Cell");
var _ReactUtils = require("../util/ReactUtils");
var _PolarUtils = require("../util/PolarUtils");
var _DataUtils = require("../util/DataUtils");
var _ChartUtils = require("../util/ChartUtils");
var _types = require("../util/types");
var _ActiveShapeUtils = require("../util/ActiveShapeUtils");
var _tooltipContext = require("../context/tooltipContext");
var _SetTooltipEntrySettings = require("../state/SetTooltipEntrySettings");
var _tooltipSelectors = require("../state/selectors/tooltipSelectors");
var _SetLegendPayload = require("../state/SetLegendPayload");
var _Constants = require("../util/Constants");
var _useAnimationId = require("../util/useAnimationId");
var _resolveDefaultProps = require("../util/resolveDefaultProps");
var _RegisterGraphicalItemId = require("../context/RegisterGraphicalItemId");
var _SetGraphicalItem = require("../state/SetGraphicalItem");
var _svgPropertiesNoEvents = require("../util/svgPropertiesNoEvents");
var _JavascriptAnimate = require("../animation/JavascriptAnimate");
var _LabelList = require("../component/LabelList");
var _ZIndexLayer = require("../zIndex/ZIndexLayer");
var _DefaultZIndexes = require("../zIndex/DefaultZIndexes");
var _getClassNameFromUnknown = require("../util/getClassNameFromUnknown");
var _excluded = ["key"],
  _excluded2 = ["onMouseEnter", "onClick", "onMouseLeave"],
  _excluded3 = ["id"],
  _excluded4 = ["id"];
function _interopRequireDefault(e) { return e && e.__esModule ? e : { default: e }; }
function _interopRequireWildcard(e, t) { if ("function" == typeof WeakMap) var r = new WeakMap(), n = new WeakMap(); return (_interopRequireWildcard = function _interopRequireWildcard(e, t) { if (!t && e && e.__esModule) return e; var o, i, f = { __proto__: null, default: e }; if (null === e || "object" != typeof e && "function" != typeof e) return f; if (o = t ? n : r) { if (o.has(e)) return o.get(e); o.set(e, f); } for (var _t in e) "default" !== _t && {}.hasOwnProperty.call(e, _t) && ((i = (o = Object.defineProperty) && Object.getOwnPropertyDescriptor(e, _t)) && (i.get || i.set) ? o(f, _t, i) : f[_t] = e[_t]); return f; })(e, t); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
/**
 * The `label` prop in Pie accepts a variety of alternatives.
 */

/**
 * We spread the data object into the sector data item,
 * so we can't really know what is going to be inside.
 *
 * This type represents our best effort, but it all depends on the input data
 * and what is inside of it.
 *
 * https://github.com/recharts/recharts/issues/6380
 * https://github.com/recharts/recharts/discussions/6375
 */

/**
 * Internal props, combination of external props + defaultProps + private Recharts state
 */

function SetPiePayloadLegend(props) {
  var cells = (0, _react.useMemo)(() => (0, _ReactUtils.findAllByType)(props.children, _Cell.Cell), [props.children]);
  var legendPayload = (0, _hooks.useAppSelector)(state => (0, _pieSelectors.selectPieLegend)(state, props.id, cells));
  if (legendPayload == null) {
    return null;
  }
  return /*#__PURE__*/React.createElement(_SetLegendPayload.SetPolarLegendPayload, {
    legendPayload: legendPayload
  });
}
var SetPieTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
    nameKey,
    sectors,
    stroke,
    strokeWidth,
    fill,
    name,
    hide,
    tooltipType,
    id
  } = _ref;
  var tooltipEntrySettings = {
    dataDefinedOnItem: sectors.map(p => p.tooltipPayload),
    getPosition: index => {
      var _sectors$Number;
      return (_sectors$Number = sectors[Number(index)]) === null || _sectors$Number === void 0 ? void 0 : _sectors$Number.tooltipPosition;
    },
    settings: {
      stroke,
      strokeWidth,
      fill,
      dataKey,
      nameKey,
      name: (0, _ChartUtils.getTooltipNameProp)(name, dataKey),
      hide,
      type: tooltipType,
      color: fill,
      unit: '',
      // why doesn't Pie support unit?
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(_SetTooltipEntrySettings.SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});
var getTextAnchor = (x, cx) => {
  if (x > cx) {
    return 'start';
  }
  if (x < cx) {
    return 'end';
  }
  return 'middle';
};
var getOuterRadius = (dataPoint, outerRadius, maxPieRadius) => {
  if (typeof outerRadius === 'function') {
    return (0, _DataUtils.getPercentValue)(outerRadius(dataPoint), maxPieRadius, maxPieRadius * 0.8);
  }
  return (0, _DataUtils.getPercentValue)(outerRadius, maxPieRadius, maxPieRadius * 0.8);
};
var parseCoordinateOfPie = (pieSettings, offset, dataPoint) => {
  var {
    top,
    left,
    width,
    height
  } = offset;
  var maxPieRadius = (0, _PolarUtils.getMaxRadius)(width, height);
  var cx = left + (0, _DataUtils.getPercentValue)(pieSettings.cx, width, width / 2);
  var cy = top + (0, _DataUtils.getPercentValue)(pieSettings.cy, height, height / 2);
  var innerRadius = (0, _DataUtils.getPercentValue)(pieSettings.innerRadius, maxPieRadius, 0);
  var outerRadius = getOuterRadius(dataPoint, pieSettings.outerRadius, maxPieRadius);
  var maxRadius = pieSettings.maxRadius || Math.sqrt(width * width + height * height) / 2;
  return {
    cx,
    cy,
    innerRadius,
    outerRadius,
    maxRadius
  };
};
var parseDeltaAngle = (startAngle, endAngle) => {
  var sign = (0, _DataUtils.mathSign)(endAngle - startAngle);
  var deltaAngle = Math.min(Math.abs(endAngle - startAngle), 360);
  return sign * deltaAngle;
};
var renderLabelLineItem = (option, props) => {
  if (/*#__PURE__*/React.isValidElement(option)) {
    // @ts-expect-error we can't know if the type of props matches the element
    return /*#__PURE__*/React.cloneElement(option, props);
  }
  if (typeof option === 'function') {
    return option(props);
  }
  var className = (0, _clsx.clsx)('recharts-pie-label-line', typeof option !== 'boolean' ? option.className : '');
  // React doesn't like it when we spread a key property onto an element
  var {
      key
    } = props,
    otherProps = _objectWithoutProperties(props, _excluded);
  return /*#__PURE__*/React.createElement(_Curve.Curve, _extends({}, otherProps, {
    type: "linear",
    className: className
  }));
};
var renderLabelItem = (option, props, value) => {
  if (/*#__PURE__*/React.isValidElement(option)) {
    // @ts-expect-error element cloning is not typed
    return /*#__PURE__*/React.cloneElement(option, props);
  }
  var label = value;
  if (typeof option === 'function') {
    label = option(props);
    if (/*#__PURE__*/React.isValidElement(label)) {
      return label;
    }
  }
  var className = (0, _clsx.clsx)('recharts-pie-label-text', (0, _getClassNameFromUnknown.getClassNameFromUnknown)(option));
  return /*#__PURE__*/React.createElement(_Text.Text, _extends({}, props, {
    alignmentBaseline: "middle",
    className: className
  }), label);
};
function PieLabels(_ref2) {
  var {
    sectors,
    props,
    showLabels
  } = _ref2;
  var {
    label,
    labelLine,
    dataKey
  } = props;
  if (!showLabels || !label || !sectors) {
    return null;
  }
  var pieProps = (0, _svgPropertiesNoEvents.svgPropertiesNoEvents)(props);
  var customLabelProps = (0, _svgPropertiesNoEvents.svgPropertiesNoEventsFromUnknown)(label);
  var customLabelLineProps = (0, _svgPropertiesNoEvents.svgPropertiesNoEventsFromUnknown)(labelLine);
  var offsetRadius = typeof label === 'object' && 'offsetRadius' in label && typeof label.offsetRadius === 'number' && label.offsetRadius || 20;
  var labels = sectors.map((entry, i) => {
    var midAngle = (entry.startAngle + entry.endAngle) / 2;
    var endPoint = (0, _PolarUtils.polarToCartesian)(entry.cx, entry.cy, entry.outerRadius + offsetRadius, midAngle);
    var labelProps = _objectSpread(_objectSpread(_objectSpread(_objectSpread({}, pieProps), entry), {}, {
      // @ts-expect-error customLabelProps is contributing unknown props
      stroke: 'none'
    }, customLabelProps), {}, {
      index: i,
      textAnchor: getTextAnchor(endPoint.x, entry.cx)
    }, endPoint);
    var lineProps = _objectSpread(_objectSpread(_objectSpread(_objectSpread({}, pieProps), entry), {}, {
      // @ts-expect-error customLabelLineProps is contributing unknown props
      fill: 'none',
      // @ts-expect-error customLabelLineProps is contributing unknown props
      stroke: entry.fill
    }, customLabelLineProps), {}, {
      index: i,
      points: [(0, _PolarUtils.polarToCartesian)(entry.cx, entry.cy, entry.outerRadius, midAngle), endPoint],
      key: 'line'
    });
    return /*#__PURE__*/React.createElement(_ZIndexLayer.ZIndexLayer, {
      zIndex: _DefaultZIndexes.DefaultZIndexes.label,
      key: "label-".concat(entry.startAngle, "-").concat(entry.endAngle, "-").concat(entry.midAngle, "-").concat(i)
    }, /*#__PURE__*/React.createElement(_Layer.Layer, null, labelLine && renderLabelLineItem(labelLine, lineProps), renderLabelItem(label, labelProps, (0, _ChartUtils.getValueByDataKey)(entry, dataKey))));
  });
  return /*#__PURE__*/React.createElement(_Layer.Layer, {
    className: "recharts-pie-labels"
  }, labels);
}
function PieLabelList(_ref3) {
  var {
    sectors,
    props,
    showLabels
  } = _ref3;
  var {
    label
  } = props;
  if (typeof label === 'object' && label != null && 'position' in label) {
    return /*#__PURE__*/React.createElement(_LabelList.LabelListFromLabelProp, {
      label: label
    });
  }
  return /*#__PURE__*/React.createElement(PieLabels, {
    sectors: sectors,
    props: props,
    showLabels: showLabels
  });
}
function PieSectors(props) {
  var {
    sectors,
    activeShape,
    inactiveShape: inactiveShapeProp,
    allOtherPieProps,
    shape,
    id
  } = props;
  var activeIndex = (0, _hooks.useAppSelector)(_tooltipSelectors.selectActiveTooltipIndex);
  var activeDataKey = (0, _hooks.useAppSelector)(_tooltipSelectors.selectActiveTooltipDataKey);
  var activeGraphicalItemId = (0, _hooks.useAppSelector)(_tooltipSelectors.selectActiveTooltipGraphicalItemId);
  var {
      onMouseEnter: onMouseEnterFromProps,
      onClick: onItemClickFromProps,
      onMouseLeave: onMouseLeaveFromProps
    } = allOtherPieProps,
    restOfAllOtherProps = _objectWithoutProperties(allOtherPieProps, _excluded2);
  var onMouseEnterFromContext = (0, _tooltipContext.useMouseEnterItemDispatch)(onMouseEnterFromProps, allOtherPieProps.dataKey, id);
  var onMouseLeaveFromContext = (0, _tooltipContext.useMouseLeaveItemDispatch)(onMouseLeaveFromProps);
  var onClickFromContext = (0, _tooltipContext.useMouseClickItemDispatch)(onItemClickFromProps, allOtherPieProps.dataKey, id);
  if (sectors == null || sectors.length === 0) {
    return null;
  }
  return /*#__PURE__*/React.createElement(React.Fragment, null, sectors.map((entry, i) => {
    if ((entry === null || entry === void 0 ? void 0 : entry.startAngle) === 0 && (entry === null || entry === void 0 ? void 0 : entry.endAngle) === 0 && sectors.length !== 1) return null;

    // For Pie charts, when multiple Pies share the same dataKey, we need to ensure only the hovered Pie's sector is active.
    // We do this by checking if the active graphical item ID matches this Pie's ID.
    var graphicalItemMatches = activeGraphicalItemId == null || activeGraphicalItemId === id;
    var isActive = String(i) === activeIndex && (activeDataKey == null || allOtherPieProps.dataKey === activeDataKey) && graphicalItemMatches;
    var inactiveShape = activeIndex ? inactiveShapeProp : null;
    var sectorOptions = activeShape && isActive ? activeShape : inactiveShape;
    var sectorProps = _objectSpread(_objectSpread({}, entry), {}, {
      stroke: entry.stroke,
      tabIndex: -1,
      [_Constants.DATA_ITEM_INDEX_ATTRIBUTE_NAME]: i,
      [_Constants.DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME]: id
    });
    return /*#__PURE__*/React.createElement(_Layer.Layer, _extends({
      key: "sector-".concat(entry === null || entry === void 0 ? void 0 : entry.startAngle, "-").concat(entry === null || entry === void 0 ? void 0 : entry.endAngle, "-").concat(entry.midAngle, "-").concat(i),
      tabIndex: -1,
      className: "recharts-pie-sector"
    }, (0, _types.adaptEventsOfChild)(restOfAllOtherProps, entry, i), {
      // @ts-expect-error the types need a bit of attention
      onMouseEnter: onMouseEnterFromContext(entry, i)
      // @ts-expect-error the types need a bit of attention
      ,
      onMouseLeave: onMouseLeaveFromContext(entry, i)
      // @ts-expect-error the types need a bit of attention
      ,
      onClick: onClickFromContext(entry, i)
    }), /*#__PURE__*/React.createElement(_ActiveShapeUtils.Shape, _extends({
      option: shape !== null && shape !== void 0 ? shape : sectorOptions,
      index: i,
      shapeType: "sector",
      isActive: isActive
    }, sectorProps)));
  }));
}
function computePieSectors(_ref4) {
  var _pieSettings$paddingA;
  var {
    pieSettings,
    displayedData,
    cells,
    offset
  } = _ref4;
  var {
    cornerRadius,
    startAngle,
    endAngle,
    dataKey,
    nameKey,
    tooltipType
  } = pieSettings;
  var minAngle = Math.abs(pieSettings.minAngle);
  var deltaAngle = parseDeltaAngle(startAngle, endAngle);
  var absDeltaAngle = Math.abs(deltaAngle);
  var paddingAngle = displayedData.length <= 1 ? 0 : (_pieSettings$paddingA = pieSettings.paddingAngle) !== null && _pieSettings$paddingA !== void 0 ? _pieSettings$paddingA : 0;
  var notZeroItemCount = displayedData.filter(entry => (0, _ChartUtils.getValueByDataKey)(entry, dataKey, 0) !== 0).length;
  var totalPaddingAngle = (absDeltaAngle >= 360 ? notZeroItemCount : notZeroItemCount - 1) * paddingAngle;
  var realTotalAngle = absDeltaAngle - notZeroItemCount * minAngle - totalPaddingAngle;
  var sum = displayedData.reduce((result, entry) => {
    var val = (0, _ChartUtils.getValueByDataKey)(entry, dataKey, 0);
    return result + ((0, _DataUtils.isNumber)(val) ? val : 0);
  }, 0);
  var sectors;
  if (sum > 0) {
    var prev;
    sectors = displayedData.map((entry, i) => {
      // @ts-expect-error getValueByDataKey does not validate the output type
      var val = (0, _ChartUtils.getValueByDataKey)(entry, dataKey, 0);
      // @ts-expect-error getValueByDataKey does not validate the output type
      var name = (0, _ChartUtils.getValueByDataKey)(entry, nameKey, i);
      var coordinate = parseCoordinateOfPie(pieSettings, offset, entry);
      var percent = ((0, _DataUtils.isNumber)(val) ? val : 0) / sum;
      var tempStartAngle;

      // @ts-expect-error can't spread unknown
      var entryWithCellInfo = _objectSpread(_objectSpread({}, entry), cells && cells[i] && cells[i].props);
      if (i) {
        tempStartAngle = prev.endAngle + (0, _DataUtils.mathSign)(deltaAngle) * paddingAngle * (val !== 0 ? 1 : 0);
      } else {
        tempStartAngle = startAngle;
      }
      var tempEndAngle = tempStartAngle + (0, _DataUtils.mathSign)(deltaAngle) * ((val !== 0 ? minAngle : 0) + percent * realTotalAngle);
      var midAngle = (tempStartAngle + tempEndAngle) / 2;
      var middleRadius = (coordinate.innerRadius + coordinate.outerRadius) / 2;
      var tooltipPayload = [{
        name,
        value: val,
        payload: entryWithCellInfo,
        dataKey,
        type: tooltipType,
        graphicalItemId: pieSettings.id
      }];
      var tooltipPosition = (0, _PolarUtils.polarToCartesian)(coordinate.cx, coordinate.cy, middleRadius, midAngle);
      prev = _objectSpread(_objectSpread(_objectSpread(_objectSpread({}, pieSettings.presentationProps), {}, {
        percent,
        cornerRadius: typeof cornerRadius === 'string' ? parseFloat(cornerRadius) : cornerRadius,
        name,
        tooltipPayload,
        midAngle,
        middleRadius,
        tooltipPosition
      }, entryWithCellInfo), coordinate), {}, {
        value: val,
        dataKey,
        startAngle: tempStartAngle,
        endAngle: tempEndAngle,
        payload: entryWithCellInfo,
        paddingAngle: (0, _DataUtils.mathSign)(deltaAngle) * paddingAngle
      });
      return prev;
    });
  }
  return sectors;
}
function PieLabelListProvider(_ref5) {
  var {
    showLabels,
    sectors,
    children
  } = _ref5;
  var labelListEntries = (0, _react.useMemo)(() => {
    if (!showLabels || !sectors) {
      return [];
    }
    return sectors.map(entry => ({
      value: entry.value,
      payload: entry.payload,
      clockWise: false,
      parentViewBox: undefined,
      viewBox: {
        cx: entry.cx,
        cy: entry.cy,
        innerRadius: entry.innerRadius,
        outerRadius: entry.outerRadius,
        startAngle: entry.startAngle,
        endAngle: entry.endAngle,
        clockWise: false
      },
      fill: entry.fill
    }));
  }, [sectors, showLabels]);
  return /*#__PURE__*/React.createElement(_LabelList.PolarLabelListContextProvider, {
    value: showLabels ? labelListEntries : undefined
  }, children);
}
function SectorsWithAnimation(_ref6) {
  var {
    props,
    previousSectorsRef,
    id
  } = _ref6;
  var {
    sectors,
    isAnimationActive,
    animationBegin,
    animationDuration,
    animationEasing,
    activeShape,
    inactiveShape,
    onAnimationStart,
    onAnimationEnd
  } = props;
  var animationId = (0, _useAnimationId.useAnimationId)(props, 'recharts-pie-');
  var prevSectors = previousSectorsRef.current;
  var [isAnimating, setIsAnimating] = (0, _react.useState)(false);
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
  return /*#__PURE__*/React.createElement(PieLabelListProvider, {
    showLabels: !isAnimating,
    sectors: sectors
  }, /*#__PURE__*/React.createElement(_JavascriptAnimate.JavascriptAnimate, {
    animationId: animationId,
    begin: animationBegin,
    duration: animationDuration,
    isActive: isAnimationActive,
    easing: animationEasing,
    onAnimationStart: handleAnimationStart,
    onAnimationEnd: handleAnimationEnd,
    key: animationId
  }, t => {
    var _first$startAngle;
    var stepData = [];
    var first = sectors && sectors[0];
    var curAngle = (_first$startAngle = first === null || first === void 0 ? void 0 : first.startAngle) !== null && _first$startAngle !== void 0 ? _first$startAngle : 0;
    sectors === null || sectors === void 0 || sectors.forEach((entry, index) => {
      var prev = prevSectors && prevSectors[index];
      var paddingAngle = index > 0 ? (0, _get.default)(entry, 'paddingAngle', 0) : 0;
      if (prev) {
        var angle = (0, _DataUtils.interpolate)(prev.endAngle - prev.startAngle, entry.endAngle - entry.startAngle, t);
        var latest = _objectSpread(_objectSpread({}, entry), {}, {
          startAngle: curAngle + paddingAngle,
          endAngle: curAngle + angle + paddingAngle
        });
        stepData.push(latest);
        curAngle = latest.endAngle;
      } else {
        var {
          endAngle,
          startAngle
        } = entry;
        var deltaAngle = (0, _DataUtils.interpolate)(0, endAngle - startAngle, t);
        var _latest = _objectSpread(_objectSpread({}, entry), {}, {
          startAngle: curAngle + paddingAngle,
          endAngle: curAngle + deltaAngle + paddingAngle
        });
        stepData.push(_latest);
        curAngle = _latest.endAngle;
      }
    });

    // eslint-disable-next-line no-param-reassign
    previousSectorsRef.current = stepData;
    return /*#__PURE__*/React.createElement(_Layer.Layer, null, /*#__PURE__*/React.createElement(PieSectors, {
      sectors: stepData,
      activeShape: activeShape,
      inactiveShape: inactiveShape,
      allOtherPieProps: props,
      shape: props.shape,
      id: id
    }));
  }), /*#__PURE__*/React.createElement(PieLabelList, {
    showLabels: !isAnimating,
    sectors: sectors,
    props: props
  }), props.children);
}
var defaultPieProps = exports.defaultPieProps = {
  animationBegin: 400,
  animationDuration: 1500,
  animationEasing: 'ease',
  cx: '50%',
  cy: '50%',
  dataKey: 'value',
  endAngle: 360,
  fill: '#808080',
  hide: false,
  innerRadius: 0,
  isAnimationActive: 'auto',
  label: false,
  labelLine: true,
  legendType: 'rect',
  minAngle: 0,
  nameKey: 'name',
  outerRadius: '80%',
  paddingAngle: 0,
  rootTabIndex: 0,
  startAngle: 0,
  stroke: '#fff',
  zIndex: _DefaultZIndexes.DefaultZIndexes.area
};
function PieImpl(props) {
  var {
      id
    } = props,
    propsWithoutId = _objectWithoutProperties(props, _excluded3);
  var {
    hide,
    className,
    rootTabIndex
  } = props;
  var cells = (0, _react.useMemo)(() => (0, _ReactUtils.findAllByType)(props.children, _Cell.Cell), [props.children]);
  var sectors = (0, _hooks.useAppSelector)(state => (0, _pieSelectors.selectPieSectors)(state, id, cells));
  var previousSectorsRef = (0, _react.useRef)(null);
  var layerClass = (0, _clsx.clsx)('recharts-pie', className);
  if (hide || sectors == null) {
    previousSectorsRef.current = null;
    return /*#__PURE__*/React.createElement(_Layer.Layer, {
      tabIndex: rootTabIndex,
      className: layerClass
    });
  }
  return /*#__PURE__*/React.createElement(_ZIndexLayer.ZIndexLayer, {
    zIndex: props.zIndex
  }, /*#__PURE__*/React.createElement(SetPieTooltipEntrySettings, {
    dataKey: props.dataKey,
    nameKey: props.nameKey,
    sectors: sectors,
    stroke: props.stroke,
    strokeWidth: props.strokeWidth,
    fill: props.fill,
    name: props.name,
    hide: props.hide,
    tooltipType: props.tooltipType,
    id: id
  }), /*#__PURE__*/React.createElement(_Layer.Layer, {
    tabIndex: rootTabIndex,
    className: layerClass
  }, /*#__PURE__*/React.createElement(SectorsWithAnimation, {
    props: _objectSpread(_objectSpread({}, propsWithoutId), {}, {
      sectors
    }),
    previousSectorsRef: previousSectorsRef,
    id: id
  })));
}
/**
 * @consumes PolarChartContext
 * @provides LabelListContext
 * @provides CellReader
 */
function Pie(outsideProps) {
  var props = (0, _resolveDefaultProps.resolveDefaultProps)(outsideProps, defaultPieProps);
  var {
      id: externalId
    } = props,
    propsWithoutId = _objectWithoutProperties(props, _excluded4);
  var presentationProps = (0, _svgPropertiesNoEvents.svgPropertiesNoEvents)(propsWithoutId);
  return /*#__PURE__*/React.createElement(_RegisterGraphicalItemId.RegisterGraphicalItemId, {
    id: externalId,
    type: "pie"
  }, id => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(_SetGraphicalItem.SetPolarGraphicalItem, {
    type: "pie",
    id: id,
    data: propsWithoutId.data,
    dataKey: propsWithoutId.dataKey,
    hide: propsWithoutId.hide,
    angleAxisId: 0,
    radiusAxisId: 0,
    name: propsWithoutId.name,
    nameKey: propsWithoutId.nameKey,
    tooltipType: propsWithoutId.tooltipType,
    legendType: propsWithoutId.legendType,
    fill: propsWithoutId.fill,
    cx: propsWithoutId.cx,
    cy: propsWithoutId.cy,
    startAngle: propsWithoutId.startAngle,
    endAngle: propsWithoutId.endAngle,
    paddingAngle: propsWithoutId.paddingAngle,
    minAngle: propsWithoutId.minAngle,
    innerRadius: propsWithoutId.innerRadius,
    outerRadius: propsWithoutId.outerRadius,
    cornerRadius: propsWithoutId.cornerRadius,
    presentationProps: presentationProps,
    maxRadius: props.maxRadius
  }), /*#__PURE__*/React.createElement(SetPiePayloadLegend, _extends({}, propsWithoutId, {
    id: id
  })), /*#__PURE__*/React.createElement(PieImpl, _extends({}, propsWithoutId, {
    id: id
  }))));
}
Pie.displayName = 'Pie';
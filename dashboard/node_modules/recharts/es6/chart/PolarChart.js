var _excluded = ["layout"];
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
import { forwardRef } from 'react';
import * as React from 'react';
import { RechartsStoreProvider } from '../state/RechartsStoreProvider';
import { ChartDataContextProvider } from '../context/chartDataContext';
import { ReportMainChartProps } from '../state/ReportMainChartProps';
import { ReportChartProps } from '../state/ReportChartProps';
import { ReportPolarOptions } from '../state/ReportPolarOptions';
import { CategoricalChart } from './CategoricalChart';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
var defaultMargin = {
  top: 5,
  right: 5,
  bottom: 5,
  left: 5
};

/**
 * These default props are the same for all PolarChart components.
 */
export var defaultPolarChartProps = {
  accessibilityLayer: true,
  stackOffset: 'none',
  barCategoryGap: '10%',
  barGap: 4,
  margin: defaultMargin,
  reverseStackOrder: false,
  syncMethod: 'index',
  layout: 'radial',
  responsive: false,
  cx: '50%',
  cy: '50%',
  innerRadius: 0,
  outerRadius: '80%'
};

/**
 * These props are required for the PolarChart to function correctly.
 * Users usually would not need to specify these explicitly,
 * because the convenience components like PieChart, RadarChart, etc.
 * will provide these defaults.
 * We can't have the defaults in this file because each of those convenience components
 * have their own opinions about what they should be.
 */

/**
 * These are one-time, immutable options that decide the chart's behavior.
 * Users who wish to call CartesianChart may decide to pass these options explicitly,
 * but usually we would expect that they use one of the convenience components like PieChart, RadarChart, etc.
 */

export var PolarChart = /*#__PURE__*/forwardRef(function PolarChart(props, ref) {
  var _polarChartProps$id;
  var polarChartProps = resolveDefaultProps(props.categoricalChartProps, defaultPolarChartProps);
  var {
      layout
    } = polarChartProps,
    otherCategoricalProps = _objectWithoutProperties(polarChartProps, _excluded);
  var {
    chartName,
    defaultTooltipEventType,
    validateTooltipEventTypes,
    tooltipPayloadSearcher
  } = props;
  var options = {
    chartName,
    defaultTooltipEventType,
    validateTooltipEventTypes,
    tooltipPayloadSearcher,
    eventEmitter: undefined
  };
  return /*#__PURE__*/React.createElement(RechartsStoreProvider, {
    preloadedState: {
      options
    },
    reduxStoreName: (_polarChartProps$id = polarChartProps.id) !== null && _polarChartProps$id !== void 0 ? _polarChartProps$id : chartName
  }, /*#__PURE__*/React.createElement(ChartDataContextProvider, {
    chartData: polarChartProps.data
  }), /*#__PURE__*/React.createElement(ReportMainChartProps, {
    layout: layout,
    margin: polarChartProps.margin
  }), /*#__PURE__*/React.createElement(ReportChartProps, {
    baseValue: undefined,
    accessibilityLayer: polarChartProps.accessibilityLayer,
    barCategoryGap: polarChartProps.barCategoryGap,
    maxBarSize: polarChartProps.maxBarSize,
    stackOffset: polarChartProps.stackOffset,
    barGap: polarChartProps.barGap,
    barSize: polarChartProps.barSize,
    syncId: polarChartProps.syncId,
    syncMethod: polarChartProps.syncMethod,
    className: polarChartProps.className,
    reverseStackOrder: polarChartProps.reverseStackOrder
  }), /*#__PURE__*/React.createElement(ReportPolarOptions, {
    cx: polarChartProps.cx,
    cy: polarChartProps.cy,
    startAngle: polarChartProps.startAngle,
    endAngle: polarChartProps.endAngle,
    innerRadius: polarChartProps.innerRadius,
    outerRadius: polarChartProps.outerRadius
  }), /*#__PURE__*/React.createElement(CategoricalChart, _extends({}, otherCategoricalProps, {
    ref: ref
  })));
});
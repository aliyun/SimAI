"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.d3ScaleToRechartsScale = d3ScaleToRechartsScale;
exports.rechartsScaleFactory = rechartsScaleFactory;
var d3Scales = _interopRequireWildcard(require("victory-vendor/d3-scale"));
var _DataUtils = require("../DataUtils");
function _interopRequireWildcard(e, t) { if ("function" == typeof WeakMap) var r = new WeakMap(), n = new WeakMap(); return (_interopRequireWildcard = function _interopRequireWildcard(e, t) { if (!t && e && e.__esModule) return e; var o, i, f = { __proto__: null, default: e }; if (null === e || "object" != typeof e && "function" != typeof e) return f; if (o = t ? n : r) { if (o.has(e)) return o.get(e); o.set(e, f); } for (var _t in e) "default" !== _t && {}.hasOwnProperty.call(e, _t) && ((i = (o = Object.defineProperty) && Object.getOwnPropertyDescriptor(e, _t)) && (i.get || i.set) ? o(f, _t, i) : f[_t] = e[_t]); return f; })(e, t); }
/**
 * This is internal representation of scale used in Recharts.
 * Users will provide CustomScaleDefinition or a string, which we will parse into RechartsScale.
 * Most importantly, RechartsScale is fully immutable - there are no setters that mutate the scale in place.
 * This is important for React integration - if the scale changes, we want to trigger re-renders.
 * Mutating the scale in place would not trigger re-renders, leading to stale UI.
 */

/**
 * Position within a band for banded scales.
 * In scales that are not banded, this parameter is ignored.
 *
 * @inline
 */

function getD3ScaleFromType(realScaleType) {
  if (realScaleType in d3Scales) {
    // @ts-expect-error we should do better type verification here
    return d3Scales[realScaleType]();
  }
  var name = "scale".concat((0, _DataUtils.upperFirst)(realScaleType));
  if (name in d3Scales) {
    // @ts-expect-error we should do better type verification here
    return d3Scales[name]();
  }
  return undefined;
}
function d3ScaleToRechartsScale(d3Scale) {
  var ticksFn = d3Scale.ticks;
  var bandwidthFn = d3Scale.bandwidth;
  var d3Range = d3Scale.range();
  var range = [Math.min(...d3Range), Math.max(...d3Range)];
  return {
    domain: () => d3Scale.domain(),
    range: function (_range) {
      function range() {
        return _range.apply(this, arguments);
      }
      range.toString = function () {
        return _range.toString();
      };
      return range;
    }(() => range),
    rangeMin: () => range[0],
    rangeMax: () => range[1],
    isInRange(value) {
      var first = range[0];
      var last = range[1];
      return first <= last ? value >= first && value <= last : value >= last && value <= first;
    },
    bandwidth: bandwidthFn ? () => bandwidthFn.call(d3Scale) : undefined,
    ticks: ticksFn ? count => ticksFn.call(d3Scale, count) : undefined,
    map: (input, options) => {
      var baseValue = d3Scale(input);
      if (baseValue == null) {
        return undefined;
      }
      if (d3Scale.bandwidth && options !== null && options !== void 0 && options.position) {
        var bandWidth = d3Scale.bandwidth();
        switch (options.position) {
          case 'middle':
            baseValue += bandWidth / 2;
            break;
          case 'end':
            baseValue += bandWidth;
            break;
          default:
            // 'start' requires no adjustment
            break;
        }
      }
      return baseValue;
    }
  };
}

/**
 * Converts external scale definition into internal RechartsScale definition.
 * @param scale custom function scale - if you have the string, use `combineRealScaleType` first
 * @param axisDomain
 * @param axisRange
 */

function rechartsScaleFactory(scale, axisDomain, axisRange) {
  if (typeof scale === 'function') {
    return d3ScaleToRechartsScale(scale.copy().domain(axisDomain).range(axisRange));
  }
  if (scale == null) {
    return undefined;
  }
  var d3ScaleFunction = getD3ScaleFromType(scale);
  if (d3ScaleFunction == null) {
    return undefined;
  }
  d3ScaleFunction.domain(axisDomain).range(axisRange);
  return d3ScaleToRechartsScale(d3ScaleFunction);
}
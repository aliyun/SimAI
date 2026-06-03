"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.BarRectangle = BarRectangle;
exports.minPointSizeCallback = void 0;
var React = _interopRequireWildcard(require("react"));
var _tinyInvariant = _interopRequireDefault(require("tiny-invariant"));
var _ActiveShapeUtils = require("./ActiveShapeUtils");
var _DataUtils = require("./DataUtils");
function _interopRequireDefault(e) { return e && e.__esModule ? e : { default: e }; }
function _interopRequireWildcard(e, t) { if ("function" == typeof WeakMap) var r = new WeakMap(), n = new WeakMap(); return (_interopRequireWildcard = function _interopRequireWildcard(e, t) { if (!t && e && e.__esModule) return e; var o, i, f = { __proto__: null, default: e }; if (null === e || "object" != typeof e && "function" != typeof e) return f; if (o = t ? n : r) { if (o.has(e)) return o.get(e); o.set(e, f); } for (var _t in e) "default" !== _t && {}.hasOwnProperty.call(e, _t) && ((i = (o = Object.defineProperty) && Object.getOwnPropertyDescriptor(e, _t)) && (i.get || i.set) ? o(f, _t, i) : f[_t] = e[_t]); return f; })(e, t); }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function BarRectangle(props) {
  return /*#__PURE__*/React.createElement(_ActiveShapeUtils.Shape, _extends({
    shapeType: "rectangle",
    activeClassName: "recharts-active-bar"
  }, props));
}
/**
 * Safely gets minPointSize from the minPointSize prop if it is a function
 * @param minPointSize minPointSize as passed to the Bar component
 * @param defaultValue default minPointSize
 * @returns minPointSize
 */
var minPointSizeCallback = exports.minPointSizeCallback = function minPointSizeCallback(minPointSize) {
  var defaultValue = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0;
  return (value, index) => {
    if ((0, _DataUtils.isNumber)(minPointSize)) return minPointSize;
    var isValueNumberOrNil = (0, _DataUtils.isNumber)(value) || (0, _DataUtils.isNullish)(value);
    if (isValueNumberOrNil) {
      return minPointSize(value, index);
    }
    !isValueNumberOrNil ? true ? (0, _tinyInvariant.default)(false, "minPointSize callback function received a value with type of ".concat(typeof value, ". Currently only numbers or null/undefined are supported.")) : (0, _tinyInvariant.default)(false) : void 0;
    return defaultValue;
  };
};
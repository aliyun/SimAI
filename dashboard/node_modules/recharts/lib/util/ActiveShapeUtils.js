"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.Shape = Shape;
exports.getPropsFromShapeOption = getPropsFromShapeOption;
var _react = _interopRequireWildcard(require("react"));
var React = _react;
var _isPlainObject = _interopRequireDefault(require("es-toolkit/compat/isPlainObject"));
var _Rectangle = require("../shape/Rectangle");
var _Trapezoid = require("../shape/Trapezoid");
var _Sector = require("../shape/Sector");
var _Layer = require("../container/Layer");
var _Symbols = require("../shape/Symbols");
var _Curve = require("../shape/Curve");
var _excluded = ["option", "shapeType", "activeClassName"];
function _interopRequireDefault(e) { return e && e.__esModule ? e : { default: e }; }
function _interopRequireWildcard(e, t) { if ("function" == typeof WeakMap) var r = new WeakMap(), n = new WeakMap(); return (_interopRequireWildcard = function _interopRequireWildcard(e, t) { if (!t && e && e.__esModule) return e; var o, i, f = { __proto__: null, default: e }; if (null === e || "object" != typeof e && "function" != typeof e) return f; if (o = t ? n : r) { if (o.has(e)) return o.get(e); o.set(e, f); } for (var _t in e) "default" !== _t && {}.hasOwnProperty.call(e, _t) && ((i = (o = Object.defineProperty) && Object.getOwnPropertyDescriptor(e, _t)) && (i.get || i.set) ? o(f, _t, i) : f[_t] = e[_t]); return f; })(e, t); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
/**
 * This is an abstraction for rendering a user defined prop for a customized shape in several forms.
 *
 * <Shape /> is the root and will handle taking in:
 *  - an object of svg properties
 *  - a boolean
 *  - a render prop(inline function that returns jsx)
 *  - a React element
 *
 * <ShapeSelector /> is a subcomponent of <Shape /> and used to match a component
 * to the value of props.shapeType that is passed to the root.
 *
 */

function defaultPropTransformer(option, props) {
  return _objectSpread(_objectSpread({}, props), option);
}
function isSymbolsProps(shapeType, _elementProps) {
  return shapeType === 'symbols';
}
function ShapeSelector(_ref) {
  var {
    shapeType,
    elementProps
  } = _ref;
  switch (shapeType) {
    case 'rectangle':
      return /*#__PURE__*/React.createElement(_Rectangle.Rectangle, elementProps);
    case 'trapezoid':
      return /*#__PURE__*/React.createElement(_Trapezoid.Trapezoid, elementProps);
    case 'sector':
      return /*#__PURE__*/React.createElement(_Sector.Sector, elementProps);
    case 'symbols':
      if (isSymbolsProps(shapeType, elementProps)) {
        return /*#__PURE__*/React.createElement(_Symbols.Symbols, elementProps);
      }
      break;
    case 'curve':
      return /*#__PURE__*/React.createElement(_Curve.Curve, elementProps);
    default:
      return null;
  }
}
function getPropsFromShapeOption(option) {
  if (/*#__PURE__*/(0, _react.isValidElement)(option)) {
    return option.props;
  }
  return option;
}
function Shape(_ref2) {
  var {
      option,
      shapeType,
      activeClassName = 'recharts-active-shape'
    } = _ref2,
    props = _objectWithoutProperties(_ref2, _excluded);
  var shape;
  if (/*#__PURE__*/(0, _react.isValidElement)(option)) {
    // @ts-expect-error we can't know the type of cloned element props
    shape = /*#__PURE__*/(0, _react.cloneElement)(option, _objectSpread(_objectSpread({}, props), getPropsFromShapeOption(option)));
  } else if (typeof option === 'function') {
    shape = option(props, props.index);
  } else if ((0, _isPlainObject.default)(option) && typeof option !== 'boolean') {
    var nextProps = defaultPropTransformer(option, props);
    shape = /*#__PURE__*/React.createElement(ShapeSelector, {
      shapeType: shapeType,
      elementProps: nextProps
    });
  } else {
    var elementProps = props;
    shape = /*#__PURE__*/React.createElement(ShapeSelector, {
      shapeType: shapeType,
      elementProps: elementProps
    });
  }
  if (props.isActive) {
    return /*#__PURE__*/React.createElement(_Layer.Layer, {
      className: activeClassName
    }, shape);
  }
  return shape;
}
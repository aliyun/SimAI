"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.TooltipBoundingBox = void 0;
var _react = _interopRequireWildcard(require("react"));
var React = _react;
var _translate = require("../util/tooltip/translate");
function _interopRequireWildcard(e, t) { if ("function" == typeof WeakMap) var r = new WeakMap(), n = new WeakMap(); return (_interopRequireWildcard = function _interopRequireWildcard(e, t) { if (!t && e && e.__esModule) return e; var o, i, f = { __proto__: null, default: e }; if (null === e || "object" != typeof e && "function" != typeof e) return f; if (o = t ? n : r) { if (o.has(e)) return o.get(e); o.set(e, f); } for (var _t in e) "default" !== _t && {}.hasOwnProperty.call(e, _t) && ((i = (o = Object.defineProperty) && Object.getOwnPropertyDescriptor(e, _t)) && (i.get || i.set) ? o(f, _t, i) : f[_t] = e[_t]); return f; })(e, t); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
class TooltipBoundingBox extends _react.PureComponent {
  constructor() {
    super(...arguments);
    _defineProperty(this, "state", {
      dismissed: false,
      dismissedAtCoordinate: {
        x: 0,
        y: 0
      }
    });
    _defineProperty(this, "handleKeyDown", event => {
      if (event.key === 'Escape') {
        var _this$props$coordinat, _this$props$coordinat2, _this$props$coordinat3, _this$props$coordinat4;
        this.setState({
          dismissed: true,
          dismissedAtCoordinate: {
            x: (_this$props$coordinat = (_this$props$coordinat2 = this.props.coordinate) === null || _this$props$coordinat2 === void 0 ? void 0 : _this$props$coordinat2.x) !== null && _this$props$coordinat !== void 0 ? _this$props$coordinat : 0,
            y: (_this$props$coordinat3 = (_this$props$coordinat4 = this.props.coordinate) === null || _this$props$coordinat4 === void 0 ? void 0 : _this$props$coordinat4.y) !== null && _this$props$coordinat3 !== void 0 ? _this$props$coordinat3 : 0
          }
        });
      }
    });
  }
  componentDidMount() {
    document.addEventListener('keydown', this.handleKeyDown);
  }
  componentWillUnmount() {
    document.removeEventListener('keydown', this.handleKeyDown);
  }
  componentDidUpdate() {
    var _this$props$coordinat5, _this$props$coordinat6;
    if (!this.state.dismissed) {
      return;
    }
    if (((_this$props$coordinat5 = this.props.coordinate) === null || _this$props$coordinat5 === void 0 ? void 0 : _this$props$coordinat5.x) !== this.state.dismissedAtCoordinate.x || ((_this$props$coordinat6 = this.props.coordinate) === null || _this$props$coordinat6 === void 0 ? void 0 : _this$props$coordinat6.y) !== this.state.dismissedAtCoordinate.y) {
      this.state.dismissed = false;
    }
  }
  render() {
    var {
      active,
      allowEscapeViewBox,
      animationDuration,
      animationEasing,
      children,
      coordinate,
      hasPayload,
      isAnimationActive,
      offset,
      position,
      reverseDirection,
      useTranslate3d,
      viewBox,
      wrapperStyle,
      lastBoundingBox,
      innerRef,
      hasPortalFromProps
    } = this.props;
    var offsetLeft = typeof offset === 'number' ? offset : offset.x;
    var offsetTop = typeof offset === 'number' ? offset : offset.y;
    var {
      cssClasses,
      cssProperties
    } = (0, _translate.getTooltipTranslate)({
      allowEscapeViewBox,
      coordinate,
      offsetLeft,
      offsetTop,
      position,
      reverseDirection,
      tooltipBox: {
        height: lastBoundingBox.height,
        width: lastBoundingBox.width
      },
      useTranslate3d,
      viewBox
    });

    // do not use absolute styles if the user has passed a custom portal prop
    var positionStyles = hasPortalFromProps ? {} : _objectSpread(_objectSpread({
      transition: isAnimationActive && active ? "transform ".concat(animationDuration, "ms ").concat(animationEasing) : undefined
    }, cssProperties), {}, {
      pointerEvents: 'none',
      visibility: !this.state.dismissed && active && hasPayload ? 'visible' : 'hidden',
      position: 'absolute',
      top: 0,
      left: 0
    });
    var outerStyle = _objectSpread(_objectSpread({}, positionStyles), {}, {
      visibility: !this.state.dismissed && active && hasPayload ? 'visible' : 'hidden'
    }, wrapperStyle);
    return (
      /*#__PURE__*/
      // This element allow listening to the `Escape` key. See https://github.com/recharts/recharts/pull/2925
      React.createElement("div", {
        // @ts-expect-error typescript library does not recognize xmlns attribute, but it's required for an HTML chunk inside SVG.
        xmlns: "http://www.w3.org/1999/xhtml",
        tabIndex: -1,
        className: cssClasses,
        style: outerStyle,
        ref: innerRef
      }, children)
    );
  }
}
exports.TooltipBoundingBox = TooltipBoundingBox;
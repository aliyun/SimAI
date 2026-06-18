"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.JavascriptAnimate = JavascriptAnimate;
var _react = require("react");
var _DataUtils = require("../util/DataUtils");
var _resolveDefaultProps = require("../util/resolveDefaultProps");
var _configUpdate = _interopRequireDefault(require("./configUpdate"));
var _easing = require("./easing");
var _useAnimationManager = require("./useAnimationManager");
var _Global = require("../util/Global");
function _interopRequireDefault(e) { return e && e.__esModule ? e : { default: e }; }
var defaultJavascriptAnimateProps = {
  begin: 0,
  duration: 1000,
  easing: 'ease',
  isActive: true,
  canBegin: true,
  onAnimationEnd: () => {},
  onAnimationStart: () => {}
};
var from = {
  t: 0
};
var to = {
  t: 1
};
function JavascriptAnimate(outsideProps) {
  var props = (0, _resolveDefaultProps.resolveDefaultProps)(outsideProps, defaultJavascriptAnimateProps);
  var {
    isActive: isActiveProp,
    canBegin,
    duration,
    easing,
    begin,
    onAnimationEnd,
    onAnimationStart,
    children
  } = props;
  var isActive = isActiveProp === 'auto' ? !_Global.Global.isSsr : isActiveProp;
  var animationManager = (0, _useAnimationManager.useAnimationManager)(props.animationId, props.animationManager);
  var [style, setStyle] = (0, _react.useState)(isActive ? from : to);
  var stopJSAnimation = (0, _react.useRef)(null);
  (0, _react.useEffect)(() => {
    if (!isActive) {
      setStyle(to);
    }
  }, [isActive]);
  (0, _react.useEffect)(() => {
    if (!isActive || !canBegin) {
      return _DataUtils.noop;
    }
    var startAnimation = (0, _configUpdate.default)(from, to, (0, _easing.configEasing)(easing), duration, setStyle, animationManager.getTimeoutController());
    var onAnimationActive = () => {
      stopJSAnimation.current = startAnimation();
    };
    animationManager.start([onAnimationStart, begin, onAnimationActive, duration, onAnimationEnd]);
    return () => {
      animationManager.stop();
      if (stopJSAnimation.current) {
        stopJSAnimation.current();
      }
      onAnimationEnd();
    };
  }, [isActive, canBegin, duration, easing, begin, onAnimationStart, onAnimationEnd, animationManager]);
  return children(style.t);
}
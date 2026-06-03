"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.CSSTransitionAnimate = CSSTransitionAnimate;
var _react = require("react");
var _DataUtils = require("../util/DataUtils");
var _resolveDefaultProps = require("../util/resolveDefaultProps");
var _useAnimationManager = require("./useAnimationManager");
var _util = require("./util");
var _Global = require("../util/Global");
var defaultProps = {
  begin: 0,
  duration: 1000,
  easing: 'ease',
  isActive: true,
  canBegin: true,
  onAnimationEnd: () => {},
  onAnimationStart: () => {}
};
function CSSTransitionAnimate(outsideProps) {
  var props = (0, _resolveDefaultProps.resolveDefaultProps)(outsideProps, defaultProps);
  var {
    animationId,
    from,
    to,
    attributeName,
    isActive: isActiveProp,
    canBegin,
    duration,
    easing,
    begin,
    onAnimationEnd,
    onAnimationStart: onAnimationStartFromProps,
    children
  } = props;
  var isActive = isActiveProp === 'auto' ? !_Global.Global.isSsr : isActiveProp;
  var animationManager = (0, _useAnimationManager.useAnimationManager)(animationId + attributeName, props.animationManager);
  var [style, setStyle] = (0, _react.useState)(() => {
    if (!isActive) {
      return to;
    }
    return from;
  });
  var initialized = (0, _react.useRef)(false);
  var onAnimationStart = (0, _react.useCallback)(() => {
    setStyle(from);
    onAnimationStartFromProps();
  }, [from, onAnimationStartFromProps]);
  (0, _react.useEffect)(() => {
    if (!isActive || !canBegin) {
      return _DataUtils.noop;
    }
    initialized.current = true;
    var unsubscribe = animationManager.subscribe(setStyle);
    animationManager.start([onAnimationStart, begin, to, duration, onAnimationEnd]);
    return () => {
      animationManager.stop();
      if (unsubscribe) {
        unsubscribe();
      }
      onAnimationEnd();
    };
  }, [isActive, canBegin, duration, easing, begin, onAnimationStart, onAnimationEnd, animationManager, to, from]);
  if (!isActive) {
    /*
     * With isActive=false, the component always renders with the final style, immediately,
     * and ignores all other props.
     * Also there is no transition applied.
     */
    return children({
      [attributeName]: to
    });
  }
  if (!canBegin) {
    return children({
      [attributeName]: from
    });
  }
  if (initialized.current) {
    var transition = (0, _util.getTransitionVal)([attributeName], duration, easing);
    return children({
      transition,
      [attributeName]: style
    });
  }
  return children({
    [attributeName]: from
  });
}
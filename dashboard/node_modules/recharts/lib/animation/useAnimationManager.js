"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.AnimationManagerContext = void 0;
exports.useAnimationManager = useAnimationManager;
var _react = require("react");
var _createDefaultAnimationManager = require("./createDefaultAnimationManager");
var AnimationManagerContext = exports.AnimationManagerContext = /*#__PURE__*/(0, _react.createContext)(_createDefaultAnimationManager.createDefaultAnimationManager);
function useAnimationManager(animationId, animationManagerFromProps) {
  var contextAnimationManager = (0, _react.useContext)(AnimationManagerContext);
  return (0, _react.useMemo)(() => animationManagerFromProps !== null && animationManagerFromProps !== void 0 ? animationManagerFromProps : contextAnimationManager(animationId), [animationId, animationManagerFromProps, contextAnimationManager]);
}
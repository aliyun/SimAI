"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.createDefaultAnimationManager = createDefaultAnimationManager;
var _AnimationManager = require("./AnimationManager");
var _timeoutController = require("./timeoutController");
function createDefaultAnimationManager() {
  return (0, _AnimationManager.createAnimateManager)(new _timeoutController.RequestAnimationFrameTimeoutController());
}
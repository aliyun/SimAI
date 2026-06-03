"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.keyboardEventsMiddleware = exports.keyDownAction = exports.focusAction = void 0;
var _toolkit = require("@reduxjs/toolkit");
var _tooltipSlice = require("./tooltipSlice");
var _tooltipSelectors = require("./selectors/tooltipSelectors");
var _selectors = require("./selectors/selectors");
var _axisSelectors = require("./selectors/axisSelectors");
var _combineActiveTooltipIndex = require("./selectors/combiners/combineActiveTooltipIndex");
var keyDownAction = exports.keyDownAction = (0, _toolkit.createAction)('keyDown');
var focusAction = exports.focusAction = (0, _toolkit.createAction)('focus');
var keyboardEventsMiddleware = exports.keyboardEventsMiddleware = (0, _toolkit.createListenerMiddleware)();
keyboardEventsMiddleware.startListening({
  actionCreator: keyDownAction,
  effect: (action, listenerApi) => {
    var state = listenerApi.getState();
    var accessibilityLayerIsActive = state.rootProps.accessibilityLayer !== false;
    if (!accessibilityLayerIsActive) {
      return;
    }
    var {
      keyboardInteraction
    } = state.tooltip;
    var key = action.payload;
    if (key !== 'ArrowRight' && key !== 'ArrowLeft' && key !== 'Enter') {
      return;
    }

    // TODO this is lacking index for charts that do not support numeric indexes
    var resolvedIndex = (0, _combineActiveTooltipIndex.combineActiveTooltipIndex)(keyboardInteraction, (0, _tooltipSelectors.selectTooltipDisplayedData)(state), (0, _axisSelectors.selectTooltipAxisDataKey)(state), (0, _tooltipSelectors.selectTooltipAxisDomain)(state));
    var currentIndex = resolvedIndex == null ? -1 : Number(resolvedIndex);
    if (!Number.isFinite(currentIndex) || currentIndex < 0) {
      return;
    }
    var tooltipTicks = (0, _tooltipSelectors.selectTooltipAxisTicks)(state);
    if (key === 'Enter') {
      var _coordinate = (0, _selectors.selectCoordinateForDefaultIndex)(state, 'axis', 'hover', String(keyboardInteraction.index));
      listenerApi.dispatch((0, _tooltipSlice.setKeyboardInteraction)({
        active: !keyboardInteraction.active,
        activeIndex: keyboardInteraction.index,
        activeCoordinate: _coordinate
      }));
      return;
    }
    var direction = (0, _axisSelectors.selectChartDirection)(state);
    var directionMultiplier = direction === 'left-to-right' ? 1 : -1;
    var movement = key === 'ArrowRight' ? 1 : -1;
    var nextIndex = currentIndex + movement * directionMultiplier;
    if (tooltipTicks == null || nextIndex >= tooltipTicks.length || nextIndex < 0) {
      return;
    }
    var coordinate = (0, _selectors.selectCoordinateForDefaultIndex)(state, 'axis', 'hover', String(nextIndex));
    listenerApi.dispatch((0, _tooltipSlice.setKeyboardInteraction)({
      active: true,
      activeIndex: nextIndex.toString(),
      activeCoordinate: coordinate
    }));
  }
});
keyboardEventsMiddleware.startListening({
  actionCreator: focusAction,
  effect: (_action, listenerApi) => {
    var state = listenerApi.getState();
    var accessibilityLayerIsActive = state.rootProps.accessibilityLayer !== false;
    if (!accessibilityLayerIsActive) {
      return;
    }
    var {
      keyboardInteraction
    } = state.tooltip;
    if (keyboardInteraction.active) {
      return;
    }
    if (keyboardInteraction.index == null) {
      var nextIndex = '0';
      var coordinate = (0, _selectors.selectCoordinateForDefaultIndex)(state, 'axis', 'hover', String(nextIndex));
      listenerApi.dispatch((0, _tooltipSlice.setKeyboardInteraction)({
        active: true,
        activeIndex: nextIndex,
        activeCoordinate: coordinate
      }));
    }
  }
});
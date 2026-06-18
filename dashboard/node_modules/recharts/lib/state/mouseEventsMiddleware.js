"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.mouseMoveMiddleware = exports.mouseMoveAction = exports.mouseClickMiddleware = exports.mouseClickAction = void 0;
var _toolkit = require("@reduxjs/toolkit");
var _tooltipSlice = require("./tooltipSlice");
var _selectActivePropsFromChartPointer = require("./selectors/selectActivePropsFromChartPointer");
var _selectTooltipEventType = require("./selectors/selectTooltipEventType");
var _getChartPointer = require("../util/getChartPointer");
var mouseClickAction = exports.mouseClickAction = (0, _toolkit.createAction)('mouseClick');
var mouseClickMiddleware = exports.mouseClickMiddleware = (0, _toolkit.createListenerMiddleware)();

// TODO: there's a bug here when you click the chart the activeIndex resets to zero
mouseClickMiddleware.startListening({
  actionCreator: mouseClickAction,
  effect: (action, listenerApi) => {
    var mousePointer = action.payload;
    var activeProps = (0, _selectActivePropsFromChartPointer.selectActivePropsFromChartPointer)(listenerApi.getState(), (0, _getChartPointer.getChartPointer)(mousePointer));
    if ((activeProps === null || activeProps === void 0 ? void 0 : activeProps.activeIndex) != null) {
      listenerApi.dispatch((0, _tooltipSlice.setMouseClickAxisIndex)({
        activeIndex: activeProps.activeIndex,
        activeDataKey: undefined,
        activeCoordinate: activeProps.activeCoordinate
      }));
    }
  }
});
var mouseMoveAction = exports.mouseMoveAction = (0, _toolkit.createAction)('mouseMove');
var mouseMoveMiddleware = exports.mouseMoveMiddleware = (0, _toolkit.createListenerMiddleware)();

/*
 * This single rafId is safe because:
 * 1. Each chart has its own Redux store instance with its own middleware
 * 2. mouseMoveAction only fires from one DOM element (the chart wrapper)
 * 3. Rapid mousemove events from the same element SHOULD debounce - we only care about the latest position
 * This is different from externalEventsMiddleware which handles multiple event types
 * (click, mouseenter, mouseleave, etc.) that should NOT cancel each other.
 */
var rafId = null;
mouseMoveMiddleware.startListening({
  actionCreator: mouseMoveAction,
  effect: (action, listenerApi) => {
    var mousePointer = action.payload;

    // Cancel any pending animation frame
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
    }
    var chartPointer = (0, _getChartPointer.getChartPointer)(mousePointer);

    // Schedule the dispatch for the next animation frame
    rafId = requestAnimationFrame(() => {
      var state = listenerApi.getState();
      var tooltipEventType = (0, _selectTooltipEventType.selectTooltipEventType)(state, state.tooltip.settings.shared);
      // this functionality only applies to charts that have axes
      if (tooltipEventType === 'axis') {
        var activeProps = (0, _selectActivePropsFromChartPointer.selectActivePropsFromChartPointer)(state, chartPointer);
        if ((activeProps === null || activeProps === void 0 ? void 0 : activeProps.activeIndex) != null) {
          listenerApi.dispatch((0, _tooltipSlice.setMouseOverAxisIndex)({
            activeIndex: activeProps.activeIndex,
            activeDataKey: undefined,
            activeCoordinate: activeProps.activeCoordinate
          }));
        } else {
          // this is needed to clear tooltip state when the mouse moves out of the inRange (svg - offset) function, but not yet out of the svg
          listenerApi.dispatch((0, _tooltipSlice.mouseLeaveChart)());
        }
      }
      rafId = null;
    });
  }
});
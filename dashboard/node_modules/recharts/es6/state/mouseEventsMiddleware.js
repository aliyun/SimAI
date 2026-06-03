import { createAction, createListenerMiddleware } from '@reduxjs/toolkit';
import { mouseLeaveChart, setMouseClickAxisIndex, setMouseOverAxisIndex } from './tooltipSlice';
import { selectActivePropsFromChartPointer } from './selectors/selectActivePropsFromChartPointer';
import { selectTooltipEventType } from './selectors/selectTooltipEventType';
import { getChartPointer } from '../util/getChartPointer';
export var mouseClickAction = createAction('mouseClick');
export var mouseClickMiddleware = createListenerMiddleware();

// TODO: there's a bug here when you click the chart the activeIndex resets to zero
mouseClickMiddleware.startListening({
  actionCreator: mouseClickAction,
  effect: (action, listenerApi) => {
    var mousePointer = action.payload;
    var activeProps = selectActivePropsFromChartPointer(listenerApi.getState(), getChartPointer(mousePointer));
    if ((activeProps === null || activeProps === void 0 ? void 0 : activeProps.activeIndex) != null) {
      listenerApi.dispatch(setMouseClickAxisIndex({
        activeIndex: activeProps.activeIndex,
        activeDataKey: undefined,
        activeCoordinate: activeProps.activeCoordinate
      }));
    }
  }
});
export var mouseMoveAction = createAction('mouseMove');
export var mouseMoveMiddleware = createListenerMiddleware();

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
    var chartPointer = getChartPointer(mousePointer);

    // Schedule the dispatch for the next animation frame
    rafId = requestAnimationFrame(() => {
      var state = listenerApi.getState();
      var tooltipEventType = selectTooltipEventType(state, state.tooltip.settings.shared);
      // this functionality only applies to charts that have axes
      if (tooltipEventType === 'axis') {
        var activeProps = selectActivePropsFromChartPointer(state, chartPointer);
        if ((activeProps === null || activeProps === void 0 ? void 0 : activeProps.activeIndex) != null) {
          listenerApi.dispatch(setMouseOverAxisIndex({
            activeIndex: activeProps.activeIndex,
            activeDataKey: undefined,
            activeCoordinate: activeProps.activeCoordinate
          }));
        } else {
          // this is needed to clear tooltip state when the mouse moves out of the inRange (svg - offset) function, but not yet out of the svg
          listenerApi.dispatch(mouseLeaveChart());
        }
      }
      rafId = null;
    });
  }
});
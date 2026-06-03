import { createAction, createListenerMiddleware } from '@reduxjs/toolkit';
import { setKeyboardInteraction } from './tooltipSlice';
import { selectTooltipAxisDomain, selectTooltipAxisTicks, selectTooltipDisplayedData } from './selectors/tooltipSelectors';
import { selectCoordinateForDefaultIndex } from './selectors/selectors';
import { selectChartDirection, selectTooltipAxisDataKey } from './selectors/axisSelectors';
import { combineActiveTooltipIndex } from './selectors/combiners/combineActiveTooltipIndex';
export var keyDownAction = createAction('keyDown');
export var focusAction = createAction('focus');
export var keyboardEventsMiddleware = createListenerMiddleware();
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
    var resolvedIndex = combineActiveTooltipIndex(keyboardInteraction, selectTooltipDisplayedData(state), selectTooltipAxisDataKey(state), selectTooltipAxisDomain(state));
    var currentIndex = resolvedIndex == null ? -1 : Number(resolvedIndex);
    if (!Number.isFinite(currentIndex) || currentIndex < 0) {
      return;
    }
    var tooltipTicks = selectTooltipAxisTicks(state);
    if (key === 'Enter') {
      var _coordinate = selectCoordinateForDefaultIndex(state, 'axis', 'hover', String(keyboardInteraction.index));
      listenerApi.dispatch(setKeyboardInteraction({
        active: !keyboardInteraction.active,
        activeIndex: keyboardInteraction.index,
        activeCoordinate: _coordinate
      }));
      return;
    }
    var direction = selectChartDirection(state);
    var directionMultiplier = direction === 'left-to-right' ? 1 : -1;
    var movement = key === 'ArrowRight' ? 1 : -1;
    var nextIndex = currentIndex + movement * directionMultiplier;
    if (tooltipTicks == null || nextIndex >= tooltipTicks.length || nextIndex < 0) {
      return;
    }
    var coordinate = selectCoordinateForDefaultIndex(state, 'axis', 'hover', String(nextIndex));
    listenerApi.dispatch(setKeyboardInteraction({
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
      var coordinate = selectCoordinateForDefaultIndex(state, 'axis', 'hover', String(nextIndex));
      listenerApi.dispatch(setKeyboardInteraction({
        active: true,
        activeIndex: nextIndex,
        activeCoordinate: coordinate
      }));
    }
  }
});
import { createAction, createListenerMiddleware } from '@reduxjs/toolkit';
import { setActiveMouseOverItemIndex, setMouseOverAxisIndex } from './tooltipSlice';
import { selectActivePropsFromChartPointer } from './selectors/selectActivePropsFromChartPointer';
import { getChartPointer } from '../util/getChartPointer';
import { selectTooltipEventType } from './selectors/selectTooltipEventType';
import { DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME, DATA_ITEM_INDEX_ATTRIBUTE_NAME } from '../util/Constants';
import { selectTooltipCoordinate } from './selectors/touchSelectors';
import { selectAllGraphicalItemsSettings } from './selectors/tooltipSelectors';
export var touchEventAction = createAction('touchMove');
export var touchEventMiddleware = createListenerMiddleware();
touchEventMiddleware.startListening({
  actionCreator: touchEventAction,
  effect: (action, listenerApi) => {
    var touchEvent = action.payload;
    if (touchEvent.touches == null || touchEvent.touches.length === 0) {
      return;
    }
    var state = listenerApi.getState();
    var tooltipEventType = selectTooltipEventType(state, state.tooltip.settings.shared);
    if (tooltipEventType === 'axis') {
      var touch = touchEvent.touches[0];
      if (touch == null) {
        return;
      }
      var activeProps = selectActivePropsFromChartPointer(state, getChartPointer({
        clientX: touch.clientX,
        clientY: touch.clientY,
        currentTarget: touchEvent.currentTarget
      }));
      if ((activeProps === null || activeProps === void 0 ? void 0 : activeProps.activeIndex) != null) {
        listenerApi.dispatch(setMouseOverAxisIndex({
          activeIndex: activeProps.activeIndex,
          activeDataKey: undefined,
          activeCoordinate: activeProps.activeCoordinate
        }));
      }
    } else if (tooltipEventType === 'item') {
      var _target$getAttribute;
      var _touch = touchEvent.touches[0];
      if (document.elementFromPoint == null || _touch == null) {
        return;
      }
      var target = document.elementFromPoint(_touch.clientX, _touch.clientY);
      if (!target || !target.getAttribute) {
        return;
      }
      var itemIndex = target.getAttribute(DATA_ITEM_INDEX_ATTRIBUTE_NAME);
      var graphicalItemId = (_target$getAttribute = target.getAttribute(DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME)) !== null && _target$getAttribute !== void 0 ? _target$getAttribute : undefined;
      var settings = selectAllGraphicalItemsSettings(state).find(item => item.id === graphicalItemId);
      if (itemIndex == null || settings == null || graphicalItemId == null) {
        return;
      }
      var {
        dataKey
      } = settings;
      var coordinate = selectTooltipCoordinate(state, itemIndex, graphicalItemId);
      listenerApi.dispatch(setActiveMouseOverItemIndex({
        activeDataKey: dataKey,
        activeIndex: itemIndex,
        activeCoordinate: coordinate,
        activeGraphicalItemId: graphicalItemId
      }));
    }
  }
});
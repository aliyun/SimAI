"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.touchEventMiddleware = exports.touchEventAction = void 0;
var _toolkit = require("@reduxjs/toolkit");
var _tooltipSlice = require("./tooltipSlice");
var _selectActivePropsFromChartPointer = require("./selectors/selectActivePropsFromChartPointer");
var _getChartPointer = require("../util/getChartPointer");
var _selectTooltipEventType = require("./selectors/selectTooltipEventType");
var _Constants = require("../util/Constants");
var _touchSelectors = require("./selectors/touchSelectors");
var _tooltipSelectors = require("./selectors/tooltipSelectors");
var touchEventAction = exports.touchEventAction = (0, _toolkit.createAction)('touchMove');
var touchEventMiddleware = exports.touchEventMiddleware = (0, _toolkit.createListenerMiddleware)();
touchEventMiddleware.startListening({
  actionCreator: touchEventAction,
  effect: (action, listenerApi) => {
    var touchEvent = action.payload;
    if (touchEvent.touches == null || touchEvent.touches.length === 0) {
      return;
    }
    var state = listenerApi.getState();
    var tooltipEventType = (0, _selectTooltipEventType.selectTooltipEventType)(state, state.tooltip.settings.shared);
    if (tooltipEventType === 'axis') {
      var touch = touchEvent.touches[0];
      if (touch == null) {
        return;
      }
      var activeProps = (0, _selectActivePropsFromChartPointer.selectActivePropsFromChartPointer)(state, (0, _getChartPointer.getChartPointer)({
        clientX: touch.clientX,
        clientY: touch.clientY,
        currentTarget: touchEvent.currentTarget
      }));
      if ((activeProps === null || activeProps === void 0 ? void 0 : activeProps.activeIndex) != null) {
        listenerApi.dispatch((0, _tooltipSlice.setMouseOverAxisIndex)({
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
      var itemIndex = target.getAttribute(_Constants.DATA_ITEM_INDEX_ATTRIBUTE_NAME);
      var graphicalItemId = (_target$getAttribute = target.getAttribute(_Constants.DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME)) !== null && _target$getAttribute !== void 0 ? _target$getAttribute : undefined;
      var settings = (0, _tooltipSelectors.selectAllGraphicalItemsSettings)(state).find(item => item.id === graphicalItemId);
      if (itemIndex == null || settings == null || graphicalItemId == null) {
        return;
      }
      var {
        dataKey
      } = settings;
      var coordinate = (0, _touchSelectors.selectTooltipCoordinate)(state, itemIndex, graphicalItemId);
      listenerApi.dispatch((0, _tooltipSlice.setActiveMouseOverItemIndex)({
        activeDataKey: dataKey,
        activeIndex: itemIndex,
        activeCoordinate: coordinate,
        activeGraphicalItemId: graphicalItemId
      }));
    }
  }
});
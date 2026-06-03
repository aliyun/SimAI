"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useReportScale = useReportScale;
var _react = require("react");
var _hooks = require("../state/hooks");
var _containerSelectors = require("../state/selectors/containerSelectors");
var _layoutSlice = require("../state/layoutSlice");
var _isWellBehavedNumber = require("./isWellBehavedNumber");
function useReportScale() {
  var dispatch = (0, _hooks.useAppDispatch)();
  var [ref, setRef] = (0, _react.useState)(null);
  var scale = (0, _hooks.useAppSelector)(_containerSelectors.selectContainerScale);
  (0, _react.useEffect)(() => {
    if (ref == null) {
      return;
    }
    var rect = ref.getBoundingClientRect();
    var newScale = rect.width / ref.offsetWidth;
    if ((0, _isWellBehavedNumber.isWellBehavedNumber)(newScale) && newScale !== scale) {
      dispatch((0, _layoutSlice.setScale)(newScale));
    }
  }, [ref, dispatch, scale]);
  return setRef;
}
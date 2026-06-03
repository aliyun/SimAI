import { RechartsRootState } from './store';
export declare const keyDownAction: import("@reduxjs/toolkit").ActionCreatorWithPayload<string, string>;
export declare const focusAction: import("@reduxjs/toolkit").ActionCreatorWithoutPayload<"focus">;
export declare const keyboardEventsMiddleware: import("@reduxjs/toolkit").ListenerMiddlewareInstance<RechartsRootState, import("@reduxjs/toolkit").ThunkDispatch<RechartsRootState, unknown, import("redux").AnyAction>, unknown>;

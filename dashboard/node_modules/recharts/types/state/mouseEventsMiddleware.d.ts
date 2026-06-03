import { RechartsRootState } from './store';
import { MousePointer } from '../util/types';
export declare const mouseClickAction: import("@reduxjs/toolkit").ActionCreatorWithPayload<MousePointer, string>;
export declare const mouseClickMiddleware: import("@reduxjs/toolkit").ListenerMiddlewareInstance<RechartsRootState, import("@reduxjs/toolkit").ThunkDispatch<RechartsRootState, unknown, import("redux").AnyAction>, unknown>;
export declare const mouseMoveAction: import("@reduxjs/toolkit").ActionCreatorWithPayload<MousePointer, string>;
export declare const mouseMoveMiddleware: import("@reduxjs/toolkit").ListenerMiddlewareInstance<RechartsRootState, import("@reduxjs/toolkit").ThunkDispatch<RechartsRootState, unknown, import("redux").AnyAction>, unknown>;

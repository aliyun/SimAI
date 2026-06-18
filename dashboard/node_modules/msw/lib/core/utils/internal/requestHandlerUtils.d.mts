import { R as RequestHandler, t as RequestHandlerDefaultInfo, c as RequestHandlerOptions } from '../../HttpResponse-Cw4ELwIN.mjs';
import '@mswjs/interceptors';
import './isIterable.mjs';
import '../../typeUtils.mjs';
import 'graphql';
import '../matching/matchRequestUrl.mjs';

declare function use(currentHandlers: Array<RequestHandler>, ...handlers: Array<RequestHandler>): void;
declare function restoreHandlers(handlers: Array<RequestHandler>): void;
declare function resetHandlers(initialHandlers: Array<RequestHandler>, ...nextHandlers: Array<RequestHandler>): RequestHandler<RequestHandlerDefaultInfo, any, any, RequestHandlerOptions>[];

export { resetHandlers, restoreHandlers, use };

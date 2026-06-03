import { Emitter } from 'strict-event-emitter';
import { SharedOptions, LifeCycleEventsMap } from '../sharedOptions.js';
import { RequiredDeep } from '../typeUtils.js';
import { m as ResponseResolutionContext, u as HandlersExecutionResult, R as RequestHandler } from '../HttpResponse-CVs3ngx3.js';
import './request/onUnhandledRequest.js';
import '@mswjs/interceptors';
import './internal/isIterable.js';
import 'graphql';
import './matching/matchRequestUrl.js';

interface HandleRequestOptions {
    /**
     * `resolutionContext` is not part of the general public api
     * but is exposed to aid in creating extensions like
     * `@mswjs/http-middleware`.
     */
    resolutionContext?: ResponseResolutionContext;
    /**
     * Invoked whenever a request is performed as-is.
     */
    onPassthroughResponse?(request: Request): void;
    /**
     * Invoked when the mocked response is ready to be sent.
     */
    onMockedResponse?(response: Response, handler: RequiredDeep<HandlersExecutionResult>): void;
}
declare function handleRequest(request: Request, requestId: string, handlers: Array<RequestHandler>, options: RequiredDeep<SharedOptions>, emitter: Emitter<LifeCycleEventsMap>, handleRequestOptions?: HandleRequestOptions): Promise<Response | undefined>;

export { type HandleRequestOptions, handleRequest };

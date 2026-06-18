import type { EdgeChange } from '@xyflow/system';
import type { Edge } from '../types';
/**
 * Registers a middleware function to transform edge changes.
 *
 * @public
 * @param fn - Middleware function. Should be memoized with useCallback to avoid re-registration.
 */
export declare function experimental_useOnEdgesChangeMiddleware<EdgeType extends Edge = Edge>(fn: (changes: EdgeChange<EdgeType>[]) => EdgeChange<EdgeType>[]): void;
//# sourceMappingURL=useOnEdgesChangeMiddleware.d.ts.map
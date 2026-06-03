import type { NodeChange } from '@xyflow/system';
import type { Node } from '../types';
/**
 * Registers a middleware function to transform node changes.
 *
 * @public
 * @param fn - Middleware function. Should be memoized with useCallback to avoid re-registration.
 */
export declare function experimental_useOnNodesChangeMiddleware<NodeType extends Node = Node>(fn: (changes: NodeChange<NodeType>[]) => NodeChange<NodeType>[]): void;
//# sourceMappingURL=useOnNodesChangeMiddleware.d.ts.map
import { NodeOrigin, CoordinateExtent, ZIndexMode } from '@xyflow/system';
import type { ReactFlowState, Node, Edge, FitViewOptions } from '../types';
declare const createStore: ({ nodes, edges, defaultNodes, defaultEdges, width, height, fitView, fitViewOptions, minZoom, maxZoom, nodeOrigin, nodeExtent, zIndexMode, }: {
    nodes?: Node[];
    edges?: Edge[];
    defaultNodes?: Node[];
    defaultEdges?: Edge[];
    width?: number;
    height?: number;
    fitView?: boolean;
    fitViewOptions?: FitViewOptions;
    minZoom?: number;
    maxZoom?: number;
    nodeOrigin?: NodeOrigin;
    nodeExtent?: CoordinateExtent;
    zIndexMode?: ZIndexMode;
}) => import("zustand/traditional").UseBoundStoreWithEqualityFn<import("zustand").StoreApi<ReactFlowState>>;
export { createStore };
//# sourceMappingURL=index.d.ts.map
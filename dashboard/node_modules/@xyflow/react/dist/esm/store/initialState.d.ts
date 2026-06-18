import { NodeOrigin, CoordinateExtent, ZIndexMode } from '@xyflow/system';
import type { Edge, FitViewOptions, Node, ReactFlowStore } from '../types';
declare const getInitialState: ({ nodes, edges, defaultNodes, defaultEdges, width, height, fitView, fitViewOptions, minZoom, maxZoom, nodeOrigin, nodeExtent, zIndexMode, }?: {
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
}) => ReactFlowStore;
export default getInitialState;
//# sourceMappingURL=initialState.d.ts.map
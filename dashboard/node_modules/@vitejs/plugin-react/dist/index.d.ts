import { Plugin, ResolvedConfig } from "vite";
import { ParserOptions, TransformOptions } from "@babel/core";

//#region src/index.d.ts
interface Options {
  /**
   * Can be used to process extra files like `.mdx`
   * @example include: /\.(mdx|js|jsx|ts|tsx)$/
   * @default /\.[tj]sx?$/
   */
  include?: string | RegExp | Array<string | RegExp>;
  /**
   * Can be used to exclude JSX/TSX files that runs in a worker or are not React files.
   * Except if explicitly desired, you should keep node_modules in the exclude list
   * @example exclude: [/\/pdf\//, /\.solid\.tsx$/, /\/node_modules\//]
   * @default /\/node_modules\//
   */
  exclude?: string | RegExp | Array<string | RegExp>;
  /**
   * Control where the JSX factory is imported from.
   * https://esbuild.github.io/api/#jsx-import-source
   * @default 'react'
   */
  jsxImportSource?: string;
  /**
   * Note: Skipping React import with classic runtime is not supported from v4
   * @default "automatic"
   */
  jsxRuntime?: 'classic' | 'automatic';
  /**
   * Babel configuration applied in both dev and prod.
   */
  babel?: BabelOptions | ((id: string, options: {
    ssr?: boolean;
  }) => BabelOptions);
  /**
   * React Fast Refresh runtime URL prefix.
   * Useful in a module federation context to enable HMR by specifying
   * the host application URL in the Vite config of a remote application.
   * @example
   * reactRefreshHost: 'http://localhost:3000'
   */
  reactRefreshHost?: string;
}
type BabelOptions = Omit<TransformOptions, 'ast' | 'filename' | 'root' | 'sourceFileName' | 'sourceMaps' | 'inputSourceMap'>;
/**
 * The object type used by the `options` passed to plugins with
 * an `api.reactBabel` method.
 */
interface ReactBabelOptions extends BabelOptions {
  plugins: Extract<BabelOptions['plugins'], any[]>;
  presets: Extract<BabelOptions['presets'], any[]>;
  overrides: Extract<BabelOptions['overrides'], any[]>;
  parserOpts: ParserOptions & {
    plugins: Extract<ParserOptions['plugins'], any[]>;
  };
}
type ReactBabelHook = (babelConfig: ReactBabelOptions, context: ReactBabelHookContext, config: ResolvedConfig) => void;
type ReactBabelHookContext = {
  ssr: boolean;
  id: string;
};
type ViteReactPluginApi = {
  /**
   * Manipulate the Babel options of `@vitejs/plugin-react`
   */
  reactBabel?: ReactBabelHook;
};
declare function viteReact(opts?: Options): Plugin[];
declare namespace viteReact {
  var preambleCode: string;
}
declare function viteReactForCjs(this: unknown, options: Options): Plugin[];
//#endregion
export { BabelOptions, Options, ReactBabelOptions, ViteReactPluginApi, viteReact as default, viteReactForCjs as "module.exports" };
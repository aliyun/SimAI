//#region src/lens-list.d.ts
declare class LensList<T> {
  #private;
  constructor();
  get [Symbol.iterator](): any;
  entries(): MapIterator<[string, T[]]>;
  /**
   * Return an order-sensitive list of values by the given key.
   */
  get(key: string): Array<T>;
  /**
   * Return an order-sensitive list of all values.
   */
  getAll(): Array<T>;
  /**
   * Append a new value to the given key.
   */
  append(key: string, value: T): void;
  /**
   * Prepend a new value to the given key.
   */
  prepend(key: string, value: T): void;
  /**
   * Delete the value belonging to the given key.
   */
  delete(key: string, value: T): void;
  /**
   * Delete all values belogning to the given key.
   */
  deleteAll(key: string): void;
  get size(): number;
  clear(): void;
}
//#endregion
export { LensList };
//# sourceMappingURL=lens-list.d.mts.map
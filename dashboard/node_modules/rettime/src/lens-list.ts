export class LensList<T> {
  #list: Array<[string, T]>
  #lens: Map<string, Array<T>>

  constructor() {
    this.#list = []
    this.#lens = new Map()
  }

  get [Symbol.iterator]() {
    // Return the list's iterator so iteration is order-sensitive.
    return this.#list[Symbol.iterator].bind(this.#list)
  }

  public entries() {
    return this.#lens.entries()
  }

  /**
   * Return an order-sensitive list of values by the given key.
   */
  public get(key: string): Array<T> {
    return this.#lens.get(key) || []
  }

  /**
   * Return an order-sensitive list of all values.
   */
  public getAll(): Array<T> {
    return this.#list.map(([, value]) => value)
  }

  /**
   * Append a new value to the given key.
   */
  public append(key: string, value: T): void {
    this.#list.push([key, value])
    this.#openLens(key, (list) => list.push(value))
  }

  /**
   * Prepend a new value to the given key.
   */
  public prepend(key: string, value: T): void {
    this.#list.unshift([key, value])
    this.#openLens(key, (list) => list.unshift(value))
  }

  /**
   * Delete the value belonging to the given key.
   */
  public delete(key: string, value: T): void {
    if (this.size === 0) {
      return
    }

    this.#list = this.#list.filter((item) => item[1] !== value)

    for (const [existingKey, values] of this.#lens) {
      if (existingKey === key && values.includes(value)) {
        values.splice(values.indexOf(value), 1)
      }
    }
  }

  /**
   * Delete all values belogning to the given key.
   */
  public deleteAll(key: string): void {
    if (this.size === 0) {
      return
    }

    this.#list = this.#list.filter((item) => item[0] !== key)
    this.#lens.delete(key)
  }

  get size(): number {
    return this.#list.length
  }

  public clear(): void {
    if (this.size === 0) {
      return
    }

    this.#list.length = 0
    this.#lens.clear()
  }

  #openLens(key: string, setter: (target: Array<T>) => void): void {
    setter(this.#lens.get(key) || this.#lens.set(key, []).get(key))
  }
}

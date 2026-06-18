//#region src/index.d.ts
type DefaultEventMap = {
  [eventType: string]: TypedEvent<any, any>;
};
/**
 * Reserved event map containing special event types like '*' for catch-all listeners.
 */
type ReservedEventMap = {
  '*': TypedEvent<any, any, '*'>;
};
type IsReservedEvent<Type extends string> = Type extends keyof ReservedEventMap ? true : false;
interface TypedEvent<DataType = void, ReturnType = void, EventType extends string = string> extends Omit<MessageEvent<DataType>, 'type'> {
  type: EventType;
}
declare const kDefaultPrevented: unique symbol;
declare const kPropagationStopped: unique symbol;
declare const kImmediatePropagationStopped: unique symbol;
declare class TypedEvent<DataType = void, ReturnType = void, EventType extends string = string> extends MessageEvent<DataType> implements TypedEvent<DataType, ReturnType, EventType> {
  #private;
  [kDefaultPrevented]: boolean;
  [kPropagationStopped]?: Emitter<any>;
  [kImmediatePropagationStopped]?: boolean;
  constructor(...args: [DataType] extends [void] ? [type: EventType] : [type: EventType, init: {
    data: DataType;
  }]);
  get defaultPrevented(): boolean;
  preventDefault(): void;
  stopImmediatePropagation(): void;
}
/**
 * Brands a TypedEvent or its subclass while preserving its (narrower) type.
 */
type Brand<Event extends TypedEvent, EventType extends string, Loose extends boolean = false> = Loose extends true ? Event extends TypedEvent<infer Data, any, any> ?
/**
* @note Omit the `ReturnType` so emit methods can accept type events
* where infering the return type is impossible.
*/
TypedEvent<Data, any, EventType> & {
  type: EventType;
} : never : Event & {
  type: EventType;
};
type InferEventMap<Target extends Emitter<any>> = Target extends Emitter<infer EventMap> ? MergedEventMap<EventMap> : never;
/**
 * Extracts only user-defined events, excluding reserved event types.
 */
type UserEventMap<EventMap$1 extends DefaultEventMap> = Omit<EventMap$1, keyof ReservedEventMap>;
/**
 * Merges the user EventMap with the ReservedEventMap.
 * The '*' event type accepts a union of all user-defined events.
 */
type MergedEventMap<EventMap$1 extends DefaultEventMap> = EventMap$1 & ReservedEventMap;
/**
 * Creates a union of all events in the EventMap with their literal type strings.
 */
type AllEvents<EventMap$1 extends DefaultEventMap> = { [K in keyof EventMap$1 & string]: Brand<EventMap$1[K], K> }[keyof EventMap$1 & string];
type TypedListenerOptions = {
  once?: boolean;
  signal?: AbortSignal;
};
declare namespace Emitter {
  /**
   * Returns an appropriate `Event` type for the given event type.
   *
   * @example
   * const emitter = new Emitter<{ greeting: TypedEvent<string> }>()
   * type GreetingEvent = Emitter.InferEventType<typeof emitter, 'greeting'>
   * // TypedEvent<string>
   */
  type EventType<Target extends Emitter<any>, EventType extends keyof EventMap$1 & string, EventMap$1 extends DefaultEventMap = InferEventMap<Target>> = IsReservedEvent<EventType> extends true ? AllEvents<UserEventMap<EventMap$1>> : Brand<EventMap$1[EventType], EventType>;
  type EventDataType<Target extends Emitter<any>, EventType extends keyof EventMap$1 & string, EventMap$1 extends DefaultEventMap = InferEventMap<Target>> = EventMap$1[EventType] extends TypedEvent<infer DataType> ? DataType : never;
  /**
   * Returns the listener type for the given event type.
   *
   * @example
   * const emitter = new Emitter<{ getTotalPrice: TypedEvent<Cart, number> }>()
   * type Listener = Emitter.ListenerType<typeof emitter, 'getTotalPrice'>
   * // (event: TypedEvent<Cart>) => number
   */
  type ListenerType<Target extends Emitter<any>, EventType extends keyof EventMap$1 & string, EventMap$1 extends DefaultEventMap = InferEventMap<Target>> = IsReservedEvent<EventType> extends true ? (event: AllEvents<UserEventMap<EventMap$1>>) => void : (event: Emitter.EventType<Target, EventType, EventMap$1>) => Emitter.ListenerReturnType<Target, EventType, EventMap$1> extends [void] ? void : Emitter.ListenerReturnType<Target, EventType, EventMap$1>;
  /**
   * Returns the return type of the listener for the given event type.
   *
   * @example
   * const emitter = new Emitter<{ getTotalPrice: TypedEvent<Cart, number> }>()
   * type ListenerReturnType = Emitter.InferListenerReturnType<typeof emitter, 'getTotalPrice'>
   * // number
   */
  type ListenerReturnType<Target extends Emitter<any>, EventType extends keyof EventMap$1 & string, EventMap$1 extends DefaultEventMap = InferEventMap<Target>> = IsReservedEvent<EventType> extends true ? void : EventMap$1[EventType] extends TypedEvent<unknown, infer ReturnType> ? ReturnType : never;
}
declare class Emitter<EventMap$1 extends DefaultEventMap> {
  #private;
  constructor();
  /**
   * Adds a listener for the given event type.
   */
  on<EventType extends keyof MergedEventMap<EventMap$1> & string>(type: EventType, listener: Emitter.ListenerType<typeof this, EventType, MergedEventMap<EventMap$1>>, options?: TypedListenerOptions): typeof this;
  /**
   * Adds a one-time listener for the given event type.
   */
  once<EventType extends keyof MergedEventMap<EventMap$1> & string>(type: EventType, listener: Emitter.ListenerType<typeof this, EventType, MergedEventMap<EventMap$1>>, options?: Omit<TypedListenerOptions, 'once'>): typeof this;
  /**
   * Prepends a listener for the given event type.
   */
  earlyOn<EventType extends keyof MergedEventMap<EventMap$1> & string>(type: EventType, listener: Emitter.ListenerType<typeof this, EventType, MergedEventMap<EventMap$1>>, options?: TypedListenerOptions): typeof this;
  /**
   * Prepends a one-time listener for the given event type.
   */
  earlyOnce<EventType extends keyof MergedEventMap<EventMap$1> & string>(type: EventType, listener: Emitter.ListenerType<typeof this, EventType, MergedEventMap<EventMap$1>>, options?: Omit<TypedListenerOptions, 'once'>): typeof this;
  /**
   * Emits the given typed event.
   *
   * @returns {boolean} Returns `true` if the event had any listeners, `false` otherwise.
   */
  emit<EventType extends keyof EventMap$1 & string>(event: Brand<EventMap$1[EventType], EventType, true>): boolean;
  /**
   * Emits the given typed event and returns a promise that resolves
   * when all the listeners for that event have settled.
   *
   * @returns {Promise<Array<Emitter.ListenerReturnType>>} A promise that resolves
   * with the return values of all listeners.
   */
  emitAsPromise<EventType extends keyof EventMap$1 & string>(event: Brand<EventMap$1[EventType], EventType, true>): Promise<Array<Emitter.ListenerReturnType<typeof this, EventType, EventMap$1>>>;
  /**
   * Emits the given event and returns a generator that yields
   * the result of each listener in the order of their registration.
   * This way, you stop exhausting the listeners once you get the expected value.
   */
  emitAsGenerator<EventType extends keyof EventMap$1 & string>(event: Brand<EventMap$1[EventType], EventType, true>): Generator<Emitter.ListenerReturnType<typeof this, EventType, EventMap$1>>;
  /**
   * Removes a listener for the given event type.
   */
  removeListener<EventType extends keyof MergedEventMap<EventMap$1> & string>(type: EventType, listener: Emitter.ListenerType<typeof this, EventType, MergedEventMap<EventMap$1>>): void;
  /**
   * Removes all listeners for the given event type.
   * If no event type is provided, removes all existing listeners.
   */
  removeAllListeners<EventType extends keyof MergedEventMap<EventMap$1> & string>(type?: EventType): void;
  /**
   * Returns the list of listeners for the given event type.
   * If no even type is provided, returns all listeners.
   */
  listeners<EventType extends keyof MergedEventMap<EventMap$1> & string>(type?: EventType): Array<Emitter.ListenerType<typeof this, EventType, MergedEventMap<EventMap$1>>>;
  /**
   * Returns the number of listeners for the given event type.
   * If no even type is provided, returns the total number of listeners.
   */
  listenerCount<EventType extends keyof MergedEventMap<EventMap$1> & string>(type?: EventType): number;
}
//#endregion
export { DefaultEventMap, Emitter, ReservedEventMap, TypedEvent, TypedListenerOptions };
//# sourceMappingURL=index.d.mts.map
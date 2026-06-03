import { LensList } from "./lens-list.mjs";

//#region src/index.ts
const kDefaultPrevented = Symbol("kDefaultPrevented");
const kPropagationStopped = Symbol("kPropagationStopped");
const kImmediatePropagationStopped = Symbol("kImmediatePropagationStopped");
var TypedEvent = class extends MessageEvent {
	/**
	* @note Keep a placeholder property with the return type
	* because the type must be set somewhere in order to be
	* correctly associated and inferred from the event.
	*/
	#returnType;
	[kDefaultPrevented];
	[kPropagationStopped];
	[kImmediatePropagationStopped];
	constructor(...args) {
		super(args[0], args[1]);
		this[kDefaultPrevented] = false;
	}
	get defaultPrevented() {
		return this[kDefaultPrevented];
	}
	preventDefault() {
		super.preventDefault();
		this[kDefaultPrevented] = true;
	}
	stopImmediatePropagation() {
		/**
		* @note Despite `.stopPropagation()` and `.stopImmediatePropagation()` being defined
		* in Node.js, they do nothing. It is safe to re-define them.
		*/
		super.stopImmediatePropagation();
		this[kImmediatePropagationStopped] = true;
	}
};
const kListenerOptions = Symbol("kListenerOptions");
var Emitter = class {
	#listeners;
	constructor() {
		this.#listeners = new LensList();
	}
	/**
	* Adds a listener for the given event type.
	*/
	on(type, listener, options) {
		this.#addListener(type, listener, options);
		return this;
	}
	/**
	* Adds a one-time listener for the given event type.
	*/
	once(type, listener, options) {
		return this.on(type, listener, {
			...options || {},
			once: true
		});
	}
	/**
	* Prepends a listener for the given event type.
	*/
	earlyOn(type, listener, options) {
		this.#addListener(type, listener, options, "prepend");
		return this;
	}
	/**
	* Prepends a one-time listener for the given event type.
	*/
	earlyOnce(type, listener, options) {
		return this.earlyOn(type, listener, {
			...options || {},
			once: true
		});
	}
	/**
	* Emits the given typed event.
	*
	* @returns {boolean} Returns `true` if the event had any listeners, `false` otherwise.
	*/
	emit(event) {
		if (this.#listeners.size === 0) return false;
		/**
		* @note Calculate matching listeners before calling them
		* since one-time listeners will self-destruct.
		*/
		const hasListeners = this.listenerCount(event.type) > 0;
		const proxiedEvent = this.#proxyEvent(event);
		for (const listener of this.#matchListeners(event.type)) {
			if (proxiedEvent.event[kPropagationStopped] != null && proxiedEvent.event[kPropagationStopped] !== this) {
				proxiedEvent.revoke();
				return false;
			}
			if (proxiedEvent.event[kImmediatePropagationStopped]) break;
			this.#callListener(proxiedEvent.event, listener);
		}
		proxiedEvent.revoke();
		return hasListeners;
	}
	/**
	* Emits the given typed event and returns a promise that resolves
	* when all the listeners for that event have settled.
	*
	* @returns {Promise<Array<Emitter.ListenerReturnType>>} A promise that resolves
	* with the return values of all listeners.
	*/
	async emitAsPromise(event) {
		if (this.#listeners.size === 0) return [];
		const pendingListeners = [];
		const proxiedEvent = this.#proxyEvent(event);
		for (const listener of this.#matchListeners(event.type)) {
			if (proxiedEvent.event[kPropagationStopped] != null && proxiedEvent.event[kPropagationStopped] !== this) {
				proxiedEvent.revoke();
				return [];
			}
			if (proxiedEvent.event[kImmediatePropagationStopped]) break;
			const returnValue = await Promise.resolve(this.#callListener(proxiedEvent.event, listener));
			if (!this.#isTypelessListener(listener)) pendingListeners.push(returnValue);
		}
		proxiedEvent.revoke();
		return Promise.allSettled(pendingListeners).then((results) => {
			return results.map((result) => result.status === "fulfilled" ? result.value : result.reason);
		});
	}
	/**
	* Emits the given event and returns a generator that yields
	* the result of each listener in the order of their registration.
	* This way, you stop exhausting the listeners once you get the expected value.
	*/
	*emitAsGenerator(event) {
		if (this.#listeners.size === 0) return;
		const proxiedEvent = this.#proxyEvent(event);
		for (const listener of this.#matchListeners(event.type)) {
			if (proxiedEvent.event[kPropagationStopped] != null && proxiedEvent.event[kPropagationStopped] !== this) {
				proxiedEvent.revoke();
				return;
			}
			if (proxiedEvent.event[kImmediatePropagationStopped]) break;
			const returnValue = this.#callListener(proxiedEvent.event, listener);
			if (!this.#isTypelessListener(listener)) yield returnValue;
		}
		proxiedEvent.revoke();
	}
	/**
	* Removes a listener for the given event type.
	*/
	removeListener(type, listener) {
		this.#listeners.delete(type, listener);
	}
	/**
	* Removes all listeners for the given event type.
	* If no event type is provided, removes all existing listeners.
	*/
	removeAllListeners(type) {
		if (type == null) {
			this.#listeners.clear();
			return;
		}
		this.#listeners.deleteAll(type);
	}
	/**
	* Returns the list of listeners for the given event type.
	* If no even type is provided, returns all listeners.
	*/
	listeners(type) {
		if (type == null) return this.#listeners.getAll();
		return this.#listeners.get(type);
	}
	/**
	* Returns the number of listeners for the given event type.
	* If no even type is provided, returns the total number of listeners.
	*/
	listenerCount(type) {
		if (type == null) return this.#listeners.size;
		return this.listeners(type).length;
	}
	#addListener(type, listener, options, insertMode = "append") {
		if (insertMode === "prepend") this.#listeners.prepend(type, listener);
		else this.#listeners.append(type, listener);
		if (options) {
			Object.defineProperty(listener, kListenerOptions, {
				value: options,
				enumerable: false,
				writable: false
			});
			if (options.signal) options.signal.addEventListener("abort", () => {
				this.removeListener(type, listener);
			}, { once: true });
		}
	}
	#proxyEvent(event) {
		const { stopPropagation } = event;
		event.stopPropagation = new Proxy(event.stopPropagation, { apply: (target, thisArg, argArray) => {
			event[kPropagationStopped] = this;
			return Reflect.apply(target, thisArg, argArray);
		} });
		return {
			event,
			revoke() {
				event.stopPropagation = stopPropagation;
			}
		};
	}
	#callListener(event, listener) {
		const returnValue = listener.call(this, event);
		if (listener[kListenerOptions]?.once) {
			const key = this.#isTypelessListener(listener) ? "*" : event.type;
			this.#listeners.delete(key, listener);
		}
		return returnValue;
	}
	/**
	* Return a list of all event listeners relevant for the given event type.
	* This includes the explicit event listeners and also typeless event listeners.
	*/
	*#matchListeners(type) {
		for (const [key, listener] of this.#listeners) if (key === "*" || key === type) yield listener;
	}
	#isTypelessListener(listener) {
		return this.#listeners.get("*").includes(listener);
	}
};

//#endregion
export { Emitter, TypedEvent };
//# sourceMappingURL=index.mjs.map